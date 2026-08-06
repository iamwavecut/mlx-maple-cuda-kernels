# Copyright © 2026 DeepGrove AI.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

# Absolute imports so this file also works standalone when shipped inside a
# checkpoint and loaded via the config's `model_file` (trust_remote_code).
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

# SwiGLU clamp for the MoE experts only (the dense MapleMLP is unclamped);
# part of the trained forward pass, not an optional guard.
MLP_CLAMP = 7.0


@dataclass(frozen=True)
class _CudaProfile:
    name: str
    elementwise_threads: int = 128
    router_threads: int = 256
    router_rows_per_warp: int = 4
    router_reference_gemv: bool = False


_UNSET = object()
_kernel_backend_cache = _UNSET
_cuda_capability_cache = _UNSET
_cuda_profile_cache = _UNSET

# The fused router is numerically close but not array-exact: its normalized
# scores can eventually change a greedy token.  It is therefore an explicit
# semantic/experimental lane, never part of strict auto mode.
_use_approximate_router = False
_use_approximate_add_rms = False


def _kernel_backend():
    global _kernel_backend_cache
    if _kernel_backend_cache is not _UNSET:
        return _kernel_backend_cache
    try:
        if mx.metal.is_available():
            _kernel_backend_cache = "metal"
            return _kernel_backend_cache
    except (AttributeError, RuntimeError):
        pass
    try:
        if mx.cuda.is_available():
            _kernel_backend_cache = "cuda"
            return _kernel_backend_cache
    except (AttributeError, RuntimeError):
        pass
    _kernel_backend_cache = None
    return _kernel_backend_cache


def _cuda_capability():
    global _cuda_capability_cache
    if _cuda_capability_cache is not _UNSET:
        return _cuda_capability_cache
    if _kernel_backend() != "cuda":
        _cuda_capability_cache = None
        return None
    try:
        info = mx.device_info(mx.gpu)
        _cuda_capability_cache = (
            int(info["compute_capability_major"]),
            int(info["compute_capability_minor"]),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        _cuda_capability_cache = None
    return _cuda_capability_cache


def _cuda_profile_for_capability(capability):
    major, minor = capability
    if (major, minor) < (8, 6):
        return None
    for target, name in (
        ((12, 0), "sm120"),
        ((10, 0), "sm100"),
        ((9, 0), "sm90"),
        ((8, 9), "sm89"),
        ((8, 6), "sm86"),
    ):
        if (major, minor) == target:
            if target == (8, 6):
                return _CudaProfile(
                    name,
                    elementwise_threads=256,
                    router_threads=128,
                    router_rows_per_warp=2,
                )
            if target == (8, 9):
                return _CudaProfile(
                    name,
                    elementwise_threads=512,
                    router_threads=128,
                    router_rows_per_warp=1,
                )
            if target in ((9, 0), (10, 0)):
                return _CudaProfile(
                    name,
                    elementwise_threads=512,
                    router_threads=512,
                    router_rows_per_warp=1,
                    router_reference_gemv=True,
                )
            if target == (12, 0):
                return _CudaProfile(
                    name,
                    elementwise_threads=512,
                    router_threads=256,
                    router_rows_per_warp=1,
                    router_reference_gemv=True,
                )
            return _CudaProfile(name, router_reference_gemv=target >= (9, 0))
    return _CudaProfile("future", router_reference_gemv=True)


def _cuda_profile():
    global _cuda_profile_cache
    if _cuda_profile_cache is not _UNSET:
        return _cuda_profile_cache
    capability = _cuda_capability()
    _cuda_profile_cache = (
        None if capability is None else _cuda_profile_for_capability(capability)
    )
    return _cuda_profile_cache


def _cuda_router_grid_size(num_experts, profile):
    block_rows = profile.router_threads // 32 * profile.router_rows_per_warp
    if num_experts % block_rows:
        raise ValueError("router profile does not evenly cover every expert")
    return num_experts // block_rows * profile.router_threads


@partial(mx.compile, shapeless=True)
def clamped_swiglu(gate, x):
    # Python floats, not 0-d arrays, so bf16 activations stay bf16.
    return nn.silu(mx.minimum(gate, MLP_CLAMP)) * mx.clip(x, -MLP_CLAMP, MLP_CLAMP)


def _matches(fast, reference):
    """Fail closed unless a hand-written kernel is array-exact.

    Every fast path below has a portable equivalent and is enabled only after
    a one-time comparison on live weights.  A tolerance check is not a strict
    correctness gate: tiny activation differences can eventually change a
    greedy token on long or near-tied generations.  Shape, dtype, and every
    value must therefore match before the fast path can latch on.  Unsupported
    devices or future MLX changes fall back to the portable implementation.
    """
    try:
        got, want = fast(), reference()
        mx.eval(got, want)
    except Exception:
        return False
    return len(got) == len(want) and all(
        g.shape == w.shape
        and g.dtype == w.dtype
        and bool(mx.array_equal(g, w))
        for g, w in zip(got, want)
    )


class MapleRMSNorm(nn.Module):
    """RMSNorm with the weight multiply in float32.

    The reference rounds only the finished product; mx.fast.rms_norm rounds
    the normalized activation first (~1% per element). Float32 inputs to the
    same kernel reproduce the reference bit-for-bit.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32), self.eps
        ).astype(x.dtype)


def _make_add_rms_norm_kernel(eps):
    """Residual add + RMSNorm in ONE dispatch for single-token decode.

    Emits both h = x + r (the residual stream, rounded once like a bf16 add)
    and hn = rmsnorm(h) with the weight multiply in fp32 (reference
    semantics, identical to MapleRMSNorm). Folding the add into the norm and
    skipping the astype round-trips replaces ~4 dispatches with 1, and the
    decode step is bounded by its serial dispatch chain, not by this math.
    """
    source = """
        uint tid = thread_position_in_threadgroup.x;
        constexpr uint N = DIM;
        constexpr uint PT = N / 256u;
        float hb[PT];
        float ss = 0.0f;
        for (uint i = 0; i < PT; ++i) {
            uint j = tid * PT + i;
            float v = (float)x[j] + (float)r[j];
            T_ vb = (T_)v;              // one rounding, same as a bf16 add
            h_out[j] = vb;
            hb[i] = (float)vb;          // norm sees the rounded stream
            ss += hb[i] * hb[i];
        }
        ss = simd_sum(ss);
        threadgroup float sums[8];
        uint sg = tid / 32u;
        uint lane = tid % 32u;
        if (lane == 0u) sums[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8u; ++i) tot += sums[i];
        float scale = metal::rsqrt(tot / (float)N + EPS_);
        for (uint i = 0; i < PT; ++i) {
            uint j = tid * PT + i;
            hn_out[j] = (T_)(hb[i] * scale * (float)w[j]);
        }
    """.replace("EPS_", f"{eps:.10e}f")
    tag = f"{eps:.3e}".replace(".", "_").replace("-", "m").replace("+", "p")
    return mx.fast.metal_kernel(
        name=f"maple_add_rms_norm_{tag}",
        input_names=["x", "r", "w"],
        output_names=["h_out", "hn_out"],
        source=source,
    )


def _make_cuda_add_rms_norm_kernel(eps, profile):
    """CUDA residual add + RMSNorm for one decode token."""
    source = """
        constexpr int VECTOR_WIDTH = 4;
        constexpr int CHUNKS = DIM / (THREADS * VECTOR_WIDTH);
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        float values[CHUNKS][VECTOR_WIDTH];
        float local_ss = 0.0f;

        #pragma unroll
        for (int chunk = 0; chunk < CHUNKS; ++chunk) {
            int base = (chunk * THREADS + tid) * VECTOR_WIDTH;
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; ++i) {
                int j = base + i;
                float value = static_cast<float>(x[j]) + static_cast<float>(r[j]);
                T_ rounded = static_cast<T_>(value);
                h_out[j] = rounded;
                values[chunk][i] = static_cast<float>(rounded);
                local_ss += values[chunk][i] * values[chunk][i];
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
        }

        __shared__ float warp_sums[THREADS / 32];
        if (lane == 0) warp_sums[warp] = local_ss;
        __syncthreads();

        float total = lane < THREADS / 32 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        __shared__ float block_sum;
        if (tid == 0) block_sum = total;
        __syncthreads();
        total = block_sum;
        float scale = rsqrtf(total / static_cast<float>(DIM) + EPSILON);

        #pragma unroll
        for (int chunk = 0; chunk < CHUNKS; ++chunk) {
            int base = (chunk * THREADS + tid) * VECTOR_WIDTH;
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; ++i) {
                int j = base + i;
                hn_out[j] = static_cast<T_>(
                    values[chunk][i] * scale * static_cast<float>(w[j]));
            }
        }
    """.replace("EPSILON", f"{eps:.10e}f")
    tag = f"{eps:.3e}".replace(".", "_").replace("-", "m").replace("+", "p")
    return mx.fast.cuda_kernel(
        name=f"maple_add_rms_norm_{profile.name}_{tag}",
        input_names=["x", "r", "w"],
        output_names=["h_out", "hn_out"],
        source=source,
    )


_add_rms_kernel_cache = {}
_last_add_rms_backend = None


def _add_rms_norm(h, r, w, eps):
    global _last_add_rms_backend
    backend = _kernel_backend()
    if backend == "metal":
        threads = 256
        key = (backend, eps)

        def make_kernel():
            return _make_add_rms_norm_kernel(eps)

    elif backend == "cuda":
        profile = _cuda_profile()
        if profile is None:
            raise RuntimeError("unsupported CUDA architecture")
        threads = profile.elementwise_threads
        key = (backend, profile.name, eps)

        def make_kernel():
            return _make_cuda_add_rms_norm_kernel(eps, profile)

    else:
        raise RuntimeError("no residual add + RMSNorm kernel for this backend")
    alignment = threads * 4 if backend == "cuda" else threads
    if h.shape[-1] % alignment:
        raise ValueError("unsupported residual add + RMSNorm dimension")
    kernel = _add_rms_kernel_cache.get(key)
    if kernel is None:
        kernel = _add_rms_kernel_cache[key] = make_kernel()
    template = [("T_", h.dtype), ("DIM", h.shape[-1])]
    if backend == "cuda":
        template.extend([("W_", w.dtype), ("THREADS", threads)])
    outputs = kernel(
        inputs=[h.reshape(-1), r.reshape(-1), w],
        template=template,
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[h.dtype, h.dtype],
    )
    _last_add_rms_backend = backend
    return outputs


def _add_rms_norm_ok(dim, dtype, w, eps):
    backend = _kernel_backend()
    if backend == "cuda":
        profile = _cuda_profile()
        if profile is None or dim % (profile.elementwise_threads * 4):
            return False
    elif backend == "metal":
        if dim % 256:
            return False
    else:
        return False
    x = mx.random.normal((1, 1, dim), key=mx.random.key(0)).astype(dtype)
    r = mx.random.normal((1, 1, dim), key=mx.random.key(1)).astype(dtype)
    return _matches(
        lambda: _add_rms_norm(x, r, w, eps),
        lambda: (
            x + r,
            mx.fast.rms_norm(
                (x + r).astype(mx.float32), w.astype(mx.float32), eps
            ).astype(dtype),
        ),
    )


# Inlined rather than imported from switch_layers: those helpers are private
# (underscore-prefixed), and this file must keep loading against whatever
# mlx-lm a user has installed when it ships inside a checkpoint.
def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "maple"
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 140000
    vocab_size: int = 151936
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    use_qk_norm: bool = True
    use_bias: bool = False
    tie_word_embeddings: bool = False
    # FlashHead metadata written by `mlx_lm.ternary --flash-head`. The exact
    # lm_head is the default; opt in to the approximate fast head with
    # mlx_lm.load(..., model_config={"use_flash_head": True}).
    flash_head: Optional[dict] = None
    use_flash_head: bool = False
    # Populated from the checkpoint's config; sanitize() reads group_size from
    # it to expand row-scale (`row_alpha`) ternary tensors.
    quantization: Optional[dict] = None

    def __post_init__(self):
        # Single source of truth for per-layer attention types: attention
        # (RoPE/NoPE), masks, and caches all read this resolved list.
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


def _make_qk_norm_rope_kernel():
    """Fused per-head RMSNorm + partial RoPE for single-token decode.

    One dispatch replaces q_norm, k_norm and two rope calls. One simdgroup per
    head: normalize head_dim values, scale by the head's norm weight, and
    rotate the first ROPE_DIM dims (non-traditional pairing i, i+R/2) at the
    given position. NoPE layers pass ROPE_DIM=0.
    """
    source = """
        uint head = thread_position_in_grid.y;
        uint lane = thread_position_in_grid.x;

        constexpr int per_lane = HEAD_DIM / 32;
        const device T_* xh = x + head * HEAD_DIM;
        const device W_* wh = w + head * HEAD_DIM;
        device T_* oh = out + head * HEAD_DIM;

        float ss = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            float v = (float)xh[lane * per_lane + i];
            ss += v * v;
        }
        ss = simd_sum(ss);
        float pos = pos_eps[0];
        float eps = pos_eps[1];
        float scale = metal::rsqrt(ss / HEAD_DIM + eps);

        for (int i = 0; i < per_lane; ++i) {
            int j = lane * per_lane + i;
            float v = (float)xh[j] * scale * (float)wh[j];
            if (ROPE_DIM > 0 && j < ROPE_DIM) {
                constexpr int rhalf = ROPE_DIM > 0 ? ROPE_DIM / 2 : 1;
                int p = j < rhalf ? j : j - rhalf;
                float theta = pos * inv_freq[p];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                int j2 = j < rhalf ? j + rhalf : j - rhalf;
                float u = (float)xh[j2] * scale * (float)wh[j2];
                v = j < rhalf ? (v * c - u * s) : (v * c + u * s);
            }
            oh[j] = (T_)v;
        }
    """
    return mx.fast.metal_kernel(
        name="maple_qk_norm_rope",
        input_names=["x", "w", "inv_freq", "pos_eps"],
        output_names=["out"],
        source=source,
    )


def _make_cuda_qk_norm_rope_kernel(profile, use_rope):
    """CUDA Q/K RMSNorm with separate RoPE and NoPE kernel bodies."""
    if use_rope:
        # NVRTC on Blackwell otherwise contracts the sine product, while MLX's
        # stock RoPE contracts the cosine product.  Pin that one rounding
        # boundary explicitly so the fused kernel stays array-exact.
        second_half = (
            "__fmaf_rn(value, rope_cos[p], "
            "__fmul_rn(paired, rope_sin[p]))"
            if profile.name in ("sm100", "sm120")
            else "paired * rope_sin[p] + value * rope_cos[p]"
        )
        transform = """
        __shared__ float rope_sin[ROPE_DIM / 2];
        __shared__ float rope_cos[ROPE_DIM / 2];
        for (int p = lane; p < ROPE_DIM / 2; p += 32) {
            float fraction = static_cast<float>(p)
                / static_cast<float>(ROPE_DIM / 2);
            float frequency = exp2f(-fraction * pos_eps[2]);
            float angle = pos_eps[0] * frequency;
            rope_cos[p] = cosf(angle);
            rope_sin[p] = sinf(angle);
        }
        __syncwarp();

        #pragma unroll
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            int j = lane * VALUES_PER_LANE + i;
            T_ normalized = static_cast<T_>(
                static_cast<float>(xh[j]) * scale * static_cast<float>(wh[j]));
            float value = static_cast<float>(normalized);
            if (j < ROPE_DIM) {
                constexpr int HALF = ROPE_DIM / 2;
                int p = j < HALF ? j : j - HALF;
                int pair = j < HALF ? j + HALF : j - HALF;
                T_ paired_normalized = static_cast<T_>(
                    static_cast<float>(xh[pair]) * scale
                    * static_cast<float>(wh[pair]));
                float paired = static_cast<float>(paired_normalized);
                value = j < HALF
                    ? value * rope_cos[p] - paired * rope_sin[p]
                    : MAPLE_SECOND_HALF;
            }
            oh[j] = static_cast<T_>(value);
        }
        """.replace("MAPLE_SECOND_HALF", second_half)
    else:
        transform = """
        #pragma unroll
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            int j = lane * VALUES_PER_LANE + i;
            oh[j] = static_cast<T_>(
                static_cast<float>(xh[j]) * scale * static_cast<float>(wh[j]));
        }
        """

    source = (
        """
        int head = blockIdx.y;
        int lane = threadIdx.x;
        constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
        const T_* xh = x + head * HEAD_DIM;
        const W_ * wh = w + head * HEAD_DIM;
        T_* oh = out + head * HEAD_DIM;

        float sum_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            float value = static_cast<float>(xh[lane * VALUES_PER_LANE + i]);
            sum_sq += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);
        float scale = rsqrtf(sum_sq / static_cast<float>(HEAD_DIM) + pos_eps[1]);
    """
        + transform
    )
    return mx.fast.cuda_kernel(
        name=f"maple_qk_norm_{'rope' if use_rope else 'nope'}_{profile.name}",
        input_names=["x", "w", "inv_freq", "pos_eps"],
        output_names=["out"],
        source=source,
    )


_qk_kernel_cache = {}


def _get_qk_norm_rope_kernel(use_rope):
    backend = _kernel_backend()
    if backend == "metal":
        key = (backend,)
        make_kernel = _make_qk_norm_rope_kernel
    elif backend == "cuda":
        profile = _cuda_profile()
        if profile is None:
            raise RuntimeError("unsupported CUDA architecture")
        key = (backend, profile.name, use_rope)

        def make_kernel():
            return _make_cuda_qk_norm_rope_kernel(profile, use_rope)

    else:
        raise RuntimeError("no Q/K norm + RoPE kernel for this backend")
    kernel = _qk_kernel_cache.get(key)
    if kernel is None:
        kernel = _qk_kernel_cache[key] = make_kernel()
    return kernel


class MapleAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        # q/k/v are stored fused (one matmul per step); sanitize() concatenates
        # the checkpoint's split projections.
        self.qkv_proj = nn.Linear(
            args.hidden_size,
            (args.num_attention_heads + 2 * args.num_key_value_heads) * self.head_dim,
            bias=args.use_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.q_norm = MapleRMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = MapleRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._eps = args.rms_norm_eps
        self._rope_base = args.rope_theta
        self._rope_log2_base = math.log2(args.rope_theta)
        self._qk_w = None
        self._inv_freq = None
        self._fused_qk = None  # None = unprobed, then True/False
        self._fused_qk_backend = None

        # Maple applies RoPE only on sliding-window layers; full-attention
        # layers use no positional encoding (NoPE).
        self.use_rope = args.layer_types[layer_idx] == "sliding_attention"
        # The custom kernel implements the checkpoint's plain RoPE exactly.
        # Scaling policies have different position math and must stay on MLX's
        # portable implementation until they receive their own specialization.
        self._can_fuse_qk = not (self.use_rope and args.rope_scaling is not None)
        if self.use_rope:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
            self.rope = initialize_rope(
                rope_dim,
                args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def _qk_fused(self, qk, offset):
        """Both norms and both rope applications in one dispatch."""
        if not self._can_fuse_qk:
            raise ValueError("scaled RoPE is unsupported by the fused Q/K kernel")
        if self._qk_w is None:
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            self._qk_w = mx.contiguous(
                mx.concatenate(
                    [
                        mx.broadcast_to(self.q_norm.weight[None], (n_q, self.head_dim)),
                        mx.broadcast_to(
                            self.k_norm.weight[None], (n_kv, self.head_dim)
                        ),
                    ]
                )
            )
            if self.use_rope:
                half = self.rope.dims // 2
                self._inv_freq = self._rope_base ** (
                    -mx.arange(half, dtype=mx.float32) / half
                )
            else:
                self._inv_freq = mx.ones((1,), dtype=mx.float32)
            mx.eval(self._qk_w, self._inv_freq)

        # cache.offset is a Python int for a plain cache but an mx.array for
        # the batched caches; coerce so the pos/eps pair is always uniform.
        pos_eps = mx.array(
            [float(offset), self._eps, self._rope_log2_base], dtype=mx.float32
        )
        backend = _kernel_backend()
        template = [
            ("T_", qk.dtype),
            ("HEAD_DIM", self.head_dim),
            ("ROPE_DIM", self.rope.dims if self.use_rope else 0),
        ]
        template.append(("W_", self._qk_w.dtype))
        output = _get_qk_norm_rope_kernel(self.use_rope)(
            inputs=[qk, self._qk_w, self._inv_freq, pos_eps],
            template=template,
            grid=(32, qk.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[qk.shape],
            output_dtypes=[qk.dtype],
        )[0]
        self._fused_qk_backend = backend
        return output

    def _qk_reference(self, qk, offset):
        """The same result from stock ops: fallback, and the yardstick the
        fused kernel is checked against."""
        n_q = self.num_attention_heads
        q = self.q_norm(qk[None, :n_q, None, :])
        k = self.k_norm(qk[None, n_q:, None, :])
        if self.use_rope:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        return mx.concatenate([q, k], axis=1).reshape(qk.shape)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)

        if B == 1 and L == 1 and self.use_qk_norm:
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            qk_size = (n_q + n_kv) * self.head_dim
            qk = qkv.reshape(-1)[:qk_size].reshape(n_q + n_kv, self.head_dim)
            if self._fused_qk is None:
                # A nonzero position, so a broken rotation cannot pass.
                self._fused_qk = self._can_fuse_qk and _matches(
                    lambda: (self._qk_fused(qk, 7),),
                    lambda: (self._qk_reference(qk, 7),),
                )
            offset = cache.offset if cache is not None else 0
            out = (self._qk_fused if self._fused_qk else self._qk_reference)(qk, offset)
            queries = out[:n_q].reshape(1, n_q, 1, self.head_dim)
            keys = out[n_q:].reshape(1, n_kv, 1, self.head_dim)
            values = qkv.reshape(-1)[qk_size:].reshape(1, n_kv, 1, self.head_dim)
        else:
            q_size = self.num_attention_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

            queries = q.reshape(B, L, self.num_attention_heads, self.head_dim)
            keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
            values = v.reshape(B, L, self.num_key_value_heads, self.head_dim)

            if self.use_qk_norm:
                queries = self.q_norm(queries)
                keys = self.k_norm(keys)

            queries = queries.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            values = values.transpose(0, 2, 1, 3)

            if self.use_rope:
                offset = cache.offset if cache is not None else 0
                queries = self.rope(queries, offset=offset)
                keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MapleMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        # Dense / shared-expert MLP: no clamp; only the MoE experts clamp.
        # Unused at first_k_dense_replace=0 with no shared experts, but keep
        # it faithful.
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def group_expert_select(gates, top_k):
    # Maple routes with a plain softmax over all experts followed by top-k
    # selection and renormalization, computed in float32.
    scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(scores, inds, axis=-1)
    scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, scores


def _make_fused_router_kernel():
    """Router gemv + softmax + top-8 + renormalize in ONE dispatch (+18%).

    Replaces ~6 kernels per layer. NE/32 threadgroups each compute 32 logits,
    keep them in float32 (`router_dtype: fp32`), and publish through an
    atomic-float scratch (plain device stores are not reliably visible across
    threadgroups on Apple GPUs); the last threadgroup to arrive does the
    softmax + top-8 + renorm.

    `ctr_in` is a persistent arrival counter, not an input: every dispatch
    must see it at zero, so the electing threadgroup resets it on its way out
    and each MapleGate keeps its own. Election on a stale counter would read
    unwritten scratch, so nothing else may share the buffer.
    """
    source = """
    constexpr uint NE = NEXP;
    constexpr uint D = DIM;
    constexpr uint NTG = NE / 32u;
    constexpr uint TM = 4u;
    constexpr uint TN = 4u;
    constexpr uint BLOCKN = 32u * TN;
    constexpr uint NITER = D / BLOCKN;

    uint tid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint n_threads = 256u;
    uint sg_id = tid / 32u;
    uint lane = tid % 32u;
    uint n_sg = n_threads / 32u;

    uint row0 = tgid * (n_sg * TM) + sg_id * TM;
    float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint bn = lane * TN;
    for (uint i = 0u; i < NITER; ++i) {
        float v[TN];
        for (uint tn = 0u; tn < TN; ++tn) v[tn] = float(x[bn + tn]);
        for (uint tm = 0u; tm < TM; ++tm) {
            const device T_* wrow = w + (ulong)(row0 + tm) * D;
            T_ inter[TN];
            for (uint tn = 0u; tn < TN; ++tn) inter[tn] = wrow[bn + tn];
            for (uint tn = 0u; tn < TN; ++tn) result[tm] += inter[tn] * v[tn];
        }
        bn += BLOCKN;
    }
    for (uint tm = 0u; tm < TM; ++tm) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    device atomic_float* ls = (device atomic_float*)logits_scratch;
    if (lane == 0u) {
        for (uint tm = 0u; tm < TM; ++tm) {
            atomic_store_explicit(&ls[row0 + tm], result[tm],
                                  memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint last_flag;
    if (tid == 0u) {
        device atomic_uint* ctr = (device atomic_uint*)ctr_in;
        uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        uint last = (prev == NTG - 1u) ? 1u : 0u;
        if (last == 1u) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
        last_flag = last;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_flag == 0u) return;
    threadgroup_barrier(mem_flags::mem_device);

    float my_max = -1e30f;
    for (uint e = tid; e < NE; e += n_threads) {
        float v = atomic_load_explicit(&ls[e], memory_order_relaxed);
        if (v > my_max) my_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = simd_shuffle_down(my_max, off);
        if (other > my_max) my_max = other;
    }
    threadgroup float sg_red[16];
    if (lane == 0u) sg_red[sg_id] = my_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = sg_red[0];
        for (uint s = 1u; s < n_sg; s++) if (sg_red[s] > m) m = sg_red[s];
        sg_red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lmax = sg_red[0];

    threadgroup float scores[NE];
    float my_sum = 0.0f;
    for (uint e = tid; e < NE; e += n_threads) {
        float lv = atomic_load_explicit(&ls[e], memory_order_relaxed);
        float v = metal::exp(lv - lmax);
        scores[e] = v;
        my_sum += v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        my_sum += simd_shuffle_down(my_sum, off);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sg_red[sg_id] = my_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float ssum = sg_red[0];
        for (uint i = 1u; i < n_sg; i++) ssum += sg_red[i];
        sg_red[0] = ssum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_total = 1.0f / (sg_red[0] + 1e-20f);
    for (uint e = tid; e < NE; e += n_threads) {
        scores[e] = scores[e] * inv_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup int topk_idx[8];
    threadgroup float topk_val[8];
    threadgroup uint8_t used[NE];
    for (uint e = tid; e < NE; e += n_threads) used[e] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < 8; k++) {
        float my_best = -1e30f;
        int my_idx = 0;
        for (int e = int(tid); e < int(NE); e += int(n_threads)) {
            if (!used[e] && scores[e] > my_best) {
                my_best = scores[e];
                my_idx = e;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float other_v = simd_shuffle_down(my_best, off);
            int other_i = simd_shuffle_down(my_idx, off);
            if (other_v > my_best) { my_best = other_v; my_idx = other_i; }
        }
        threadgroup float sg_vals[16];
        threadgroup int sg_idxs[16];
        if (lane == 0u) { sg_vals[sg_id] = my_best; sg_idxs[sg_id] = my_idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float bv = sg_vals[0]; int bi = sg_idxs[0];
            for (uint s = 1u; s < n_sg; s++) {
                if (sg_vals[s] > bv) { bv = sg_vals[s]; bi = sg_idxs[s]; }
            }
            topk_val[k] = bv; topk_idx[k] = bi;
            used[bi] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 8u) {
        float sel_sum = 0.0f;
        for (int i = 0; i < 8; i++) sel_sum += topk_val[i];
        out_indices[tid] = topk_idx[tid];
        out_scores[tid] = float(topk_val[tid] / (sel_sum + 1e-20f));
    }
"""
    return mx.fast.metal_kernel(
        name="maple_fused_router",
        input_names=["x", "w", "ctr_in"],
        output_names=["out_indices", "out_scores", "logits_scratch"],
        source=source,
    )


def _make_cuda_router_kernel(profile):
    """CUDA router GEMV and selected-only top-8 normalization."""
    source = """
        constexpr int VALUES_PER_LANE = 4;
        constexpr int WARP_WIDTH = 32;
        constexpr int COLS_PER_ITER = WARP_WIDTH * VALUES_PER_LANE;
        constexpr int WARPS = ROUTER_THREADS / WARP_WIDTH;
        constexpr int BLOCK_ROWS = WARPS * ROWS_PER_WARP;
        constexpr int NUM_BLOCKS = NEXP / BLOCK_ROWS;

        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        int row0 = blockIdx.x * BLOCK_ROWS + warp * ROWS_PER_WARP;
        float accum[ROWS_PER_WARP] = {0.0f};

        for (int col = lane * VALUES_PER_LANE; col < DIM; col += COLS_PER_ITER) {
            float xv[VALUES_PER_LANE];
            #pragma unroll
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                xv[i] = static_cast<float>(x[col + i]);
            }
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                const auto* wrow = w + static_cast<long long>(row0 + row) * DIM;
                #pragma unroll
                for (int i = 0; i < VALUES_PER_LANE; ++i) {
                    accum[row] += static_cast<float>(wrow[col + i]) * xv[i];
                }
            }
        }

        #pragma unroll
        for (int row = 0; row < ROWS_PER_WARP; ++row) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                accum[row] += __shfl_down_sync(0xffffffff, accum[row], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                logits_scratch[row0 + row] = accum[row];
            }
        }

        __syncthreads();
        __threadfence();
        __syncthreads();
        __shared__ int is_last;
        if (tid == 0) {
            unsigned int* counter = (unsigned int*)ctr_out;
            unsigned int previous = atomicAdd(counter, 1u);
            is_last = previous == static_cast<unsigned int>(NUM_BLOCKS - 1);
        }
        __syncthreads();
        if (!is_last) return;

        volatile float* logits = logits_scratch;
        __shared__ unsigned char used[NEXP];
        __shared__ float warp_values[WARPS];
        __shared__ int warp_indices[WARPS];
        __shared__ float selected_values[8];
        __shared__ int selected_indices[8];
        __shared__ float selected_exp[8];
        __shared__ float selected_sum;

        for (int expert = tid; expert < NEXP; expert += ROUTER_THREADS) {
            used[expert] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int pick = 0; pick < 8; ++pick) {
            float best = -3.402823466e+38F;
            int best_idx = -1;
            for (int expert = tid; expert < NEXP; expert += ROUTER_THREADS) {
                if (!used[expert]) {
                    float value = logits[expert];
                    if (value > best || (value == best && expert < best_idx)) {
                        best = value;
                        best_idx = expert;
                    }
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_value = __shfl_down_sync(0xffffffff, best, offset);
                int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
                if (other_idx >= 0 &&
                    (other_value > best ||
                     (other_value == best && (best_idx < 0 || other_idx < best_idx)))) {
                    best = other_value;
                    best_idx = other_idx;
                }
            }
            if (lane == 0) {
                warp_values[warp] = best;
                warp_indices[warp] = best_idx;
            }
            __syncthreads();
            if (tid == 0) {
                float block_best = warp_values[0];
                int block_idx = warp_indices[0];
                #pragma unroll
                for (int candidate = 1; candidate < WARPS; ++candidate) {
                    float value = warp_values[candidate];
                    int index = warp_indices[candidate];
                    if (index >= 0 &&
                        (value > block_best ||
                         (value == block_best && index < block_idx))) {
                        block_best = value;
                        block_idx = index;
                    }
                }
                selected_values[pick] = block_best;
                selected_indices[pick] = block_idx;
                used[block_idx] = 1;
            }
            __syncthreads();
        }

        if (tid < 8) {
            selected_exp[tid] = expf(selected_values[tid] - selected_values[0]);
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) sum += selected_exp[i];
            selected_sum = sum + 1e-20f;
        }
        __syncthreads();
        if (tid < 8) {
            // MLX argpartition exposes the selected suffix in ascending
            // score order.  Expert aggregation is not bitwise commutative,
            // so preserve that order exactly.
            int output_pick = 7 - tid;
            out_indices[tid] = selected_indices[output_pick];
            out_scores[tid] = selected_exp[output_pick] / selected_sum;
        }
        __syncthreads();
        if (tid == 0) {
            atomicExch((unsigned int*)ctr_out, 0u);
            __threadfence();
        }
    """
    return mx.fast.cuda_kernel(
        name=f"maple_fused_router_{profile.name}_safe_counter",
        input_names=["x", "w"],
        output_names=[
            "out_indices", "out_scores", "logits_scratch", "ctr_out"
        ],
        source=source,
    )


def _make_cuda_router_select_kernel(profile):
    """CUDA softmax/top-8/renorm over logits produced by MLX matmul.

    Hopper and newer MLX matmuls may use an architecture-specific FP32/TF32
    accumulation path. Reusing that exact GEMV preserves expert selection;
    this kernel still collapses the remaining softmax and selection chain to
    one dispatch.
    """
    source = """
        constexpr int WARP_WIDTH = 32;
        constexpr int WARPS = ROUTER_THREADS / WARP_WIDTH;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;

        __shared__ unsigned char used[NEXP];
        __shared__ float warp_values[WARPS];
        __shared__ int warp_indices[WARPS];
        __shared__ float selected_values[8];
        __shared__ int selected_indices[8];
        __shared__ float selected_exp[8];
        __shared__ float selected_sum;

        for (int expert = tid; expert < NEXP; expert += ROUTER_THREADS) {
            used[expert] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int pick = 0; pick < 8; ++pick) {
            float best = -3.402823466e+38F;
            int best_idx = -1;
            for (int expert = tid; expert < NEXP; expert += ROUTER_THREADS) {
                if (!used[expert]) {
                    float value = logits[expert];
                    if (value > best ||
                        (value == best && (best_idx < 0 || expert < best_idx))) {
                        best = value;
                        best_idx = expert;
                    }
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_value = __shfl_down_sync(0xffffffff, best, offset);
                int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
                if (other_idx >= 0 &&
                    (other_value > best ||
                     (other_value == best && (best_idx < 0 || other_idx < best_idx)))) {
                    best = other_value;
                    best_idx = other_idx;
                }
            }
            if (lane == 0) {
                warp_values[warp] = best;
                warp_indices[warp] = best_idx;
            }
            __syncthreads();
            if (tid == 0) {
                float block_best = warp_values[0];
                int block_idx = warp_indices[0];
                #pragma unroll
                for (int candidate = 1; candidate < WARPS; ++candidate) {
                    float value = warp_values[candidate];
                    int index = warp_indices[candidate];
                    if (index >= 0 &&
                        (value > block_best ||
                         (value == block_best && index < block_idx))) {
                        block_best = value;
                        block_idx = index;
                    }
                }
                selected_values[pick] = block_best;
                selected_indices[pick] = block_idx;
                used[block_idx] = 1;
            }
            __syncthreads();
        }

        if (tid < 8) {
            selected_exp[tid] = expf(selected_values[tid] - selected_values[0]);
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) sum += selected_exp[i];
            selected_sum = sum + 1e-20f;
        }
        __syncthreads();
        if (tid < 8) {
            // MLX argpartition returns its selected suffix from the smallest
            // to the largest score. Preserve that order: expert aggregation
            // is mathematically commutative, but its FP32 reduction is not.
            int output_pick = 7 - tid;
            out_indices[tid] = selected_indices[output_pick];
            out_scores[tid] = selected_exp[output_pick] / selected_sum;
        }
    """
    return mx.fast.cuda_kernel(
        name=f"maple_router_select_{profile.name}",
        input_names=["logits"],
        output_names=["out_indices", "out_scores"],
        source=source,
    )


_router_kernel_cache = {}
_router_select_kernel_cache = {}


def _get_router_select_kernel():
    profile = _cuda_profile()
    if profile is None or not profile.router_reference_gemv:
        raise RuntimeError("router selection kernel requires a modern CUDA profile")
    key = (profile.name, profile.router_threads)
    kernel = _router_select_kernel_cache.get(key)
    if kernel is None:
        kernel = _router_select_kernel_cache[key] = _make_cuda_router_select_kernel(
            profile
        )
    return kernel


def _get_fused_router_kernel():
    backend = _kernel_backend()
    if backend == "metal":
        key = (backend,)
        make_kernel = _make_fused_router_kernel
    elif backend == "cuda":
        profile = _cuda_profile()
        if profile is None:
            raise RuntimeError("unsupported CUDA architecture")
        key = (
            backend,
            profile.name,
            profile.router_threads,
            profile.router_rows_per_warp,
        )

        def make_kernel():
            return _make_cuda_router_kernel(profile)

    else:
        raise RuntimeError("no fused router kernel for this backend")
    kernel = _router_kernel_cache.get(key)
    if kernel is None:
        kernel = _router_kernel_cache[key] = make_kernel()
    return kernel


class MapleGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.hidden_size = args.hidden_size
        # Kept as a raw parameter (not nn.Linear) so quantization never
        # touches it. The matmul accumulates in float32 and selection runs on
        # float32 scores.
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self._router_ctr = None
        self._fused = None  # None = unprobed, then True/False
        self._fused_backend = None

    def _fused_call(self, x):
        backend = _kernel_backend()
        profile = _cuda_profile() if backend == "cuda" else None
        threads = profile.router_threads if profile is not None else 256
        rows_per_warp = profile.router_rows_per_warp if profile is not None else 4
        if profile is not None and profile.router_reference_gemv:
            if self.top_k != 8:
                raise ValueError("unsupported fused router top-k")
            logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
            inds, scores = _get_router_select_kernel()(
                inputs=[logits.reshape(-1)],
                template=[
                    ("NEXP", self.num_experts),
                    ("ROUTER_THREADS", threads),
                ],
                grid=(threads, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[(8,), (8,)],
                output_dtypes=[(
                    mx.uint32
                    if backend == "cuda" and _cuda_router_indices_uint32
                    else mx.int32
                ), mx.float32],
            )
            self._fused_backend = backend
            shape = x.shape[:-1] + (self.top_k,)
            return inds.reshape(shape), scores.reshape(shape)
        block_rows = threads // 32 * rows_per_warp
        if self.top_k != 8 or self.num_experts % block_rows or self.hidden_size % 128:
            raise ValueError("unsupported fused router dimensions")
        grid_size = (
            (self.num_experts // 32) * 256
            if backend == "metal"
            else _cuda_router_grid_size(self.num_experts, profile)
        )
        template = [
            ("T_", self.weight.dtype),
            ("NEXP", self.num_experts),
            ("DIM", self.hidden_size),
            ("ROUTER_THREADS", threads),
            ("ROWS_PER_WARP", rows_per_warp),
        ]
        index_dtype = (
            mx.uint32
            if backend == "cuda" and _cuda_router_indices_uint32
            else mx.int32
        )
        if backend == "cuda":
            # The multi-block completion counter is a zero-initialized output,
            # not hidden mutable input state.  This makes graph dependencies
            # explicit and concurrent calls independent.
            inds, scores, _, _ = _get_fused_router_kernel()(
                inputs=[x.reshape(-1), self.weight],
                template=template,
                grid=(grid_size, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[(8,), (8,), (self.num_experts,), (1,)],
                output_dtypes=[
                    index_dtype, mx.float32, mx.float32, mx.uint32
                ],
                init_value=0,
            )
        else:
            if self._router_ctr is None:
                self._router_ctr = mx.zeros((8,), dtype=mx.uint32)
                mx.eval(self._router_ctr)
            inds, scores, _ = _get_fused_router_kernel()(
                inputs=[x.reshape(-1), self.weight, self._router_ctr],
                template=template,
                grid=(grid_size, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[(8,), (8,), (self.num_experts,)],
                output_dtypes=[index_dtype, mx.float32, mx.float32],
            )
        self._fused_backend = backend
        shape = x.shape[:-1] + (self.top_k,)
        return inds.reshape(shape), scores.reshape(shape)

    def _reference(self, x):
        # `router_dtype: fp32`. In bf16 the near-tied top-8 boundary flips a
        # few percent of picks per layer, which compounds over 24 layers.
        gates = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(gates, self.top_k)

    def _probe(self, x):
        # Not _matches(): the two paths may order the selected experts
        # differently, and an exact tie at the top-k boundary may legitimately
        # pick either of the tied experts. Compare the sorted score vectors,
        # and bound-check the ids since a bad one indexes the expert gather.
        try:
            inds, scores = self._fused_call(x)
            ref_inds, ref_scores = self._reference(x)
            mx.eval(inds, scores, ref_inds, ref_scores)
        except Exception:
            return False
        valid = inds.shape == ref_inds.shape and bool(
            mx.all((inds >= 0) & (inds < self.num_experts))
        )
        # Expert order and routing weights both affect the ordered fp32 MoE
        # reduction.  An allclose score is not sufficient for strict decode:
        # the fixed common-slice regression found eventual greedy-token
        # divergence from sub-ULP routing drift.  Keep this experimental
        # router available for explicit studies, but auto mode must fall back
        # unless it reproduces the portable selector exactly.
        return (
            valid
            and inds.dtype == ref_inds.dtype
            and scores.dtype == ref_scores.dtype
            and bool(mx.array_equal(inds, ref_inds))
            and bool(mx.array_equal(scores, ref_scores))
        )

    def __call__(self, x):
        if not _use_approximate_router:
            self._fused = False
            return self._reference(x)
        if self._fused is not False and x.size == self.hidden_size:
            if self._fused is None:
                self._fused = self._probe(x)
            if self._fused:
                return self._fused_call(x)
        return self._reference(x)


@partial(mx.compile, shapeless=True)
def aggregate_expert_outputs(expert_outputs, scores):
    # Combined in float32, rounded once at the end (reference `moe_infer`).
    return (
        (expert_outputs.astype(mx.float32) * scores[..., None])
        .sum(axis=-2)
        .astype(expert_outputs.dtype)
    )



# Maple Preview's checkpoint stores true ternary expert weights: codes 0/1/2,
# one alpha per output row, and affine bias == -alpha.  This decode-only
# specialization is deliberately Maple-owned; generic affine W2 layers still
# use MLX GatherQMM.
_TERNARY_UP_GATE_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int WARPS = THREADS / WARP;
    constexpr int ROWS_PER_BLOCK = WARPS * ROWS_PER_WARP;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int slot = blockIdx.y;
    unsigned int expert_u = static_cast<unsigned int>(rhs_indices[slot]);
    int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;

    if (expert_u >= static_cast<unsigned int>(NUM_EXPERTS)) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                int row = row0 + r;
                if (row < N) out[slot * N + row] = static_cast<T_>(0);
            }
        }
        return;
    }
    int expert = static_cast<int>(expert_u);
    if (row0 >= N) return;

    float sums[ROWS_PER_WARP] = {0.0f};
    float alpha[ROWS_PER_WARP] = {0.0f};
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = row0 + r;
        if (row < N) {
            long long scale_off =
                (static_cast<long long>(expert) * N + row) * GROUPS;
            alpha[r] = static_cast<float>(scales[scale_off]);
        }
    }

    for (int col = lane; col < K; col += WARP) {
        float xv = static_cast<float>(x[col]);
        int word = col >> 4;
        int shift = (col & 15) << 1;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = row0 + r;
            if (row < N) {
                long long weight_off =
                    (static_cast<long long>(expert) * N + row) * WORDS + word;
                unsigned int code = (weights[weight_off] >> shift) & 3u;
                float weight = (static_cast<int>(code) - 1) * alpha[r];
                sums[r] = fmaf(xv, weight, sums[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], offset);
        }
        if (lane == 0 && row0 + r < N) {
            out[slot * N + row0 + r] = static_cast<T_>(sums[r]);
        }
    }
"""

_cuda_ternary_up_gate_kernel = None
_use_cuda_ternary_up_gate = False


def _get_cuda_ternary_up_gate_kernel():
    global _cuda_ternary_up_gate_kernel
    if _cuda_ternary_up_gate_kernel is None:
        _cuda_ternary_up_gate_kernel = mx.fast.cuda_kernel(
            name="maple_ternary_up_gate_sm86",
            input_names=["x", "weights", "scales", "rhs_indices"],
            output_names=["out"],
            source=_TERNARY_UP_GATE_SOURCE,
        )
    return _cuda_ternary_up_gate_kernel


def _cuda_ternary_up_gate(x, layer, indices):
    threads = 256
    rows_per_warp = 2
    k = layer.input_dims
    n = layer.output_dims
    rows_per_block = (threads // 32) * rows_per_warp
    blocks_x = (n + rows_per_block - 1) // rows_per_block
    return _get_cuda_ternary_up_gate_kernel()(
        inputs=[
            x.reshape(-1),
            layer["weight"],
            layer["scales"],
            indices.reshape(-1),
        ],
        template=[
            ("T_", x.dtype),
            ("K", k),
            ("N", n),
            ("NUM_EXPERTS", layer.num_experts),
            ("WORDS", k // 16),
            ("GROUPS", k // layer.group_size),
            ("ROWS_PER_WARP", rows_per_warp),
            ("THREADS", threads),
        ],
        grid=(blocks_x * threads, indices.size, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[tuple(indices.shape) + (1, n)],
        output_dtypes=[x.dtype],
    )[0]


_decode_lhs_cache = {}
# Exact candidates retained for explicit experiments. Cached LHS showed no
# independently significant marginal win in the within-process factorial;
# uint32 indices only affect the approximate router. Keep both off by default.
_use_cached_decode_lhs = False
_cuda_router_indices_uint32 = False


def _decode_lhs_indices(top_k):
    indices = _decode_lhs_cache.get(top_k)
    if indices is None:
        indices = (
            mx.zeros((top_k,), dtype=mx.uint32),
            mx.arange(top_k, dtype=mx.uint32),
        )
        mx.eval(*indices)
        _decode_lhs_cache[top_k] = indices
    return indices


class MapleSwitchGLU(nn.Module):
    """SwitchGLU with the up and gate projections fused into one gather
    matmul; sanitize() concatenates the checkpoint's split tensors."""

    def __init__(self, input_dims, hidden_dims, num_experts, bias=False):
        super().__init__()
        self.up_gate_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self._ternary_row_alpha = False
        self._ternary_up_gate = None  # None = unprobed, then True/False

    def _ternary_up_gate_eligible(self, x, indices, single_token):
        layer = self.up_gate_proj
        profile = _cuda_profile()
        return (
            _use_cuda_ternary_up_gate
            and single_token
            and not self.training
            and profile is not None
            and profile.name == "sm86"
            and self._ternary_row_alpha
            and isinstance(layer, QuantizedSwitchLinear)
            and x.dtype == mx.bfloat16
            and x.shape[-1] == 2048
            and indices.dtype == mx.uint32
            and indices.size == 8
            and indices.shape[-1] == 8
            and layer.bits == 2
            and layer.mode == "affine"
            and layer.group_size == 128
            and "bias" not in layer
            and layer.weight.dtype == mx.uint32
            and layer.scales.dtype == mx.bfloat16
            and layer.biases is not None
            and layer.biases.dtype == mx.bfloat16
            and layer.weight.shape == (256, 1024, 128)
            and layer.scales.shape == (256, 1024, 16)
            and layer.biases.shape == (256, 1024, 16)
        )

    def _up_gate(self, x, indices, single_token, sorted_indices, lhs_indices):
        def stock():
            return self.up_gate_proj(
                x,
                indices,
                sorted_indices=sorted_indices,
                lhs_indices=lhs_indices,
            )

        if not self._ternary_up_gate_eligible(x, indices, single_token):
            return stock()

        try:
            candidate = _cuda_ternary_up_gate(x, self.up_gate_proj, indices)
            if self._ternary_up_gate is None:
                reference = stock()
                weight = self.up_gate_proj.weight
                code3 = (weight & (weight >> 1)) & mx.array(
                    0x55555555, dtype=mx.uint32
                )
                no_code3 = mx.all(code3 == 0)
                indices_in_bounds = mx.all(indices < self.up_gate_proj.num_experts)
                same = mx.array_equal(candidate, reference)
                mx.eval(candidate, reference, no_code3, indices_in_bounds, same)
                self._ternary_up_gate = (
                    candidate.shape == reference.shape
                    and candidate.dtype == reference.dtype
                    and bool(no_code3)
                    and bool(indices_in_bounds)
                    and bool(same)
                )
                return candidate if self._ternary_up_gate else reference
            if self._ternary_up_gate:
                return candidate
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._ternary_up_gate = False
        return stock()

    def __call__(self, x, indices):
        single_token = x.size == self.up_gate_proj.input_dims and indices.size == 8
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        up_lhs = down_lhs = None
        if single_token and _use_cached_decode_lhs:
            up_lhs, down_lhs = _decode_lhs_indices(indices.shape[-1])
        x_up, x_gate = mx.split(
            self._up_gate(x, idx, single_token, do_sort, up_lhs),
            2,
            axis=-1,
        )
        x = self.down_proj(
            clamped_swiglu(x_gate, x_up),
            idx,
            sorted_indices=do_sort,
            lhs_indices=down_lhs,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)

class MapleSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MapleGate(args)
        self.switch_mlp = MapleSwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        return aggregate_expert_outputs(y, scores)


class MapleDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MapleAttention(args, layer_idx)
        self.mlp = (
            MapleSparseMoeBlock(args)
            if layer_idx >= args.first_k_dense_replace
            else MapleMLP(args)
        )
        self.input_layernorm = MapleRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MapleModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MapleDecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = MapleRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.layer_types = args.layer_types
        self.window_size = args.sliding_window
        self.swa_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )
        self.ga_idx = (
            self.layer_types.index("full_attention")
            if "full_attention" in self.layer_types
            else None
        )
        self._fused_add_norm = None  # None = unprobed, then True/False
        self._zero = None

    def _decode_fused(self, h, cache, full_mask, swa_mask):
        """Decode loop with residual adds folded into the norms.

        Carries (h, r) instead of adding r back each step, so every
        add+norm pair is one dispatch. Identical arithmetic: the kernel
        rounds the sum once (as the bf16 add did) and norms the rounded
        stream with an fp32 weight multiply.
        """
        if self._zero is None:
            self._zero = mx.zeros(h.shape, h.dtype)
            mx.eval(self._zero)
        r = self._zero  # x + 0 is exact in bf16
        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            ln = layer.input_layernorm
            h, hn = _add_rms_norm(h, r, ln.weight, ln.eps)
            r = layer.self_attn(hn, mask, c)
            ln = layer.post_attention_layernorm
            h, hn = _add_rms_norm(h, r, ln.weight, ln.eps)
            r = layer.mlp(hn)
        return _add_rms_norm(h, r, self.norm.weight, self.norm.eps)[1]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        swa_mask = None
        if self.ga_idx is not None:
            full_mask = create_attention_mask(h, cache[self.ga_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.window_size
            )

        if h.size == h.shape[-1]:
            if not _use_approximate_add_rms:
                # The fused residual carrier passes local array-equality probes
                # but diverged on a deterministic long-generation regression.
                # Keep it in the explicit semantic lane until its full 49-norm
                # chain is exact for every decode activation.
                self._fused_add_norm = False
            elif self._fused_add_norm is None:
                self._fused_add_norm = _add_rms_norm_ok(
                    h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                )
            if self._fused_add_norm:
                return self._decode_fused(h, cache, full_mask, swa_mask)

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class FlashHead(nn.Module):
    """Two-phase approximate lm_head for single-stream decode.

    Phase one scores quantized cluster centroids of the vocabulary; phase two
    computes exact logits only for the tokens of the top ``n_probes`` clusters
    (plus a fixed set of forced control tokens such as EOS). All other logits
    are -inf, so greedy decoding is exact whenever the true argmax lies in the
    probed clusters. Prefill and batched calls use the exact lm_head.

    Reference: FlashHead — Efficient Drop-in Replacement for the
    Classification Head in Language Model Inference.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        meta = args.flash_head
        if not meta.get("scaled_centroids"):
            raise ValueError(
                "FlashHead metadata predates scaled centroids; regenerate with "
                "`python -m mlx_lm.ternary <checkpoint> --flash-head-only`."
            )
        n_clusters = meta["n_clusters"]
        cluster_size = meta["cluster_size"]
        # Default matches the converter's `--probes` default; every generated
        # checkpoint records the value explicitly.
        self.n_probes = min(meta.get("n_probes", 512), n_clusters)
        self.head_group_size = meta.get("head_group_size", 64)
        self.head_bits = meta.get("head_bits", 4)
        # Centroids are directions, pre-scaled at generation time by the
        # largest lm_head row norm in their cluster: that upper-bounds the
        # cluster's best logit, so high-frequency small-norm tokens are still
        # probed, and scoring stays a single matmul.
        self.centroids = nn.QuantizedLinear(
            args.hidden_size,
            n_clusters,
            bias=False,
            group_size=meta.get("group_size", 64),
            bits=meta.get("bits", 4),
        )
        self.token_map = mx.zeros((n_clusters, cluster_size), dtype=mx.int32)
        # Cluster-ordered copy of the quantized lm_head: subset logits are one
        # gather_qmm over the probed 32-row blocks, with no per-step gather.
        # It is a row-permutation of lm_head by token_map and nothing more, so
        # it is derived rather than stored: Model.sanitize rebuilds it at load.
        hidden = args.hidden_size
        self.head = {
            "weight": mx.zeros(
                (n_clusters, cluster_size, hidden * self.head_bits // 32),
                dtype=mx.uint32,
            ),
            "scales": mx.zeros(
                (n_clusters, cluster_size, hidden // self.head_group_size),
                dtype=mx.bfloat16,
            ),
            "biases": mx.zeros(
                (n_clusters, cluster_size, hidden // self.head_group_size),
                dtype=mx.bfloat16,
            ),
        }
        self._force_ids = mx.array(meta.get("force_tokens", []), dtype=mx.int32)
        self._force_rows = None

    def __call__(self, h: mx.array, lm_head: nn.Module) -> mx.array:
        hv = h[:, -1, :]
        top = mx.argpartition(self.centroids(hv), kth=-self.n_probes, axis=-1)[
            ..., -self.n_probes :
        ]  # [1, n_probes]
        oids = self.token_map[top[0]].reshape(-1)

        logits = mx.gather_qmm(
            hv.reshape(1, 1, 1, 1, -1),
            self.head["weight"],
            self.head["scales"],
            self.head["biases"],
            rhs_indices=top[:, None, :],
            transpose=True,
            group_size=self.head_group_size,
            bits=self.head_bits,
        ).reshape(-1)

        if self._force_ids.size:
            if self._force_rows is None:
                self._force_rows = (
                    lm_head.weight[self._force_ids],
                    lm_head.scales[self._force_ids],
                    lm_head.biases[self._force_ids],
                )
                mx.eval(*self._force_rows)
            fw, fs, fb = self._force_rows
            force_logits = mx.quantized_matmul(
                hv,
                fw,
                scales=fs,
                biases=fb,
                transpose=True,
                group_size=lm_head.group_size,
                bits=lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            )[0]
            oids = mx.concatenate([oids, self._force_ids])
            logits = mx.concatenate([logits, force_logits])

        vocab_size = lm_head.weight.shape[0]
        full = mx.full((1, 1, vocab_size), float("-inf"), dtype=logits.dtype)
        full[0, 0, oids] = logits
        return full


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MapleModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.flash_head and args.use_flash_head and not args.tie_word_embeddings:
            self.lm_head_flash = FlashHead(args)
        else:
            self.lm_head_flash = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        if (
            self.lm_head_flash is not None
            and out.shape[0] == 1
            and out.shape[1] == 1
            and isinstance(self.lm_head, nn.QuantizedLinear)
            and getattr(self.lm_head, "mode", "affine") == "affine"
        ):
            return self.lm_head_flash(out, self.lm_head)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            # Drop the head entirely (weight + quantization scales/biases).
            weights = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}

        # FlashHead disabled (e.g. model_config={"flash_head": None}): drop its
        # tensors so checkpoints that carry them still load.
        if self.lm_head_flash is None:
            weights = {
                k: v for k, v in weights.items() if not k.startswith("lm_head_flash.")
            }
        else:
            # Folded into the centroid rows at generation time; older shards
            # still carry the tensor.
            weights.pop("lm_head_flash.cluster_scale", None)
            # `lm_head_flash.head.*` is lm_head permuted by token_map (see
            # mlx_lm.ternary.generate_flash_head), so it is pure redundancy on
            # disk. Checkpoints may ship it or omit it; reconcile both here.
            if "lm_head_flash.head.weight" not in weights:
                token_map = weights["lm_head_flash.token_map"]
                order = token_map.reshape(-1)
                for k in ("weight", "scales", "biases"):
                    weights[f"lm_head_flash.head.{k}"] = weights[f"lm_head.{k}"][
                        order
                    ].reshape(*token_map.shape, -1)

        # Ternary tensors carry one scale per output row, so checkpoints store
        # it once as `row_alpha` and omit biases entirely (bias == -scale).
        # Expand here so everything downstream — fusion below, and mlx's own
        # quantized kernels — sees the per-group layout. Checkpoints written
        # with `--group-scales` have no row_alpha and pass straight through.
        row_alpha_keys = [k for k in weights if k.endswith(".row_alpha")]
        row_alpha_prefixes = {
            k[: -len(".row_alpha")] for k in row_alpha_keys
        }
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            switch_mlp = getattr(
                getattr(self.model.layers[layer_idx], "mlp", None),
                "switch_mlp",
                None,
            )
            if switch_mlp is None:
                continue
            split = (
                f"{prefix}.switch_mlp.up_proj" in row_alpha_prefixes
                and f"{prefix}.switch_mlp.gate_proj" in row_alpha_prefixes
            )
            fused = (
                f"{prefix}.switch_mlp.up_gate_proj" in row_alpha_prefixes
            )
            expert_split = all(
                f"{prefix}.experts.{expert}.up_proj" in row_alpha_prefixes
                and f"{prefix}.experts.{expert}.gate_proj" in row_alpha_prefixes
                for expert in range(self.args.num_experts)
            )
            switch_mlp._ternary_row_alpha = split or fused or expert_split

        if row_alpha_keys:
            group_size = (self.args.quantization or {}).get("group_size", 128)
            for key in row_alpha_keys:
                alpha = weights.pop(key)
                prefix = key[: -len(".row_alpha")]
                packed = weights.get(f"{prefix}.weight")
                if packed is None:
                    continue
                # 2-bit packing stores 16 codes per uint32 word.
                n_groups = (packed.shape[-1] * 16) // group_size
                scales = mx.contiguous(
                    mx.broadcast_to(alpha[..., None], (*alpha.shape, n_groups))
                )
                weights[f"{prefix}.scales"] = scales
                weights[f"{prefix}.biases"] = -scales

        # Stack per-expert weights from the Hugging Face layout into the
        # SwitchGLU layout. Already-converted checkpoints pass through.
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases", "bias"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            # Fuse split projections: q/k/v -> qkv_proj (rows), MoE up/gate ->
            # up_gate_proj (per-expert rows). Row-wise quantized tensors
            # (weight/scales/biases) concatenate losslessly along the output
            # axis.
            for suffix in ["weight", "scales", "biases", "bias"]:
                qkv = [
                    f"{prefix}.self_attn.{p}.{suffix}"
                    for p in ("q_proj", "k_proj", "v_proj")
                ]
                if qkv[0] in weights:
                    weights[f"{prefix}.self_attn.qkv_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(k) for k in qkv], axis=0
                    )
                up = f"{prefix}.mlp.switch_mlp.up_proj.{suffix}"
                gate = f"{prefix}.mlp.switch_mlp.gate_proj.{suffix}"
                if up in weights:
                    weights[f"{prefix}.mlp.switch_mlp.up_gate_proj.{suffix}"] = (
                        mx.concatenate([weights.pop(up), weights.pop(gate)], axis=1)
                    )

        return weights

    def make_cache(self):
        caches = []
        for layer_type in self.model.layer_types:
            if layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("lm_head") or "word_embeddings" in path:
                return {"group_size": 64, "bits": 4}
            return True

        return predicate
