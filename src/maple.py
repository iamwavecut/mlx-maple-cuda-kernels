# Copyright © 2026 DeepGrove AI.

import math
import os
import weakref
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

def _env_flag(name, default):
    """Let a deployment choose a lane without importing and patching this module.

    The flags below are module attributes because a test needs to flip them
    mid-process.  That is the right shape for a test and the wrong shape for a
    server, which has to make the choice before the model loads.  The
    environment is consulted once, at import; the attribute stays authoritative
    afterwards, so a test that sets it still wins.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# The hand-written fused router is numerically close but not array-exact: its
# normalized scores can eventually change a greedy token.  It is therefore an
# explicit semantic/experimental lane, never part of strict auto mode.  Same
# for the original fused add/RMSNorm carrier, whose thread mapping did not
# match mx.fast.rms_norm; the corrected kernel below replaces it.
_use_approximate_router = False
_use_approximate_add_rms = False

# Strict-lane fusions.  Each is array-exact against the stock chain and is on
# by default; each falls back the moment its live probe fails.
#
# Decode on CUDA is host-bound: with a warm cache the GPU wait per step
# measures ~0.003 ms, so wall clock is the sum of per-operation host costs and
# the currency of optimization is operation count, not arithmetic.  These three
# fusions exist to remove operations, not to make the GPU work less.
_use_fused_add_rms = _env_flag("MAPLE_FUSED_ADD_RMS", True)   # add + RMSNorm
_use_fused_qkv = _env_flag("MAPLE_FUSED_QKV", True)           # split + norm + RoPE

# The stock router chain under mx.compile.  Array-exact, and in isolation it
# cuts the router's host cost from 96.5 us to 77.9 us per layer -- but paired
# over ten fresh processes the end-to-end effect was 1.0062x with a 95%
# interval of 0.9927-1.0198 and 6/10 wins, so it is not distinguishable from
# zero on the measured host.  Exact but unproven stays opt-in.
_use_compiled_router = _env_flag("MAPLE_COMPILED_ROUTER", False)

# The fast lane, on by default: the whole MoE block (router, experts,
# activation, score-weighted aggregation and the preceding add/RMSNorm) in one
# dispatch, worth 73-88%.  It is within ~1 ULP of bf16 rather than array-exact,
# because a software fp32 reduction cannot reproduce what qmm_naive gets from a
# tensor-core MMA, so it can change a greedy token on a near-tie.
#
# What that costs was measured rather than assumed: an 846-token teacher-forced
# suite through the decode path moves corpus perplexity by -0.8% to -1.3% on all
# five supported architectures -- unbiased last-bit noise, and if anything in
# the favourable direction.  What it does cost is a token stream reproducible
# against stock, which roughly 9% of top-1 predictions no longer are.
#
# `MAPLE_MOE_MEGAKERNEL=0` returns the array-exact strict lane, which is what a
# reproducibility claim, a regression baseline or a bisect needs.  It falls back
# on its own for anything it cannot serve: a non-CUDA backend, experts that are
# not 2-bit affine at group size 128, a top_k other than 8, or a hidden size the
# block partition does not divide.
_use_moe_megakernel = _env_flag("MAPLE_MOE_MEGAKERNEL", True)

# The array-exact megakernel, the default lane: the same one-dispatch MoE
# block, and every phase reproduces the stock chain's bits (see the recipes
# above its source).  It survived the full screen on sm86 and sm89 -- decode
# stream identical to stock on 8/8 screened prompts, quality NLL to the last
# digit, throughput at parity with the ~1 ULP megakernel (345 vs 341 and 320
# vs 319 medians) -- so the default no longer trades the reproducible stream
# for speed.  The ~1 ULP lane below stays on as the fallback for geometries
# the exact plan declines; MAPLE_MOE_MEGAKERNEL_EXACT=0 disables this lane.
_use_moe_megakernel_exact = _env_flag("MAPLE_MOE_MEGAKERNEL_EXACT", True)

# The batch (2 <= B <= 8) decode lane: both proven M=B megakernels behind
# the stock per-layer structure.  Every batched row reproduces its solo
# stream bit for bit -- a contract stock batching does not hold (its own
# rows drift through batch-variant GEMM tails, down to 2/8 on a 5090).
# Default is data-driven per profile: ON where the full bit battery has
# run (solo-exact E2E, kernel-vs-kernel, LRU isolation), OFF elsewhere
# until scale-out.  MAPLE_BATCH_MEGAKERNELS=0/1 always wins.  Covers
# kL <= 1024 (the pair carries the 1-pass SDPA port); layers past that
# fall back to the stock path per step with an exact writeback.
# Per-profile default ceiling on the batch size the lane takes: the full
# battery ran everywhere below, and the aggregate curve decides the cap
# (sm90 measured +32%/+13%/par at B=1/2/4 but -22% at B8, where the 132
# SMs make stock batching genuinely strong -- so its default stops at 4).
_BATCH_MEGAKERNEL_PROVEN_PROFILES = {"sm86": 8, "sm120": 8, "sm90": 4,
                                      "sm89": 4}
_use_batch_megakernels = (
    _env_flag("MAPLE_BATCH_MEGAKERNELS", False)
    if "MAPLE_BATCH_MEGAKERNELS" in os.environ else None
)


def _batch_megakernel_max_rows():
    if _use_batch_megakernels is not None:
        return 8 if _use_batch_megakernels else 0
    prof = _cuda_profile()
    if prof is None:
        return 0
    return _BATCH_MEGAKERNEL_PROVEN_PROFILES.get(prof.name, 0)


def _batch_megakernels_enabled():
    return _batch_megakernel_max_rows() > 0

# The one-dispatch decode attention block.  The full bit battery is green
# on sm86/sm89/sm90 (stream identity, the window-rotation boundary, the
# whole kL>1024 boundary suite incl. cap growth), but the SPEED verdict is
# per-architecture: +20.6% on sm86 and +42.9% on sm89, while H100 measured
# a consistent -12..-14% in interleaved in-process pairs at every context
# length (132 SMs are starved by the 64-block grid).  The default is
# therefore data-driven: on where the lane is measured faster, off
# elsewhere until profiled.  An explicit MAPLE_ATTENTION_MEGAKERNEL=0/1
# always wins over the auto choice.
# Data-driven per-architecture default.  The lane is auto-on where it is
# measured faster (sm86 +20.6%, sm89 +42.9% / 357->439 on the fix build)
# and auto-off elsewhere (sm90/sm120 measured slower on healthy-CPU
# hosts).  Request isolation against serving LRU caches is gated by
# benchmarks/maple_lru_service_repro.py -- green 3/3 after the
# materialize-on-any-detach fix (chronicle #18): stored caches get real
# copies the moment the lane detaches from them, on EVERY detach path.
# An explicit MAPLE_ATTENTION_MEGAKERNEL=0/1 always wins.
_ATTENTION_MEGAKERNEL_FAST_PROFILES = ("sm86", "sm89")
_use_attention_megakernel = _env_flag("MAPLE_ATTENTION_MEGAKERNEL", None)


def _attention_megakernel_enabled():
    if _use_attention_megakernel is not None:
        return _use_attention_megakernel
    profile = _cuda_profile()
    return (profile is not None
            and profile.name in _ATTENTION_MEGAKERNEL_FAST_PROFILES)


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


# Residual add + RMSNorm in one dispatch, array-exact against the stock chain.
#
# MLX picks `rms_norm_small<float, 512, 32, 4>` for a 2048-wide row: 512
# threads, thread t owning exactly x[4t .. 4t+3], reduced by a descending
# __shfl_down tree inside the warp and then across the 16 warp results.  The
# earlier fused kernel used the elementwise thread count (256 on sm86) and two
# chunks per thread, so a thread summed 4t..4t+3 together with 4(t+256)..  --
# a different partition of the same 2048 squares, and therefore a different
# rounding.  Reproducing the partition is what makes this exact; the reduction
# tree was already right.
_EXACT_ADD_RMS_SOURCE = r"""
    constexpr int VEC = 4;
    constexpr int WARPS = THREADS_ / 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int base = tid * VEC;

    float hb[VEC];
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        const T_ rounded = static_cast<T_>(
            static_cast<float>(x[base + i]) + static_cast<float>(r[base + i]));
        h_out[base + i] = rounded;      // one rounding, exactly like a bf16 add
        hb[i] = static_cast<float>(rounded);
        ss += hb[i] * hb[i];            // the norm sees the rounded stream
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);

    __shared__ float temp[WARPS];
    if (lane == 0) temp[warp] = ss;
    __syncthreads();
    float tot = (lane < WARPS) ? temp[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
    __shared__ float total_s;
    if (tid == 0) total_s = tot;
    __syncthreads();

    const float scale = rsqrtf(total_s / static_cast<float>(DIM_) + EPS_);
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        hn_out[base + i] = static_cast<T_>(
            hb[i] * scale * static_cast<float>(w[base + i]));
    }
"""

_exact_add_rms_kernels = {}
_exact_add_rms_plans = {}


def _exact_add_rms_kernel(eps):
    kernel = _exact_add_rms_kernels.get(eps)
    if kernel is None:
        tag = f"{eps:.3e}".replace(".", "_").replace("-", "m").replace("+", "p")
        kernel = _exact_add_rms_kernels[eps] = mx.fast.cuda_kernel(
            name=f"maple_exact_add_rms_{tag}",
            input_names=["x", "r", "w"],
            output_names=["h_out", "hn_out"],
            source=_EXACT_ADD_RMS_SOURCE.replace("EPS_", f"{eps:.10e}f"),
        )
    return kernel


def _exact_add_rms_supported(dim):
    if _kernel_backend() != "cuda" or _cuda_profile() is None:
        return False
    threads = dim // 4
    return dim % 4 == 0 and threads % 32 == 0 and 32 <= threads <= 1024


def _exact_add_rms_norm(h, r, w, eps):
    """(h + r, rmsnorm(h + r) * w) in a single dispatch."""
    dim = h.shape[-1]
    key = (dim, h.dtype, w.dtype, eps, h.shape)
    plan = _exact_add_rms_plans.get(key)
    if plan is None:
        threads = dim // 4
        plan = _exact_add_rms_plans[key] = (
            _exact_add_rms_kernel(eps),
            {
                "template": [
                    ("T_", h.dtype), ("W_", w.dtype), ("DIM_", dim),
                    ("THREADS_", threads),
                ],
                "grid": (threads, 1, 1),
                "threadgroup": (threads, 1, 1),
                "output_shapes": [h.shape, h.shape],
                "output_dtypes": [h.dtype, h.dtype],
            },
        )
    kernel, kwargs = plan
    return kernel(inputs=[h, r, w], **kwargs)


def _exact_add_rms_ok(dim, dtype, w, eps):
    """Probe on a wide-dynamic-range vector, not on Gaussian noise.

    Measured: on N(0, 1) test vectors a butterfly reduction and a __shfl_down
    reduction agree on 300/300 trials, so a Gaussian probe cannot tell a wrong
    reduction from a right one.  It takes the outliers of a real residual
    stream to separate them, so the probe injects them.
    """
    if not _exact_add_rms_supported(dim):
        return False
    x = mx.random.normal((1, 1, dim), key=mx.random.key(0)).astype(dtype)
    r = mx.random.normal((1, 1, dim), key=mx.random.key(1)).astype(dtype)
    spikes = mx.random.normal((1, 1, dim), key=mx.random.key(2))
    x = (x + mx.where(spikes > 2.5, spikes * 24.0, 0.0)).astype(dtype)
    r = (r + mx.where(spikes < -2.5, spikes * 24.0, 0.0)).astype(dtype)
    return _matches(
        lambda: _exact_add_rms_norm(x, r, w, eps),
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


# The Q/K norm + RoPE kernel, widened so it also splits the fused qkv
# projection.  The decode path used to slice that projection three ways with a
# reshape around each piece:
#
#     qk      = qkv.reshape(-1)[:qk_size].reshape(n_q + n_kv, head_dim)
#     out     = qk_norm_rope(qk, offset)
#     queries = out[:n_q].reshape(1, n_q, 1, head_dim)
#     keys    = out[n_q:].reshape(1, n_kv, 1, head_dim)
#     values  = qkv.reshape(-1)[qk_size:].reshape(1, n_kv, 1, head_dim)
#
# Decode is host-bound, so those slices cost wall clock even though they move
# almost no data.  This kernel takes the whole qkv vector and emits the three
# tensors already in their final shapes.  The per-head arithmetic is unchanged
# -- same reduction, same rope expression, same rounding -- so q and k are
# bit-identical by construction and v is an exact copy; only the plumbing goes.
def _make_cuda_qkv_split_kernel(profile, use_rope):
    if use_rope:
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
        const int head = blockIdx.y;
        const int lane = threadIdx.x;
        constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
        const T_* xh = x + head * HEAD_DIM;

        if (head >= NQ_ + NKV_) {
            T_* vh = values_out + (head - NQ_ - NKV_) * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                const int j = lane * VALUES_PER_LANE + i;
                vh[j] = xh[j];
            }
            return;
        }

        const W_* wh = w + head * HEAD_DIM;
        T_* oh = (head < NQ_) ? (queries_out + head * HEAD_DIM)
                              : (keys_out + (head - NQ_) * HEAD_DIM);

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
        name=f"maple_qkv_split_{'rope' if use_rope else 'nope'}_{profile.name}",
        input_names=["x", "w", "inv_freq", "pos_eps"],
        output_names=["queries_out", "keys_out", "values_out"],
        source=source,
    )


_qkv_split_cache = {}


def _get_qkv_split_kernel(use_rope):
    profile = _cuda_profile()
    if _kernel_backend() != "cuda" or profile is None:
        raise RuntimeError("no fused QKV split kernel for this backend")
    key = (profile.name, use_rope)
    kernel = _qkv_split_cache.get(key)
    if kernel is None:
        kernel = _qkv_split_cache[key] = _make_cuda_qkv_split_kernel(
            profile, use_rope
        )
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
        self._split_qkv = None  # None = unprobed, then True/False

        # Maple applies RoPE only on sliding-window layers; full-attention
        # layers use no positional encoding (NoPE).
        self.use_rope = args.layer_types[layer_idx] == "sliding_attention"
        # The custom kernel implements the checkpoint's plain RoPE exactly.
        # Scaling policies have different position math and must stay on MLX's
        # portable implementation until they receive their own specialization.
        self._can_fuse_qk = not (self.use_rope and args.rope_scaling is not None)
        if self.use_rope:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
            self._rope_dim = rope_dim
            self.rope = initialize_rope(
                rope_dim,
                args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def _ensure_qk_constants(self):
        """Per-head norm weights and rope frequencies, built once.

        Both the Q/K kernel and the widened QKV kernel need these, so neither
        may depend on the other having run first.
        """
        if self._qk_w is not None:
            return
        n_q = self.num_attention_heads
        n_kv = self.num_key_value_heads
        self._qk_w = mx.contiguous(
            mx.concatenate(
                [
                    mx.broadcast_to(self.q_norm.weight[None], (n_q, self.head_dim)),
                    mx.broadcast_to(self.k_norm.weight[None], (n_kv, self.head_dim)),
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

    def _qk_fused(self, qk, offset):
        """Both norms and both rope applications in one dispatch."""
        if not self._can_fuse_qk:
            raise ValueError("scaled RoPE is unsupported by the fused Q/K kernel")
        self._ensure_qk_constants()

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

        # Any stock-path call that will touch the cache must first flush the
        # attention megakernel's buffers back into it: during fused decode
        # the stock buffers go stale, and a multi-turn prefill would other-
        # wise concatenate against history that is missing our appends.
        state = getattr(self, "_mega_state", None)
        if (
            cache is not None
            and state is not None
            and state.synced_offset >= 0
        ):
            _attn_mega_writeback(self, cache)

        qkv = self.qkv_proj(x)

        if B == 1 and L == 1 and self.use_qk_norm:
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            qk_size = (n_q + n_kv) * self.head_dim
            offset = cache.offset if cache is not None else 0

            def slice_qk():
                return qkv.reshape(-1)[:qk_size].reshape(n_q + n_kv, self.head_dim)

            if self._fused_qk is None:
                qk = slice_qk()
                # A nonzero position, so a broken rotation cannot pass.
                self._fused_qk = self._can_fuse_qk and _matches(
                    lambda: (self._qk_fused(qk, 7),),
                    lambda: (self._qk_reference(qk, 7),),
                )
            # The widened kernel consumes qkv directly, so the slice stays out
            # of the hot path entirely; it is only built for the probe and for
            # the fallback.
            if _use_fused_qkv and self._fused_qk:
                if self._split_qkv is None:
                    self._split_qkv = self._probe_qkv_split(qkv, slice_qk())
                if self._split_qkv:
                    queries, keys, values = self._qkv_split(qkv, offset)
                    return self._attend(
                        queries, keys, values, mask, cache, B, L
                    )
            qk = slice_qk()
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

        return self._attend(queries, keys, values, mask, cache, B, L)

    def _attend(self, queries, keys, values, mask, cache, B, L):
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def _qkv_split(self, qkv, offset):
        """queries, keys and values straight out of the fused projection."""
        self._ensure_qk_constants()
        n_q = self.num_attention_heads
        n_kv = self.num_key_value_heads
        hd = self.head_dim
        pos_eps = mx.array(
            [float(offset), self._eps, self._rope_log2_base], dtype=mx.float32
        )
        return _get_qkv_split_kernel(self.use_rope)(
            # qkv is passed unreshaped: the kernel indexes it linearly, and a
            # reshape is a real operation on a host-bound path.
            inputs=[qkv, self._qk_w, self._inv_freq, pos_eps],
            template=[
                ("T_", qkv.dtype), ("W_", self._qk_w.dtype),
                ("HEAD_DIM", hd), ("ROPE_DIM", self.rope.dims if self.use_rope else 0),
                ("NQ_", n_q), ("NKV_", n_kv),
            ],
            grid=(32, n_q + 2 * n_kv, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(1, n_q, 1, hd), (1, n_kv, 1, hd), (1, n_kv, 1, hd)],
            output_dtypes=[qkv.dtype] * 3,
        )

    def _probe_qkv_split(self, qkv, qk):
        """Fail closed unless the widened kernel matches the sliced path."""
        n_q = self.num_attention_heads
        n_kv = self.num_key_value_heads
        hd = self.head_dim
        qk_size = (n_q + n_kv) * hd
        try:
            return _matches(
                lambda: self._qkv_split(qkv, 7),
                lambda: (
                    self._qk_fused(qk, 7)[:n_q].reshape(1, n_q, 1, hd),
                    self._qk_fused(qk, 7)[n_q:].reshape(1, n_kv, 1, hd),
                    qkv.reshape(-1)[qk_size:].reshape(1, n_kv, 1, hd),
                ),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


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


_compiled_router_cache = {}


def _compiled_router(top_k):
    """mx.compile over the stock router chain, cached once per top-k.

    One shared closure, not one per layer: 24 separate compiled closures
    measured slower than a single shared one (20.75 us against 15.63 us of
    host time per call), because each carries its own cache.
    """
    fn = _compiled_router_cache.get(top_k)
    if fn is None:
        def router(x, w):
            gates = x.astype(mx.float32) @ w.astype(mx.float32).T
            scores = mx.softmax(gates, axis=-1)
            inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
            return inds, scores
        fn = _compiled_router_cache[top_k] = mx.compile(router)
    return fn


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
        self._compiled_ok = None  # None = unprobed, then True/False
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

    def _compiled(self, x):
        """The same chain under mx.compile: array-exact, fewer dispatches.

        The hand-written fused router is cheaper still but differs by ~1 ULP in
        the normalized weights, which flips greedy decisions.  Compiling the
        stock chain is exact by construction -- indices and scores both compare
        array-equal -- and removes most of the per-layer dispatch cost.

        shapeless=True cannot be used: it fails to infer shapes through the
        top-k slice.
        """
        return _compiled_router(self.top_k)(x, self.weight)

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
            if _use_compiled_router and x.size == self.hidden_size:
                if self._compiled_ok is None:
                    self._compiled_ok = _matches(
                        lambda: self._compiled(x), lambda: self._reference(x)
                    )
                if self._compiled_ok:
                    return self._compiled(x)
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

# Opt-in fast lane: the whole MoE block in one dispatch.
#
# Five phases separated by four atomic-counter grid barriers -- there is no
# cooperative launch, so ordering across blocks has to be built by hand, and a
# barrier is only safe while every block is resident.  MLX does not expose the
# multiprocessor count, so the grid is chosen by `_moe_megakernel_grid` from
# compute capability and memory (see its docstring for the measured sweep).
#
# Phase 0 recomputes the residual add and RMSNorm redundantly in every block.
# That is 2048 elements against an otherwise idle GPU, and doing it everywhere
# removes the need for an extra barrier before the router.
#
# Phase E is the tail: it folds the *next* layer's residual add + RMSNorm into
# this dispatch, so the Python loop never issues the standalone fuse between
# one layer's MoE and the next layer's attention.  Decode is host-bound, and
# that fuse costs ~13 us of host time per MoE layer per step; the tail replaces
# it with one extra barrier and 2048 elements of work on an idle GPU.  Its
# arithmetic mirrors _EXACT_ADD_RMS_SOURCE line for line, so the tail adds no
# inexactness of its own -- the lane's ~1 ULP story is confined to the MoE
# math, exactly as before.
#
# The single scratch buffer is deliberate: each additional kernel output costs
# ~5-7 us of host time (measured), which is the same order as a whole extra
# dispatch, so barrier counters, logits, indices, weights, activations and the
# staged MoE output all live at offsets in one array.  The three outputs are
# `out` (the next layer's normed attention input), `hout` (the new residual
# carrier h+attn+moe) and the scratch itself.
_MOE_MEGAKERNEL_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int WARPS = THREADS_ / WARP;

    // up/gate geometry
    constexpr int VPL_A = KH_ / LPRA_;
    constexpr int U32_A = VPL_A / 16;
    constexpr int U32ROW_A = KH_ / 16;
    // down geometry
    constexpr int VPL_B = KD_ / LPRB_;
    constexpr int U32_B = VPL_B / 16;
    constexpr int U32ROW_B = KD_ / 16;
    constexpr int SUBROWS_B = WARP / LPRB_;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid >> 5;
    const int blk = blockIdx.x;

    // Every extra kernel output costs host time (measured ~5-7 us each), so
    // all of the block's working state lives in one buffer.
    constexpr int OFF_IDX = 8;
    constexpr int OFF_SCO = 24;
    constexpr int OFF_LOG = 64;
    constexpr int OFF_ACT = 512;
    constexpr int OFF_STG = OFF_ACT + NEXP_ * KD_;
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch);
    float* idxf = scratch + OFF_IDX;
    float* scoref = scratch + OFF_SCO;
    float* logits = scratch + OFF_LOG;
    float* actf = scratch + OFF_ACT;
    float* stagef = scratch + OFF_STG;

    __shared__ float xs_lin[KH_];
    __shared__ float xs_t[KH_];
    __shared__ float acts[NEXP_ * KD_];
    __shared__ float red[WARPS];
    __shared__ float gmax_s;
    __shared__ float gsum_s;
    __shared__ int sel[NEXP_];
    __shared__ float selv[NEXP_];

    // ---- phase 0: residual add + RMSNorm, recomputed in every block --------
    // The normalized vector is consumed only inside this kernel, so it never
    // has to be materialized; nothing leaves this phase (the residual stream
    // is re-derived in phase E, where the MoE output can be folded in at the
    // same time).  Thread t owns x[4t..4t+3] and the reduction is a descending
    // __shfl_down tree, which is what mx.fast.rms_norm does for a 2048-wide
    // row.
    {
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float hb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const T_ rounded = static_cast<T_>(
                static_cast<float>(hin[base + i]) + static_cast<float>(rin[base + i]));
            hb[i] = static_cast<float>(rounded);
            ss += hb[i] * hb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            const float v = static_cast<float>(static_cast<T_>(
                hb[i] * nscale * static_cast<float>(nw[idx])));
            xs_lin[idx] = v;
            const int l = idx / VPL_A;
            const int j = idx - l * VPL_A;
            xs_t[j * LPRA_ + l] = v;
        }
    }
    __syncthreads();

    // ---- phase A: router logits, one warp per expert row -------------------
    for (int e = blk * WARPS + warp; e < NROUT_; e += GRID_ * WARPS) {
        const RW_* wrow = rw + static_cast<long long>(e) * KH_;
        float acc = 0.0f;
        for (int k = lane; k < KH_; k += WARP) {
            acc = fmaf(static_cast<float>(wrow[k]), xs_lin[k], acc);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) logits[e] = acc;
    }

    // ---- barrier 1 ---------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[0], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[3], 1u);
        else while (atomicAdd(&ctr[3], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase B: softmax + top-K + renormalize, on block 0 ----------------
    if (blk == 0) {
        float m = -INFINITY;
        for (int e = tid; e < NROUT_; e += THREADS_) m = fmaxf(m, logits[e]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (lane == 0) red[warp] = m;
        __syncthreads();
        if (tid == 0) {
            float g = red[0];
            for (int i = 1; i < WARPS; ++i) g = fmaxf(g, red[i]);
            gmax_s = g;
        }
        __syncthreads();
        const float gmax = gmax_s;
        float s = 0.0f;
        for (int e = tid; e < NROUT_; e += THREADS_) s += expf(logits[e] - gmax);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) red[warp] = s;
        __syncthreads();
        if (tid == 0) {
            float g = 0.0f;
            for (int i = 0; i < WARPS; ++i) g += red[i];
            gsum_s = g;
        }
        __syncthreads();

        if (warp == 0) {
            for (int t = 0; t < NEXP_; ++t) {
                float best = -INFINITY;
                int bi = NROUT_;
                for (int e = lane; e < NROUT_; e += WARP) {
                    const float v = logits[e];
                    if (v > best || (v == best && e < bi)) { best = v; bi = e; }
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const float ov = __shfl_down_sync(0xffffffffu, best, off);
                    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
                }
                if (lane == 0) { sel[t] = bi; selv[t] = best; logits[bi] = -INFINITY; }
                __syncwarp();
            }
        }
        __syncthreads();
        if (tid == 0) {
            const float inv = 1.0f / gsum_s;
            float den = 0.0f;
            float sc[NEXP_];
            #pragma unroll
            for (int t = 0; t < NEXP_; ++t) {
                sc[t] = expf(selv[t] - gmax_s) * inv;
                den += sc[t];
            }
            den += 1e-20f;
            #pragma unroll
            for (int t = 0; t < NEXP_; ++t) {
                idxf[NEXP_ - 1 - t] = static_cast<float>(sel[t]);
                scoref[NEXP_ - 1 - t] = sc[t] / den;
            }
        }
    }

    // ---- barrier 2 ---------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[1], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[4], 1u);
        else while (atomicAdd(&ctr[4], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase C: up/gate projection + clamped SwiGLU ----------------------
    {
        const int rows_per_iter = WARPS;
        for (int base_row = blk * rows_per_iter; base_row < NOUT_ * NEXP_;
             base_row += GRID_ * rows_per_iter) {
            const int flat = base_row + warp;
            if (flat >= NOUT_ * NEXP_) break;
            const int expert = flat / NOUT_;
            const int r = flat - expert * NOUT_;
            const int w_e = static_cast<int>(idxf[expert]);
            const int half = lane / LPRA_;
            const int slane = lane % LPRA_;
            const int row = r + half * NOUT_;

            const long long wbase =
                (static_cast<long long>(w_e) * (2 * NOUT_) + row) * U32ROW_A
                + static_cast<long long>(slane) * U32_A;
            uint32_t wv[U32_A];
            if (U32_A == 8) {
                const uint4 t0 = *reinterpret_cast<const uint4*>(ugw + wbase);
                const uint4 t1 = *reinterpret_cast<const uint4*>(ugw + wbase + 4);
                wv[0]=t0.x; wv[1]=t0.y; wv[2]=t0.z; wv[3]=t0.w;
                wv[4]=t1.x; wv[5]=t1.y; wv[6]=t1.z; wv[7]=t1.w;
            } else {
                #pragma unroll
                for (int i = 0; i < U32_A; ++i) wv[i] = ugw[wbase + i];
            }
            const int g = (slane * VPL_A) / GS_;
            const long long sbase =
                (static_cast<long long>(w_e) * (2 * NOUT_) + row) * GRPA_ + g;
            const float sc = static_cast<float>(ugs[sbase]);
            const float bs = static_cast<float>(ugb[sbase]);
            const float s2 = sc + sc;
            const float b2 = fmaf(-4.0f, sc, bs);

            float part[U32_A];
            #pragma unroll
            for (int i = 0; i < U32_A; ++i) {
                const uint32_t v = wv[i];
                const int off = i * 16;
                float a = 0.0f;
                #pragma unroll
                for (int j = 0; j < 11; ++j) {
                    const float f = __uint_as_float(
                        ((v << (21 - 2 * j)) & 0x00600000u) | 0x40000000u);
                    a = fmaf(fmaf(f, s2, b2), xs_t[(off + j) * LPRA_ + slane], a);
                }
                #pragma unroll
                for (int j = 11; j < 16; ++j) {
                    const float f = __uint_as_float(
                        ((v >> (2 * j - 21)) & 0x00600000u) | 0x40000000u);
                    a = fmaf(fmaf(f, s2, b2), xs_t[(off + j) * LPRA_ + slane], a);
                }
                part[i] = a;
            }
            #pragma unroll
            for (int stride = U32_A / 2; stride > 0; stride >>= 1) {
                #pragma unroll
                for (int i = 0; i < stride; ++i) part[i] += part[i + stride];
            }
            float acc = part[0];
            #pragma unroll
            for (int off = LPRA_ / 2; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            acc = static_cast<float>(static_cast<T_>(acc));
            const float other = __shfl_sync(0xffffffffu, acc, lane + LPRA_);
            if (lane == 0) {
                const float up = fminf(fmaxf(acc, -7.0f), 7.0f);
                const float gate = fminf(other, 7.0f);
                // round through bf16 so the value matches the separate
                // kernel's stored activation exactly
                actf[expert * NOUT_ + r] = static_cast<float>(static_cast<T_>(
                    (gate / (1.0f + expf(-gate))) * up));
            }
        }
    }

    // ---- barrier 3 ---------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[2], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[5], 1u);
        else while (atomicAdd(&ctr[5], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase D: down projection + score-weighted aggregation -------------
    for (int i = tid; i < NEXP_ * KD_; i += THREADS_) {
        const int e = i / KD_;
        const int k = i - e * KD_;
        const int l = k / VPL_B;
        const int j = k - l * VPL_B;
        acts[e * KD_ + j * LPRB_ + l] = actf[i];
    }
    __syncthreads();
    {
        const int sub = lane / LPRB_;
        const int slane = lane % LPRB_;
        const int g = (slane * VPL_B) / GS_;
        const int rows_per_iter = WARPS * SUBROWS_B;
        for (int base = blk * rows_per_iter; base < ND_;
             base += GRID_ * rows_per_iter) {
            const int row = base + warp * SUBROWS_B + sub;
            if (row >= ND_) break;
            float total = 0.0f;
            for (int e = 0; e < NEXP_; ++e) {
                const int w_e = static_cast<int>(idxf[e]);
                const long long wbase =
                    (static_cast<long long>(w_e) * ND_ + row) * U32ROW_B
                    + static_cast<long long>(slane) * U32_B;
                uint32_t wv[U32_B];
                if (U32_B == 4) {
                    const uint4 t = *reinterpret_cast<const uint4*>(dnw + wbase);
                    wv[0]=t.x; wv[1]=t.y; wv[2]=t.z; wv[3]=t.w;
                } else {
                    #pragma unroll
                    for (int i = 0; i < U32_B; ++i) wv[i] = dnw[wbase + i];
                }
                const long long sbase =
                    (static_cast<long long>(w_e) * ND_ + row) * GRPB_ + g;
                const float sc = static_cast<float>(dns[sbase]);
                const float bs = static_cast<float>(dnb[sbase]);
                const float s2 = sc + sc;
                const float b2 = fmaf(-4.0f, sc, bs);
                const float* xe = acts + e * KD_;
                float part[U32_B];
                #pragma unroll
                for (int i = 0; i < U32_B; ++i) {
                    const uint32_t v = wv[i];
                    const int off = i * 16;
                    float a = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 11; ++j) {
                        const float f = __uint_as_float(
                            ((v << (21 - 2 * j)) & 0x00600000u) | 0x40000000u);
                        a = fmaf(fmaf(f, s2, b2), xe[(off + j) * LPRB_ + slane], a);
                    }
                    #pragma unroll
                    for (int j = 11; j < 16; ++j) {
                        const float f = __uint_as_float(
                            ((v >> (2 * j - 21)) & 0x00600000u) | 0x40000000u);
                        a = fmaf(fmaf(f, s2, b2), xe[(off + j) * LPRB_ + slane], a);
                    }
                    part[i] = a;
                }
                #pragma unroll
                for (int stride = U32_B / 2; stride > 0; stride >>= 1) {
                    #pragma unroll
                    for (int i = 0; i < stride; ++i) part[i] += part[i + stride];
                }
                float acc = part[0];
                #pragma unroll
                for (int off = LPRB_ / 2; off > 0; off >>= 1)
                    acc += __shfl_down_sync(0xffffffffu, acc, off);
                total = fmaf(static_cast<float>(static_cast<T_>(acc)),
                             scoref[e], total);
            }
            // Staged as the bf16-rounded value: phase E must see exactly what
            // the standalone MoE output array used to hold.
            if (slane == 0)
                stagef[row] = static_cast<float>(static_cast<T_>(total));
        }
    }

    // ---- barrier 4 ---------------------------------------------------------
    // Every block must arrive (a waiter spins on the release flag), but only
    // block 0 has work left.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[6], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[7], 1u);
        else while (atomicAdd(&ctr[7], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();
    if (blk != 0) return;

    // ---- phase E: next layer's residual add + RMSNorm ----------------------
    // Mirrors _EXACT_ADD_RMS_SOURCE exactly: one bf16 rounding per add, the
    // norm sees the rounded stream, descending __shfl_down tree, warp-leader
    // array, T_(x * scale * float(w)).  `hout` becomes the residual carrier
    // h + attn + moe; `out` becomes the next attention input, already normed
    // with the next layer's weight.
    {
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float sb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            const T_ s = static_cast<T_>(
                static_cast<float>(hin[idx]) + static_cast<float>(rin[idx]));
            const T_ s2 = static_cast<T_>(
                static_cast<float>(s) + stagef[idx]);
            hout[idx] = s2;
            sb[i] = static_cast<float>(s2);
            ss += sb[i] * sb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            out[idx] = static_cast<T_>(
                sb[i] * nscale * static_cast<float>(nw2[idx]));
        }
    }
"""



# The array-exact megakernel.  Same shape as the fast megakernel -- one
# dispatch, grid barriers, three outputs -- but every phase reproduces the
# stock chain's bits, each recipe proven in isolation on hardware first
# (benchmarks/maple_qmm_naive_repro.py, maple_exact_lane_semantics.py, and
# the exhaustive silu sweep recorded in results):
#
#   logits      fp32 gemv: four consecutive columns per lane, stride 128,
#               fma-contracted, descending shuffle tree
#   softmax     online single-pass port of softmax.cu (BLOCK_DIM=64,
#               N_READS=4, xor-butterfly all-reduces, identity padding)
#   top-8       argsort's ascending-(value, index) tail, ties included
#   renorm      linear 8-term fp32 sum, + 1e-20, div.rn
#   experts     qmm_naive's tensor-core atom: dequant bf16(bf16(q*s)+z),
#               mma.sync m16n8k16 bf16 with row 0 of A populated, k-tiles
#               of 128 accumulated in order, one bf16 epilogue rounding
#   activation  the bf16-typed sigmoid chain with accurate expf (bit-equal
#               on every finite bf16), silu multiply and up multiply each
#               rounded once
#   aggregate   col_reduce_small's linear loop with the multiply rounded
#               separately from the sum: __fmul_rn then __fadd_rn, never fma
#   tail        the proven phase E (residual fold + next layer's RMSNorm)
_MOE_EXACT_MEGAKERNEL_HEADER = r"""
__device__ __forceinline__ float bf16f(float v) {
    return __bfloat162float(__nv_bfloat16(v));
}

// One full 128-k tile of the qmm_naive reproduction: eight m16n8k16 atoms
// in k order for the 8-column octet at `col0`.  The tile's sixteen packed
// words per column arrive as two uint4 loads and one scale/bias pair covers
// the whole tile (group_size == 128) -- same data as loading per fragment
// element, same bits, an eighth of the transactions.
__device__ __forceinline__ void qmm_tile(
    const __nv_bfloat16* xb,
    const unsigned int* wq,
    const __nv_bfloat16* sc,
    const __nv_bfloat16* bi,
    long long wrow_stride_u32,
    long long grow_stride,
    int col0,
    int ktile,
    int gs,
    int lane,
    float& acc0,
    float& acc1) {
    const int col = col0 + (lane >> 2);
    const unsigned int* wrow = wq + col * wrow_stride_u32 + (ktile >> 4);
    const uint4 wa = *reinterpret_cast<const uint4*>(wrow);
    const uint4 wb = *reinterpret_cast<const uint4*>(wrow + 4);
    const unsigned words[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
    const __nv_bfloat16 s = sc[col * grow_stride + ktile / gs];
    const __nv_bfloat16 z = bi[col * grow_stride + ktile / gs];
    float d2 = 0.0f, d3 = 0.0f;  // rows 8..15 of the atom: discarded

    #pragma unroll
    for (int ka = 0; ka < 8; ++ka) {
        const int kbase = ktile + ka * 16;
        const unsigned int word = words[ka];
        const int akol = (lane & 3) * 2;
        const bool arow0 = (lane >> 2) == 0;
        const __nv_bfloat16 zero = __nv_bfloat16(0.0f);
        const __nv_bfloat16 a0 = arow0 ? xb[kbase + akol] : zero;
        const __nv_bfloat16 a1 = arow0 ? xb[kbase + akol + 1] : zero;
        const __nv_bfloat16 a4 = arow0 ? xb[kbase + 8 + akol] : zero;
        const __nv_bfloat16 a5 = arow0 ? xb[kbase + 8 + akol + 1] : zero;
        __nv_bfloat16 bfrag[4];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            #pragma unroll
            for (int piece = 0; piece < 2; ++piece) {
                const int sub = half * 8 + (lane & 3) * 2 + piece;
                const int q = (word >> (2 * sub)) & 3;
                bfrag[half * 2 + piece] =
                    __hadd(__hmul(__nv_bfloat16(float(q)), s), z);
            }
        }
        const unsigned azz = 0u;
        const unsigned a01 = (unsigned(__bfloat16_as_ushort(a1)) << 16)
                           | unsigned(__bfloat16_as_ushort(a0));
        const unsigned a45 = (unsigned(__bfloat16_as_ushort(a5)) << 16)
                           | unsigned(__bfloat16_as_ushort(a4));
        const unsigned b01 = (unsigned(__bfloat16_as_ushort(bfrag[1])) << 16)
                           | unsigned(__bfloat16_as_ushort(bfrag[0]));
        const unsigned b23 = (unsigned(__bfloat16_as_ushort(bfrag[3])) << 16)
                           | unsigned(__bfloat16_as_ushort(bfrag[2]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc0), "+f"(acc1), "+f"(d2), "+f"(d3)
            : "r"(a01), "r"(azz), "r"(a45), "r"(azz), "r"(b01), "r"(b23));
    }
}

// The original per-atom form, kept for reference paths.
__device__ __forceinline__ void qmm_atom(
    const __nv_bfloat16* xb,
    const unsigned int* wq,
    const __nv_bfloat16* sc,
    const __nv_bfloat16* bi,
    long long wrow_stride_u32,
    long long grow_stride,
    int col0,
    int kbase,
    int gs,
    int lane,
    float& acc0,
    float& acc1) {
    const int akol = (lane & 3) * 2;
    const bool arow0 = (lane >> 2) == 0;
    const __nv_bfloat16 zero = __nv_bfloat16(0.0f);
    const __nv_bfloat16 a0 = arow0 ? xb[kbase + akol] : zero;
    const __nv_bfloat16 a1 = arow0 ? xb[kbase + akol + 1] : zero;
    const __nv_bfloat16 a4 = arow0 ? xb[kbase + 8 + akol] : zero;
    const __nv_bfloat16 a5 = arow0 ? xb[kbase + 8 + akol + 1] : zero;

    const int col = col0 + (lane >> 2);
    // One atom's 16 k-values of one column live in a single packed word,
    // and group_size >= 16 means one scale/bias pair covers them all.
    // Loading once per lane instead of once per fragment element does not
    // change a single bit -- same data, fewer transactions.
    const unsigned int word = wq[col * wrow_stride_u32 + (kbase >> 4)];
    const __nv_bfloat16 s = sc[col * grow_stride + kbase / gs];
    const __nv_bfloat16 z = bi[col * grow_stride + kbase / gs];
    __nv_bfloat16 bfrag[4];
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int piece = 0; piece < 2; ++piece) {
            const int sub = half * 8 + (lane & 3) * 2 + piece;
            const int q = (word >> (2 * sub)) & 3;
            bfrag[half * 2 + piece] =
                __hadd(__hmul(__nv_bfloat16(float(q)), s), z);
        }
    }
    const unsigned azz = 0u;
    const unsigned a01 = (unsigned(__bfloat16_as_ushort(a1)) << 16)
                       | unsigned(__bfloat16_as_ushort(a0));
    const unsigned a45 = (unsigned(__bfloat16_as_ushort(a5)) << 16)
                       | unsigned(__bfloat16_as_ushort(a4));
    const unsigned b01 = (unsigned(__bfloat16_as_ushort(bfrag[1])) << 16)
                       | unsigned(__bfloat16_as_ushort(bfrag[0]));
    const unsigned b23 = (unsigned(__bfloat16_as_ushort(bfrag[3])) << 16)
                       | unsigned(__bfloat16_as_ushort(bfrag[2]));
    float d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc0), "+f"(acc1), "+f"(d2), "+f"(d3)
        : "r"(a01), "r"(azz), "r"(a45), "r"(azz), "r"(b01), "r"(b23));
}

// clamped_swiglu, bit-equal to the stock chain on every finite bf16 input:
// silu's sigmoid is the bf16-typed chain with accurate expf, the silu
// multiply and the up multiply each round once.
__device__ __forceinline__ __nv_bfloat16 exact_swiglu(
    float gate_f, float up_f) {
    const float gm = fminf(gate_f, 7.0f);
    const float a = fabsf(gm);
    const float e = bf16f(expf(a));
    const float d = bf16f(1.0f + e);
    const float y = bf16f(__fdiv_rn(1.0f, d));
    const float sig = (gm < 0.0f) ? y : 1.0f - y;
    const float sb = bf16f(sig);
    const float t = bf16f(gm * sb);
    const float uc = fmaxf(-7.0f, fminf(up_f, 7.0f));
    return __nv_bfloat16(__fmul_rn(t, uc));
}
"""

_MOE_EXACT_MEGAKERNEL_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int WARPS = THREADS_ / WARP;
    constexpr int GS = 128;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid >> 5;
    const int blk = blockIdx.x;
    const int wglobal = blk * WARPS + warp;

    // Scratch layout (floats): 16 barrier uints, then indices, scores,
    // logits, softmax probabilities, bf16-rounded activations and the
    // bf16-rounded per-expert down outputs.
    constexpr int OFF_IDX = 16;
    constexpr int OFF_SCO = OFF_IDX + 8;
    constexpr int OFF_LOG = OFF_SCO + 8;
    constexpr int OFF_PRB = OFF_LOG + NROUT_;
    constexpr int OFF_UGS = OFF_PRB + NROUT_;
    constexpr int OFF_DST = OFF_UGS + NEXP_ * 2 * KD_;
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch);
    float* idxf = scratch + OFF_IDX;
    float* scoref = scratch + OFF_SCO;
    float* logits = scratch + OFF_LOG;
    float* probs = scratch + OFF_PRB;
    float* ugstage = scratch + OFF_UGS;
    float* dstage = scratch + OFF_DST;

    __shared__ float xs_lin[KH_];
    __shared__ __nv_bfloat16 xbs[KH_];
    __shared__ float red[WARPS];
    __shared__ float gsum_s;

    // ---- phase 0: residual add + RMSNorm, recomputed in every block ------
    // Identical to the proven phase 0 of the fast megakernel; the normed
    // vector is kept both as exact floats (for the fp32 router gemv) and as
    // bf16 (the A operand of every expert atom).
    {
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float hb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const T_ rounded = static_cast<T_>(
                static_cast<float>(hin[base + i]) + static_cast<float>(rin[base + i]));
            hb[i] = static_cast<float>(rounded);
            ss += hb[i] * hb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            const float v = static_cast<float>(static_cast<T_>(
                hb[i] * nscale * static_cast<float>(nw[idx])));
            xs_lin[idx] = v;
            xbs[idx] = __nv_bfloat16(v);
        }
    }
    __syncthreads();

    // ---- phase A: router logits, the stock fp32 gemv order ---------------
    // One warp per expert row: four consecutive columns per lane, stride
    // 128, fma, descending shuffle tree.  16/16 bitwise in isolation.
    for (int row = wglobal; row < NROUT_; row += GRID_ * WARPS) {
        // The stock chain materializes weight.astype(float32) first; reading
        // the bf16 weight and widening per element is the same values.
        const __nv_bfloat16* wrow = rw + (long long)row * KH_;
        float sum = 0.0f;
        for (int col = 4 * lane; col < KH_; col += 128) {
            const __nv_bfloat162 wab =
                *reinterpret_cast<const __nv_bfloat162*>(wrow + col);
            const __nv_bfloat162 wcd =
                *reinterpret_cast<const __nv_bfloat162*>(wrow + col + 2);
            sum = fmaf(__bfloat162float(wab.x), xs_lin[col], sum);
            sum = fmaf(__bfloat162float(wab.y), xs_lin[col + 1], sum);
            sum = fmaf(__bfloat162float(wcd.x), xs_lin[col + 2], sum);
            sum = fmaf(__bfloat162float(wcd.y), xs_lin[col + 3], sum);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, o);
        if (lane == 0) logits[row] = sum;
    }

    // ---- barrier 1 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[0], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[8], 1u);
        else while (atomicAdd(&ctr[8], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase B: softmax + top-8 + renorm, on block 0 --------------------
    // The softmax is the online port (100/100 bitwise at 256 experts): the
    // first 64 threads carry the data, every thread of the block reaches the
    // same __syncthreads() calls.
    __shared__ float local_max[2];
    __shared__ float local_norm[2];
    if (blk == 0) {
        constexpr int N_READS = 4;
        constexpr int SWARPS = 2;  // 64 active threads
        const bool active = tid < 64;
        const int slane = tid & 31;
        const int swarp = tid >> 5;
        float vals[N_READS];
        float maxval = -INFINITY;
        float normalizer = 0.0f;
        float prevmax;
        if (active) {
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                vals[i] = logits[tid * N_READS + i];
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                maxval = fmaxf(maxval, vals[i]);
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                normalizer = normalizer + __expf(vals[i] - maxval);
            prevmax = maxval;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
            normalizer = normalizer * __expf(prevmax - maxval);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);
            prevmax = maxval;
            if (slane == 0) local_max[swarp] = maxval;
        }
        __syncthreads();
        if (active) {
            maxval = (slane < SWARPS) ? local_max[slane] : -INFINITY;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
            normalizer = normalizer * __expf(prevmax - maxval);
            if (slane == 0) local_norm[swarp] = normalizer;
        }
        __syncthreads();
        if (active) {
            normalizer = (slane < SWARPS) ? local_norm[slane] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);
            normalizer = 1.0f / normalizer;
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                probs[tid * N_READS + i] =
                    __expf(vals[i] - maxval) * normalizer;
        }
        __syncthreads();

        // top-8 by ascending (value, index) -- argsort's tail -- then the
        // linear renorm.  One thread; the work is 8 passes over 256 floats
        // against an idle GPU.
        if (tid < 32) {
            int chosen[NEXP_];
            #pragma unroll
            for (int k = 0; k < NEXP_; ++k) {
                float bestv = -INFINITY;
                int besti = -1;
                for (int i = lane; i < NROUT_; i += 32) {
                    bool taken = false;
                    #pragma unroll
                    for (int t = 0; t < NEXP_; ++t)
                        if (t < k && chosen[t] == i) taken = true;
                    if (taken) continue;
                    const float v = probs[i];
                    if (v > bestv || (v == bestv && i > besti)) {
                        bestv = v;
                        besti = i;
                    }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_down_sync(0xffffffffu, bestv, o);
                    const int oi = __shfl_down_sync(0xffffffffu, besti, o);
                    if (ov > bestv || (ov == bestv && oi > besti)) {
                        bestv = ov;
                        besti = oi;
                    }
                }
                besti = __shfl_sync(0xffffffffu, besti, 0);
                chosen[k] = besti;
            }
            if (tid == 0) {
            // descending pick order -> ascending storage order
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e) {
                const int src = NEXP_ - 1 - e;
                idxf[e] = static_cast<float>(chosen[src]);
                scoref[e] = probs[chosen[src]];
            }
            // The stock sum(axis=-1) over (1,1,8) dispatches to
            // row_reduce_simple with N_READS=4: two four-term sequential
            // partials, then one add.  300/300 bitwise on live routers;
            // the naive linear order missed a third of them.
            const float lo = __fadd_rn(__fadd_rn(__fadd_rn(
                scoref[0], scoref[1]), scoref[2]), scoref[3]);
            const float hi = __fadd_rn(__fadd_rn(__fadd_rn(
                scoref[4], scoref[5]), scoref[6]), scoref[7]);
            const float sum = __fadd_rn(lo, hi);
            const float denom = sum + 1e-20f;
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e)
                scoref[e] = __fdiv_rn(scoref[e], denom);
            }
        }
    }

    // ---- barrier 2 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[1], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[9], 1u);
        else while (atomicAdd(&ctr[9], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase C: up_gate experts on the qmm_naive tile -------------------
    // One warp per (expert, up_gate octet): all 2*KD columns as independent
    // tasks, which fills the grid twice as densely as pairing up with gate
    // did.  Raw bf16-rounded projection outputs are staged; the activation
    // moves to phase D's shared-load, where both halves are visible -- same
    // formula, same inputs, same bits.
    {
        constexpr int OCTS = 2 * KD_ / 8;
        const long long wstride = (2LL * KD_) * (KH_ / 16);
        const long long gstride = (2LL * KD_) * (KH_ / GS);
        for (int task = wglobal; task < NEXP_ * OCTS; task += GRID_ * WARPS) {
            const int e = task / OCTS;
            const int oct = task - e * OCTS;
            const int we = static_cast<int>(idxf[e]);
            const unsigned int* wq =
                reinterpret_cast<const unsigned int*>(ugw) + we * wstride;
            const __nv_bfloat16* sc = ugs + we * gstride;
            const __nv_bfloat16* bi = ugb + we * gstride;
            float o0 = 0.0f, o1 = 0.0f;
            for (int kt = 0; kt < KH_ / 128; ++kt) {
                qmm_tile(xbs, wq, sc, bi, KH_ / 16, KH_ / GS,
                         oct * 8, kt * 128, GS, lane, o0, o1);
            }
            if ((lane >> 2) == 0) {
                const int c = oct * 8 + 2 * (lane & 3);
                ugstage[e * 2 * KD_ + c] =
                    __bfloat162float(__nv_bfloat16(o0));
                ugstage[e * 2 * KD_ + c + 1] =
                    __bfloat162float(__nv_bfloat16(o1));
            }
        }
    }

    // ---- barrier 3 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[2], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[10], 1u);
        else while (atomicAdd(&ctr[10], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase D: down experts, staged per expert -------------------------
    {
        constexpr int OCTS = ND_ / 8;
        const long long wstride = (long long)ND_ * (KD_ / 16);
        const long long gstride = (long long)ND_ * (KD_ / GS);
        __shared__ __nv_bfloat16 abs_[NEXP_ * KD_];
        for (int i = tid; i < NEXP_ * KD_; i += THREADS_) {
            const int e = i / KD_;
            const int c = i - e * KD_;
            // stock layout: up = cols [0, KD), gate = cols [KD, 2*KD)
            abs_[i] = exact_swiglu(
                ugstage[e * 2 * KD_ + KD_ + c],
                ugstage[e * 2 * KD_ + c]);
        }
        __syncthreads();
        for (int task = wglobal; task < NEXP_ * OCTS; task += GRID_ * WARPS) {
            const int e = task / OCTS;
            const int oct = task - e * OCTS;
            const int we = static_cast<int>(idxf[e]);
            const unsigned int* wq =
                reinterpret_cast<const unsigned int*>(dnw) + we * wstride;
            const __nv_bfloat16* sc = dns + we * gstride;
            const __nv_bfloat16* bi = dnb + we * gstride;
            float a0 = 0.0f, a1 = 0.0f;
            for (int kt = 0; kt < KD_ / 128; ++kt) {
                qmm_tile(abs_ + e * KD_, wq, sc, bi, KD_ / 16, KD_ / GS,
                         oct * 8, kt * 128, GS, lane, a0, a1);
            }
            if ((lane >> 2) == 0) {
                const int c = oct * 8 + 2 * (lane & 3);
                dstage[e * ND_ + c] = __bfloat162float(__nv_bfloat16(a0));
                dstage[e * ND_ + c + 1] = __bfloat162float(__nv_bfloat16(a1));
            }
        }
    }

    // ---- barrier 4 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[3], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[11], 1u);
        else while (atomicAdd(&ctr[11], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();
    if (blk != 0) return;

    // ---- phase E: aggregation + residual fold + next norm -----------------
    // Aggregation is col_reduce_small's linear loop with the multiply
    // rounded on its own (128/128 bitwise); the tail is the proven exact
    // fuse.
    {
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float sb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            float agg = 0.0f;
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e)
                agg = __fadd_rn(agg,
                    __fmul_rn(dstage[e * ND_ + idx], scoref[e]));
            const float aggb = __bfloat162float(__nv_bfloat16(agg));
            const T_ s = static_cast<T_>(
                static_cast<float>(hin[idx]) + static_cast<float>(rin[idx]));
            const T_ s2 = static_cast<T_>(static_cast<float>(s) + aggb);
            hout[idx] = s2;
            sb[i] = static_cast<float>(s2);
            ss += sb[i] * sb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            out[idx] = static_cast<T_>(
                sb[i] * nscale * static_cast<float>(nw2[idx]));
        }
    }
"""

# The batch (M=B) port of the exact-MoE megakernel: ROWS_ independent
# decode rows through every proven recipe in ONE dispatch.  Differences
# from the production source are purely structural:
#   - the per-block shared normed vector becomes a global fp32+bf16 stage
#     (phase 0 runs row-per-block, then a grid barrier) -- fp32/bf16
#     round-trips through global memory are value-exact;
#   - phase B runs its block-0 recipe on block r for row r;
#   - the SwiGLU staging moves from per-block shared memory to a global
#     bf16 stage behind one extra barrier (B*8*KD_ no longer fits shared);
#   - every task grid gains a leading row factor.
# Each row's arithmetic chains are untouched, so row r must equal the
# production kernel run alone on row r bit for bit -- that is the gate
# (benchmarks/maple_moe_batch_check.py).
_MOE_BATCH_MEGAKERNEL_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int WARPS = THREADS_ / WARP;
    constexpr int GS = 128;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid >> 5;
    const int blk = blockIdx.x;
    const int wglobal = blk * WARPS + warp;

    constexpr int OFF_IDX = 16;
    constexpr int OFF_SCO = OFF_IDX + ROWS_ * 8;
    constexpr int OFF_LOG = OFF_SCO + ROWS_ * 8;
    constexpr int OFF_PRB = OFF_LOG + ROWS_ * NROUT_;
    constexpr int OFF_XF = OFF_PRB + ROWS_ * NROUT_;
    constexpr int OFF_XB = OFF_XF + ROWS_ * KH_;
    constexpr int OFF_UGS = OFF_XB + ROWS_ * KH_ / 2;
    constexpr int OFF_AST = OFF_UGS + ROWS_ * NEXP_ * 2 * KD_;
    constexpr int OFF_DST = OFF_AST + ROWS_ * NEXP_ * KD_ / 2;
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch);
    float* idxf = scratch + OFF_IDX;
    float* scoref = scratch + OFF_SCO;
    float* logits = scratch + OFF_LOG;
    float* probs = scratch + OFF_PRB;
    float* xf = scratch + OFF_XF;
    __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(scratch + OFF_XB);
    float* ugstage = scratch + OFF_UGS;
    __nv_bfloat16* astage = reinterpret_cast<__nv_bfloat16*>(scratch + OFF_AST);
    float* dstage = scratch + OFF_DST;

    __shared__ float red[WARPS];
    __shared__ float gsum_s;

    // ---- phase 0: add + RMSNorm, row r on block r, staged to global ------
    if (blk < ROWS_) {
        const int row = blk;
        const T_* hrow = hin + (long long)row * KH_;
        const T_* rrow = rin + (long long)row * KH_;
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float hb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const T_ rounded = static_cast<T_>(
                static_cast<float>(hrow[base + i]) + static_cast<float>(rrow[base + i]));
            hb[i] = static_cast<float>(rounded);
            ss += hb[i] * hb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            const float v = static_cast<float>(static_cast<T_>(
                hb[i] * nscale * static_cast<float>(nw[idx])));
            xf[(long long)row * KH_ + idx] = v;
            xb[(long long)row * KH_ + idx] = __nv_bfloat16(v);
        }
    }

    // ---- barrier 1 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[0], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[8], 1u);
        else while (atomicAdd(&ctr[8], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase A: router logits, tasks (row, expert-row) ------------------
    for (int task = wglobal; task < ROWS_ * NROUT_; task += GRID_ * WARPS) {
        const int row = task / NROUT_;
        const int rrow = task - row * NROUT_;
        const float* xrow = xf + (long long)row * KH_;
        const __nv_bfloat16* wrow = rw + (long long)rrow * KH_;
        float sum = 0.0f;
        for (int col = 4 * lane; col < KH_; col += 128) {
            const __nv_bfloat162 wab =
                *reinterpret_cast<const __nv_bfloat162*>(wrow + col);
            const __nv_bfloat162 wcd =
                *reinterpret_cast<const __nv_bfloat162*>(wrow + col + 2);
            sum = fmaf(__bfloat162float(wab.x), xrow[col], sum);
            sum = fmaf(__bfloat162float(wab.y), xrow[col + 1], sum);
            sum = fmaf(__bfloat162float(wcd.x), xrow[col + 2], sum);
            sum = fmaf(__bfloat162float(wcd.y), xrow[col + 3], sum);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, o);
        if (lane == 0) logits[(long long)row * NROUT_ + rrow] = sum;
    }

    // ---- barrier 2 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[1], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[9], 1u);
        else while (atomicAdd(&ctr[9], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase B: softmax + top-8 + renorm, row r on block r --------------
    __shared__ float local_max[2];
    __shared__ float local_norm[2];
    if (blk < ROWS_) {
        const int row = blk;
        float* rlog = logits + (long long)row * NROUT_;
        float* rprb = probs + (long long)row * NROUT_;
        float* ridx = idxf + row * 8;
        float* rsco = scoref + row * 8;
        constexpr int N_READS = 4;
        constexpr int SWARPS = 2;  // 64 active threads
        const bool active = tid < 64;
        const int slane = tid & 31;
        const int swarp = tid >> 5;
        float vals[N_READS];
        float maxval = -INFINITY;
        float normalizer = 0.0f;
        float prevmax;
        if (active) {
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                vals[i] = rlog[tid * N_READS + i];
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                maxval = fmaxf(maxval, vals[i]);
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                normalizer = normalizer + __expf(vals[i] - maxval);
            prevmax = maxval;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
            normalizer = normalizer * __expf(prevmax - maxval);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);
            prevmax = maxval;
            if (slane == 0) local_max[swarp] = maxval;
        }
        __syncthreads();
        if (active) {
            maxval = (slane < SWARPS) ? local_max[slane] : -INFINITY;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
            normalizer = normalizer * __expf(prevmax - maxval);
            if (slane == 0) local_norm[swarp] = normalizer;
        }
        __syncthreads();
        if (active) {
            normalizer = (slane < SWARPS) ? local_norm[slane] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);
            normalizer = 1.0f / normalizer;
            #pragma unroll
            for (int i = 0; i < N_READS; ++i)
                rprb[tid * N_READS + i] =
                    __expf(vals[i] - maxval) * normalizer;
        }
        __syncthreads();

        if (tid < 32) {
            int chosen[NEXP_];
            #pragma unroll
            for (int k = 0; k < NEXP_; ++k) {
                float bestv = -INFINITY;
                int besti = -1;
                for (int i = lane; i < NROUT_; i += 32) {
                    bool taken = false;
                    #pragma unroll
                    for (int t = 0; t < NEXP_; ++t)
                        if (t < k && chosen[t] == i) taken = true;
                    if (taken) continue;
                    const float v = rprb[i];
                    if (v > bestv || (v == bestv && i > besti)) {
                        bestv = v;
                        besti = i;
                    }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_down_sync(0xffffffffu, bestv, o);
                    const int oi = __shfl_down_sync(0xffffffffu, besti, o);
                    if (ov > bestv || (ov == bestv && oi > besti)) {
                        bestv = ov;
                        besti = oi;
                    }
                }
                besti = __shfl_sync(0xffffffffu, besti, 0);
                chosen[k] = besti;
            }
            if (tid == 0) {
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e) {
                const int src = NEXP_ - 1 - e;
                ridx[e] = static_cast<float>(chosen[src]);
                rsco[e] = rprb[chosen[src]];
            }
            const float lo = __fadd_rn(__fadd_rn(__fadd_rn(
                rsco[0], rsco[1]), rsco[2]), rsco[3]);
            const float hi = __fadd_rn(__fadd_rn(__fadd_rn(
                rsco[4], rsco[5]), rsco[6]), rsco[7]);
            const float sum = __fadd_rn(lo, hi);
            const float denom = sum + 1e-20f;
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e)
                rsco[e] = __fdiv_rn(rsco[e], denom);
            }
        }
    }

    // ---- barrier 3 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[2], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[10], 1u);
        else while (atomicAdd(&ctr[10], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase C: up_gate experts, tasks (row, expert, octet) -------------
    {
        constexpr int OCTS = 2 * KD_ / 8;
        const long long wstride = (2LL * KD_) * (KH_ / 16);
        const long long gstride = (2LL * KD_) * (KH_ / GS);
        for (int task = wglobal; task < ROWS_ * NEXP_ * OCTS;
             task += GRID_ * WARPS) {
            const int row = task / (NEXP_ * OCTS);
            const int rem = task - row * NEXP_ * OCTS;
            const int e = rem / OCTS;
            const int oct = rem - e * OCTS;
            const int we = static_cast<int>(idxf[row * 8 + e]);
            const unsigned int* wq =
                reinterpret_cast<const unsigned int*>(ugw) + we * wstride;
            const __nv_bfloat16* sc = ugs + we * gstride;
            const __nv_bfloat16* bi = ugb + we * gstride;
            const __nv_bfloat16* xrow = xb + (long long)row * KH_;
            float o0 = 0.0f, o1 = 0.0f;
            for (int kt = 0; kt < KH_ / 128; ++kt) {
                qmm_tile(xrow, wq, sc, bi, KH_ / 16, KH_ / GS,
                         oct * 8, kt * 128, GS, lane, o0, o1);
            }
            if ((lane >> 2) == 0) {
                const int c = oct * 8 + 2 * (lane & 3);
                float* ug = ugstage + ((long long)row * NEXP_ + e) * 2 * KD_;
                ug[c] = __bfloat162float(__nv_bfloat16(o0));
                ug[c + 1] = __bfloat162float(__nv_bfloat16(o1));
            }
        }
    }

    // ---- barrier 4 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[3], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[11], 1u);
        else while (atomicAdd(&ctr[11], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase S: SwiGLU into the global bf16 stage -----------------------
    // The production kernel recomputes this per block into shared memory;
    // B*8*KD_ bf16 no longer fits, so it is computed once (same formula,
    // same inputs, same bits) and staged globally behind one extra barrier.
    for (int i = blk * THREADS_ + tid; i < ROWS_ * NEXP_ * KD_;
         i += GRID_ * THREADS_) {
        const int re = i / KD_;
        const int c = i - re * KD_;
        const float* ug = ugstage + (long long)re * 2 * KD_;
        astage[i] = exact_swiglu(ug[KD_ + c], ug[c]);
    }

    // ---- barrier 5 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[4], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[12], 1u);
        else while (atomicAdd(&ctr[12], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase D: down experts, tasks (row, expert, octet) ----------------
    {
        constexpr int OCTS = ND_ / 8;
        const long long wstride = (long long)ND_ * (KD_ / 16);
        const long long gstride = (long long)ND_ * (KD_ / GS);
        for (int task = wglobal; task < ROWS_ * NEXP_ * OCTS;
             task += GRID_ * WARPS) {
            const int row = task / (NEXP_ * OCTS);
            const int rem = task - row * NEXP_ * OCTS;
            const int e = rem / OCTS;
            const int oct = rem - e * OCTS;
            const int we = static_cast<int>(idxf[row * 8 + e]);
            const unsigned int* wq =
                reinterpret_cast<const unsigned int*>(dnw) + we * wstride;
            const __nv_bfloat16* sc = dns + we * gstride;
            const __nv_bfloat16* bi = dnb + we * gstride;
            const __nv_bfloat16* arow =
                astage + ((long long)row * NEXP_ + e) * KD_;
            float a0 = 0.0f, a1 = 0.0f;
            for (int kt = 0; kt < KD_ / 128; ++kt) {
                qmm_tile(arow, wq, sc, bi, KD_ / 16, KD_ / GS,
                         oct * 8, kt * 128, GS, lane, a0, a1);
            }
            if ((lane >> 2) == 0) {
                const int c = oct * 8 + 2 * (lane & 3);
                float* ds = dstage + ((long long)row * NEXP_ + e) * ND_;
                ds[c] = __bfloat162float(__nv_bfloat16(a0));
                ds[c + 1] = __bfloat162float(__nv_bfloat16(a1));
            }
        }
    }

    // ---- barrier 6 --------------------------------------------------------
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[5], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[13], 1u);
        else while (atomicAdd(&ctr[13], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();
    if (blk >= ROWS_) return;

    // ---- phase E: aggregation + residual + next norm, row r on block r ----
    {
        const int row = blk;
        const T_* hrow = hin + (long long)row * KH_;
        const T_* rrow = rin + (long long)row * KH_;
        T_* horow = hout + (long long)row * KH_;
        T_* orow = out + (long long)row * KH_;
        const float* rsco = scoref + row * 8;
        const float* ds0 = dstage + (long long)row * NEXP_ * ND_;
        constexpr int VEC = KH_ / THREADS_;
        const int base = tid * VEC;
        float sb[VEC];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            float agg = 0.0f;
            #pragma unroll
            for (int e = 0; e < NEXP_; ++e)
                agg = __fadd_rn(agg,
                    __fmul_rn(ds0[e * ND_ + idx], rsco[e]));
            const float aggb = __bfloat162float(__nv_bfloat16(agg));
            const T_ s = static_cast<T_>(
                static_cast<float>(hrow[idx]) + static_cast<float>(rrow[idx]));
            const T_ s2 = static_cast<T_>(static_cast<float>(s) + aggb);
            horow[idx] = s2;
            sb[i] = static_cast<float>(s2);
            ss += sb[i] * sb[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = (lane < WARPS) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (tid == 0) gsum_s = tot;
        __syncthreads();
        const float nscale = rsqrtf(gsum_s / static_cast<float>(KH_) + EPS_);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = base + i;
            orow[idx] = static_cast<T_>(
                sb[i] * nscale * static_cast<float>(nw2[idx]));
        }
    }
"""


_moe_batch_megakernel_cache = {}


def _moe_batch_megakernel(eps):
    """The M=B research kernel; gated by maple_moe_batch_check.py before
    any production wiring."""
    kernel = _moe_batch_megakernel_cache.get(eps)
    if kernel is None:
        kernel = _moe_batch_megakernel_cache[eps] = mx.fast.cuda_kernel(
            name="maple_moe_batch_megakernel",
            input_names=["hin", "rin", "nw", "rw", "ugw", "ugs", "ugb",
                         "dnw", "dns", "dnb", "nw2"],
            output_names=["out", "hout", "scratch"],
            source=_MOE_BATCH_MEGAKERNEL_SOURCE.replace(
                "EPS_", f"{eps:.10e}f"),
            header=_MOE_EXACT_MEGAKERNEL_HEADER,
        )
    return kernel


def _moe_batch_megakernel_plan(block, ln, dtype, rows, grid=None,
                               threads=512):
    """Geometry for the batch kernel: the exact lane's gates plus the
    row-count bound (every row-per-block phase needs rows <= grid)."""
    base = _moe_exact_megakernel_plan(block, ln, dtype, grid=grid,
                                      threads=threads)
    if base is False:
        return False
    _, kwargs = base
    grid = _moe_megakernel_grid() if grid is None else grid
    if not (1 <= rows <= min(8, grid)):
        return False
    kh = block.switch_mlp.up_gate_proj.input_dims
    kd = block.switch_mlp.down_proj.input_dims
    nd = block.switch_mlp.down_proj.output_dims
    ne = block.gate.top_k
    nr = block.gate.num_experts
    scratch = (16 + rows * 8 * 2 + rows * nr * 2 + rows * kh
               + rows * kh // 2 + rows * ne * 2 * kd
               + rows * ne * kd // 2 + rows * ne * nd)
    return (
        _moe_batch_megakernel(ln.eps),
        {
            "template": [t for t in kwargs["template"]
                         if t[0] != "GRID_"] + [
                ("GRID_", grid), ("ROWS_", rows)],
            "grid": (grid * threads, 1, 1),
            "threadgroup": (threads, 1, 1),
            "output_shapes": [(rows, 1, nd), (rows, 1, kh), (scratch,)],
            "output_dtypes": [dtype, dtype, mx.float32],
            "init_value": 0,
        },
    )


def _moe_batch_call(layer, h, r, ln, next_w):
    """The M=B MoE step; same contract as `_moe_exact_megakernel_call`."""
    block = layer.mlp
    rows = h.shape[0]
    plans = getattr(block, "_batch_megakernel_plans", None)
    if plans is None:
        plans = block._batch_megakernel_plans = {}
    plan = plans.get(rows)
    if plan is None:
        # The batch kernel is register-heavier than the single-row one, so
        # its residency ceiling is lower (192 deadlocks on 82 SMs where
        # the production kernel runs it -- roughly 2 blocks/SM at 512
        # threads).  Per-profile defaults clamp to the smallest class
        # member (sm120: RTX 5080's 84 SMs -> 160); big hosts opt higher
        # via env -- measured: 128 on the 3090 (-10..12%), 256 on the
        # 5090 (-41..47%).
        try:
            grid = int(os.environ.get("MAPLE_BATCH_MOE_GRID", "0")) or None
        except ValueError:
            grid = None
        if grid is None:
            prof_name = _cuda_profile().name
            grid = {"sm100": 160, "sm120": 160}.get(prof_name)
        try:
            plan = _moe_batch_megakernel_plan(block, ln, h.dtype, rows,
                                              grid=grid)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            plan = False
        plans[rows] = plan
    if plan is False:
        return None
    kernel, kwargs = plan
    mlp = block.switch_mlp
    ug, dp = mlp.up_gate_proj, mlp.down_proj
    try:
        hn, hout, _ = kernel(
            inputs=[h, r, ln.weight, block.gate.weight, ug.weight, ug.scales,
                    ug.biases, dp.weight, dp.scales, dp.biases, next_w],
            **kwargs,
        )
    except (RuntimeError, TypeError, ValueError):
        plans[rows] = False
        return None
    return hout, hn


# The attention megakernel: the whole decode attention block -- qkv gemv,
# per-head RMSNorm + RoPE, the KV-cache append, single-token SDPA and the
# output projection -- in one dispatch behind three grid barriers.  Every
# phase is a proven bit recipe:
#   qkv / o_proj   the dense bf16 gemv order (maple_attention_semantics.py,
#                  12/12 at both shapes)
#   split+norm+rope  a line-for-line port of the shipped exact split kernel
#   sdpa           kL <= 1024: the kernel_sdpav_1pass port (12/12 at five
#                  context lengths); kL > 1024: the kernel_sdpav_2pass port
#                  (benchmarks/maple_attention_2pass_semantics.py, 12/12 at
#                  six lengths up to 8192) -- 32 fp32 slab partials per head
#                  behind one extra grid barrier, merged by a 32x32 block
#   cache append   value-identical writes into caller-owned contiguous
#                  buffers at the same physical slot the stock rotating
#                  cache uses -- SDPA walks the same physical order, so the
#                  bits cannot move
_ATTN_MEGAKERNEL_HEADER = r"""
// The stock qmv row: 2-bit affine, group_size 128, elems_per_thread 16.
// Lane l owns 16 consecutive elements at stride 512; each tile is exactly
// one packed word and one scale/bias pair; the per-element accumulators are
// bf16 and take one HFMA rounding per tile; the tail is a linear float sum
// of the sixteen lanesums, then the butterfly.  Matches qmv.cu bit for bit.
__device__ __forceinline__ float qmv_row(
    const float* xf,             // activations, exact float(bf16) values
    const unsigned int* wq,      // packed rows [n][k/16]
    const __nv_bfloat16* sc,     // scales [n][k/128]
    const __nv_bfloat16* bi,     // biases [n][k/128]
    int row,
    int k,
    int lane) {
    const unsigned int* wrow = wq + (long long)row * (k >> 4);
    const __nv_bfloat16* srow = sc + (long long)row * (k >> 7);
    const __nv_bfloat16* brow = bi + (long long)row * (k >> 7);
    __nv_bfloat16 sums[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) sums[i] = __nv_bfloat16(0.0f);
    for (int base = lane * 16; base < k; base += 512) {
        const unsigned int word = wrow[base >> 4];
        const __nv_bfloat16 scale = srow[base >> 7];
        const __nv_bfloat16 bias = brow[base >> 7];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = (word >> (2 * i)) & 3;
            const __nv_bfloat16 wdq = __hadd(
                __hmul(__nv_bfloat16(float(q)), scale), bias);
            sums[i] = __hfma(__nv_bfloat16(xf[base + i]), wdq, sums[i]);
        }
    }
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) sum += __bfloat162float(sums[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}
"""

_ATTN_MEGAKERNEL_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int HD = 128;
    constexpr int VPL = HD / WARP;      // per-lane values in a head row
    constexpr int NH = NQ_ + 2 * NKV_;
    constexpr int QKVROWS = NH * HD;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blk = blockIdx.x;
    const int wglobal = blk * (THREADS_ / 32) + warp;

    constexpr int OFF_QKV = 16;
    constexpr int OFF_Q = OFF_QKV + QKVROWS;
    constexpr int OFF_O = OFF_Q + NQ_ * HD;
    constexpr int OFF_P = OFF_O + NQ_ * HD;      // 2-pass slab partials
    constexpr int OFF_PS = OFF_P + NQ_ * 32 * HD;  // 2-pass slab sums
    constexpr int OFF_PM = OFF_PS + NQ_ * 32;      // 2-pass slab maxs
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch);
    float* stq_kv = scratch + OFF_QKV;
    float* stq = scratch + OFF_Q;
    float* sto = scratch + OFF_O;
    float* partials = scratch + OFF_P;
    float* psums = scratch + OFF_PS;
    float* pmaxs = scratch + OFF_PM;

    // Persistent step counters, advanced by the kernel itself so every
    // input pointer stays stable and the CUDA graph is captured once.
    float* live = const_cast<float*>(scalars);
    const float pos = live[0];
    const int kL = static_cast<int>(live[1]);
    const int slot = static_cast<int>(live[2]);
    constexpr float eps = EPS_;
    constexpr float log2b = LOG2B_;

    // ---- phase A: the fused qkv projection, the stock qmv recipe ----------
    __shared__ float hn_s[KH_];
    for (int i = tid; i < KH_; i += THREADS_)
        hn_s[i] = static_cast<float>(hn[i]);
    __syncthreads();
    for (int row = wglobal; row < QKVROWS; row += GRID_ * (THREADS_ / 32)) {
        const float sum = qmv_row(
            hn_s, reinterpret_cast<const unsigned int*>(wqkv),
            reinterpret_cast<const __nv_bfloat16*>(sqkv),
            reinterpret_cast<const __nv_bfloat16*>(bqkv),
            row, KH_, lane);
        if (lane == 0)
            stq_kv[row] = static_cast<float>(static_cast<T_>(sum));
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[0], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[8], 1u);
        else while (atomicAdd(&ctr[8], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase B: split + QK norm + RoPE + cache append, block 0 ----------
    // One warp per head, the shipped exact split kernel line for line.
    if (blk == 0 && warp < NH) {
        const int head = warp;
        const float* xh = stq_kv + head * HD;
        T_* kc = const_cast<T_*>(kcache);
        T_* vc = const_cast<T_*>(vcache);
        if (head >= NQ_ + NKV_) {
            const int kvh = head - NQ_ - NKV_;
            T_* vh = vc + ((long long)kvh * CAP_ + slot) * HD;
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const int j = lane * VPL + i;
                vh[j] = static_cast<T_>(xh[j]);
            }
        } else {
            const T_* wh = wqk + head * HD;
            float sum_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const float value = xh[lane * VPL + i];
                sum_sq += value * value;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
            sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);
            const float nscale = rsqrtf(sum_sq / (float)HD + eps);

            // Partial RoPE: only the first RD_ dims rotate, paired
            // (p, p + RD_/2), exactly as the shipped split kernel.
            float rope_cos[VPL], rope_sin[VPL];
            if (ROPE_) {
                #pragma unroll
                for (int i = 0; i < VPL; ++i) {
                    const int j = lane * VPL + i;
                    if (j < RD_) {
                        constexpr int rhalf = RD_ > 0 ? RD_ / 2 : 1;
                        const int pp = (j < rhalf) ? j : j - rhalf;
                        const float fraction =
                            (float)pp / (float)(RD_ / 2);
                        const float frequency = exp2f(-fraction * log2b);
                        const float angle = pos * frequency;
                        rope_cos[i] = cosf(angle);
                        rope_sin[i] = sinf(angle);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const int j = lane * VPL + i;
                const T_ normalized = static_cast<T_>(
                    xh[j] * nscale * static_cast<float>(wh[j]));
                float value = static_cast<float>(normalized);
                if (ROPE_ && j < RD_) {
                    constexpr int rhalf = RD_ > 0 ? RD_ / 2 : 1;
                    const int pair = j < rhalf ? j + rhalf : j - rhalf;
                    const T_ paired_normalized = static_cast<T_>(
                        xh[pair] * nscale * static_cast<float>(wh[pair]));
                    const float paired = static_cast<float>(paired_normalized);
                    value = (j < rhalf)
                        ? value * rope_cos[i] - paired * rope_sin[i]
                        : SECOND_HALF_;
                }
                const T_ out_b = static_cast<T_>(value);
                if (head < NQ_) {
                    stq[head * HD + j] = static_cast<float>(out_b);
                } else {
                    const int kvh = head - NQ_;
                    kc[((long long)kvh * CAP_ + slot) * HD + j] = out_b;
                }
            }
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[1], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[9], 1u);
        else while (atomicAdd(&ctr[9], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase C: single-token SDPA ----------------------------------------
    // kL <= 1024: the kernel_sdpav_1pass port, one head per block -- 32 warps
    // interleave the keys, base-2 online softmax, xor butterflies, __frcp_rn,
    // shared transpose merge.  kL > 1024: the kernel_sdpav_2pass port -- 32
    // slabs per head (8 warps each, keys at stride 256), fp32 partials scaled
    // to the slab max, one extra grid barrier, then a 32x32 merge block per
    // head walking the slab maxs/sums exactly like the stock second kernel.
    __shared__ float outs[32][33];
    __shared__ float maxs[32];
    __shared__ float sums[32];
    const float scale_log2 = SCALE_ * 1.44269504088896340736f;
    if (kL <= 1024) {
    if (blk < NQ_) {
        const int head = blk;
        const int kvh = head / (NQ_ / NKV_);
        float q[VPL], k[VPL], o[VPL];
        #pragma unroll
        for (int i = 0; i < VPL; ++i) {
            q[i] = scale_log2 * stq[head * HD + VPL * lane + i];
            o[i] = 0.0f;
        }
        float max_score = -3.402823466e38f;
        float sum_exp = 0.0f;
        const long long kh = (long long)kvh * CAP_ * HD;
        for (int i = warp; i < kL; i += 32) {
            #pragma unroll
            for (int j = 0; j < VPL; ++j)
                k[j] = static_cast<float>(
                    kcache[kh + (long long)i * HD + VPL * lane + j]);
            float score = 0.0f;
            #pragma unroll
            for (int j = 0; j < VPL; ++j) score += q[j] * k[j];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_xor_sync(0xffffffffu, score, off);
            const float new_max = fmaxf(max_score, score);
            const float factor = exp2f(max_score - new_max);
            const float exp_score = exp2f(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;
            #pragma unroll
            for (int j = 0; j < VPL; ++j)
                o[j] = o[j] * factor + exp_score * static_cast<float>(
                    vcache[kh + (long long)i * HD + VPL * lane + j]);
        }
        if (lane == 0) { maxs[warp] = max_score; sums[warp] = sum_exp; }
        __syncthreads();
        max_score = maxs[lane];
        float new_max = max_score;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            new_max = fmaxf(new_max,
                            __shfl_xor_sync(0xffffffffu, new_max, off));
        const float factor = exp2f(max_score - new_max);
        float se = sums[lane] * factor;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            se += __shfl_xor_sync(0xffffffffu, se, off);
        se = (se == 0.0f) ? 0.0f : __frcp_rn(se);
        #pragma unroll
        for (int i = 0; i < VPL; ++i) {
            outs[lane][warp] = o[i];
            __syncthreads();
            float ot = outs[warp][lane] * factor;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ot += __shfl_xor_sync(0xffffffffu, ot, off);
            o[i] = ot * se;
            __syncthreads();
        }
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < VPL; ++i)
                sto[head * HD + VPL * warp + i] = static_cast<float>(
                    static_cast<T_>(o[i]));
        }
    }
    } else {
    // 2-pass, pass 1: every block hosts four 8-warp slabs; slab s of head h
    // walks keys s*8+w :: 256 with a warp-local online softmax, merges its
    // eight warps through the -1e9 lane mask and a linear j=1..7 fold, and
    // writes fp32 partials scaled to the slab max.  All 64 blocks make the
    // same fixed slab-wave count, so the shared reuse stays in lockstep.
    {
        const int sub = warp >> 3;         // four slabs per real block
        const int swrp = warp & 7;         // warp within the slab
        for (int vb = blk * 4 + sub; vb < NQ_ * 32; vb += GRID_ * 4) {
            const int head = vb >> 5;
            const int slab = vb & 31;
            const int kvh = head / (NQ_ / NKV_);
            float q[VPL], k[VPL], o[VPL];
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                q[i] = scale_log2 * stq[head * HD + VPL * lane + i];
                o[i] = 0.0f;
            }
            float max_score = -3.402823466e38f;
            float sum_exp = 0.0f;
            const long long kh = (long long)kvh * CAP_ * HD;
            for (int i = slab * 8 + swrp; i < kL; i += 256) {
                #pragma unroll
                for (int j = 0; j < VPL; ++j)
                    k[j] = static_cast<float>(
                        kcache[kh + (long long)i * HD + VPL * lane + j]);
                float score = 0.0f;
                #pragma unroll
                for (int j = 0; j < VPL; ++j) score += q[j] * k[j];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    score += __shfl_xor_sync(0xffffffffu, score, off);
                const float new_max = fmaxf(max_score, score);
                const float factor = exp2f(max_score - new_max);
                const float exp_score = exp2f(score - new_max);
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                #pragma unroll
                for (int j = 0; j < VPL; ++j)
                    o[j] = o[j] * factor + exp_score * static_cast<float>(
                        vcache[kh + (long long)i * HD + VPL * lane + j]);
            }
            if (lane == 0) {
                maxs[sub * 8 + swrp] = max_score;
                sums[sub * 8 + swrp] = sum_exp;
            }
            __syncthreads();
            const float wmax = (lane < 8) ? maxs[sub * 8 + lane] : -1e9f;
            float new_max = wmax;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                new_max = fmaxf(new_max,
                                __shfl_xor_sync(0xffffffffu, new_max, off));
            const float factor = exp2f(wmax - new_max);
            float se = (lane < 8) ? sums[sub * 8 + lane] : 0.0f;
            se *= factor;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                se += __shfl_xor_sync(0xffffffffu, se, off);
            const long long p = (long long)head * 32 + slab;
            if (swrp == 0 && lane == 0) { psums[p] = se; pmaxs[p] = new_max; }
            const float ff = exp2f(maxs[sub * 8 + swrp] - new_max);
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                outs[sub * 8 + swrp][lane] = o[i] * ff;
                __syncthreads();
                if (swrp == 0) {
                    float ot = outs[sub * 8][lane];
                    #pragma unroll
                    for (int j = 1; j < 8; ++j)
                        ot += outs[sub * 8 + j][lane];
                    o[i] = ot;
                }
                __syncthreads();
            }
            if (swrp == 0) {
                #pragma unroll
                for (int i = 0; i < VPL; ++i)
                    partials[p * HD + VPL * lane + i] = o[i];
            }
            __syncthreads();
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[3], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[11], 1u);
        else while (atomicAdd(&ctr[11], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // 2-pass, pass 2: one 32x32 block per head; lane l owns slab l's
    // max/sum, warp w owns slab w's partial, transposed shared merge,
    // __frcp_rn of the global sum, bf16 rounding into the staging row.
    if (blk < NQ_) {
        const int head = blk;
        const long long p0 = (long long)head * 32;
        const float bmax = pmaxs[p0 + lane];
        float new_max = bmax;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            new_max = fmaxf(new_max,
                            __shfl_xor_sync(0xffffffffu, new_max, off));
        const float factor = exp2f(bmax - new_max);
        float se = psums[p0 + lane] * factor;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            se += __shfl_xor_sync(0xffffffffu, se, off);
        se = (se == 0.0f) ? 0.0f : __frcp_rn(se);
        float o[VPL];
        #pragma unroll
        for (int i = 0; i < VPL; ++i)
            o[i] = partials[(p0 + warp) * HD + VPL * lane + i];
        #pragma unroll
        for (int i = 0; i < VPL; ++i) {
            outs[lane][warp] = o[i];
            __syncthreads();
            float ot = outs[warp][lane] * factor;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ot += __shfl_xor_sync(0xffffffffu, ot, off);
            o[i] = ot * se;
            __syncthreads();
        }
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < VPL; ++i)
                sto[head * HD + VPL * warp + i] = static_cast<float>(
                    static_cast<T_>(o[i]));
        }
    }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[2], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[10], 1u);
        else while (atomicAdd(&ctr[10], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase D: the output projection, the stock qmv recipe -------------
    for (int row = wglobal; row < KH_; row += GRID_ * (THREADS_ / 32)) {
        const float sum = qmv_row(
            sto, reinterpret_cast<const unsigned int*>(wo),
            reinterpret_cast<const __nv_bfloat16*>(so_),
            reinterpret_cast<const __nv_bfloat16*>(bo_),
            row, NQ_ * HD, lane);
        if (lane == 0) out[row] = static_cast<T_>(sum);
    }

    if (blk == 0 && tid == 0) {
        live[0] = pos + 1.0f;
        live[1] = (kL < CAP_) ? (float)(kL + 1) : (float)CAP_;
        const int nslot = slot + 1;
        live[2] = (nslot == CAP_) ? 0.0f : (float)nslot;
    }
"""


_ATTN_VERIFY_AB_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int HD = 128;
    constexpr int VPL = HD / WARP;
    constexpr int NH = NQ_ + 2 * NKV_;
    constexpr int QKVROWS = NH * HD;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blk = blockIdx.x;
    const int wglobal = blk * (THREADS_ / 32) + warp;

    constexpr int OFF_QKV = 16;
    constexpr int OFF_Q = OFF_QKV + ROWS_ * QKVROWS;
    constexpr int OFF_O = OFF_Q + ROWS_ * NQ_ * HD;
    constexpr int OFF_HNF = OFF_O + ROWS_ * NQ_ * HD;
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch);
    float* stq_kv = scratch + OFF_QKV;
    float* stq = scratch + OFF_Q;
    float* sto = scratch + OFF_O;
    float* hnf = scratch + OFF_HNF;

    float* live = const_cast<float*>(scalars);
    const float pos0 = live[0];
    const int kl0 = static_cast<int>(live[1]);
    const int slot0 = static_cast<int>(live[2]);
    constexpr float eps = EPS_;
    constexpr float log2b = LOG2B_;

    // ---- phase A0: stage the pack inputs as exact float(bf16) ----------
    for (int i = blk * THREADS_ + tid; i < ROWS_ * KH_;
         i += GRID_ * THREADS_) {
        hnf[i] = static_cast<float>(hn[i]);
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old0 = atomicAdd(&ctr[3], 1u);
        if (old0 == GRID_ - 1) atomicExch(&ctr[11], 1u);
        else while (atomicAdd(&ctr[11], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase A: qkv for every pack row (the stock qmv recipe) --------
    for (int task = wglobal; task < ROWS_ * QKVROWS;
         task += GRID_ * (THREADS_ / 32)) {
        const int r = task / QKVROWS;
        const int qrow = task % QKVROWS;
        const float sum = qmv_row(
            hnf + (long long)r * KH_,
            reinterpret_cast<const unsigned int*>(wqkv),
            reinterpret_cast<const __nv_bfloat16*>(sqkv),
            reinterpret_cast<const __nv_bfloat16*>(bqkv),
            qrow, KH_, lane);
        if (lane == 0)
            stq_kv[(long long)r * QKVROWS + qrow] =
                static_cast<float>(static_cast<T_>(sum));
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[0], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[8], 1u);
        else while (atomicAdd(&ctr[8], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase B: split + QK norm + RoPE + cache append, per (row, head)
    for (int task = wglobal; task < ROWS_ * NH;
         task += GRID_ * (THREADS_ / 32)) {
        const int r = task / NH;
        const int head = task % NH;
        const float* xh = stq_kv + (long long)r * QKVROWS + head * HD;
        const float pos = RAGGED_ ? live[3 + r * 3 + 0]
            : (BATCH_ ? pos0 : (pos0 + (float)r));
        const int slot = RAGGED_ ? (int)live[3 + r * 3 + 2]
            : (BATCH_ ? slot0 : (slot0 + r));
        const long long bplane = (RAGGED_ || BATCH_)
            ? (long long)r * NKV_ * CAP_ * HD : 0;
        T_* kc = const_cast<T_*>(kcache) + bplane;
        T_* vc = const_cast<T_*>(vcache) + bplane;
        if (head >= NQ_ + NKV_) {
            const int kvh = head - NQ_ - NKV_;
            T_* vh = vc + ((long long)kvh * CAP_ + slot) * HD;
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const int j = lane * VPL + i;
                vh[j] = static_cast<T_>(xh[j]);
            }
        } else {
            const T_* wh = wqk + head * HD;
            float sum_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const float value = xh[lane * VPL + i];
                sum_sq += value * value;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
            sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);
            const float nscale = rsqrtf(sum_sq / (float)HD + eps);
            float rope_cos[VPL], rope_sin[VPL];
            if (ROPE_) {
                #pragma unroll
                for (int i = 0; i < VPL; ++i) {
                    const int j = lane * VPL + i;
                    if (j < RD_) {
                        constexpr int rhalf = RD_ > 0 ? RD_ / 2 : 1;
                        const int pp = (j < rhalf) ? j : j - rhalf;
                        const float fraction = (float)pp / (float)(RD_ / 2);
                        const float frequency = exp2f(-fraction * log2b);
                        const float angle = pos * frequency;
                        rope_cos[i] = cosf(angle);
                        rope_sin[i] = sinf(angle);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                const int j = lane * VPL + i;
                const T_ normalized = static_cast<T_>(
                    xh[j] * nscale * static_cast<float>(wh[j]));
                float value = static_cast<float>(normalized);
                if (ROPE_ && j < RD_) {
                    constexpr int rhalf = RD_ > 0 ? RD_ / 2 : 1;
                    const int pair = j < rhalf ? j + rhalf : j - rhalf;
                    const T_ paired_normalized = static_cast<T_>(
                        xh[pair] * nscale * static_cast<float>(wh[pair]));
                    const float paired = static_cast<float>(paired_normalized);
                    value = (j < rhalf)
                        ? value * rope_cos[i] - paired * rope_sin[i]
                        : SECOND_HALF_;
                }
                const T_ out_b = static_cast<T_>(value);
                if (head < NQ_) {
                    stq[((long long)r * NQ_ + head) * HD + j] =
                        static_cast<float>(out_b);
                } else {
                    const int kvh = head - NQ_;
                    kc[((long long)kvh * CAP_ + slot) * HD + j] = out_b;
                }
            }
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[1], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[9], 1u);
        else while (atomicAdd(&ctr[9], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

"""


_ATTN_VERIFY_CD_SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int HD = 128;
    constexpr int VPL = HD / WARP;
    constexpr int NH = NQ_ + 2 * NKV_;
    constexpr int QKVROWS = NH * HD;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blk = blockIdx.x;
    const int wglobal = blk * (THREADS_ / 32) + warp;

    constexpr int OFF_QKV = 16;
    constexpr int OFF_Q = OFF_QKV + ROWS_ * QKVROWS;
    constexpr int OFF_O = OFF_Q + ROWS_ * NQ_ * HD;
    constexpr int OFF_HNF = OFF_O + ROWS_ * NQ_ * HD;
    constexpr int OFF_P = OFF_HNF + ROWS_ * KH_;   // 2-pass slab partials
    constexpr int OFF_PS = OFF_P + ROWS_ * NQ_ * 32 * HD;
    constexpr int OFF_PM = OFF_PS + ROWS_ * NQ_ * 32;
    float* scratch_w = const_cast<float*>(scratch_in);
    unsigned int* ctr = reinterpret_cast<unsigned int*>(scratch_w);
    float* stq_kv = scratch_w + OFF_QKV;
    float* stq = scratch_w + OFF_Q;
    float* sto = scratch_w + OFF_O;
    float* hnf = scratch_w + OFF_HNF;

    float* live = const_cast<float*>(scalars);
    const float pos0 = live[0];
    const int kl0 = static_cast<int>(live[1]);
    const int slot0 = static_cast<int>(live[2]);
    constexpr float eps = EPS_;
    constexpr float log2b = LOG2B_;
    (void)pos0; (void)eps; (void)log2b; (void)hnf; (void)stq_kv;

    // ---- phase C: SDPA per (head, row), causal via kL = kl0 + r ---------
    // Each row runs the algorithm the stock dispatch would pick for its
    // own kL: the 1-pass port at kL <= 1024, the 2-pass slab port past
    // it.  Rows in 1-pass mode skip the slab sections (whole-block
    // continue -- every thread of a block shares one task) and 2-pass
    // rows skip the 1-pass loop; the slab barrier is unconditional.
    {
        __shared__ float outs[32][33];
        __shared__ float maxs[32];
        __shared__ float sums[32];
        const float scale_log2 = SCALE_ * 1.44269504088896340736f;
        for (int task = blk; task < NQ_ * ROWS_; task += GRID_) {
            const int head = task / ROWS_;
            const int r = task % ROWS_;
            const int kvh = head / (NQ_ / NKV_);
            const int kL = RAGGED_ ? (int)live[3 + r * 3 + 1] : (BATCH_ ? kl0 : (kl0 + r));
            if (kL > 1024) continue;  // whole block: no divergence
            float q[VPL], k[VPL], o[VPL];
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                q[i] = scale_log2
                     * stq[((long long)r * NQ_ + head) * HD + VPL * lane + i];
                o[i] = 0.0f;
            }
            float max_score = -3.402823466e38f;
            float sum_exp = 0.0f;
            const long long kh = (long long)kvh * CAP_ * HD
                + ((RAGGED_ || BATCH_) ? (long long)r * NKV_ * CAP_ * HD : 0);
            for (int i = warp; i < kL; i += 32) {
                #pragma unroll
                for (int j = 0; j < VPL; ++j)
                    k[j] = static_cast<float>(
                        kcache[kh + (long long)i * HD + VPL * lane + j]);
                float score = 0.0f;
                #pragma unroll
                for (int j = 0; j < VPL; ++j) score += q[j] * k[j];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    score += __shfl_xor_sync(0xffffffffu, score, off);
                const float new_max = fmaxf(max_score, score);
                const float factor = exp2f(max_score - new_max);
                const float exp_score = exp2f(score - new_max);
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                #pragma unroll
                for (int j = 0; j < VPL; ++j)
                    o[j] = o[j] * factor + exp_score * static_cast<float>(
                        vcache[kh + (long long)i * HD + VPL * lane + j]);
            }
            if (lane == 0) { maxs[warp] = max_score; sums[warp] = sum_exp; }
            __syncthreads();
            max_score = maxs[lane];
            float new_max = max_score;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                new_max = fmaxf(new_max,
                                __shfl_xor_sync(0xffffffffu, new_max, off));
            const float factor = exp2f(max_score - new_max);
            float se = sums[lane] * factor;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                se += __shfl_xor_sync(0xffffffffu, se, off);
            se = (se == 0.0f) ? 0.0f : __frcp_rn(se);
            #pragma unroll
            for (int i = 0; i < VPL; ++i) {
                outs[lane][warp] = o[i];
                __syncthreads();
                float ot = outs[warp][lane] * factor;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    ot += __shfl_xor_sync(0xffffffffu, ot, off);
                o[i] = ot * se;
                __syncthreads();
            }
            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < VPL; ++i)
                    sto[((long long)r * NQ_ + head) * HD + VPL * warp + i] =
                        static_cast<float>(static_cast<T_>(o[i]));
            }
            __syncthreads();
        }

        // -- 2-pass pass 1: the production slab recipe with a row plane.
        // Four 8-warp slabs per block; sub-warps of one block can sit on
        // DIFFERENT rows, so the loop is wave-shaped: barriers run
        // unconditionally, work is gated per slab (shared traffic stays
        // inside each sub's own maxs/sums/outs slots).
        {
            constexpr int PHD = HD + 2;
            float* partials = scratch_w + OFF_P;
            float* psums = scratch_w + OFF_PS;
            float* pmaxs = scratch_w + OFF_PM;
            (void)partials; (void)psums; (void)pmaxs; (void)PHD;
            const int sub = warp >> 3;
            const int swrp = warp & 7;
            const int total1 = ROWS_ * NQ_ * 32;
            const int waves1 = (total1 + GRID_ * 4 - 1) / (GRID_ * 4);
            for (int w = 0; w < waves1; ++w) {
                const int vb = (w * GRID_ + blk) * 4 + sub;
                int r = 0, head = 0, slab = 0, kL = 0;
                bool act = vb < total1;
                if (act) {
                    r = vb / (NQ_ * 32);
                    const int rem = vb - r * NQ_ * 32;
                    head = rem >> 5;
                    slab = rem & 31;
                    kL = RAGGED_ ? (int)live[3 + r * 3 + 1] : (BATCH_ ? kl0 : (kl0 + r));
                    act = kL > 1024;
                }
                const int kvh = head / (NQ_ / NKV_);
                float q[VPL], k[VPL], o[VPL];
                float max_score = -3.402823466e38f;
                float sum_exp = 0.0f;
                if (act) {
                    #pragma unroll
                    for (int i = 0; i < VPL; ++i) {
                        q[i] = scale_log2 * stq[
                            ((long long)r * NQ_ + head) * HD + VPL * lane + i];
                        o[i] = 0.0f;
                    }
                    const long long kh = (long long)kvh * CAP_ * HD
                        + ((RAGGED_ || BATCH_) ? (long long)r * NKV_ * CAP_ * HD : 0);
                    for (int i = slab * 8 + swrp; i < kL; i += 256) {
                        #pragma unroll
                        for (int j = 0; j < VPL; ++j)
                            k[j] = static_cast<float>(
                                kcache[kh + (long long)i * HD + VPL * lane + j]);
                        float score = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < VPL; ++j) score += q[j] * k[j];
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            score += __shfl_xor_sync(0xffffffffu, score, off);
                        const float new_max = fmaxf(max_score, score);
                        const float factor = exp2f(max_score - new_max);
                        const float exp_score = exp2f(score - new_max);
                        max_score = new_max;
                        sum_exp = sum_exp * factor + exp_score;
                        #pragma unroll
                        for (int j = 0; j < VPL; ++j)
                            o[j] = o[j] * factor + exp_score * static_cast<float>(
                                vcache[kh + (long long)i * HD + VPL * lane + j]);
                    }
                    if (lane == 0) {
                        maxs[sub * 8 + swrp] = max_score;
                        sums[sub * 8 + swrp] = sum_exp;
                    }
                }
                __syncthreads();
                float new_max = -3.402823466e38f;
                if (act) {
                    const float wmax = (lane < 8) ? maxs[sub * 8 + lane] : -1e9f;
                    new_max = wmax;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        new_max = fmaxf(new_max,
                                        __shfl_xor_sync(0xffffffffu, new_max, off));
                    const float factor = exp2f(wmax - new_max);
                    float se = (lane < 8) ? sums[sub * 8 + lane] : 0.0f;
                    se *= factor;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        se += __shfl_xor_sync(0xffffffffu, se, off);
                    const long long p = ((long long)r * NQ_ + head) * 32 + slab;
                    if (swrp == 0 && lane == 0) {
                        psums[p] = se;
                        pmaxs[p] = new_max;
                    }
                }
                const float ff = act
                    ? exp2f(maxs[sub * 8 + swrp] - new_max) : 0.0f;
                #pragma unroll
                for (int i = 0; i < VPL; ++i) {
                    if (act) outs[sub * 8 + swrp][lane] = o[i] * ff;
                    __syncthreads();
                    if (act && swrp == 0) {
                        float ot = outs[sub * 8][lane];
                        #pragma unroll
                        for (int j = 1; j < 8; ++j)
                            ot += outs[sub * 8 + j][lane];
                        o[i] = ot;
                    }
                    __syncthreads();
                }
                if (act && swrp == 0) {
                    const long long p = ((long long)r * NQ_ + head) * 32 + slab;
                    #pragma unroll
                    for (int i = 0; i < VPL; ++i)
                        partials[p * HD + VPL * lane + i] = o[i];
                }
                __syncthreads();
            }
        }

        // -- slab barrier ---------------------------------------------------
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            // AB consumed counter pairs 0-3; the CD kernel shares the
            // scratch, so its barriers live on fresh slots 4 and 5.
            const unsigned int old = atomicAdd(&ctr[5], 1u);
            if (old == GRID_ - 1) atomicExch(&ctr[13], 1u);
            else while (atomicAdd(&ctr[13], 0u) == 0u) __nanosleep(48);
        }
        __syncthreads();
        __threadfence();

        // -- 2-pass pass 2: one 32x32 merge per (row, head), wave-shaped.
        {
            float* partials = scratch_w + OFF_P;
            float* psums = scratch_w + OFF_PS;
            float* pmaxs = scratch_w + OFF_PM;
            const int total2 = ROWS_ * NQ_;
            const int waves2 = (total2 + GRID_ - 1) / GRID_;
            for (int w = 0; w < waves2; ++w) {
                const int t = w * GRID_ + blk;
                int r = 0, head = 0, kL = 0;
                bool act = t < total2;
                if (act) {
                    r = t / NQ_;
                    head = t - r * NQ_;
                    kL = RAGGED_ ? (int)live[3 + r * 3 + 1] : (BATCH_ ? kl0 : (kl0 + r));
                    act = kL > 1024;
                }
                float o[VPL];
                float factor = 0.0f, se = 0.0f;
                if (act) {
                    const long long p0 = ((long long)r * NQ_ + head) * 32;
                    const float bmax = pmaxs[p0 + lane];
                    float new_max = bmax;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        new_max = fmaxf(new_max,
                                        __shfl_xor_sync(0xffffffffu, new_max, off));
                    factor = exp2f(bmax - new_max);
                    se = psums[p0 + lane] * factor;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        se += __shfl_xor_sync(0xffffffffu, se, off);
                    se = (se == 0.0f) ? 0.0f : __frcp_rn(se);
                    #pragma unroll
                    for (int i = 0; i < VPL; ++i)
                        o[i] = partials[(p0 + warp) * HD + VPL * lane + i];
                }
                #pragma unroll
                for (int i = 0; i < VPL; ++i) {
                    if (act) outs[lane][warp] = o[i];
                    __syncthreads();
                    if (act) {
                        float ot = outs[warp][lane] * factor;
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            ot += __shfl_xor_sync(0xffffffffu, ot, off);
                        o[i] = ot * se;
                    }
                    __syncthreads();
                }
                if (act && lane == 0) {
                    #pragma unroll
                    for (int i = 0; i < VPL; ++i)
                        sto[((long long)r * NQ_ + head) * HD + VPL * warp + i]
                            = static_cast<float>(static_cast<T_>(o[i]));
                }
                __syncthreads();
            }
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int old = atomicAdd(&ctr[4], 1u);
        if (old == GRID_ - 1) atomicExch(&ctr[12], 1u);
        else while (atomicAdd(&ctr[12], 0u) == 0u) __nanosleep(48);
    }
    __syncthreads();
    __threadfence();

    // ---- phase D: o_proj for every pack row -----------------------------
    for (int task = wglobal; task < ROWS_ * KH_;
         task += GRID_ * (THREADS_ / 32)) {
        const int r = task / KH_;
        const int orow = task % KH_;
        const float sum = qmv_row(
            sto + (long long)r * NQ_ * HD,
            reinterpret_cast<const unsigned int*>(wo),
            reinterpret_cast<const __nv_bfloat16*>(so_),
            reinterpret_cast<const __nv_bfloat16*>(bo_),
            orow, NQ_ * HD, lane);
        if (lane == 0)
            out[(long long)r * KH_ + orow] = static_cast<T_>(sum);
    }

    if (blk == 0 && tid == 0) {
        if (RAGGED_) {
            // every row advances its own counters, production clamp/wrap
            for (int r = 0; r < ROWS_; ++r) {
                const float pr = live[3 + r * 3 + 0];
                const int kr = (int)live[3 + r * 3 + 1];
                const int sr = (int)live[3 + r * 3 + 2];
                live[3 + r * 3 + 0] = pr + 1.0f;
                live[3 + r * 3 + 1] =
                    (kr < CAP_) ? (float)(kr + 1) : (float)CAP_;
                const int ns = sr + 1;
                live[3 + r * 3 + 2] = (ns == CAP_) ? 0.0f : (float)ns;
            }
        } else if (BATCH_) {
            // one token per stream: the counters advance like a single
            // sequential step, shared by every row -- with the SAME
            // clamp and ring wrap the production tail applies, or a
            // wrapped sliding window walks its device counters off the
            // buffer while the host counters stay sane.
            live[0] = pos0 + 1.0f;
            live[1] = (kl0 < CAP_) ? (float)(kl0 + 1) : (float)CAP_;
            const int nslot = slot0 + 1;
            live[2] = (nslot == CAP_) ? 0.0f : (float)nslot;
        } else {
            live[0] = pos0 + (float)ROWS_;
            live[1] = (float)(kl0 + ROWS_);
            live[2] = (float)(slot0 + ROWS_);
        }
    }
"""


_attn_verify_cache = {}


def _attn_verify_kernels(profile_name, use_rope, scale, eps, log2b,
                         batch=False, second_half_form=None):
    if second_half_form is None:
        # Bit-gated per profile: form 1 reproduces the production sm86
        # compilation byte-for-byte (batch_kv_debug, 0 diffs x8); the
        # Blackwell profiles pin form 0 in production already. Re-confirm
        # on each new profile during scale-out.
        second_half_form = 0 if profile_name in ("sm100", "sm120") else 1
    key = (profile_name, use_rope, scale, eps, log2b, batch,
           second_half_form)
    pair = _attn_verify_cache.get(key)
    if pair is None:
        # The upper-half RoPE expression is contraction-sensitive (the
        # sm100/sm120 story, chronicle #3): a fresh compilation of the
        # same source can fma it differently from the production kernel.
        # The verify pair therefore PINS the form; form 0/1 exist so the
        # bit gate can select whichever reproduces production bytes on
        # the profile at hand.
        if profile_name in ("sm100", "sm120") or second_half_form == 0:
            second_half = ("__fmaf_rn(value, rope_cos[i], "
                           "__fmul_rn(paired, rope_sin[i]))")
        else:
            second_half = ("__fmaf_rn(paired, rope_sin[i], "
                           "__fmul_rn(value, rope_cos[i]))")

        def bake(src):
            return (src
                    .replace("SCALE_", f"{scale:.17e}f")
                    .replace("EPS_", f"{eps:.10e}f")
                    .replace("LOG2B_", f"{log2b:.17e}f")
                    .replace("SECOND_HALF_", second_half))

        suffix = "_batch" if batch else ""
        ab = mx.fast.cuda_kernel(
            name="maple_attn_verify_ab" + suffix,
            input_names=["hn", "wqkv", "sqkv", "bqkv", "wqk",
                         "kcache", "vcache", "scalars"],
            output_names=["scratch"],
            source=bake(_ATTN_VERIFY_AB_SOURCE),
            header=_ATTN_MEGAKERNEL_HEADER,
        )
        cd = mx.fast.cuda_kernel(
            name="maple_attn_verify_cd" + suffix,
            input_names=["scratch_in", "wo", "so_", "bo_",
                         "kcache", "vcache", "scalars"],
            output_names=["out"],
            source=bake(_ATTN_VERIFY_CD_SOURCE),
            header=_ATTN_MEGAKERNEL_HEADER,
        )
        pair = _attn_verify_cache[key] = (ab, cd)
    return pair


_attn_megakernel_cache = {}


def _attn_megakernel(profile_name, use_rope, scale, eps, log2b):
    key = (profile_name, use_rope, scale, eps, log2b)
    kernel = _attn_megakernel_cache.get(key)
    if kernel is None:
        second_half = (
            "__fmaf_rn(value, rope_cos[i], "
            "__fmul_rn(paired, rope_sin[i]))"
            if profile_name in ("sm100", "sm120")
            else "paired * rope_sin[i] + value * rope_cos[i]"
        )
        src = (_ATTN_MEGAKERNEL_SOURCE
               .replace("SECOND_HALF_", second_half)
               .replace("SCALE_", f"{scale:.17e}f")
               .replace("EPS_", f"{eps:.10e}f")
               .replace("LOG2B_", f"{log2b:.17e}f"))
        kernel = _attn_megakernel_cache[key] = mx.fast.cuda_kernel(
            name=f"maple_attn_megakernel_{'rope' if use_rope else 'nope'}"
                 f"_{profile_name}",
            input_names=["hn", "wqkv", "sqkv", "bqkv", "wqk",
                         "wo", "so_", "bo_", "kcache", "vcache", "scalars"],
            output_names=["out", "scratch"],
            source=src,
            header=_ATTN_MEGAKERNEL_HEADER,
        )
    return kernel


def _stock_phys_rows(n, cap):
    """The PHYSICAL row count the stock cache would hold for n live rows.

    Stock caches grow in 256-row blocks (both KVCache and the rotating
    cache) and downstream cache code branches on the physical shape, not
    just the logical offset -- the LRU repro caught exact-length buffers
    changing those paths. Everything the lane publishes therefore carries
    the stock-grown shape; rows past the logical offset are padding that
    no stock path reads.
    """
    if n <= 0:
        return 0
    return min(cap, ((n + 255) // 256) * 256)


class _AttnMegaState:
    """Caller-owned contiguous KV buffers mirroring the stock cache.

    The stock cache object stays the source of truth for counters; the
    buffers here are the truth for contents while the fused path runs, and
    are written back whenever the layer leaves the fused regime.

    The device buffers must NEVER be reassigned once the megakernel has
    seen them: a python-side slice assignment copies-on-write into a fresh
    buffer, and the kernel's const_cast appends then land in the orphaned
    one.  All (re)seeding therefore goes through _attn_seed_kernel, which
    writes into the very input buffers the megakernel reads.
    """

    __slots__ = ("kbuf", "vbuf", "ctr", "cap", "rows", "synced_offset",
                 "cache_ref")

    def __init__(self, kv_heads, cap, dtype, rows=1):
        self.kbuf = mx.zeros((rows, kv_heads, cap, 128), dtype)
        self.vbuf = mx.zeros((rows, kv_heads, cap, 128), dtype)
        self.ctr = mx.zeros((8,), mx.float32)
        mx.eval(self.kbuf, self.vbuf, self.ctr)
        self.cap = cap
        self.rows = rows
        self.synced_offset = -1
        self.cache_ref = None

    def bound_to(self, c):
        return self.cache_ref is not None and self.cache_ref() is c

    def materialize_old(self):
        """Turn the previously-bound cache's zero-copy views into real
        copies. MUST run before synced_offset is cleared on ANY detach
        path: a stored cache (the service's LRU) that keeps views into
        these buffers would otherwise silently mutate as the next request
        overwrites them -- that exact ordering gap rotted stored histories
        when the multi-turn guard cleared the sync before bind() ran."""
        old = self.cache_ref() if self.cache_ref is not None else None
        if old is not None and self.synced_offset > 0:
            n = min(self.synced_offset, self.cap)
            phys = _stock_phys_rows(n, self.cap)
            old.keys = mx.contiguous(self.kbuf[..., :phys, :])
            old.values = mx.contiguous(self.vbuf[..., :phys, :])
            mx.eval(old.keys, old.values)

    def bind(self, c):
        if self.bound_to(c):
            return
        # A different cache object means a different request: materialize
        # the old cache's views, then resync from the new one.
        self.materialize_old()
        self.synced_offset = -1
        self.cache_ref = weakref.ref(c)


# Seeds the persistent attention-mega buffers in place: copies n rows of
# K/V per kv-head from the stock cache slices and stamps the three live
# counters, all through const_cast writes into the kernel INPUTS so their
# device pointers never change.
_ATTN_SEED_SOURCE = r"""
    constexpr int HD = 128;
    const int n = static_cast<int>(meta[0]);
    T_* kd = const_cast<T_*>(kbuf);
    T_* vd = const_cast<T_*>(vbuf);
    const long long total = (long long)KVH_ * n * HD;
    for (long long i = (long long)blockIdx.x * THREADS_ + threadIdx.x;
         i < total; i += (long long)GRID_ * THREADS_) {
        const int h = (int)(i / ((long long)n * HD));
        const long long r = i - (long long)h * n * HD;
        const int row = (int)(r / HD);
        const int j = (int)(r - (long long)row * HD);
        const long long dst = ((long long)h * CAP_ + row) * HD + j;
        kd[dst] = ks[i];
        vd[dst] = vs[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* live = const_cast<float*>(ctr);
        live[0] = meta[1];
        live[1] = meta[2];
        live[2] = meta[3];
    }
"""

# Row variant: seeds ONE row's plane of a ragged batch state and stamps
# that row's counter triple at live[3 + row*3 .. +2] -- same const_cast
# discipline (buffer pointers never move).
_ATTN_SEED_ROW_SOURCE = r"""
    constexpr int HD = 128;
    const int n = static_cast<int>(meta[0]);
    const int row = static_cast<int>(meta[4]);
    T_* kd = const_cast<T_*>(kbuf) + (long long)row * KVH_ * CAP_ * HD;
    T_* vd = const_cast<T_*>(vbuf) + (long long)row * KVH_ * CAP_ * HD;
    const long long total = (long long)KVH_ * n * HD;
    for (long long i = (long long)blockIdx.x * THREADS_ + threadIdx.x;
         i < total; i += (long long)GRID_ * THREADS_) {
        const int h = (int)(i / ((long long)n * HD));
        const long long r = i - (long long)h * n * HD;
        const int rw = (int)(r / HD);
        const int j = (int)(r - (long long)rw * HD);
        const long long dst = ((long long)h * CAP_ + rw) * HD + j;
        kd[dst] = ks[i];
        vd[dst] = vs[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* live = const_cast<float*>(ctr);
        live[3 + row * 3 + 0] = meta[1];
        live[3 + row * 3 + 1] = meta[2];
        live[3 + row * 3 + 2] = meta[3];
    }
"""


def _attn_seed_row_kernel():
    kernel = _attn_seed_kernel_cache.get("row")
    if kernel is None:
        kernel = _attn_seed_kernel_cache["row"] = mx.fast.cuda_kernel(
            name="maple_attn_seed_row",
            input_names=["kbuf", "vbuf", "ctr", "ks", "vs", "meta"],
            output_names=["ok"],
            source=_ATTN_SEED_ROW_SOURCE,
        )
    return kernel


def _attn_seed_row(state, row, keys_src, values_src, pos, kl, slot, dtype):
    """Seed one plane of a ragged state; src is (1, kvh, n, 128)."""
    n = keys_src.shape[2] if keys_src is not None else 0
    if n == 0:
        keys_src = mx.zeros((1, state.kbuf.shape[1], 1, 128), dtype)
        values_src = keys_src
    meta = mx.array([float(n), float(pos), float(kl), float(slot),
                     float(row)], mx.float32)
    (ok,) = _attn_seed_row_kernel()(
        inputs=[state.kbuf, state.vbuf, state.ctr,
                keys_src.astype(dtype), values_src.astype(dtype), meta],
        template=[("T_", dtype), ("KVH_", int(state.kbuf.shape[1])),
                  ("CAP_", state.cap), ("THREADS_", 256), ("GRID_", 64)],
        grid=(64 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1,)], output_dtypes=[mx.float32],
    )
    mx.eval(ok)


_attn_seed_kernel_cache = {}


def _attn_seed_kernel():
    kernel = _attn_seed_kernel_cache.get("k")
    if kernel is None:
        kernel = _attn_seed_kernel_cache["k"] = mx.fast.cuda_kernel(
            name="maple_attn_seed",
            input_names=["kbuf", "vbuf", "ctr", "ks", "vs", "meta"],
            output_names=["ok"],
            source=_ATTN_SEED_SOURCE,
        )
    return kernel


def _attn_seed(state, keys_src, values_src, pos, kl, slot, dtype):
    """Copy the stock rows and stamp counters without moving any buffer."""
    n = keys_src.shape[2] if keys_src is not None else 0
    if n == 0:
        keys_src = mx.zeros((1, state.kbuf.shape[1], 1, 128), dtype)
        values_src = keys_src
        n = 0
    meta = mx.array([float(n), float(pos), float(kl), float(slot)],
                    mx.float32)
    # The copy is a flat walk over (rows*kv_heads, n, 128) planes, so a
    # batch state simply fuses its leading axes into the plane index.
    (ok,) = _attn_seed_kernel()(
        inputs=[state.kbuf, state.vbuf, state.ctr,
                keys_src.astype(dtype), values_src.astype(dtype), meta],
        template=[("T_", dtype),
                  ("KVH_", int(state.kbuf.shape[0] * state.kbuf.shape[1])),
                  ("CAP_", state.cap), ("THREADS_", 256), ("GRID_", 64)],
        grid=(64 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1,)], output_dtypes=[mx.float32],
    )
    mx.eval(ok)


def _attn_mega_call(layer, hn, c):
    """One-dispatch decode attention, or None to fall back to the stock path."""
    attn = layer.self_attn
    qkv = attn.qkv_proj
    op = attn.o_proj
    if (
        _cuda_profile() is None
        or hn.dtype != mx.bfloat16
        or attn.head_dim != 128
        or not attn.use_qk_norm
        or not attn._can_fuse_qk
        or attn._fused_qk is not True
        or getattr(qkv, "bits", None) != 2
        or getattr(op, "bits", None) != 2
        or getattr(qkv, "group_size", None) != 128
        or getattr(op, "group_size", None) != 128
        or getattr(qkv, "mode", "affine") != "affine"
        or getattr(qkv, "biases", None) is None
        or getattr(op, "biases", None) is None
        or "bias" in qkv or "bias" in op
        or hn.shape[-1] % 512 != 0
        or (attn.num_attention_heads * 128) % 512 != 0
    ):
        return None
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    rotating = isinstance(c, RotatingKVCache)
    if not rotating and not isinstance(c, KVCache):
        return None
    if rotating and (c.keep != 0 or c.max_size is None):
        return None

    state = getattr(attn, "_mega_state", None)
    if state is not None and state.rows != 1:
        # A batch-lane state: its buffers carry B planes the single-row
        # kernel would misindex. Materialize and rebuild.
        state.materialize_old()
        state = None
        attn._mega_state = None
    if state is not None:
        state.bind(c)
    if rotating:
        cap = c.max_size
        if state is None or state.cap != cap:
            state = _AttnMegaState(attn.num_key_value_heads, cap, hn.dtype)
            attn._mega_state = state
            state.bind(c)
    else:
        # Full-attention layers grow: start at 1024 (the 1-pass limit) and
        # double up to 8192 -- each step is one recompile/regraph, and the
        # kernel's 2-pass branch covers everything past 1024.
        needed = c.offset + 1
        if needed > 8192:
            _attn_mega_writeback(attn, c)
            return None
        if state is None:
            cap = 1024
            while cap < needed:
                cap *= 2
            state = _AttnMegaState(attn.num_key_value_heads, cap, hn.dtype)
            attn._mega_state = state
            state.bind(c)
        elif needed > state.cap:
            cap = state.cap
            while cap < needed:
                cap *= 2
            grown = _AttnMegaState(attn.num_key_value_heads, cap, hn.dtype)
            if state.synced_offset == c.offset and state.synced_offset > 0:
                # Mid-fused-run growth: the stock cache is stale, so carry
                # our own buffers over and reseed the on-device counters.
                n = min(state.synced_offset, state.cap)
                _attn_seed(grown, state.kbuf[..., :n, :],
                           state.vbuf[..., :n, :],
                           c.offset, c.offset + 1, c.offset, hn.dtype)
                grown.synced_offset = c.offset
            # Otherwise stay fresh (-1); the resync below seeds from the
            # stock cache, which is current when we are not mid-run.
            grown.cache_ref = state.cache_ref
            state = grown
            attn._mega_state = state
        cap = state.cap

    offset = c.offset
    # After a multi-token _update_concat on an overflowing ring the stock
    # buffer is TEMPORAL order and longer than the window (max_size + S - 1);
    # the next stock in-place step trims it to the last max_size rows and
    # restarts the ring at slot keep(=0).  Mirror that state exactly -- but
    # only on (re)entry: once synced, the stale stock buffer keeps its
    # concat shape while our ring and c._idx advance in lockstep.
    concat_tail = (
        rotating and state.synced_offset != offset
        and c.keys is not None and c.keys.shape[2] > cap)
    if rotating:
        if concat_tail:
            idx = 0
        else:
            idx = c._idx
            if idx == c.max_size:
                idx = 0
        kl_after = min(offset + 1, cap)
        slot = idx
    else:
        slot = offset
        kl_after = offset + 1
    if kl_after > cap:
        _attn_mega_writeback(attn, c)
        return None

    if state.synced_offset != offset:
        # (Re)enter the fused regime: copy the stock cache's physical layout
        # into our buffers and seed the on-device step counters once -- all
        # kernel-side, so the persistent buffer pointers never move.
        ks = vs = None
        if c.keys is not None and offset > 0:
            if concat_tail:
                n_phys = c.keys.shape[2]
                ks = c.keys[..., n_phys - cap:, :]
                vs = c.values[..., n_phys - cap:, :]
            else:
                n = min(c.keys.shape[2], cap)
                ks = c.keys[..., :n, :]
                vs = c.values[..., :n, :]
        _attn_seed(state, ks, vs, offset, kl_after, slot, hn.dtype)
        state.synced_offset = offset

    if attn._qk_w is None:
        attn._ensure_qk_constants()

    kernel = _attn_megakernel(
        _cuda_profile().name, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base),
    )
    kh = hn.shape[-1]
    grid = _attn_megakernel_grid()
    try:
        out, _ = kernel(
            inputs=[hn.reshape(-1), qkv.weight, qkv.scales, qkv.biases,
                    attn._qk_w, op.weight, op.scales, op.biases,
                    state.kbuf, state.vbuf, state.ctr],
            template=[
                ("T_", hn.dtype), ("KH_", kh),
                ("NQ_", attn.num_attention_heads),
                ("NKV_", attn.num_key_value_heads),
                ("CAP_", cap), ("ROPE_", 1 if attn.use_rope else 0),
                ("RD_", getattr(attn, "_rope_dim", 0) if attn.use_rope else 0),
                ("THREADS_", 1024), ("GRID_", grid),
            ],
            grid=(grid * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[
                (1, 1, kh),
                (16 + (attn.num_attention_heads
                       + 2 * attn.num_key_value_heads) * 128
                 + attn.num_attention_heads * 128 * 2
                 + attn.num_attention_heads * 32 * (128 + 2),),
            ],
            output_dtypes=[hn.dtype, mx.float32],
            init_value=0,
        )
    except (RuntimeError, TypeError, ValueError):
        return None

    # Advance the stock counters exactly as update_and_fetch would, and
    # leave the stock buffers as LIVE zero-copy views into ours: anything
    # that deep-copies, stores or trims this cache object between calls
    # (the service's LRU prompt cache does all three) then sees the real
    # history instead of a stale snapshot -- that staleness is exactly how
    # one request's context leaked into another's answer.
    c.offset = offset + 1
    if rotating:
        c._idx = slot + 1
    phys = _stock_phys_rows(kl_after, cap)
    c.keys = state.kbuf[..., :phys, :]
    c.values = state.vbuf[..., :phys, :]
    state.synced_offset = offset + 1  # our buffers are current for this offset
    return out


def _attn_mega_call_batch(layer, hn, c):
    """The M=B decode attention step through the AB/CD pair, or None.

    Mirrors `_attn_mega_call`'s state discipline exactly -- persistent
    buffers seeded kernel-side, stock counters in lockstep, live zero-copy
    views published each step -- with a leading batch axis everywhere.
    The pair carries the 1-pass SDPA port only, so kL must stay <= 1024;
    a layer past that writes back and declines (stock takes the step).
    """
    attn = layer.self_attn
    qkv = attn.qkv_proj
    op = attn.o_proj
    B = hn.shape[0]
    if (
        _cuda_profile() is None
        or hn.dtype != mx.bfloat16
        or attn.head_dim != 128
        or not attn.use_qk_norm
        or not attn._can_fuse_qk
        or attn._fused_qk is not True
        or getattr(qkv, "bits", None) != 2
        or getattr(op, "bits", None) != 2
        or getattr(qkv, "group_size", None) != 128
        or getattr(op, "group_size", None) != 128
        or getattr(qkv, "mode", "affine") != "affine"
        or getattr(qkv, "biases", None) is None
        or getattr(op, "biases", None) is None
        or "bias" in qkv or "bias" in op
        or hn.shape[-1] % 512 != 0
        or (attn.num_attention_heads * 128) % 512 != 0
    ):
        return None
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    rotating = isinstance(c, RotatingKVCache)
    if not rotating and not isinstance(c, KVCache):
        return None
    if rotating and (c.keep != 0 or c.max_size is None):
        return None

    state = getattr(attn, "_mega_state", None)
    if state is not None and state.rows != B:
        state.materialize_old()
        state = None
        attn._mega_state = None
    if state is not None:
        state.bind(c)
    if rotating:
        cap = c.max_size
        if cap > 1024:
            return None
        if state is None or state.cap != cap:
            state = _AttnMegaState(attn.num_key_value_heads, cap,
                                   hn.dtype, rows=B)
            attn._mega_state = state
            state.bind(c)
    else:
        # Full-attention layers grow 1024 -> 8192 exactly like the B=1
        # lane; the CD kernel's 2-pass branch covers every kL past 1024.
        needed = c.offset + 1
        if needed > 8192:
            _attn_mega_writeback(attn, c)
            return None
        if state is None:
            cap = 1024
            while cap < needed:
                cap *= 2
            state = _AttnMegaState(attn.num_key_value_heads, cap,
                                   hn.dtype, rows=B)
            attn._mega_state = state
            state.bind(c)
        elif needed > state.cap:
            cap = state.cap
            while cap < needed:
                cap *= 2
            grown = _AttnMegaState(attn.num_key_value_heads, cap,
                                   hn.dtype, rows=B)
            if state.synced_offset == c.offset and state.synced_offset > 0:
                n = min(state.synced_offset, state.cap)
                _attn_seed(grown, state.kbuf[..., :n, :],
                           state.vbuf[..., :n, :],
                           c.offset, c.offset + 1, c.offset, hn.dtype)
                grown.synced_offset = c.offset
            grown.cache_ref = state.cache_ref
            state = grown
            attn._mega_state = state
        cap = state.cap

    offset = c.offset
    concat_tail = (
        rotating and state.synced_offset != offset
        and c.keys is not None and c.keys.shape[2] > cap)
    if rotating:
        if concat_tail:
            idx = 0
        else:
            idx = c._idx
            if idx == c.max_size:
                idx = 0
        kl_after = min(offset + 1, cap)
        slot = idx
    else:
        slot = offset
        kl_after = offset + 1
    if kl_after > cap:
        _attn_mega_writeback(attn, c)
        return None

    if state.synced_offset != offset:
        ks = vs = None
        if c.keys is not None and offset > 0:
            if concat_tail:
                n_phys = c.keys.shape[2]
                ks = c.keys[..., n_phys - cap:, :]
                vs = c.values[..., n_phys - cap:, :]
            else:
                n = min(c.keys.shape[2], cap)
                ks = c.keys[..., :n, :]
                vs = c.values[..., :n, :]
        _attn_seed(state, ks, vs, offset, kl_after, slot, hn.dtype)
        state.synced_offset = offset

    if attn._qk_w is None:
        attn._ensure_qk_constants()

    ab, cd = _attn_verify_kernels(
        _cuda_profile().name, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base), batch=True,
    )
    kh = hn.shape[-1]
    nq = attn.num_attention_heads
    nkv = attn.num_key_value_heads
    # Grid residency bounds the CD kernel (1024 threads: ONE block per SM
    # on consumer parts).  Per-profile defaults are clamped to the
    # smallest member of each class (sm86: RTX 3080's 68 SMs -> 64;
    # sm120: RTX 5080's 84 SMs -> 80); big hosts opt higher via env --
    # measured: 80 on the 3090 (-6..11%), 160 on the 5090 (-39..44%).
    try:
        grid = int(os.environ.get("MAPLE_BATCH_ATTENTION_GRID", "0"))
    except ValueError:
        grid = 0
    if not grid:
        prof = _cuda_profile().name
        grid = {"sm100": 80, "sm120": 80}.get(prof, 0)
    grid = grid or _attn_megakernel_grid()
    tmpl = [
        ("T_", hn.dtype), ("KH_", kh), ("NQ_", nq), ("NKV_", nkv),
        ("CAP_", cap), ("ROPE_", 1 if attn.use_rope else 0),
        ("RD_", getattr(attn, "_rope_dim", 0) if attn.use_rope else 0),
        ("ROWS_", B), ("BATCH_", 1), ("RAGGED_", 0),
        ("GRID_", grid),
    ]
    try:
        (scr,) = ab(
            inputs=[hn.reshape(-1), qkv.weight, qkv.scales, qkv.biases,
                    attn._qk_w, state.kbuf, state.vbuf, state.ctr],
            template=tmpl + [("THREADS_", 512)],
            grid=(grid * 512, 1, 1), threadgroup=(512, 1, 1),
            output_shapes=[(16 + B * ((nq + 2 * nkv) * 128
                                      + nq * 128 * 2 + kh
                                      + nq * 32 * (128 + 2)),)],
            output_dtypes=[mx.float32],
            init_value=0,
        )
        (out,) = cd(
            inputs=[scr, op.weight, op.scales, op.biases,
                    state.kbuf, state.vbuf, state.ctr],
            template=tmpl + [("THREADS_", 1024)],
            grid=(grid * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(B, kh)],
            output_dtypes=[hn.dtype],
        )
    except (RuntimeError, TypeError, ValueError):
        return None

    c.offset = offset + 1
    if rotating:
        c._idx = slot + 1
    phys = _stock_phys_rows(kl_after, cap)
    c.keys = state.kbuf[..., :phys, :]
    c.values = state.vbuf[..., :phys, :]
    state.synced_offset = offset + 1
    return out.reshape(B, 1, kh)


class _AttnRaggedState:
    """B-plane persistent buffers with PER-ROW sync: each plane mirrors a
    different request's cache at its own offset. Same pointer discipline
    as _AttnMegaState -- all (re)seeding is kernel-side."""

    __slots__ = ("kbuf", "vbuf", "ctr", "cap", "rows", "synced",
                 "cache_refs")

    def __init__(self, kv_heads, cap, dtype, rows):
        self.kbuf = mx.zeros((rows, kv_heads, cap, 128), dtype)
        self.vbuf = mx.zeros((rows, kv_heads, cap, 128), dtype)
        self.ctr = mx.zeros((3 + 3 * rows,), mx.float32)
        mx.eval(self.kbuf, self.vbuf, self.ctr)
        self.cap = cap
        self.rows = rows
        self.synced = [-1] * rows
        self.cache_refs = [None] * rows

    def materialize_row(self, r):
        ref = self.cache_refs[r]
        old = ref() if ref is not None else None
        if old is not None and self.synced[r] > 0:
            n = min(self.synced[r], self.cap)
            phys = _stock_phys_rows(n, self.cap)
            old.keys = mx.contiguous(self.kbuf[r:r + 1, :, :phys, :])
            old.values = mx.contiguous(self.vbuf[r:r + 1, :, :phys, :])
            mx.eval(old.keys, old.values)

    def bind_row(self, r, c):
        ref = self.cache_refs[r]
        if ref is not None and ref() is c:
            return
        self.materialize_row(r)
        self.synced[r] = -1
        self.cache_refs[r] = weakref.ref(c)

    def materialize_all(self):
        for r in range(self.rows):
            self.materialize_row(r)
            self.synced[r] = -1


def _attn_mega_call_ragged(layer, hn, caches):
    """One-dispatch-pair decode attention for B requests at DIFFERENT
    offsets (their per-layer caches in `caches`), or None for stock."""
    attn = layer.self_attn
    qkv = attn.qkv_proj
    op = attn.o_proj
    B = hn.shape[0]
    if (
        _cuda_profile() is None
        or hn.dtype != mx.bfloat16
        or attn.head_dim != 128
        or not attn.use_qk_norm
        or not attn._can_fuse_qk
        or attn._fused_qk is not True
        or getattr(qkv, "bits", None) != 2
        or getattr(op, "bits", None) != 2
        or getattr(qkv, "group_size", None) != 128
        or getattr(op, "group_size", None) != 128
        or getattr(qkv, "mode", "affine") != "affine"
        or getattr(qkv, "biases", None) is None
        or getattr(op, "biases", None) is None
        or "bias" in qkv or "bias" in op
        or hn.shape[-1] % 512 != 0
        or (attn.num_attention_heads * 128) % 512 != 0
    ):
        return None
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    rotating = isinstance(caches[0], RotatingKVCache)
    for c in caches:
        if rotating != isinstance(c, RotatingKVCache):
            return None
        if not rotating and not isinstance(c, KVCache):
            return None
        if rotating and (c.keep != 0 or c.max_size is None):
            return None

    state = getattr(attn, "_ragged_state", None)
    if state is not None and state.rows != B:
        state.materialize_all()
        state = None
        attn._ragged_state = None
    if rotating:
        cap = caches[0].max_size
        if cap > 1024 or any(c.max_size != cap for c in caches):
            return None
        if state is None or state.cap != cap:
            if state is not None:
                state.materialize_all()
            state = _AttnRaggedState(attn.num_key_value_heads, cap,
                                     hn.dtype, B)
            attn._ragged_state = state
    else:
        needed = max(c.offset for c in caches) + 1
        if needed > 8192:
            if state is not None:
                state.materialize_all()
                attn._ragged_state = None
            return None
        if state is None or needed > state.cap:
            cap = 1024 if state is None else state.cap
            while cap < needed:
                cap *= 2
            if state is not None:
                # simplest correct growth: push every plane back to its
                # stock cache, rebuild, reseed below from stock
                state.materialize_all()
            state = _AttnRaggedState(attn.num_key_value_heads, cap,
                                     hn.dtype, B)
            attn._ragged_state = state
        cap = state.cap

    metas = []
    for r, c in enumerate(caches):
        state.bind_row(r, c)
        offset = c.offset
        concat_tail = (
            rotating and state.synced[r] != offset
            and c.keys is not None and c.keys.shape[2] > cap)
        if rotating:
            if concat_tail:
                idx = 0
            else:
                idx = c._idx
                if idx == c.max_size:
                    idx = 0
            kl_after = min(offset + 1, cap)
            slot = idx
        else:
            slot = offset
            kl_after = offset + 1
        if kl_after > cap:
            state.materialize_all()
            attn._ragged_state = None
            return None
        if state.synced[r] != offset:
            ks = vs = None
            if c.keys is not None and offset > 0:
                if concat_tail:
                    n_phys = c.keys.shape[2]
                    ks = c.keys[..., n_phys - cap:, :]
                    vs = c.values[..., n_phys - cap:, :]
                else:
                    n = min(c.keys.shape[2], cap)
                    ks = c.keys[..., :n, :]
                    vs = c.values[..., :n, :]
            _attn_seed_row(state, r, ks, vs, offset, kl_after, slot,
                           hn.dtype)
            state.synced[r] = offset
        metas.append((offset, kl_after, slot))

    if attn._qk_w is None:
        attn._ensure_qk_constants()

    ab, cd = _attn_verify_kernels(
        _cuda_profile().name, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base), batch=True,
    )
    kh = hn.shape[-1]
    nq = attn.num_attention_heads
    nkv = attn.num_key_value_heads
    try:
        grid = int(os.environ.get("MAPLE_BATCH_ATTENTION_GRID", "0"))
    except ValueError:
        grid = 0
    if not grid:
        grid = {"sm100": 80, "sm120": 80}.get(_cuda_profile().name, 0)
    grid = grid or _attn_megakernel_grid()
    tmpl = [
        ("T_", hn.dtype), ("KH_", kh), ("NQ_", nq), ("NKV_", nkv),
        ("CAP_", cap), ("ROPE_", 1 if attn.use_rope else 0),
        ("RD_", getattr(attn, "_rope_dim", 0) if attn.use_rope else 0),
        ("ROWS_", B), ("BATCH_", 0), ("RAGGED_", 1), ("GRID_", grid),
    ]
    try:
        (scr,) = ab(
            inputs=[hn.reshape(-1), qkv.weight, qkv.scales, qkv.biases,
                    attn._qk_w, state.kbuf, state.vbuf, state.ctr],
            template=tmpl + [("THREADS_", 512)],
            grid=(grid * 512, 1, 1), threadgroup=(512, 1, 1),
            output_shapes=[(16 + B * ((nq + 2 * nkv) * 128
                                      + nq * 128 * 2 + kh
                                      + nq * 32 * (128 + 2)),)],
            output_dtypes=[mx.float32],
            init_value=0,
        )
        (out,) = cd(
            inputs=[scr, op.weight, op.scales, op.biases,
                    state.kbuf, state.vbuf, state.ctr],
            template=tmpl + [("THREADS_", 1024)],
            grid=(grid * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(B, kh)],
            output_dtypes=[hn.dtype],
        )
    except (RuntimeError, TypeError, ValueError):
        return None

    for r, (c, (offset, kl_after, slot)) in enumerate(zip(caches, metas)):
        c.offset = offset + 1
        if rotating:
            c._idx = slot + 1
        phys = _stock_phys_rows(kl_after, cap)
        c.keys = state.kbuf[r:r + 1, :, :phys, :]
        c.values = state.vbuf[r:r + 1, :, :phys, :]
        state.synced[r] = offset + 1
    return out.reshape(B, 1, kh)


def ragged_decode_step(model, y, request_caches):
    """One decode step for B requests at DIFFERENT offsets in ONE pass.

    `y` is (B, 1) token ids; `request_caches` is a list of B per-request
    prompt caches (each the usual per-layer list). Every row's bits equal
    its solo decode step -- the serving contract. Returns (B, 1, vocab)
    logits. The caller owns sampling and cache lifecycles.
    """
    inner = model.model
    B = y.shape[0]
    h = inner.word_embeddings(y)
    fuse = None
    if _use_fused_add_rms:
        if inner._exact_add_norm is None:
            inner._exact_add_norm = _exact_add_rms_ok(
                h.shape[-1], h.dtype, inner.norm.weight, inner.norm.eps)
        if inner._exact_add_norm:
            fuse = _exact_add_rms_norm

    def bfuse(hh, rr, w, eps):
        if fuse is None:
            ss = (hh + rr).astype(hh.dtype)
            return ss, mx.fast.rms_norm(ss, w, eps)
        parts = [fuse(hh[b:b + 1], rr[b:b + 1], w, eps) for b in range(B)]
        return (mx.concatenate([pp[0] for pp in parts], axis=0),
                mx.concatenate([pp[1] for pp in parts], axis=0))

    r = mx.zeros(h.shape, h.dtype)
    hn = None
    mega_next = inner._megakernel_next_norms()
    for i, (layer, layer_type) in enumerate(
        zip(inner.layers, inner.layer_types)
    ):
        layer_caches = [rc[i] for rc in request_caches]
        if hn is None:
            ln = layer.input_layernorm
            h, hn = bfuse(h, r, ln.weight, ln.eps)
        r = _attn_mega_call_ragged(layer, hn, layer_caches)
        if r is None:
            r = mx.concatenate(
                [layer.self_attn(hn[b:b + 1], None, layer_caches[b])
                 for b in range(B)], axis=0)
        hn = None
        ln = layer.post_attention_layernorm
        fused = _moe_batch_call(layer, h, r, ln, mega_next[i])
        if fused is not None:
            h, hn = fused
            continue
        h, hn2 = bfuse(h, r, ln.weight, ln.eps)
        r = layer.mlp(hn2)
    if hn is None:
        h, hn = bfuse(h, r, inner.norm.weight, inner.norm.eps)
    out = hn
    if B > 4:
        return mx.concatenate(
            [model.lm_head(out[b:b + 1]) for b in range(B)], axis=0)
    return model.lm_head(out)


def _attn_mega_rollback(attn, c, to_offset):
    """Rewind a fused layer's cache to an earlier offset (speculation).

    Rejected draft tokens live only in slots past `to_offset` of the
    persistent buffers; the next accepted token overwrites them, so a
    rollback is purely counter surgery: reseed the on-device counters
    through the seed kernel (n=0 copies nothing) and rewind the stock
    counters in lockstep. Bit-safety: verified tokens' K/V at slots
    < to_offset were written by the same recipes a sequential decode
    would have used, so the resumed stream is the sequential stream.
    Only valid while the state is bound and synced past `to_offset`.
    """
    state = getattr(attn, "_mega_state", None)
    if state is None or not state.bound_to(c):
        return False
    if state.synced_offset < to_offset:
        return False
    from mlx_lm.models.cache import RotatingKVCache

    rotating = isinstance(c, RotatingKVCache)
    if rotating:
        if c.offset >= c.max_size or to_offset >= c.max_size:
            # A wrapped ring cannot rewind by counters alone: rejected
            # slots may have overwritten live history.
            return False
        slot = to_offset
        kl_after = min(to_offset + 1, state.cap)
    else:
        slot = to_offset
        kl_after = to_offset + 1
    # The counters describe the NEXT step: pos to write, kL after its
    # append, and the slot it lands in -- same contract as the resync.
    _attn_seed(state, None, None, to_offset, kl_after, slot,
               state.kbuf.dtype)
    c.offset = to_offset
    if rotating:
        c._idx = to_offset
    state.synced_offset = to_offset
    return True


def _attn_mega_writeback(attn, c):
    """Push the fused buffers back into the stock cache before leaving.

    The stock buffers did not grow while the fused path ran, so they are
    rebuilt outright from our slices; the counters were kept in lockstep
    all along, so update_and_fetch continues seamlessly.
    """
    state = getattr(attn, "_mega_state", None)
    if state is None or state.synced_offset < 0:
        return
    if not state.bound_to(c):
        # A fresh request reuses the module but not the cache object; our
        # buffers belong to the previous conversation. Writing them here
        # is exactly the way one user's context leaks into another's
        # answer. Materialize the PREVIOUS cache's views first -- clearing
        # the sync without doing so leaves any stored copy of it aliased
        # to buffers the new request will overwrite -- then drop the sync.
        state.materialize_old()
        state.synced_offset = -1
        return
    n = min(state.synced_offset, state.cap)
    if n > 0:
        phys = _stock_phys_rows(n, state.cap)
        c.keys = state.kbuf[..., :phys, :]
        c.values = state.vbuf[..., :phys, :]
        mx.eval(c.keys, c.values)
    state.synced_offset = -1


_moe_exact_megakernel_cache = {}


def _moe_exact_megakernel(eps):
    kernel = _moe_exact_megakernel_cache.get(eps)
    if kernel is None:
        kernel = _moe_exact_megakernel_cache[eps] = mx.fast.cuda_kernel(
            name="maple_moe_exact_megakernel",
            input_names=["hin", "rin", "nw", "rw", "ugw", "ugs", "ugb",
                         "dnw", "dns", "dnb", "nw2"],
            output_names=["out", "hout", "scratch"],
            source=_MOE_EXACT_MEGAKERNEL_SOURCE.replace(
                "EPS_", f"{eps:.10e}f"),
            header=_MOE_EXACT_MEGAKERNEL_HEADER,
        )
    return kernel


def _moe_exact_megakernel_plan(block, ln, dtype, grid=None, threads=512):
    """Geometry gates for the exact lane; strictly narrower than the fast
    lane's, because every proven bit recipe assumed its exact shape."""
    grid = _moe_megakernel_grid() if grid is None else grid
    mlp = getattr(block, "switch_mlp", None)
    if mlp is None or _cuda_profile() is None:
        return False
    ug = getattr(mlp, "up_gate_proj", None)
    dp = getattr(mlp, "down_proj", None)
    if ug is None or dp is None:
        return False
    kh, kd, nd = ug.input_dims, dp.input_dims, dp.output_dims
    if not (
        getattr(ug, "bits", None) == 2 and getattr(dp, "bits", None) == 2
        and getattr(ug, "group_size", None) == 128
        and getattr(dp, "group_size", None) == 128
        and getattr(ug, "mode", "affine") == "affine"
        and getattr(dp, "mode", "affine") == "affine"
        and getattr(ug, "biases", None) is not None
        and getattr(dp, "biases", None) is not None
        and "bias" not in ug and "bias" not in dp
        and block.gate.top_k == 8
        and block.gate.num_experts == 256  # the softmax port's shape
        and block.gate.weight.dtype == mx.bfloat16  # astype(f32) is exact
        and dtype == mx.bfloat16           # every recipe is bf16-specific
        and ug.output_dims == 2 * kd
        and nd == kh
        and kh % threads == 0
        and kh % 128 == 0 and kd % 128 == 0
        and (kh // 16) % 4 == 0 and (kd // 16) % 4 == 0  # uint4 tile loads
        and ug.scales.dtype == mx.bfloat16
        and dp.scales.dtype == mx.bfloat16
    ):
        return False
    return (
        _moe_exact_megakernel(ln.eps),
        {
            "template": [
                ("T_", dtype),
                ("KH_", kh), ("KD_", kd), ("ND_", nd),
                ("NROUT_", block.gate.num_experts),
                ("NEXP_", block.gate.top_k),
                ("THREADS_", threads), ("GRID_", grid),
            ],
            "grid": (grid * threads, 1, 1),
            "threadgroup": (threads, 1, 1),
            "output_shapes": [
                (1, 1, nd), (1, 1, kh),
                (16 + 8 + 8 + 2 * block.gate.num_experts
                 + block.gate.top_k * 2 * kd + block.gate.top_k * nd,),
            ],
            "output_dtypes": [dtype, dtype, mx.float32],
            "init_value": 0,
        },
    )


def _moe_exact_megakernel_call(layer, h, r, ln, next_w):
    """The array-exact fast lane; same contract as _moe_megakernel_call."""
    block = layer.mlp
    plan = getattr(block, "_exact_megakernel_plan", None)
    if plan is None:
        try:
            plan = _moe_exact_megakernel_plan(block, ln, h.dtype)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            plan = False
        block._exact_megakernel_plan = plan
    if plan is False:
        return None
    kernel, kwargs = plan
    mlp = block.switch_mlp
    ug, dp = mlp.up_gate_proj, mlp.down_proj
    try:
        hn, hout, _ = kernel(
            inputs=[h, r, ln.weight, block.gate.weight, ug.weight, ug.scales,
                    ug.biases, dp.weight, dp.scales, dp.biases, next_w],
            **kwargs,
        )
    except (RuntimeError, TypeError, ValueError):
        block._exact_megakernel_plan = False
        return None
    return hout, hn


_moe_megakernel_cache = {}


def _moe_megakernel(eps):
    kernel = _moe_megakernel_cache.get(eps)
    if kernel is None:
        kernel = _moe_megakernel_cache[eps] = mx.fast.cuda_kernel(
            name="maple_moe_megakernel",
            input_names=["hin", "rin", "nw", "rw", "ugw", "ugs", "ugb",
                         "dnw", "dns", "dnb", "nw2"],
            output_names=["out", "hout", "scratch"],
            source=_MOE_MEGAKERNEL_SOURCE.replace("EPS_", f"{eps:.10e}f"),
        )
    return kernel


def _moe_megakernel_lpr(k):
    """Lanes per row, preferring 64 or 128 values per lane so the packed
    weights arrive as one or two 16-byte uint4 loads."""
    for want in (64, 128, 32, 16):
        lpr = k // want
        if lpr in (4, 8, 16, 32):
            vals = k // lpr
            if 16 <= vals <= 128 and 128 % vals == 0 and (k // 16) % lpr == 0:
                return lpr
    return None


def _attn_megakernel_grid():
    """Blocks for the attention megakernel (1024 threads each).

    Residency bounds the choice and the parts differ sharply at this block
    size: consumer Ampere/Blackwell run 1536 threads per SM, so they hold
    ONE 1024-thread block per SM (RTX 3090: 82 SMs, RTX 4090: 128), while
    Hopper/B200 run 2048 and hold two (H100: 264).  The default stays at
    the validated 64 everywhere -- the sm86/sm89 wins were measured there
    -- and MAPLE_ATTENTION_MEGAKERNEL_GRID exists to scan bigger parts
    (the H100 story turned out to be CPU-class, not grid: on a starved
    8-vcpu host the lane WINS +13-14% at any grid, on a healthy CPU it
    loses ~12% — and grids 64/96/128 tie within noise while 128 already
    grazes the residency edge and 192 spins).  The override is clamped to
    a conservative per-capability ceiling so a typo becomes a slowdown
    rather than a deadlock: the kernel's register weight holds one
    1024-thread block per SM even on Hopper in practice.
    """
    raw = os.environ.get("MAPLE_ATTENTION_MEGAKERNEL_GRID")
    if raw:
        ceiling = {(8, 6): 64, (8, 9): 112, (9, 0): 112,
                   (10, 0): 112, (12, 0): 112}.get(_cuda_capability(), 64)
        try:
            return max(16, min(int(raw), ceiling))
        except ValueError:
            pass
    return 64


def _moe_megakernel_grid(default=None):
    """Blocks for the megakernel: as many as the device certainly holds.

    The grid barrier is only correct while every block is resident, and MLX
    exposes neither the multiprocessor count nor occupancy, so the grid cannot
    be read off the device.  Compute capability and memory are safe proxies in
    this class: at 512 threads and ~33 KB of shared memory a multiprocessor
    holds three blocks, so 96 blocks need 32 multiprocessors and 192 need 64 --
    comfortably inside anything with the memory to hold this checkpoint.

    Measured medians for the fast lane, four fresh processes per point:

        grid          32      64      96     128     192   chosen
        RTX 3090   357.5   375.0   365.3   370.2   333.9       64
        RTX 4090   429.1   469.1   507.7   478.5   463.3       96
        H100 80GB  344.8   359.0   365.9   358.4   356.2       96
        RTX 5090   422.9   424.9   435.3   427.2   426.7       96
        B200       293.4   327.9   324.9   332.1   335.4      192

    Ampere consumer parts peak lower than the rest and lose measurably at 96,
    so they stay at 64.  The very large parts keep scaling, so they get 192.
    `MAPLE_MOE_MEGAKERNEL_GRID` overrides all of it; the cap keeps a typo from
    turning into a deadlock rather than a slowdown.
    """
    raw = os.environ.get("MAPLE_MOE_MEGAKERNEL_GRID")
    if raw:
        try:
            return max(8, min(int(raw), 240))
        except ValueError:
            pass
    if default is not None:
        return default
    capability = _cuda_capability()
    if capability == (8, 6):
        return 64
    try:
        total = mx.device_info(mx.gpu).get("total_memory", 0)
    except (AttributeError, RuntimeError, TypeError):
        total = 0
    if total >= 100 * (1 << 30):
        return 192
    return 96


def _moe_megakernel_plan(block, ln, dtype, grid=None, threads=512):
    grid = _moe_megakernel_grid() if grid is None else grid
    mlp = getattr(block, "switch_mlp", None)
    if mlp is None or _cuda_profile() is None:
        return False
    ug = getattr(mlp, "up_gate_proj", None)
    dp = getattr(mlp, "down_proj", None)
    if ug is None or dp is None:
        return False
    kh, nout, kd, nd = (ug.input_dims, dp.input_dims, dp.input_dims,
                        dp.output_dims)
    if not (
        getattr(ug, "bits", None) == 2 and getattr(dp, "bits", None) == 2
        and getattr(ug, "group_size", None) == 128
        and getattr(dp, "group_size", None) == 128
        and getattr(ug, "mode", "affine") == "affine"
        and getattr(dp, "mode", "affine") == "affine"
        and getattr(ug, "biases", None) is not None
        and getattr(dp, "biases", None) is not None
        and "bias" not in ug and "bias" not in dp
        and block.gate.top_k == 8
        and ug.output_dims == 2 * nout
        and kh % threads == 0
        and nd == kh  # phase E folds the MoE output back into the residual
        and _moe_megakernel_lpr(kd) is not None
    ):
        return False
    return (
        _moe_megakernel(ln.eps),
        {
            "template": [
                ("T_", dtype), ("RW_", block.gate.weight.dtype),
                ("KH_", kh), ("NOUT_", nout), ("KD_", kd), ("ND_", nd),
                ("NROUT_", block.gate.num_experts), ("NEXP_", block.gate.top_k),
                ("GRPA_", ug.scales.shape[-1]), ("GRPB_", dp.scales.shape[-1]),
                ("GS_", 128), ("LPRA_", 16), ("LPRB_", _moe_megakernel_lpr(kd)),
                ("THREADS_", threads), ("GRID_", grid),
            ],
            "grid": (grid * threads, 1, 1),
            "threadgroup": (threads, 1, 1),
            "output_shapes": [(1, 1, nd), (1, 1, kh),
                              (512 + block.gate.top_k * nout + nd,)],
            "output_dtypes": [dtype, dtype, mx.float32],
            "init_value": 0,
        },
    )


def _moe_megakernel_call(layer, h, r, ln, next_w):
    """Returns (new_residual_carrier, next_attention_input) or None.

    The carrier is h + attn + moe rounded once per add, exactly as the chain
    of exact fuses produced it; the second array is that carrier already
    normed with the *next* layer's input weight (or the final norm), so the
    caller skips the standalone fuse dispatch entirely.
    """
    block = layer.mlp
    plan = getattr(block, "_megakernel_plan", None)
    if plan is None:
        try:
            plan = _moe_megakernel_plan(block, ln, h.dtype)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            plan = False
        block._megakernel_plan = plan
    if plan is False:
        return None
    kernel, kwargs = plan
    mlp = block.switch_mlp
    ug, dp = mlp.up_gate_proj, mlp.down_proj
    try:
        hn, hout, _ = kernel(
            inputs=[h, r, ln.weight, block.gate.weight, ug.weight, ug.scales,
                    ug.biases, dp.weight, dp.scales, dp.biases, next_w],
            **kwargs,
        )
    except (RuntimeError, TypeError, ValueError):
        block._megakernel_plan = False
        return None
    return hout, hn


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
        self._exact_add_norm = None  # None = unprobed, then True/False
        self._zero = None
        self._mega_next = None  # None = unbuilt, False = eps mismatch, or list

    def _megakernel_next_norms(self):
        """Per-layer (next input norm weight | final norm weight) for the tail.

        The megakernel's phase E norms with the *next* layer's weight, and the
        kernel bakes a single eps, so a model whose norms disagree on eps must
        not take the megakernel at all.  Built once; weights are fixed.
        """
        nxt = self._mega_next
        if nxt is None:
            eps = self.norm.eps
            if all(
                layer.input_layernorm.eps == eps
                and layer.post_attention_layernorm.eps == eps
                for layer in self.layers
            ):
                nxt = [layer.input_layernorm.weight for layer in self.layers[1:]]
                nxt.append(self.norm.weight)
            else:
                nxt = False
            self._mega_next = nxt
        return nxt if nxt is not False else None

    def _decode_batch_fused(self, h, cache, full_mask, swa_mask, fuse=None):
        """The B<=8 decode step through the M=B megakernels.

        Same (h, r, hn) carry as `_decode_fused`.  Boundary fuses MUST be
        bit-equal per row to whatever the solo stream uses -- and the solo
        stream's `_exact_add_rms_norm` is NOT bit-equal to the plain
        add+rms pair (nor row-safe at B>1), so when a fuse is given it is
        applied per row; only a fuse-less solo baseline uses the stock
        pair.  Each megakernel lane falls back to the stock computation
        per layer when its plan declines.
        """
        B = h.shape[0]

        def bfuse(hh, rr, w, eps):
            if fuse is None:
                s = (hh + rr).astype(hh.dtype)
                return s, mx.fast.rms_norm(s, w, eps)
            parts = [fuse(hh[b:b + 1], rr[b:b + 1], w, eps)
                     for b in range(B)]
            return (mx.concatenate([p[0] for p in parts], axis=0),
                    mx.concatenate([p[1] for p in parts], axis=0))

        r = mx.zeros(h.shape, h.dtype)
        hn = None
        mega_next = self._megakernel_next_norms()
        for i, (layer, c, layer_type) in enumerate(
            zip(self.layers, cache, self.layer_types)
        ):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            if hn is None:
                ln = layer.input_layernorm
                h, hn = bfuse(h, r, ln.weight, ln.eps)
            r = _attn_mega_call_batch(layer, hn, c)
            if r is None:
                r = layer.self_attn(hn, mask, c)
            hn = None
            ln = layer.post_attention_layernorm
            fused = _moe_batch_call(layer, h, r, ln, mega_next[i])
            if fused is not None:
                h, hn = fused
                continue
            h, hn2 = bfuse(h, r, ln.weight, ln.eps)
            r = layer.mlp(hn2)
        if hn is not None:
            return hn
        return bfuse(h, r, self.norm.weight, self.norm.eps)[1]

    def _decode_fused(self, h, cache, full_mask, swa_mask, fuse):
        """Decode loop with residual adds folded into the norms.

        Carries (h, r) instead of adding r back each step, so every
        add+norm pair is one dispatch. Identical arithmetic: the kernel
        rounds the sum once (as the bf16 add did) and norms the rounded
        stream with an fp32 weight multiply.

        When a layer takes the megakernel, its tail has already produced the
        next attention input (`hn`), so the loop skips the standalone fuse for
        that boundary; after the last layer the tail has already applied the
        final norm.
        """
        if self._zero is None:
            self._zero = mx.zeros(h.shape, h.dtype)
            mx.eval(self._zero)
        r = self._zero  # x + 0 is exact in bf16
        hn = None
        mega_next = (
            self._megakernel_next_norms()
            if (_use_moe_megakernel or _use_moe_megakernel_exact)
            else None
        )
        for i, (layer, c, layer_type) in enumerate(
            zip(self.layers, cache, self.layer_types)
        ):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            if hn is None:
                ln = layer.input_layernorm
                h, hn = fuse(h, r, ln.weight, ln.eps)
            r = None
            if _attention_megakernel_enabled():
                r = _attn_mega_call(layer, hn, c)
            if r is None:
                r = layer.self_attn(hn, mask, c)
            hn = None
            ln = layer.post_attention_layernorm
            if mega_next is not None:
                if _use_moe_megakernel_exact:
                    fused = _moe_exact_megakernel_call(
                        layer, h, r, ln, mega_next[i]
                    )
                    if fused is not None:
                        h, hn = fused
                        continue
                fused = _moe_megakernel_call(layer, h, r, ln, mega_next[i])
                if fused is not None:
                    h, hn = fused
                    continue
            h, hn2 = fuse(h, r, ln.weight, ln.eps)
            r = layer.mlp(hn2)
        if hn is not None:
            return hn
        return fuse(h, r, self.norm.weight, self.norm.eps)[1]

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

        if (
            h.ndim == 3
            and h.shape[1] == 1
            and 2 <= h.shape[0] <= _batch_megakernel_max_rows()
        ):
            # Boundary fuse: the SAME lane the solo stream would pick, so
            # each batched row's boundary bits match its solo run.
            fuse = None
            if _use_fused_add_rms:
                if self._exact_add_norm is None:
                    self._exact_add_norm = _exact_add_rms_ok(
                        h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                    )
                if self._exact_add_norm:
                    fuse = _exact_add_rms_norm
            return self._decode_batch_fused(
                h, cache, full_mask, swa_mask, fuse)

        if h.size == h.shape[-1]:
            # Strict lane first: the corrected kernel reproduces
            # mx.fast.rms_norm's thread mapping and is array-exact.
            if _use_fused_add_rms:
                if self._exact_add_norm is None:
                    self._exact_add_norm = _exact_add_rms_ok(
                        h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                    )
                if self._exact_add_norm:
                    return self._decode_fused(
                        h, cache, full_mask, swa_mask, _exact_add_rms_norm
                    )
            # Historical approximate carrier: its thread mapping differs from
            # mx.fast.rms_norm, so it stays an explicit semantic lane.
            if not _use_approximate_add_rms:
                self._fused_add_norm = False
            elif self._fused_add_norm is None:
                self._fused_add_norm = _add_rms_norm_ok(
                    h.shape[-1], h.dtype, self.norm.weight, self.norm.eps
                )
            if self._fused_add_norm:
                return self._decode_fused(
                    h, cache, full_mask, swa_mask, _add_rms_norm
                )

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
        if (
            out.ndim == 3
            and out.shape[1] == 1
            and 4 < out.shape[0] <= _batch_megakernel_max_rows()
        ):
            # The quantized-matmul head switches algorithms past M=4 and
            # stops being row-invariant at the bit level (measured at M=8:
            # every row differs from its solo run).  The batch lane's
            # solo-exact contract therefore runs the head per row here.
            return mx.concatenate(
                [self.lm_head(out[b:b + 1]) for b in range(out.shape[0])],
                axis=0)
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
