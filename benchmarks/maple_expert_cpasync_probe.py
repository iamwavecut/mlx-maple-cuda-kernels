"""cp.async k-tile double-buffering under recipe-shaped compute.

Both variants run the REAL per-word work of the exact qmv/qmm recipe
(dequant bf16(q*s)+z and 16 HFMA accumulations per uint32 word); the
async variant stages the NEXT word-batch into shared memory with
cp.async while the current one computes. Bit-neutral by construction:
the arithmetic order per word is untouched, only WHERE the word is read
from changes.

    python benchmarks/maple_expert_cpasync_probe.py
"""
import json
import statistics
import time

import mlx.core as mx

COMMON = r"""
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int e = gw / (N_ / 8);
    const int tile = gw % (N_ / 8);
    if (e >= 8) return;
"""

BODY_SYNC = COMMON + r"""
    float acc_all = 0.0f;
    for (int c = 0; c < 8; ++c) {
        const int row = tile * 8 + c;
        const long long rbase =
            ((long long)idx[e] * N_ + row) * (K_ >> 4);
        const unsigned int* wrow =
            reinterpret_cast<const unsigned int*>(wq) + rbase;
        const __nv_bfloat16* srow =
            reinterpret_cast<const __nv_bfloat16*>(sc)
            + ((long long)idx[e] * N_ + row) * (K_ >> 7);
        __nv_bfloat16 sums[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) sums[i] = __nv_bfloat16(0.0f);
        for (int base = lane * 16; base < K_; base += 512) {
            const unsigned int word = wrow[base >> 4];
            const __nv_bfloat16 scale = srow[base >> 7];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int q = (word >> (2 * i)) & 3;
                const __nv_bfloat16 wdq = __hmul(
                    __nv_bfloat16(float(q)), scale);
                sums[i] = __hfma(__nv_bfloat16(1.0f), wdq, sums[i]);
            }
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) s += __bfloat162float(sums[i]);
        acc_all += s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc_all += __shfl_xor_sync(0xffffffffu, acc_all, o);
    if (lane == 0) out[gw] = acc_all;
"""

BODY_ASYNC = COMMON + r"""
    // double-buffered word staging: 2 buffers x 32 lanes x 1 word per warp
    __shared__ unsigned int stage[THREADS_ / 32][2][32];
    float acc_all = 0.0f;
    for (int c = 0; c < 8; ++c) {
        const int row = tile * 8 + c;
        const long long rbase =
            ((long long)idx[e] * N_ + row) * (K_ >> 4);
        const unsigned int* wrow =
            reinterpret_cast<const unsigned int*>(wq) + rbase;
        const __nv_bfloat16* srow =
            reinterpret_cast<const __nv_bfloat16*>(sc)
            + ((long long)idx[e] * N_ + row) * (K_ >> 7);
        __nv_bfloat16 sums[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) sums[i] = __nv_bfloat16(0.0f);
        // prefetch batch 0: lane's first word
        {
            const unsigned int* src = wrow + (lane * 16 >> 4);
            unsigned int* dst = &stage[wrp][0][lane];
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
            asm volatile("cp.async.commit_group;\n");
        }
        int buf = 0;
        for (int base = lane * 16; base < K_; base += 512) {
            const int nbase = base + 512;
            if (nbase < K_) {
                const unsigned int* src = wrow + (nbase >> 4);
                unsigned int* dst = &stage[wrp][buf ^ 1][lane];
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                    "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
                asm volatile("cp.async.commit_group;\n");
                asm volatile("cp.async.wait_group 1;\n");
            } else {
                asm volatile("cp.async.wait_group 0;\n");
            }
            __syncwarp();
            const unsigned int word = stage[wrp][buf][lane];
            const __nv_bfloat16 scale = srow[base >> 7];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int q = (word >> (2 * i)) & 3;
                const __nv_bfloat16 wdq = __hmul(
                    __nv_bfloat16(float(q)), scale);
                sums[i] = __hfma(__nv_bfloat16(1.0f), wdq, sums[i]);
            }
            buf ^= 1;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) s += __bfloat162float(sums[i]);
        acc_all += s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc_all += __shfl_xor_sync(0xffffffffu, acc_all, o);
    if (lane == 0) out[gw] = acc_all;
"""

E, N, K = 256, 1408, 2048
mx.random.seed(1)
wq = mx.random.randint(0, 2**31 - 1, (E * N * (K // 16),)).astype(mx.uint32)
sc = (mx.random.normal((E * N * (K // 128),)) * 0.05).astype(mx.bfloat16)
idx = mx.array([3, 17, 42, 99, 120, 200, 230, 255], mx.uint32)
mx.eval(wq, sc, idx)

ks = mx.fast.cuda_kernel(name="exp_sync", input_names=["wq", "sc", "idx"],
                         output_names=["out"], source=BODY_SYNC)
ka = mx.fast.cuda_kernel(name="exp_async", input_names=["wq", "sc", "idx"],
                         output_names=["out"], source=BODY_ASYNC)

warps = 8 * (N // 8)
grid = ((warps * 32 + 255) // 256) * 256


def run(k):
    (o,) = k(inputs=[wq, sc, idx],
             template=[("N_", N), ("K_", K), ("THREADS_", 256)],
             grid=(grid, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(warps,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


a = run(ks); b = run(ka)
mx.eval(a, b)
same = bool(mx.allclose(a, b, atol=0, rtol=0).item())

rep = {"checksums_equal": same}
for name, k in (("sync", ks), ("cp_async", ka)):
    vals = []
    for r in range(3):
        run(k)
        t0 = time.perf_counter()
        for _ in range(40):
            run(k)
        vals.append((time.perf_counter() - t0) / 40)
    dt = statistics.median(vals)
    rep[name] = {"ms": round(dt * 1e3, 3),
                 "GBps": round(8 * N * (K // 4) / dt / 1e9, 1)}
print(json.dumps(rep), flush=True)
