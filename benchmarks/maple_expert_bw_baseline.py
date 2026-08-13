"""Baseline: effective weight-read bandwidth of the exact expert recipe.

Runs the pinned qmm_naive-recipe tile loop over 8 gathered experts (the
real decode shape) in isolation, times it, and reports achieved GB/s
against the device's streaming peak. The number to beat for the
memory-gap front.

    python benchmarks/maple_expert_bw_baseline.py
"""
import json
import time

import mlx.core as mx

SRC_V2 = r"""
    // Variant: warp reads its 8 rows INTERLEAVED -- lane l serves row
    // l%8 at word offset (l/8), quadrupling the distinct cache lines in
    // flight per warp at every step. Pure load-order change.
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int e = gw / (N_ / 8);
    const int tile = gw % (N_ / 8);
    if (e >= 8) return;
    const long long ebase = (long long)idx[e] * N_ * (K_ / 16);
    float acc = 0.0f;
    const int r = lane & 7;          // 8 rows per warp
    const int sub = lane >> 3;       // 4 lanes stride the words
    const int row = tile * 8 + r;
    const uint4* wr = reinterpret_cast<const uint4*>(
        wq + ebase + (long long)row * (K_ / 16));
    const int words = K_ / 16 / 4;
    for (int w = sub; w < words; w += 4) {
        const uint4 v = wr[w];
        acc += (float)(v.x ^ v.y ^ v.z ^ v.w);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[gw] = acc;
"""

SRC = r"""
    // 8 experts x (N_ x K_) 2-bit weights; one warp per (expert, 8-col
    // tile); the exact dequant+MMA recipe's LOAD pattern (uint4 pairs +
    // one scale/bias per 128-tile) -- compute reduced to a checksum so
    // the timing isolates the memory path.
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int e = gw / (N_ / 8);
    const int tile = gw % (N_ / 8);
    if (e >= 8) return;
    const long long ebase = (long long)idx[e] * N_ * (K_ / 16);
    float acc = 0.0f;
    for (int c = 0; c < 8; ++c) {
        const int row = tile * 8 + c;
        const uint4* wr = reinterpret_cast<const uint4*>(
            wq + ebase + (long long)row * (K_ / 16));
        const int words = K_ / 16 / 4;
        for (int w = lane; w < words; w += 32) {
            const uint4 v = wr[w];
            acc += (float)(v.x ^ v.y ^ v.z ^ v.w);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[gw] = acc;
"""

E, N, K = 256, 1408, 2048   # up_gate rows per expert (2*704), gs=128
mx.random.seed(1)
wq = mx.random.randint(0, 2**31 - 1, (E * N * (K // 16),)).astype(mx.uint32)
idx = mx.array([3, 17, 42, 99, 120, 200, 230, 255], mx.uint32)
mx.eval(wq, idx)

kern = mx.fast.cuda_kernel(
    name="expert_bw_probe", input_names=["wq", "idx"], output_names=["out"],
    source=SRC)
kern2 = mx.fast.cuda_kernel(
    name="expert_bw_probe_v2", input_names=["wq", "idx"],
    output_names=["out"], source=SRC_V2)

warps = 8 * (N // 8)
grid = ((warps * 32 + 255) // 256) * 256


def run(k=None):
    (o,) = (k or kern)(inputs=[wq, idx],
                template=[("N_", N), ("K_", K), ("THREADS_", 256)],
                grid=(grid, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(warps,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


bytes_read = 8 * N * (K // 4)
rep = {}
for name, k in (("recipe_order", kern), ("row_interleave", kern2)):
    run(k)
    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        run(k)
    dt = (time.perf_counter() - t0) / reps
    rep[name] = {"per_call_ms": round(dt * 1e3, 3),
                 "GBps": round(bytes_read / dt / 1e9, 1)}
print(json.dumps(rep), flush=True)
