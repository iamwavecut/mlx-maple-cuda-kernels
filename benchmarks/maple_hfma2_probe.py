"""__hfma2 vs two scalar __hfma: bits first, then rate in the recipe shape.

    python benchmarks/maple_hfma2_probe.py
"""
import json
import statistics
import time

import mlx.core as mx

BITS_SRC = r"""
    // element i: scalar __hfma(a,b,c) vs the i-th half of packed __hfma2.
    const int i = blockIdx.x * THREADS_ + threadIdx.x;
    if (i >= N_) return;
    const __nv_bfloat16 a = av[i], b = bv[i], c = cv[i];
    const __nv_bfloat16 s = __hfma(a, b, c);
    __nv_bfloat162 a2, b2, c2;
    if (i + 1 < N_) { a2 = __nv_bfloat162(a, av[i+1]); b2 = __nv_bfloat162(b, bv[i+1]); c2 = __nv_bfloat162(c, cv[i+1]); }
    else { a2 = __nv_bfloat162(a, a); b2 = __nv_bfloat162(b, b); c2 = __nv_bfloat162(c, c); }
    const __nv_bfloat162 p = __hfma2(a2, b2, c2);
    ok[i] = (__bfloat16_as_ushort(p.x) == __bfloat16_as_ushort(s)) ? 1 : 0;
"""

RATE_SRC = r"""
    // recipe-shaped: per word, dequant + 16 accumulations; PACKED_ picks
    // 16 scalar __hfma vs 8 packed __hfma2 over the same lanes of data.
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int e = gw / (N_ / 8);
    const int tile = gw % (N_ / 8);
    if (e >= 8) return;
    float acc_all = 0.0f;
    for (int c = 0; c < 8; ++c) {
        const int row = tile * 8 + c;
        const long long rb = ((long long)idx[e] * N_ + row) * (K_ >> 4);
        const unsigned int* wrow =
            reinterpret_cast<const unsigned int*>(wq) + rb;
        const __nv_bfloat16* srow =
            reinterpret_cast<const __nv_bfloat16*>(sc)
            + ((long long)idx[e] * N_ + row) * (K_ >> 7);
#if PACKED_
        __nv_bfloat162 sums2[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            sums2[i] = __nv_bfloat162(__nv_bfloat16(0.f), __nv_bfloat16(0.f));
#else
        __nv_bfloat16 sums[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) sums[i] = __nv_bfloat16(0.0f);
#endif
        for (int base = lane * 16; base < K_; base += 512) {
            const unsigned int word = wrow[base >> 4];
            const __nv_bfloat16 scale = srow[base >> 7];
#if PACKED_
            const __nv_bfloat162 scale2(scale, scale);
            const __nv_bfloat162 one2(__nv_bfloat16(1.f), __nv_bfloat16(1.f));
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int qa = (word >> (4 * i)) & 3;
                const int qb = (word >> (4 * i + 2)) & 3;
                const __nv_bfloat162 q2(
                    __nv_bfloat16(float(qa)), __nv_bfloat16(float(qb)));
                const __nv_bfloat162 wdq2 = __hmul2(q2, scale2);
                sums2[i] = __hfma2(one2, wdq2, sums2[i]);
            }
#else
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int q = (word >> (2 * i)) & 3;
                const __nv_bfloat16 wdq = __hmul(
                    __nv_bfloat16(float(q)), scale);
                sums[i] = __hfma(__nv_bfloat16(1.0f), wdq, sums[i]);
            }
#endif
        }
        float s = 0.0f;
#if PACKED_
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            s += __bfloat162float(sums2[i].x) + __bfloat162float(sums2[i].y);
#else
        #pragma unroll
        for (int i = 0; i < 16; ++i) s += __bfloat162float(sums[i]);
#endif
        acc_all += s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc_all += __shfl_xor_sync(0xffffffffu, acc_all, o);
    if (lane == 0) out[gw] = acc_all;
"""

rep = {}
Nb = 1 << 20
mx.random.seed(5)
av = mx.random.normal((Nb,)).astype(mx.bfloat16)
bv = mx.random.normal((Nb,)).astype(mx.bfloat16)
cv = mx.random.normal((Nb,)).astype(mx.bfloat16)
mx.eval(av, bv, cv)
kb = mx.fast.cuda_kernel(name="hfma2_bits", input_names=["av", "bv", "cv"],
                         output_names=["ok"], source=BITS_SRC)
(okv,) = kb(inputs=[av, bv, cv], template=[("N_", Nb), ("THREADS_", 256)],
            grid=(((Nb + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(Nb,)], output_dtypes=[mx.uint32])
mx.eval(okv)
rep["bits"] = f"{int(mx.sum(okv).item())}/{Nb}"

E, N, K = 256, 1408, 2048
mx.random.seed(1)
wq = mx.random.randint(0, 2**31 - 1, (E * N * (K // 16),)).astype(mx.uint32)
sc = (mx.random.normal((E * N * (K // 128),)) * 0.05).astype(mx.bfloat16)
idx = mx.array([3, 17, 42, 99, 120, 200, 230, 255], mx.uint32)
mx.eval(wq, sc, idx)
warps = 8 * (N // 8)
grid = ((warps * 32 + 255) // 256) * 256

outs = {}
for name, packed in (("scalar", 0), ("hfma2", 1)):
    k = mx.fast.cuda_kernel(
        name=f"rate_{name}", input_names=["wq", "sc", "idx"],
        output_names=["out"], source=RATE_SRC)

    def run():
        (o,) = k(inputs=[wq, sc, idx],
                 template=[("N_", N), ("K_", K), ("THREADS_", 256),
                           ("PACKED_", packed)],
                 grid=(grid, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(warps,)], output_dtypes=[mx.float32])
        mx.eval(o)
        return o

    outs[name] = run()
    vals = []
    for r in range(3):
        run()
        t0 = time.perf_counter()
        for _ in range(40):
            run()
        vals.append((time.perf_counter() - t0) / 40)
    dt = statistics.median(vals)
    rep[name] = {"ms": round(dt * 1e3, 3),
                 "GBps": round(8 * N * (K // 4) / dt / 1e9, 1)}
mx.eval(*outs.values())
rep["checksums_equal"] = bool(
    mx.allclose(outs["scalar"], outs["hfma2"], atol=0, rtol=0).item())
print(json.dumps(rep), flush=True)
