"""What does an extra kernel output cost on the host, and what does init_value add?

The megakernel collapsed three dispatches into one but did not get cheaper.
It has six outputs and needs init_value=0 for its barrier counter, and MLX
allocates and zero-fills every output.  Measure that directly.
"""
import json, time
import mlx.core as mx

def make_src(nout):
    body = "\n".join(f"        o{i}[i] = a[i];" for i in range(nout))
    return ("    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (i < N_) {\n" + body + "\n    }\n")

N = 2048
mx.random.seed(1)
a = mx.random.normal((N,)).astype(mx.float32)
mx.eval(a)

def build(nout):
    names = [f"o{i}" for i in range(nout)]
    return mx.fast.cuda_kernel(
        name=f"outcost{nout}", input_names=["a"], output_names=names,
        source=make_src(nout))

def measure(nout, init, n=400):
    k = build(nout)
    kw = dict(template=[("T_", mx.float32), ("N_", N)],
              grid=(N, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(N,)] * nout, output_dtypes=[mx.float32] * nout)
    if init:
        kw["init_value"] = 0
    k(inputs=[a], **kw); mx.synchronize()
    outs, ts = [], []
    for _ in range(n):
        t0 = time.perf_counter(); outs.append(k(inputs=[a], **kw)[0])
        ts.append(time.perf_counter() - t0)
    mx.synchronize(); t0 = time.perf_counter(); mx.eval(outs); mx.synchronize()
    ts.sort()
    return round(ts[len(ts)//2]*1e6, 2), round((time.perf_counter()-t0)/n*1e6, 2)

for nout in (1, 2, 3, 6):
    for init in (False, True):
        tr, su = measure(nout, init)
        print(json.dumps({"outputs": nout, "init_value_zero": init,
                          "trace_us": tr, "submit_us": su,
                          "host_us": round(tr+su, 2)}), flush=True)
