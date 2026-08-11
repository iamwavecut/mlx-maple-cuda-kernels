"""Host-side cost per MLX operation: tracing (Python) and eval (C++ scheduling).

Decode is host-bound, so the currency of optimization is microseconds of host
time per op, not GPU time.  This measures both halves for the op types Maple's
decode step is built from, including mx.fast.cuda_kernel with a rebuilt argument
list versus a hoisted one.
"""
import json, time
import mlx.core as mx

N = 400
HID, EXP, TOPK = 2048, 256, 8


def trace_cost(make, n=N):
    """Median Python seconds to construct one op (no evaluation)."""
    outs = []
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        outs.append(make(i))
        ts.append(time.perf_counter() - t0)
    mx.eval(outs)
    mx.synchronize()
    ts.sort()
    return ts[len(ts) // 2]


def submit_cost(make, n=N):
    """Seconds of mx.eval per op for a tape of n independent ops."""
    outs = [make(i) for i in range(n)]
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / n


mx.random.seed(1)
x = mx.random.normal((HID,)).astype(mx.bfloat16)
w = mx.random.normal((EXP, HID)).astype(mx.bfloat16)
gates = mx.random.normal((1, EXP)).astype(mx.float32)
mx.eval(x, w, gates)

qw, sc, bi = mx.quantize(
    mx.random.normal((EXP, 1024, HID)).astype(mx.bfloat16), group_size=128, bits=2)
lhs = mx.zeros((1, TOPK), dtype=mx.uint32)
rhs = mx.arange(TOPK, dtype=mx.uint32).reshape(1, TOPK)
x4 = x.reshape(1, 1, 1, HID)
mx.eval(qw, sc, bi, lhs, rhs)

KERNEL = mx.fast.cuda_kernel(
    name="host_cost_probe",
    input_names=["a"],
    output_names=["o"],
    source="""
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N_) o[i] = a[i];
    """,
)
TEMPLATE = [("T_", mx.bfloat16), ("N_", HID)]
SHAPES = [(HID,)]
DTYPES = [mx.bfloat16]

cases = {
    "add (bf16 2048)": lambda i: x + float(i),
    "astype bf16->f32": lambda i: (x + float(i)).astype(mx.float32),
    "softmax f32 256": lambda i: mx.softmax(gates + float(i), axis=-1),
    "argpartition 256": lambda i: mx.argpartition(gates + float(i), kth=-TOPK, axis=-1),
    "take_along_axis": lambda i: mx.take_along_axis(
        gates, mx.argpartition(gates + float(i), kth=-TOPK, axis=-1)[..., -TOPK:], axis=-1),
    "matmul router gemv": lambda i: (w @ (x + float(i))),
    "gather_qmm 2bit": lambda i: mx.gather_qmm(
        x4 + float(i), qw, sc, bi, lhs_indices=lhs, rhs_indices=rhs,
        transpose=True, group_size=128, bits=2),
    "cuda_kernel (fresh args)": lambda i: KERNEL(
        inputs=[x + float(i)],
        template=[("T_", mx.bfloat16), ("N_", HID)],
        grid=(HID, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(HID,)], output_dtypes=[mx.bfloat16])[0],
    "cuda_kernel (hoisted args)": lambda i: KERNEL(
        inputs=[x + float(i)], template=TEMPLATE,
        grid=(HID, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=SHAPES, output_dtypes=DTYPES)[0],
}

rows = []
base_trace = trace_cost(lambda i: x + float(i))
base_submit = submit_cost(lambda i: x + float(i))
for name, fn in cases.items():
    t = trace_cost(fn)
    s = submit_cost(fn)
    rows.append({"op": name, "trace_us": t * 1e6, "submit_us": s * 1e6,
                 "host_us_total": (t + s) * 1e6})
    print(json.dumps(rows[-1]), flush=True)

print(json.dumps({"note": "baseline add op", "trace_us": base_trace * 1e6,
                  "submit_us": base_submit * 1e6}, indent=1))
with open("host_op_cost.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
