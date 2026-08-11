# Copyright © 2026 DeepGrove AI.

"""Portability self-check for Maple's Metal kernels and precision rules.

    python -m pytest tests/test_maple_kernels.py -v

Kernels have runtime fallbacks, so a kernel failure here means "the fast path
is off on this machine", not "the model is broken". A precision failure does
mean the model is wrong.
"""

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.models import maple


def _args(**kw):
    base = dict(
        num_hidden_layers=2,
        num_experts=64,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        vocab_size=1024,
        layer_types=["sliding_attention", "full_attention"],
    )
    base.update(kw)
    return maple.ModelArgs(**base)


class TestMapleKernels(unittest.TestCase):
    def test_benchmark_source_sha_falls_back_outside_a_git_checkout(self):
        """Disposable source bundles must retain their pinned provenance."""
        from benchmarks import maple_kernel_benchmark as benchmark

        failure = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        with mock.patch.object(benchmark.subprocess, "run", side_effect=failure):
            with mock.patch.dict(
                benchmark.os.environ, {"MAPLE_SOURCE_SHA": "pinned-source"}
            ):
                self.assertEqual(benchmark._git_sha(), "pinned-source")

    def test_kernel_benchmark_cli_smoke(self):
        """The CLI must emit one environment and every kernel result record."""
        backend = maple._kernel_backend()
        if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
            self.skipTest("custom kernel backend unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/maple_kernel_benchmark.py",
                    "--output",
                    str(output),
                    "--warmup",
                    "1",
                    "--trials",
                    "2",
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(sum(r["type"] == "environment" for r in records), 1)
        results = {r["kernel"] for r in records if r["type"] == "result"}
        self.assertEqual(
            results,
            {"add_rms_norm", "qk_norm_rope", "qk_norm_nope", "router"},
        )

    def test_cuda_elementwise_sweep_cli_smoke(self):
        """Every supported block-size candidate must run through the CLI."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/maple_elementwise_sweep.py",
                    "--output",
                    str(output),
                    "--warmup",
                    "1",
                    "--trials",
                    "2",
                    "--stress-dispatches",
                    "2",
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]
        candidates = {
            record["threads"] for record in records if record.get("status") == "ok"
        }
        self.assertEqual(candidates, {64, 128, 256, 512})

    def test_cuda_architecture_profiles(self):
        """Known generations must map to their tuned profile floors."""
        expected = {
            (8, 6): ("sm86", 256, 128, 2, False),
            (8, 9): ("sm89", 512, 128, 1, False),
            (9, 0): ("sm90", 512, 512, 1, True),
            (10, 0): ("sm100", 512, 512, 1, True),
            (12, 0): ("sm120", 512, 256, 1, True),
            (13, 0): ("future", 128, 256, 4, True),
        }
        for capability, (
            name,
            elementwise_threads,
            router_threads,
            rows_per_warp,
            reference_gemv,
        ) in expected.items():
            with self.subTest(capability=capability):
                profile = maple._cuda_profile_for_capability(capability)
                self.assertIsNotNone(profile)
                self.assertEqual(profile.name, name)
                self.assertEqual(profile.elementwise_threads, elementwise_threads)
                self.assertEqual(profile.router_threads, router_threads)
                self.assertEqual(profile.router_rows_per_warp, rows_per_warp)
                self.assertEqual(profile.router_reference_gemv, reference_gemv)
        self.assertIsNone(maple._cuda_profile_for_capability((8, 0)))

    def test_blackwell_qk_rope_pins_stock_second_half_rounding(self):
        """sm100 and sm120 pin the FFMA association used by stock MLX RoPE."""
        sources = {}
        with mock.patch.object(mx.fast, "cuda_kernel", return_value=object()) as make:
            for name in ("sm90", "sm100", "sm120"):
                maple._make_cuda_qk_norm_rope_kernel(maple._CudaProfile(name), True)
                sources[name] = make.call_args.kwargs["source"]

        pinned = (
            "__fmaf_rn(value, rope_cos[p], "
            "__fmul_rn(paired, rope_sin[p]))"
        )
        for name in ("sm100", "sm120"):
            self.assertIn(pinned, sources[name])
        self.assertNotIn(pinned, sources["sm90"])
        self.assertIn(
            "paired * rope_sin[p] + value * rope_cos[p]", sources["sm90"]
        )

    def test_cuda_router_launch_covers_every_expert(self):
        """Every tuned launch must produce exactly one row for every expert."""
        for threads in (64, 128, 256, 512):
            for rows_per_warp in (1, 2, 4, 8):
                profile = maple._CudaProfile(
                    "test",
                    router_threads=threads,
                    router_rows_per_warp=rows_per_warp,
                )
                with self.subTest(threads=threads, rows_per_warp=rows_per_warp):
                    grid = maple._cuda_router_grid_size(256, profile)
                    block_rows = threads // 32 * rows_per_warp
                    self.assertEqual(grid // threads * block_rows, 256)

    def test_cuda_router_counter_is_an_explicit_initialized_output(self):
        """CUDA graphs must see the multi-block counter write dependency."""
        factory = inspect.getsource(maple._make_cuda_router_kernel)
        call = inspect.getsource(maple.MapleGate._fused_call)
        self.assertIn('"ctr_out"', factory)
        self.assertNotIn('"ctr_in"', factory)
        self.assertIn("init_value=0", call)
        self.assertIn("output_pick = 7 - tid", factory)

    def test_experimental_decode_paths_are_disabled_by_default(self):
        self.assertFalse(maple._use_approximate_router)
        self.assertFalse(maple._use_approximate_add_rms)
        self.assertFalse(maple._use_cached_decode_lhs)
        self.assertFalse(maple._cuda_router_indices_uint32)

    def test_ternary_up_gate_stays_opt_in(self):
        """The non-bit-exact ternary GEMV cannot enter strict mode by default."""
        self.assertFalse(maple._use_cuda_ternary_up_gate)

    def test_scaled_rope_does_not_use_the_unscaled_cuda_kernel(self):
        """A scaling policy unsupported by the JIT kernel must stay portable."""
        args = _args(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        attention = maple.MapleAttention(args, 0)
        self.assertFalse(attention._can_fuse_qk)

    def test_kernel_factories_are_lazy(self):
        """A fresh import must not construct a backend-specific kernel."""
        code = """
from mlx_lm.models import maple
assert maple._add_rms_kernel_cache == {}
assert maple._qk_kernel_cache == {}
assert maple._router_kernel_cache == {}
assert maple._router_select_kernel_cache == {}
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_fused_router_matches_reference(self):
        """Candidate router stays close, while auto mode is array-exact."""
        args = _args()
        gate = maple.MapleGate(args)
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)
        x0 = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
        backend = maple._kernel_backend()
        if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
            self.skipTest("fused router disabled on this build")

        fi0, fs0 = gate._fused_call(x0)
        ri0, rs0 = gate._reference(x0)
        mx.eval(fi0, fs0, ri0, rs0)
        candidate_is_exact = (
            fi0.dtype == ri0.dtype
            and fs0.dtype == rs0.dtype
            and bool(mx.array_equal(fi0, ri0))
            and bool(mx.array_equal(fs0, rs0))
        )
        self.assertEqual(gate._probe(x0), candidate_is_exact)
        self.assertEqual(gate._fused_backend, backend)

        # Strict auto never admits this numerically approximate router, even
        # when one live input happens to compare exactly.
        previous_approximate = maple._use_approximate_router
        maple._use_approximate_router = False
        gate._fused = None
        ai0, as0 = gate(x0)
        mx.eval(ai0, as0)
        self.assertTrue(bool(mx.array_equal(ai0, ri0)))
        self.assertTrue(bool(mx.array_equal(as0, rs0)))
        self.assertFalse(gate._fused)

        # Explicit experimental mode retains the live-probed candidate.
        maple._use_approximate_router = True
        gate._fused = None
        ei0, es0 = gate(x0)
        mx.eval(ei0, es0)
        self.assertEqual(gate._fused, candidate_is_exact)
        if candidate_is_exact:
            self.assertTrue(bool(mx.array_equal(ei0, ri0)))
            self.assertTrue(bool(mx.array_equal(es0, rs0)))
        maple._use_approximate_router = previous_approximate

        set_matches = 0
        trials = 32
        for i in range(trials):
            x = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
            fi, fs = gate._fused_call(x)
            # The router runs in float32 (`router_dtype: fp32`), so the
            # reference must not round the logits to bf16 either.
            ri, rs = maple.group_expert_select(
                x.astype(mx.float32) @ gate.weight.astype(mx.float32).T,
                args.num_experts_per_tok,
            )
            mx.eval(fi, fs, ri, rs)

            if backend == "cuda":
                self.assertEqual(
                    list(map(int, fi.reshape(-1))),
                    list(map(int, ri.reshape(-1))),
                    "CUDA must preserve argpartition order for expert aggregation",
                )
                self.assertTrue(bool(mx.allclose(fs, rs, rtol=1e-5, atol=1e-5)))
                set_matches += 1
                continue

            fused = {int(a): float(b) for a, b in zip(fi.reshape(-1), fs.reshape(-1))}
            ref = {int(a): float(b) for a, b in zip(ri.reshape(-1), rs.reshape(-1))}
            if set(fused) == set(ref):
                set_matches += 1
                for k in fused:
                    # Same experts selected -> scores must agree to ~1 ulp.
                    self.assertAlmostEqual(fused[k], ref[k], places=5)

        # Exact ties at the top-k boundary may select the other equally scored
        # expert; that is legitimate, but it must be rare.
        self.assertGreater(set_matches, trials * 0.75)

    def test_fused_router_counter_resets_across_dispatches(self):
        """Every dispatch must elect one last block and start the next at zero."""
        backend = maple._kernel_backend()
        if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
            self.skipTest("fused router disabled on this build")
        args = _args(num_experts=256)
        gate = maple.MapleGate(args)
        mx.random.seed(17)
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)

        for dispatch in range(128):
            x = (
                mx.random.normal((1, 1, args.hidden_size)) * 0.5 + dispatch / 1024.0
            ).astype(mx.bfloat16)
            inds, scores = gate._fused_call(x)
            ref_inds, ref_scores = gate._reference(x)
            mx.eval(inds, scores, ref_inds, ref_scores)
            self.assertEqual(gate._fused_backend, backend)
            self.assertTrue(bool(mx.all((inds >= 0) & (inds < args.num_experts))))
            if backend == "cuda":
                self.assertTrue(
                    bool(mx.array_equal(inds, ref_inds)),
                    f"dispatch {dispatch} reordered experts",
                )
                scores_match = mx.allclose(
                    scores, ref_scores, rtol=1e-5, atol=1e-5
                )
            else:
                scores_match = mx.allclose(
                    mx.sort(scores), mx.sort(ref_scores), rtol=1e-5, atol=1e-5
                )
            self.assertTrue(
                bool(scores_match),
                f"dispatch {dispatch} produced stale or incomplete scores",
            )

    def test_cuda_router_reference_gemv_profile_matches_portable_selection(self):
        """Modern profiles may fuse selection while retaining MLX matmul logits."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        previous = maple._cuda_profile_cache
        profile = maple._cuda_profile()
        maple._cuda_profile_cache = maple._CudaProfile(
            "test_reference_gemv",
            elementwise_threads=profile.elementwise_threads,
            router_threads=256,
            router_rows_per_warp=4,
            router_reference_gemv=True,
        )
        try:
            args = _args(num_experts=256)
            gate = maple.MapleGate(args)
            mx.random.seed(190)
            gate.weight = (
                mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
            ).astype(mx.bfloat16)
            x = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
            indices, scores = gate._fused_call(x)
            ref_indices, ref_scores = gate._reference(x)
            mx.eval(indices, scores, ref_indices, ref_scores)
            self.assertEqual(
                list(map(int, indices.reshape(-1))),
                list(map(int, ref_indices.reshape(-1))),
                "selection must preserve argpartition order for deterministic "
                "expert aggregation",
            )
            self.assertTrue(
                bool(mx.allclose(scores, ref_scores, rtol=1e-5, atol=1e-5)),
                "selection scores must remain within the routing tolerance",
            )
        finally:
            maple._cuda_profile_cache = previous

    def test_modern_cuda_router_probe_rejects_reordered_experts(self):
        """The live guard must enforce the aggregation order on modern GPUs."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        previous = maple._cuda_profile_cache
        profile = maple._cuda_profile()
        maple._cuda_profile_cache = maple._CudaProfile(
            "test_reference_gemv_probe",
            elementwise_threads=profile.elementwise_threads,
            router_threads=256,
            router_rows_per_warp=1,
            router_reference_gemv=True,
        )
        try:
            args = _args(num_experts=256)
            gate = maple.MapleGate(args)
            mx.random.seed(191)
            gate.weight = (
                mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
            ).astype(mx.bfloat16)
            x = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
            ref_indices, ref_scores = gate._reference(x)
            reverse = mx.array([7, 6, 5, 4, 3, 2, 1, 0])
            reordered = (
                mx.take(ref_indices, reverse, axis=-1),
                mx.take(ref_scores, reverse, axis=-1),
            )
            with mock.patch.object(gate, "_fused_call", return_value=reordered):
                self.assertFalse(gate._probe(x))
        finally:
            maple._cuda_profile_cache = previous

    def test_router_logits_are_float32(self):
        """Routing must not round the expert logits to bf16.

        With 256 experts the top-k boundary is routinely a near tie, and
        rounding logits of this magnitude to bf16 (spacing ~0.5 near 100)
        perturbs the renormalized scores by percent, not ulps. Both the fused
        and the fallback path are checked against a float64 computation.
        """
        args = _args(num_experts=256)
        gate = maple.MapleGate(args)
        mx.random.seed(0)
        # Logits land around 100, where bf16 has ~0.5 resolution.
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05 + 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)

        x = (mx.random.normal((1, 1, args.hidden_size)) * 0.2 + 1.0).astype(mx.bfloat16)
        w64 = np.array(gate.weight.astype(mx.float32), dtype=np.float64)
        x64 = np.array(x.astype(mx.float32), dtype=np.float64).reshape(-1)
        logits = w64 @ x64
        self.assertGreater(np.abs(logits).max(), 20.0, "test needs large logits")
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[: args.num_experts_per_tok]
        expect = dict(zip(top.tolist(), (p[top] / p[top].sum()).tolist()))

        for fused in (True, False):
            if fused and not gate._probe(x):
                continue
            gate._fused = fused
            inds, scores = gate(x)
            mx.eval(inds, scores)
            self.assertEqual(scores.dtype, mx.float32, "scores must stay float32")
            got = {
                int(i): float(s) for i, s in zip(inds.reshape(-1), scores.reshape(-1))
            }
            label = "fused" if fused else "fallback"
            self.assertEqual(set(got), set(expect), f"{label}: wrong experts selected")
            for e, want in expect.items():
                self.assertAlmostEqual(
                    got[e],
                    want,
                    delta=1e-3 * want,
                    msg=f"{label}: expert {e} score {got[e]} != {want}",
                )

    def test_add_rms_norm_matches_reference(self):
        """Fused residual add + RMSNorm vs the stock two-step path."""
        args = _args()
        dim, eps = args.hidden_size, args.rms_norm_eps
        w = (mx.random.normal((dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
        x = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        r = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        backend = maple._kernel_backend()
        if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
            self.skipTest("fused add+norm disabled on this build")
        self.assertTrue(
            maple._add_rms_norm_ok(dim, mx.bfloat16, w, eps),
            f"{backend} add+norm fast path failed its live comparison",
        )

        h, hn = maple._add_rms_norm(x, r, w, eps)
        mx.eval(h, hn)
        self.assertEqual(maple._last_add_rms_backend, backend)
        # The residual stream must be rounded exactly once, like a bf16 add.
        self.assertTrue(mx.array_equal(h, x + r), "residual add is not bit-exact")
        ref = mx.fast.rms_norm(
            (x + r).astype(mx.float32), w.astype(mx.float32), eps
        ).astype(mx.bfloat16)
        got = np.array(hn.astype(mx.float32))
        want = np.array(ref.astype(mx.float32))
        # bf16 has ~0.4% resolution; the two differ only in reduction order.
        self.assertTrue(
            np.allclose(got, want, rtol=8e-3), f"max |d| {np.abs(got - want).max()}"
        )
        if backend == "cuda":
            self.assertTrue(mx.array_equal(hn, ref), "CUDA BF16 norm must be exact")
            weight32 = w.astype(mx.float32)
            _, norm32 = maple._add_rms_norm(x, r, weight32, eps)
            ref32 = mx.fast.rms_norm((x + r).astype(mx.float32), weight32, eps).astype(
                mx.bfloat16
            )
            mx.eval(norm32, ref32)
            self.assertTrue(
                mx.array_equal(norm32, ref32),
                "CUDA kernel cache aliased BF16 and FP32 weights",
            )

    def test_add_rms_norm_rejects_unsupported_dimension_before_compile(self):
        """A partial CUDA block must never read or write past an odd width."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        dim, eps = 1537, 1e-6
        w = mx.ones((dim,), dtype=mx.bfloat16)
        cached = set(maple._add_rms_kernel_cache)
        self.assertFalse(maple._add_rms_norm_ok(dim, mx.bfloat16, w, eps))
        self.assertEqual(set(maple._add_rms_kernel_cache), cached)

    def test_blackwell_qk_frozen_rounding_boundary_matches_reference(self):
        """Frozen upper-half RoPE midpoint must stay exact on Blackwell."""
        profile = maple._cuda_profile()
        if (
            maple._kernel_backend() != "cuda"
            or profile is None
            or profile.name not in ("sm100", "sm120")
        ):
            self.skipTest("frozen Blackwell boundary requires sm100 or sm120")
        fixture_path = Path(__file__).parent / "data" / "sm100_qk_rope_boundary.npz"
        with np.load(fixture_path, allow_pickle=False) as fixture:
            qk_values = fixture["qk_bfloat16_as_float32"]
            q_weight = fixture["q_norm_weight_float32"]
            k_weight = fixture["k_norm_weight_float32"]
        args = _args(num_hidden_layers=1, layer_types=["sliding_attention"])
        attn = maple.MapleAttention(args, 0)
        attn.q_norm.weight = mx.array(q_weight)
        attn.k_norm.weight = mx.array(k_weight)
        qk = mx.array(qk_values).astype(mx.bfloat16)
        got = attn._qk_fused(qk, 613)
        want = attn._qk_reference(qk, 613)
        mx.eval(got, want)
        self.assertEqual(attn._fused_qk_backend, "cuda")
        self.assertTrue(
            mx.array_equal(got, want),
            f"{profile.name} upper-half RoPE contraction changed at head=8, dim=45",
        )

    def test_qk_norm_rope_matches_reference(self):
        """Fused per-head norm + partial RoPE vs q_norm/k_norm + mx.fast.rope,
        on both the RoPE and the NoPE layer type."""
        backend = maple._kernel_backend()
        if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
            self.skipTest("Q/K custom kernel unavailable on this backend")
        for layer_type in ("sliding_attention", "full_attention"):
            for weight_dtype in (mx.bfloat16, mx.float32):
                args = _args(num_hidden_layers=1, layer_types=[layer_type])
                attn = maple.MapleAttention(args, 0)
                n = args.num_attention_heads + args.num_key_value_heads
                attn.q_norm.weight = (
                    mx.random.normal((args.head_dim,)) * 0.1 + 1.0
                ).astype(weight_dtype)
                attn.k_norm.weight = (
                    mx.random.normal((args.head_dim,)) * 0.1 + 1.0
                ).astype(weight_dtype)
                qk = (mx.random.normal((n, args.head_dim)) * 0.5).astype(mx.bfloat16)
                mx.eval(attn.parameters(), qk)

                for offset in (0, 7, 511, 613):
                    with self.subTest(
                        layer_type=layer_type,
                        weight_dtype=weight_dtype,
                        offset=offset,
                    ):
                        got = attn._qk_fused(qk, offset)
                        want = attn._qk_reference(qk, offset)
                        mx.eval(got, want)
                        self.assertEqual(attn._fused_qk_backend, backend)
                        self.assertEqual(got.dtype, mx.bfloat16)
                        g = np.array(got.astype(mx.float32))
                        w = np.array(want.astype(mx.float32))
                        self.assertTrue(
                            np.allclose(g, w, rtol=8e-3, atol=8e-3),
                            f"max |d| {np.abs(g - w).max()}",
                        )
                        if backend == "cuda":
                            self.assertTrue(
                                mx.array_equal(got, want),
                                "CUDA fused Q/K must preserve the reference rounding; "
                                f"layer_type={layer_type}, weight_dtype={weight_dtype}, "
                                f"offset={offset}, mismatches={(g != w).sum()}, "
                                f"max |d|={np.abs(g - w).max()}",
                            )

    def test_probe_rejects_a_mismatched_kernel(self):
        """The self-check must latch the fallback rather than ship garbage."""
        if maple._kernel_backend() != "metal":
            self.skipTest("covers the Metal live-probe path")
        args = _args(num_hidden_layers=1, layer_types=["sliding_attention"])
        attn = maple.MapleAttention(args, 0)
        qk = (mx.random.normal((20, args.head_dim)) * 0.5).astype(mx.bfloat16)
        mx.eval(attn.parameters(), qk)
        self.assertFalse(
            maple._matches(
                lambda: (mx.zeros_like(qk),),
                lambda: (attn._qk_reference(qk, 7),),
            )
        )

    def test_decode_matches_with_fast_paths_off(self):
        """A decode step through the fused kernels must equal the stock path."""
        mx.random.seed(0)
        args = _args()
        model = maple.Model(args)
        for layer in model.layers:
            layer.mlp.gate.weight = (
                mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
            )
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        def decode(fused):
            cache = model.make_cache()
            model(mx.array([[3, 1, 4, 1, 5]]), cache=cache)
            model.model._fused_add_norm = fused
            for layer in model.layers:
                layer.self_attn._fused_qk = fused
                # The router is compared against float64 separately; a top-8
                # tie flipping between paths would make this test flaky.
                layer.mlp.gate._fused = False
            out = model(mx.array([[9]]), cache=cache)
            mx.eval(out)
            return np.array(out.astype(mx.float32)).reshape(-1)

        # `None` leaves the probes to decide, i.e. exactly what a user gets.
        fast = decode(None)
        if not model.model._fused_add_norm:
            self.skipTest("fused decode disabled on this build")
        if not all(layer.self_attn._fused_qk for layer in model.layers):
            self.skipTest("Q/K fused decode disabled on this build")
        slow = decode(False)
        self.assertEqual(int(fast.argmax()), int(slow.argmax()), "argmax differs")
        self.assertTrue(
            np.allclose(fast, slow, rtol=5e-2, atol=5e-2),
            f"max |d| {np.abs(fast - slow).max()}",
        )

    def test_experts_clamp_both_swiglu_branches(self):
        """silu(min(gate, 7)) * clip(up, -7, 7), in the activation dtype."""
        gate = mx.array([[-3.0, 0.5, 9.0, 40.0]], dtype=mx.bfloat16)
        up = mx.array([[100.0, -50.0, 2.0, -0.25]], dtype=mx.bfloat16)
        got = maple.clamped_swiglu(gate, up)
        mx.eval(got)
        self.assertEqual(got.dtype, mx.bfloat16, "clamp must not promote to f32")

        g = np.array(gate.astype(mx.float32))
        u = np.array(up.astype(mx.float32))
        g = np.minimum(g, maple.MLP_CLAMP)
        u = np.clip(u, -maple.MLP_CLAMP, maple.MLP_CLAMP)
        want = (g / (1 + np.exp(-g))) * u
        self.assertTrue(
            np.allclose(np.array(got.astype(mx.float32)), want, rtol=8e-3),
            f"{np.array(got.astype(mx.float32))} != {want}",
        )

    def test_row_and_group_scale_layouts_agree(self):
        """A row_alpha checkpoint must expand to the per-group tensors."""
        args = _args(quantization={"group_size": 128, "bits": 2})
        model = maple.Model(args)

        n, k, groups = 256, 2048, 2048 // 128
        alpha = mx.abs(mx.random.normal((n,))).astype(mx.bfloat16) + 0.01
        packed = mx.random.randint(0, 2**31, (n, k // 16)).astype(mx.uint32)
        # o_proj is not part of the q/k/v fusion, so it exercises the
        # row_alpha expansion on its own.
        prefix = "model.layers.0.self_attn.o_proj"

        rows = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.row_alpha": alpha}
        )
        scales = mx.broadcast_to(alpha[:, None], (n, groups))
        mx.eval(rows, scales)

        self.assertIn(f"{prefix}.scales", rows)
        self.assertIn(f"{prefix}.biases", rows)
        self.assertTrue(mx.array_equal(rows[f"{prefix}.scales"], scales))
        self.assertTrue(mx.array_equal(rows[f"{prefix}.biases"], -scales))
        # `--group-scales` checkpoints must pass through untouched.
        grouped = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.scales": scales}
        )
        self.assertTrue(mx.array_equal(grouped[f"{prefix}.scales"], scales))


    def test_exact_add_rms_needs_outliers_to_be_validated(self):
        """The corrected fused add+RMSNorm is exact, including on outliers.

        A Gaussian probe cannot validate this kernel: on N(0, 1) inputs a
        butterfly reduction and a __shfl_down reduction agree on every trial,
        so a wrong reduction passes.  Only the wide dynamic range of a real
        residual stream separates them, which is why the shipped probe injects
        outliers.
        """
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        args = _args()
        dim, eps = args.hidden_size, args.rms_norm_eps
        w = (mx.random.normal((dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
        self.assertTrue(maple._exact_add_rms_ok(dim, mx.bfloat16, w, eps))

        spikes = mx.random.normal((1, 1, dim))
        x = (mx.random.normal((1, 1, dim))
             + mx.where(spikes > 2.0, spikes * 30.0, 0.0)).astype(mx.bfloat16)
        r = (mx.random.normal((1, 1, dim))
             + mx.where(spikes < -2.0, spikes * 30.0, 0.0)).astype(mx.bfloat16)
        h, hn = maple._exact_add_rms_norm(x, r, w, eps)
        ref = mx.fast.rms_norm(
            (x + r).astype(mx.float32), w.astype(mx.float32), eps
        ).astype(mx.bfloat16)
        mx.eval(h, hn, ref)
        self.assertTrue(mx.array_equal(h, x + r), "residual add is not bit-exact")
        self.assertTrue(mx.array_equal(hn, ref), "norm is not array-exact")

    def test_exact_add_rms_rejects_unsupported_dimension(self):
        """A width that cannot be split four-per-thread must fail closed."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        self.assertFalse(maple._exact_add_rms_supported(1537))
        self.assertFalse(maple._exact_add_rms_supported(2048 * 4))

    def test_qkv_split_matches_the_sliced_path(self):
        """The widened kernel must reproduce the slice-and-reshape chain."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        args = _args()
        model = maple.Model(args)
        attn = model.model.layers[0].self_attn
        n_q, n_kv, hd = (attn.num_attention_heads, attn.num_key_value_heads,
                         attn.head_dim)
        qkv = mx.random.normal((1, 1, (n_q + 2 * n_kv) * hd)).astype(mx.bfloat16)
        qk = qkv.reshape(-1)[: (n_q + n_kv) * hd].reshape(n_q + n_kv, hd)
        mx.eval(qkv)
        self.assertTrue(attn._probe_qkv_split(qkv, qk))

        queries, keys, values = attn._qkv_split(qkv, 7)
        out = attn._qk_fused(qk, 7)
        mx.eval(queries, keys, values, out)
        self.assertTrue(mx.array_equal(
            queries, out[:n_q].reshape(1, n_q, 1, hd)))
        self.assertTrue(mx.array_equal(
            keys, out[n_q:].reshape(1, n_kv, 1, hd)))
        self.assertTrue(mx.array_equal(
            values,
            qkv.reshape(-1)[(n_q + n_kv) * hd:].reshape(1, n_kv, 1, hd)))

    def test_compiled_router_is_array_exact(self):
        """mx.compile over the stock chain must not move a single bit."""
        args = _args()
        model = maple.Model(args)
        gate = model.model.layers[0].mlp.gate
        x = mx.random.normal((1, gate.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        ri, rs = gate._reference(x)
        ci, cs = gate._compiled(x)
        mx.eval(ri, rs, ci, cs)
        self.assertTrue(mx.array_equal(ri, ci), "compiled router reordered experts")
        self.assertTrue(mx.array_equal(rs, cs), "compiled router moved the weights")

    def test_the_strict_lane_stays_one_variable_away(self):
        """The megakernel is the default, so the exact lane must stay reachable.

        Read the defaults rather than the module attributes: the attributes are
        seeded from the environment, and a run that chose a lane explicitly
        should not fail this.
        """
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(maple._env_flag("MAPLE_MOE_MEGAKERNEL", True))
            self.assertFalse(maple._env_flag("MAPLE_COMPILED_ROUTER", False))
            self.assertTrue(maple._env_flag("MAPLE_FUSED_ADD_RMS", True))
            self.assertTrue(maple._env_flag("MAPLE_FUSED_QKV", True))
        with mock.patch.dict("os.environ", {"MAPLE_MOE_MEGAKERNEL": "0"}):
            self.assertFalse(maple._env_flag("MAPLE_MOE_MEGAKERNEL", True),
                             "the array-exact lane must be one variable away")

    def test_env_flag_only_accepts_affirmative_spellings(self):
        """A deployment sets these; a typo must not silently flip a lane."""
        for raw in ("1", "true", "TRUE", "yes", "on", " on "):
            with mock.patch.dict("os.environ", {"MAPLE_X": raw}):
                self.assertTrue(maple._env_flag("MAPLE_X", False), raw)
        for raw in ("0", "false", "no", "off", "", "maybe", "2"):
            with mock.patch.dict("os.environ", {"MAPLE_X": raw}):
                self.assertFalse(maple._env_flag("MAPLE_X", True), raw)

    def test_megakernel_grid_is_clamped(self):
        """The barrier deadlocks if the grid outgrows residency, so cap it."""
        for raw, expected in (("1", 8), ("10000", 240), ("96", 96),
                              ("not-a-number", None)):
            with mock.patch.dict("os.environ",
                                 {"MAPLE_MOE_MEGAKERNEL_GRID": raw}):
                grid = maple._moe_megakernel_grid(default=64)
                self.assertEqual(grid, 64 if expected is None else expected,
                                 f"{raw!r} produced {grid}")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(maple._moe_megakernel_grid(default=64), 64)

    def test_moe_megakernel_rejects_unsupported_layers(self):
        """An unquantized MoE block must fall back instead of dispatching."""
        args = _args()
        model = maple.Model(args)
        layer = model.model.layers[0]
        ln = layer.post_attention_layernorm
        plan = maple._moe_megakernel_plan(layer.mlp, ln, mx.bfloat16)
        self.assertFalse(plan, "unquantized experts must not build a plan")

    def test_exact_megakernel_matches_the_stock_chain_bit_for_bit(self):
        """The exact lane's whole point: post-norm, router, experts,
        activation, aggregation and the next-layer fuse, one dispatch,
        array-equal to the strict chain."""
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        import mlx.nn as nn

        mx.random.seed(21)
        args = _args(num_experts=256)
        model = maple.Model(args)
        moe = None
        for layer in model.model.layers:
            if getattr(layer.mlp, "switch_mlp", None) is not None:
                moe = layer
                break
        self.assertIsNotNone(moe, "fixture has no MoE layer")
        moe.mlp.gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
        )
        model.set_dtype(mx.bfloat16)
        nn.quantize(moe.mlp.switch_mlp, group_size=128, bits=2)
        mx.eval(model.parameters())

        ln = moe.post_attention_layernorm
        next_w = model.model.norm.weight
        h = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(
            mx.bfloat16
        )
        r = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(
            mx.bfloat16
        )
        mx.eval(h, r)

        got = maple._moe_exact_megakernel_call(moe, h, r, ln, next_w)
        if got is None:
            self.skipTest("exact megakernel plan rejected the fixture")
        hout, hn = got

        s, x = maple._exact_add_rms_norm(h, r, ln.weight, ln.eps)
        ref_moe = moe.mlp(x)
        ref_h, ref_hn = maple._exact_add_rms_norm(s, ref_moe, next_w, ln.eps)
        mx.eval(hout, hn, ref_h, ref_hn)

        self.assertTrue(
            mx.array_equal(hout, ref_h).item(),
            "carrier differs from the stock chain",
        )
        self.assertTrue(
            mx.array_equal(hn, ref_hn).item(),
            "next attention input differs from the stock chain",
        )

    def test_exact_megakernel_is_the_default_lane(self):
        """The screened exact lane leads; the ~1 ULP lane is the fallback."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(
                maple._env_flag("MAPLE_MOE_MEGAKERNEL_EXACT", True)
            )
        with mock.patch.dict("os.environ", {"MAPLE_MOE_MEGAKERNEL_EXACT": "0"}):
            self.assertFalse(
                maple._env_flag("MAPLE_MOE_MEGAKERNEL_EXACT", True),
                "the escape hatch must stay one variable away",
            )

    def test_megakernel_tail_norm_is_exactly_the_fuse(self):
        """Phase E must be bit-identical to the standalone exact fuse.

        The megakernel now emits the next layer's attention input itself.  The
        lane's inexactness budget lives in the MoE math alone, so the tail's
        add+norm of its own carrier has to reproduce `_exact_add_rms_norm` to
        the last bit -- otherwise the fusion quietly widened the error story.
        """
        if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
            self.skipTest("CUDA fast path unavailable")
        import mlx.nn as nn

        mx.random.seed(7)
        args = _args()
        model = maple.Model(args)
        moe = None
        for layer in model.model.layers:
            if getattr(layer.mlp, "switch_mlp", None) is not None:
                moe = layer
                break
        self.assertIsNotNone(moe, "fixture has no MoE layer")
        moe.mlp.gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
        )
        model.set_dtype(mx.bfloat16)
        nn.quantize(moe.mlp.switch_mlp, group_size=128, bits=2)
        mx.eval(model.parameters())

        h = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
        r = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
        next_w = model.model.norm.weight
        ln = moe.post_attention_layernorm
        mx.eval(h, r)

        fused = maple._moe_megakernel_call(moe, h, r, ln, next_w)
        if fused is None:
            self.skipTest("megakernel plan rejected the quantized fixture")
        hout, hn = fused
        mx.eval(hout, hn)

        # The tail normed its own carrier; the fuse over (carrier, 0) must
        # agree exactly, because adding bf16 zero is exact.
        zero = mx.zeros_like(hout)
        ref_h, ref_hn = maple._exact_add_rms_norm(hout, zero, next_w, ln.eps)
        mx.eval(ref_h, ref_hn)
        self.assertTrue(
            mx.array_equal(ref_h, hout).item(), "carrier changed under a zero add"
        )
        self.assertTrue(
            mx.array_equal(ref_hn, hn).item(),
            "tail norm is not bit-identical to the exact fuse",
        )

        # And the carrier itself must be the residual chain, loosely: the MoE
        # values differ from stock by design, but a staging bug would miss by
        # far more than 1 ULP of bf16.
        s = (h.astype(mx.float32) + r.astype(mx.float32)).astype(mx.bfloat16)
        x = mx.fast.rms_norm(s, ln.weight, ln.eps)
        s2 = s.astype(mx.float32) + moe.mlp(x).astype(mx.float32)
        close = mx.allclose(
            hout.astype(mx.float32), s2, rtol=5e-2, atol=5e-2
        ).item()
        self.assertTrue(close, "carrier is not the residual chain")


if __name__ == "__main__":
    unittest.main()
