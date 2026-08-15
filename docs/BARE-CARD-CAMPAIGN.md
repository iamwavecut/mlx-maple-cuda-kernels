# Bare-card kernel campaign (next rental)

The shared-GPU bench cannot answer the remaining trunk questions
(140us/layer at 7x the bandwidth floor, barrier-serialized phases).
One rented quiet card (vast.ai 4090 ~$0.35/hr, ~2 hours) runs:

1. **nsys stall analysis** of the two production megakernels at B=1:
   `nsys profile --stats=true` over 200 decode steps; per-kernel SM
   occupancy, warp stall reasons (the latency chain's shape), memory
   throughput per phase. This decides the next kernel move
   (warp-specialization vs ILP restructuring vs accepting the wall).
2. **B=1 grid scans 64..128** for BOTH kernels (safe on a bare card;
   deadlock class only exists beside a foreign context) with the layer
   bit gate at each point.
3. **Wave-tail probe** rerun on silicon that is not shared (the farm
   run was non-monotonic).
4. If any of 1-3 yields a win: per-arch defaults + the full battery
   (108/162/60-gates, E2E, isolation) before any default flips.

Scripts: reuse `lab2026-08-10/{attn_paired_probe, wave_probe,
trunk_true_gpu, phase_budget}.py` plus an nsys wrapper; the standard
pod bootstrap (runpod-ssh-pod skill or vast flow in memory) with the
CUDA-12.9-headers fix applies unchanged.

## Run 1 verdicts (2026-08-15, bare 3090, ~$0.06)

- **Wave hypothesis dead**: 2560 rows ≥ speed of 2048 on clean silicon
  (54.1/64.2/59.3/59.5µs for 1024/2048/2304/2560). Nothing to rebalance.
- **nsys tooling miss**: the runpod image carries nsight-COMPUTE only;
  its bundled nsys writes no CUDA kernel data. Next run: install
  nsight-systems from NVIDIA's apt repo (or use a devtools image), and
  keep `ncu --set full` on one megakernel launch as the stall-reason
  fallback (verify ERR_NVGPUCTRPERM first).
- Remaining open question is unchanged: WHICH stalls make 140µs/layer
  out of a ~19µs read floor. Only proper stall attribution decides
  between warp-specialization, ILP restructuring, or accepting the
  wall.
