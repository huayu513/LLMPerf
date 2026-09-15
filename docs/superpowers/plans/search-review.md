# Search and lifecycle review — 2026-09-09

**Final review status: all reported findings resolved; no outstanding findings in this scoped review.**

Scope: read-only review of `automation/planner.py`, `automation/search.py`, `automation/workflow.py`, and `benchctl.py`, against tasks 3/4 of the approved single-configuration plan. No source changes, Docker, GPU activity, image operations, network access, or model execution. Reproductions used Python 3.11 and deterministic fake executors in temporary directories.

## Original findings (resolved)

### 1. P1 — Screening permanently excludes later strategies and does not replace failed candidates

Locations: `automation/search.py:143–150`; ordering source `automation/planner.py:131–132`.

`screen_count` is fixed to `(max_trials - repetitions) // 4`, and the controller only visits `plan.candidates[:screen_count]`. The cutoff counts failed smoke attempts as occupied screening slots and never draws replacements from the remaining plan.

Reproduction with `tests.test_search.plan(max_trials=64, candidates=16)`: make candidates c0–c14 return FAILED and c15 return VALID. The result is INCONCLUSIVE after 15 trials, with 49 trial slots unused; c15 never runs.

The prefix also systematically excludes strategy families under a normal eight-GPU configuration. `create_search_plan` with eight H100 GPUs, an MoE model with 64 attention heads and 80 layers, and the existing advertised triton capability produces 74 candidates. Its first 15 are exclusively ordinary TP/DP, PP=1, backend=auto, with DP attention disabled. The default search screens exactly those 15, so every DP-attention, pipeline, and explicit backend candidate is excluded. DSpark candidates would likewise sort after all non-DSpark auto candidates.

Recommendation: select the initial screening set across strategy families, and backfill failed screenings from untried candidates while preserving the final-repeat reserve and the intended deeper-search allowance. Add deterministic tests for failed-screen replacement and representative default-budget coverage of a multi-GPU MoE plan.

### 2. P2 — Open-loop scale rounding reuses the wrong trial and reports false completion

Locations: `automation/search.py:232–233` and cached result return at `automation/search.py:95–110`.

The trial suffix uses `f'-s{scale:g}'`, whose default precision collapses distinct accepted scale values. With `search.open_loop_scales = [1.0, 1.0000001]`, both requests map to the same task ID. The second call reuses the first result without checking the requested scale.

Deterministic reproduction returned PASS while the executor received only one open-loop task, at scale 1.0. `open_loop_results` contained two copies of the 1.0 result, so the length/status check falsely counted the second requested scale as completed.

Recommendation: use a lossless float representation or canonical scale hash in the task identity, and check the cached task parameters before reuse. Test close but distinct scales and resumed reuse.

## Integration observation resolved during review

The planner originally omitted explicit `model_overrides.quantization` from runtime metadata, although ReplayAdapter forwards only the metadata field to QUANTIZATION. This was reported promptly; the root agent confirmed a planner fix that forwards explicit-override quantization. It is therefore not listed as an outstanding finding.

The root agent also reported adding saved replay-index fingerprint and environment identity checks during review. No transient discovery/runtime edits were treated as findings.

## Scoped re-review of fixes

The root agent added strategy-family interleaving, replaced the fixed screening prefix with successful-screen backfill, and hashed losslessly serialized open-loop scale values. Independently ran all 17 tests in `test_auto_planner` and `test_search`; all passed. A separate close-scale reproduction now executes both 1.0 and 1.0000001, records the corresponding distinct results, and makes zero new executor calls when resumed.

Original finding 2 is resolved. Original finding 1's strategy-prefix exclusion and failed-only early termination are resolved, but the backfill change introduces the remaining budget issue below.

### Follow-up P2 (resolved) — Failed-screen backfill can consume every adaptive-search slot

Location: `automation/search.py:143–156`.

The loop now continues until `screen_count` candidates have valid smoke and baseline results, even after usable candidates have been found. Failed replacements consume the entire `explore_limit`; the subsequent doubling and neighbor probes then have no trial slots left.

Deterministic reproduction: `plan(max_trials=64, candidates=64)`, c0 always VALID at `10 * task.concurrency`, every other candidate FAILED. The controller spends 2 trials on c0's smoke/baseline, 59 trials on failed smoke replacements, and 3 on c0's final repeats. It returns PASS after 64 trials with winner concurrency 1; c0 is never tested above concurrency 1. The original fixed-prefix implementation retained budget for c0's adaptive search in this same scenario.

Recommendation: once successful candidates exist, stop replacement screening early enough to preserve at least their first concurrency doubling, or reserve an explicit deeper-search allowance. Continue replacing failed-only screens when there are no viable candidates. Add a regression for one successful early candidate followed by many failed candidates, asserting a measured concurrency above 1 while retaining the final repeats and trial bound.


## Final scoped verification

Reviewed the screening-budget fix at `automation/search.py:143–162`. Once a viable baseline exists, the controller stops replacement screening at half the exploration allowance. Its screening accounting comes from visited attempts, preserving the same scheduling path when completed trials are reused. Failed-only screening can still continue to later candidates.

Independently reran the complete planner/search test modules under Python 3.11: **18 tests passed**. The new regression reproduces the 64-candidate case with only c0 usable, confirms a verified winner above concurrency 1, and confirms resume returns the identical winner with zero new executor calls. Prior failed-only backfill, strategy-family coverage, close-scale identity, plateau/OOM, repeat reservation, and resume regressions also pass.

The follow-up budget finding is resolved. No outstanding findings remain from this review. Verification is limited to static inspection and CPU-only deterministic tests; actual Docker/GPU integration remains with the root task.
