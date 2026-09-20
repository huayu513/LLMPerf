# Single configuration automation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent discovery/runtime tasks and review their integration. Track completion here.

**Goal:** Run a reproducible throughput search with `benchctl.py auto --config configs/experiment.json`, requiring only model_path, input_path and image.

**Architecture:** Resolve one strict JSON configuration into model and workload facts, discover host/image capabilities, generate compatible candidates, and execute measured adaptive search through the bundled AUTO launcher. Save resolved facts, immutable plan, per-attempt evidence and a verified winner in a fresh result directory.

**Tech Stack:** Python 3.11 standard library on host; Docker/NVIDIA and SGLang plus replay dependencies in the pinned image; Bash bundled runtime.

**Spec:** User-approved design in this conversation: single configuration includes first-deployment image; automatically derive model/workload/policy, preserve request semantics, propagate runtime arguments, search concurrency and repeat winner, retain existing profiles for now. Delete obsolete files when no longer referenced.

## Global constraints

- Required configuration keys: model_path, input_path, image (immutable repository@sha256 digest).
- Optional keys: output_dir, gpu_indexes, search, warmup, request_timeout, ready_timeout, model_overrides, docker. Unknown keys are errors.
- No real GPU benchmarks, image pulls/builds, or model code execution during local verification.
- Preserve original request objects. Unknown model adaptation and mixed request model aliases produce actionable errors instead of guessing.
- Existing F/O/Qwen/GLM profiles remain untouched; AUTO and shared runtime may change to honor generated parameters.
- No Git repository is available. Use /tmp snapshot and file diffs in place of commits/worktrees.

## Tasks and interfaces

### Task 1: Single configuration and discovery

Files: create automation/configuration.py, automation/discovery.py, tests/test_configuration.py, tests/test_discovery.py.

Interfaces: load_config(Path) -> RunConfig; config fields model_path: Path, input_path: Path, image: str, output_dir: Path|None, gpu_indexes: tuple[int,...]|None, search: SearchConfig, warmup: int, request_timeout: float, ready_timeout: int, model_overrides: dict, docker: DockerConfig, source_path: Path. SearchConfig defaults: concurrency_max=64, repetitions=3, max_trials=64, max_seconds=14400; optional backends and open_loop_scales tuples. discover_model(model_path, overrides=None) -> existing ModelManifest. inspect_workload(input_path) -> existing WorkloadManifest with sha256 and raw count/models metadata. Model raw contains metadata, model_type, num_attention_heads, num_hidden_layers, is_moe and provenance.

- [x] Write tests for minimal config, relative paths, invalid/unknown keys and image, renamed model directories, quantization, model alias consistency, preserving requests.
- [x] Run tests and observe missing feature failures.
- [x] Implement strict configuration and read-only metadata discovery.
- [x] Run tests and review interfaces.

### Task 2: Runtime parameter and evidence contract

Files: automation/adapters.py, benchmarks/config/portable.env, benchmarks/server/profiles/AUTO.sh, benchmarks/server/common.sh, benchmarks/server/launch_server.sh, benchmarks/run_point.sh, benchmarks/replay/run_replay.sh; runtime tests.

Interfaces: retain ReplayAdapter.build_spec(task, attempt, attempt_dir), __call__ returning attempt directory. Add read_attempt(attempt_dir, task, metadata, exit_code=0) -> dict normalized result with status, reasons, output_tokens_per_second, summary_path, candidate_id, concurrency. New metadata forwards parsers, template kwargs, profile_env, quantization if required, warmup/timeouts. Candidate static_config includes tp/dp/pp, dp_attention/dp_lm_head, backend (runner), moe_a2a_backend, dspark, mem_fraction_static, max_running_requests, chunked_prefill_size. Docker device mapping is physical at allocation, dense logical inside the container. Readiness/server parameter evidence must be explicit and validated; do not rank exit code alone.

- [x] Write failing command propagation, shell dry-run and invalid-results tests.
- [x] Implement parameter forwarding, fix mode spelling and Python executable lookup, capture server_info evidence.
- [x] Run targeted tests and review.

### Task 3: Automatic candidate planning and measured search

Files: automation/planner.py, automation/search.py, tests/test_search.py, planner tests.

Interfaces: create_search_plan(config, model, workload, env, result_dir) -> Plan; execute_search(plan, run_root, executor=None, resume=False) -> dict. Executor callback(task, attempt) returns normalized dict. Search increases concurrency by 16 until plateau/failure/bound, screens candidates fairly before deeper search, and reserves repeated full-workload validation; optional open-loop evidence is separate from closed-loop winner. Budget limits new trials and stops cleanly. Results and decisions persist; resumed runs validate input/plan identity and completed artifacts.

- [x] Test GPU/model filtering, concurrency >1, plateau/OOM behavior, best measurement versus largest concurrency, repeats, no valid winner, resume and budget exhaustion with deterministic fake executors.
- [x] Implement candidate generation using hardware/model facts, adaptive search and normalized reports.
- [x] Verify targeted tests.

### Task 4: One CLI and cleanup

Files: benchctl.py, automation/workflow.py, configuration example, README.md, tests/test_workflow.py; remove unreferenced legacy manifests/configs/wrappers/schemas/tests.

- [x] Test one-command lifecycle with injected environment/runtime, copied bundle CLI, failures before expensive runs and fresh output directories.
- [x] Implement auto --config, doctor/prepare/plan --config, run --plan [--resume], collect --run using the same resolved contract.
- [x] Save resolved-config.json, environment.json, data/replay_index.json, plan.json, results-index.json and best.json.
- [x] Delete legacy model/workload/policy JSONs and loader paths only after all consumers migrate.
- [x] Update Chinese user documentation around a single configuration and truthful search limits.
- [x] Run full unit suite, shell syntax checks, copied-bundle check and independent review.

## Progress and rulings

- Baseline: 56 unit tests passed under Python 3.11; no Docker or GPU runs.
- Configuration and runtime tasks own disjoint files; root owns planner/search/workflow/CLI and cleanup. Existing dataclasses remain internal contracts during migration.
- Ruling: metadata is inspected without importing checkpoint Python code. Runtime image owns actual model loading.
- Ruling: input timestamps remain part of the existing captured-request JSONL contract; the default optimization target is full-workload closed-loop output tokens/second.

## Integration evidence and review follow-ups

- Host Python 3.11 is required. System Python 3.6 is not a supported execution environment.
- Root planner/search/workflow/artifact regression group: 31 tests passed after mandatory launch-option and native-resume checks.
- Capability component: 18 tests passed after compute_cap fallback, Docker CSV GPU selection, probe timeout and launcher alignment fixes.
- Compared all 15 preserved manual profiles byte for byte against the pre-change archive: no differences.
- Deleted unused scheduler artifact validators; native read_attempt is the one active result-validation contract.
- Independent review identified variant-specific parser discovery and DP-attention parameter normalization; these were fixed and independently rechecked before final integration verification.
- Best reports keep planned_configuration, effective configuration, and native server command paths. Repetition configurations must agree; cached production results must retain their effective configuration and native evidence fingerprint.
- Resume runtime identity includes controller Python and CLI as well as bundled shell/replay code.

## Final verification

- Complete suite: **94 tests passed**, Python 3.11, including execution of the actual shell evidence writer through the host normalizer. Log: `/tmp/automation-final-suite.log`.
- Final independent review: both original reproductions (DeepSeek V3 static Jinja and nested CUDA graph backend fields) pass; **32 focused tests passed**, no outstanding review findings.
- All **28 Python sources** compile and all **22 shell/env files** pass `bash -n`.
- Copied-directory CLI validation is included in the passing suite. All 15 preserved manual profiles match the pre-change archive byte for byte.
- `configs/` contains only `experiment.json`. Removed four cache directories and four legacy standalone `.pyc` files; no runtime references to deleted configuration/loader/registry paths remain.
- No Docker launch, image pull/build, checkpoint code execution, or real GPU benchmark was performed during verification.
- Scope is a measured best within the generated candidate set and budget, not a guarantee of global optimality. Real checkpoint/image/hardware compatibility remains a deployment-time probe and smoke-test concern.
- No Git repository was available; the pre-change archive is `/tmp/automation-before-KiypXe/Automation.tar.gz`.
