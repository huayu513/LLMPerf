# Task 1: single configuration and discovery report

Implemented automation/configuration.py and automation/discovery.py, with behavior tests in tests/test_configuration.py and tests/test_discovery.py.

## Configuration

- load_config(Path) returns the planned RunConfig and accepts only the planned top-level JSON keys. Relative model, input, and output paths resolve against the configuration file.
- Search defaults are concurrency 64, repetitions 3, 64 trials, and 14,400 seconds. Backends and open-loop scales remain optional tuples.
- Validation rejects unknown nested keys, booleans used as numbers, non-finite numbers, invalid ranges, duplicate or empty GPU selections, duplicate scales and backends, trial budgets smaller than repetitions plus two, mutable images, and invalid Docker settings.
- Image syntax uses the existing DockerRuntime validator. Docker name prefixes are limited to 114 characters so runtime suffixes fit Docker's 128-character limit.

## Discovery

- discover_model reads only static checkpoint metadata. It classifies supported Qwen, GLM, DeepSeek V3/V3.2, and DeepSeek V4 models without using directory names or importing checkpoint code.
- Tool parser selection for shared architectures uses static chat templates from config.json, tokenizer_config.json, chat_template.jinja, or chat_templates/*.jinja. DeepSeek V3 function/fenced markers select deepseekv3, DSML function-call markers select deepseekv32, and Qwen function/parameter markers select qwen3_coder. Missing or conflicting evidence requires an explicit model_overrides.tool_call_parser value.
- Unknown model families fail with an actionable model_overrides error. The override surface is limited to structural facts, parser choices, chat template arguments, quantization, and string profile environment values that downstream execution consumes.
- No thinking flag or reasoning effort is enabled by default. The default chat_template_kwargs mapping is empty.
- ModelManifest.raw records original metadata, model type, architectures, attention heads, hidden layers, MoE classification, field provenance, and a checkpoint inventory.
- The checkpoint inventory SHA-256 hashes relevant metadata and tokenizer files. Large .safetensors and .bin files use filename, size, and mtime_ns facts and are never content-hashed.
- inspect_workload validates the bundled captured OpenAI Chat Completions format through prepare_jsonl_replay.validate_request, rejects duplicate IDs and mixed request model aliases, and records exact source SHA-256, request count, and alias counts without rewriting source bytes.
- prepare_workload invokes the bundled canonical indexer with the active Python interpreter and no tokenizer, then validates the emitted format, input hash, count, and aliases. Its input token statistics are explicitly absent.

## Verification

- Required Python 3.11 targeted suite: 18 tests passed, including actual concatenated-Jinja parser markers and ambiguity regressions.
- Python compilation check passed for both implementation modules and both test modules.
- A concurrent full Automation run executed 84 tests and reported four failures in runtime adapter/shell-contract files outside Task 1. Configuration and discovery tests passed; the root integration owner is resolving those runtime failures before the final full-suite run.

## Limits

- Checkpoint identity is exact for selected metadata files and stat-based for large weight files. Replacing a weight shard while preserving its filename, size, and nanosecond mtime is not detected.
- Local verification performs no GPU benchmark, Docker pull/build/run, model load, tokenizer load, or checkpoint Python execution.
