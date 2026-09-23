# Runtime integration report

The generated AUTO parameters now reach the container, launcher and replay process, including TP/DP/PP, MoE settings, parser/template settings, concurrency cap, warmup and timeouts. Docker physical GPU selection uses CSV quoting; CUDA device IDs inside the container are dense logical IDs. Container name prefixes are honored.

Each attempt saves native requested parameters, server info, resolved evidence, command, replay summary and controller state. The host reader validates the actual nested output layout, complete workload/usage accounting, positive finite throughput arithmetic, exact loopback service port, controller completion, and current candidate identity. It reads direct ServerArgs fields so nested CUDA graph backends cannot masquerade as MoE runners. Disabled speculation and DP-attention prefill normalization are handled explicitly. Startup/configuration failures are classified separately from replay/load failures so an explicit backend can be quarantined for later candidates without treating OOM or request errors as backend incompatibility.

The verified result exports effective values from native server info, preserves launch/planning values and hashes native artifacts. Cached results are revalidated during resume, and final repetitions must agree on effective configuration. Removed the unused adapter registry and scheduler-specific result validators.

Verification: final 94-test suite passes, including the actual embedded evidence writer and copied LLMPerf CLI; 28 Python sources compile and 22 shell/env files pass syntax checks. Independent review has no outstanding findings. No Docker or GPU execution was used for these checks.
