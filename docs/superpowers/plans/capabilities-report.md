# Capability probe report

The preserved public interface is `EnvironmentSnapshot`, `GPUInfo`,
`ImageSnapshot`, and `probe_environment(config, runner)`. The config surface
uses `image`, `paths`, `docker`, and optional `gpu_indexes`.

The pinned image probe records Python, Torch, CUDA, NCCL, and SGLang versions.
It requires `torch.cuda.is_available()` to be true and at least one CUDA device
to be visible. Kernel packages are optional; versions are recorded for
`sgl_kernel` or `sglang_kernel` when installed.

The container requires `sglang serve --help` to succeed. It does not accept
the Python module launcher as a fallback because runtime launches use the
`sglang` executable. CLI failures are recorded in probe errors, and the
snapshot explains that SGLang serve capabilities are unavailable. Parsed
capabilities are stored in `snapshot.container["capabilities"]` as `options`,
`runner_backends`, `a2a_backends`, `tool_parsers`, and `reasoning_parsers`.
Undiscovered choice sets are empty lists, with no inferred compatibility.

Host GPU enumeration remains complete. When `gpu_indexes` is supplied, the
probe container uses Docker's quoted `"device=<indexes>"` selector. It publishes no
host port.

Host enumeration requests NVIDIA's documented `compute_cap` query field. If
that field is unavailable, a second query omits it, preserves GPU UUIDs, and
records compute capability as unknown with a warning instead of failing the
snapshot. Ordinary commands retain a 30-second timeout; the container probe
uses 180 seconds so image imports and embedded SGLang help inspection can
finish.

Verification used
`python3 -B -m unittest s1slow.Automation.tests.test_capabilities`: 18 tests
ran with zero failures. The embedded `_CONTAINER_PROBE` also compiled under
Python 3.11. Tests used a fake command runner, so development made no Docker,
network, or GPU probes.

An earlier broad discovery run reached unrelated concurrent-edit failures in
adapter, planner, runtime-contract, and self-contained tests. The parent agent
will run the integrated suite after the parallel edits settle.
