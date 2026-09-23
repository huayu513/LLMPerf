# Vendored benchmark runtime

This directory is the runtime payload used by `LLMPerf`. It is mounted
read-only at `/opt/s1slow/benchmarks`; task state and results are written under
`/run/results`.

The files are maintained as part of the LLMPerf bundle, so a copied
`LLMPerf` directory has no runtime dependency on a sibling tree. Keep changes local to this directory,
or explicitly resync and run both LLMPerf and benchmark regression tests.
Only runtime scripts, profiles, and configuration are included; host wrappers,
tests, caches, and historical documentation are intentionally excluded.
