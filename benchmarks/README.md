# Vendored benchmark runtime

This directory is the runtime payload used by `Automation`. It is mounted
read-only at `/opt/s1slow/benchmarks`; task state and results are written under
`/run/results`.

The files are maintained as part of the Automation bundle, so a copied
`Automation` directory has no runtime dependency on a sibling tree. Keep changes local to this directory,
or explicitly resync and run both Automation and benchmark regression tests.
Only runtime scripts, profiles, and configuration are included; host wrappers,
tests, caches, and historical documentation are intentionally excluded.
