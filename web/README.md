# Automation Web Console

A local, single-server debugging console for \`Automation\`.

Start it with Python 3 from the \`Automation\` directory or repository root:

    python3 -m web.server --host 0.0.0.0 --port 18080

or:

    python3 Automation/web/server.py --host 0.0.0.0 --port 18080

The server uses only the Python standard library and serves the static frontend
from \`Automation/web/static\`. The default result root is
\`/data/hjh/Automation/results\`; override it in the UI or with
\`AUTOMATION_RESULT_ROOT\`.

## API sketch

- \`GET /api/settings\`
- \`GET /api/runs?result_root=/data/hjh/Automation/results\`
- \`GET /api/runs/{run_id}?result_root=...\`
- \`GET /api/runs/{run_id}/candidates?result_root=...\`
- \`GET /api/runs/{run_id}/candidates/{candidate_id}?result_root=...\`
- \`GET /api/runs/{run_id}/trials?result_root=...\`
- \`GET /api/runs/{run_id}/artifact?result_root=...&path=trials/.../server.log\`
- \`POST /api/jobs/plan\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/jobs/auto\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/runs/{run_id}/run?result_root=...\`
- \`POST /api/runs/{run_id}/resume?result_root=...\`
- \`POST /api/runs/{run_id}/collect?result_root=...\`
- \`POST /api/runs/{run_id}/adopt-runtime?result_root=...\`
- \`POST /api/runs/{run_id}/debug-rerun?result_root=...\`
- \`POST /api/runs/{run_id}/repair-trial?result_root=...\`

\`adopt-runtime\` updates \`plan.json.metadata.bundle_fingerprint\` and
\`search-state.json.plan_hash\` after backing both files up under
\`runtime-adoptions/<timestamp>-<id>/\`. Use it when Automation code has been
updated and you intentionally want to continue the old run with the current
runtime.

\`debug-rerun\` reuses one candidate from the saved \`plan.json\`, runs it with
the current Automation code, and records output under \`debug-trials/\` so the
official search state and \`best.json\` remain unchanged.

\`repair-trial\` reuses one official trial's original task parameters, appends
the next \`attempt-N\` under \`trials/\`, updates \`search-state.json\`, and
refreshes \`results-index.json\`. It is intended for interrupted or unstable
service attempts that should count in later strict resume runs.
