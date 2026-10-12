# LLMPerf Web Console

## 自适应反馈调优

无需配置 `search.strategy`，直接通过按钮选择模式：

- “生成 Auto 计划”使用原有 `strict` 规则搜索，不调用大模型。
- “生成智能调优计划”明确使用 `adaptive`，大模型参与规划，并在“开始搜索”后的试验间隙分析实测结果、选择下一批候选；即使配置写了 `strict`，该按钮仍选择 `adaptive`。
- 两个入口都先生成计划，选中 run 后点击“开始搜索”才进行实测。已有 Strict 计划不会自动切换，需要重新生成智能调优计划。

计划预览显示候选池和探索/复验配额；运行概览显示搜索策略、大模型参与环节、顾问调用配额、反馈假设、实际选择和规则降级原因。AUTO 不接入 Adaptive，同一份配置无需在两种入口之间反复修改策略。

`GET /api/runs/{run_id}` 的 `summary.strategy` 标记策略，`adaptive_decisions` 提供紧凑决策列表，不重复传输完整顾问输入。决策原文保存在 `decisions/`，与试验证据一起校验；“严格 Resume”在 Adaptive 中也会复用已接受决策，不代表切回 Strict。认证不可用时降级规则，计划不会保存令牌。

A local, single-server debugging console for \`LLMPerf\`.

Start it with Python 3 from the \`LLMPerf\` directory or repository root:

    python3 -m web.server --host 0.0.0.0 --port 18080

or:

    python3 web/server.py --host 0.0.0.0 --port 18080

The server uses only the Python standard library and serves the static frontend
from \`web/static\`. The default result root is
\`/data/hjh/LLMPerf/results\`; override it in the UI or with
\`LLMPERF_RESULT_ROOT\`.

## API sketch

- \`GET /api/settings\`
- \`GET /api/runs?result_root=/data/hjh/LLMPerf/results\`
- \`GET /api/runs/{run_id}?result_root=...\`
- \`GET /api/runs/{run_id}/candidates?result_root=...\`
- \`GET /api/runs/{run_id}/candidates/{candidate_id}?result_root=...\`
- \`GET /api/runs/{run_id}/trials?result_root=...\`
- \`GET /api/runs/{run_id}/artifact?result_root=...&path=trials/.../server.log\`
- \`POST /api/jobs/plan\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/jobs/auto\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/scenario/parse\` with \`{ "config_path": "configs/experiment.json", "description": "输入最多 8K、输出最多 2K、并发 32、时延优先" }\`
- \`POST /api/jobs/tune-plan\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/runs/{run_id}/run?result_root=...\`
- \`POST /api/runs/{run_id}/resume?result_root=...\`
- \`POST /api/runs/{run_id}/collect?result_root=...\`
- \`POST /api/runs/{run_id}/adopt-runtime?result_root=...\`
- \`POST /api/runs/{run_id}/debug-rerun?result_root=...\`
- \`POST /api/runs/{run_id}/debug-search?result_root=...\`
- \`POST /api/runs/{run_id}/repair-trial?result_root=...\`

控制台中的“生成 Auto 计划”和“生成智能调优计划”都只保存待审阅的计划；选中 run 后点击同一个“开始搜索”才调用 \`/api/runs/{run_id}/run\` 实测。\`/api/jobs/auto\` 保留为旧的一步式接口，不在控制台主流程中使用。

\`adopt-runtime\` updates \`plan.json.metadata.bundle_fingerprint\` and
\`search-state.json.plan_hash\` after backing both files up under
\`runtime-adoptions/<timestamp>-<id>/\`. Use it when LLMPerf code has been
updated and you intentionally want to continue the old run with the current
runtime.

\`debug-rerun\` reuses one candidate from the saved \`plan.json\`, runs it with
the current LLMPerf code, and records output under \`debug-trials/\` so the
official search state and \`best.json\` remain unchanged.

\`debug-search\` runs one candidate through the current debug search flow:
start at the configured/default concurrency, add 16 while output throughput
improves by more than 1%, and store each point under \`debug-trials/\`.

\`repair-trial\` reuses one official trial's original task parameters, appends
the next \`attempt-N\` under \`trials/\`, updates \`search-state.json\`, and
refreshes \`results-index.json\`. It is intended for interrupted or unstable
service attempts that should count in later strict resume runs.
