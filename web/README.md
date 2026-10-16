# LLMPerf Web Console

## 自适应反馈调优

无需配置 `search.strategy`，直接通过按钮选择模式：

- “生成 Auto 计划”使用原有 `strict` 规则搜索，不调用大模型。
- “生成智能调优计划”明确使用 `adaptive`，大模型参与规划，并在“开始搜索”后的试验间隙分析实测结果、选择下一批候选；即使配置写了 `strict`，该按钮仍选择 `adaptive`。
- 两个入口都先生成计划，选中 run 后点击“开始搜索”才进行实测。已有 Strict 计划不会自动切换，需要重新生成智能调优计划。

计划预览显示候选池和探索/复验配额；运行概览显示搜索策略、大模型参与环节、顾问调用配额、反馈假设、实际选择和规则降级原因。AUTO 不接入 Adaptive，同一份配置无需在两种入口之间反复修改策略。

场景可写“并发48，时延优先，其他默认”，也可写“全部默认”；仅明确要求有歧义时才追问。解析后显示生效参数和来源。默认输入、输出保留原始请求；未提供输出上限时不新增 `max_tokens`。缺省并发优先用 `search.start_concurrency`，否则按配置 GPU 数估算且受 `search.concurrency_max` 限制；缺省目标为吞吐优先。

“智能调优负载”默认选择原始 JSONL 回放，要求配置 `input_path`；选择合成负载时必须明确输入和输出上限。生成计划后在概览核对负载来源、输入/输出策略、6666 秒业务请求总超时，以及顾问等待和思考设置。当前配置启用顾问最大思考并等待最多 2000 秒，反馈还受剩余探索预算约束；是否支持 `max` 由接口决定。已有计划不自动更新，服务更新后请重启 Web、刷新页面并重新生成计划。

顾问在本轮第 0、500、1000、1500 秒错峰追加相同请求，旧请求继续等待，最多四次；首个通过任务校验的完整结果胜出，其他结果不再应用。整轮最多 2000 秒，较短的配置超时和剩余探索预算优先；不会给最后一次额外增加 2000 秒。顾问全失败或超时后仍降级规则选点。反馈顾问配额按轮次计，每轮实际请求数单独记录；“反馈决策”显示发送次数和采用序号，`request_attempts` 提供完整时间与状态。取消连接不保证服务端停止生成，需考虑额外推理负载。代码更新后请重新生成计划，旧计划的指纹与已接受决策规则不变。

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
- \`POST /api/scenario/parse\` with \`{ "config_path": "configs/experiment.json", "description": "并发48，时延优先，其他默认" }\`
- \`POST /api/jobs/tune-plan\` with \`{ "config_path": "configs/experiment.json" }\`
- \`POST /api/runs/{run_id}/run?result_root=...\`
- \`POST /api/runs/{run_id}/resume?result_root=...\`
- \`POST /api/runs/{run_id}/collect?result_root=...\`
- \`POST /api/runs/{run_id}/adopt-runtime?result_root=...\`
- \`POST /api/runs/{run_id}/debug-rerun?result_root=...\`
- \`POST /api/runs/{run_id}/debug-search?result_root=...\`
- \`POST /api/runs/{run_id}/repair-trial?result_root=...\`

控制台中的“生成 Auto 计划”和“生成智能调优计划”都只保存待审阅的计划；选中 run 后点击同一个“开始搜索”才调用 \`/api/runs/{run_id}/run\` 实测。\`/api/jobs/auto\` 保留为旧的一步式接口，不在控制台主流程中使用。

`POST /api/jobs/tune-plan` 的 `use_captured` 缺省为 `true`，显式 `false` 才使用合成负载；只接受布尔值，不接受字符串。场景解析成功时返回 `scenario`、字段 `sources` 和业务 `request_timeout`。

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
