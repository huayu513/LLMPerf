**LLMPerf 统计与输出正确性复查（2026-10-29）**

结论：正常响应的计数、汇总和吞吐率计算在本次可控测试中通过；当前成功判定存在漏洞，不能用 `ok=true`、`VALID` 或协议探测 `PASS` 证明每条响应完整、内容正确。已有结果未保存响应正文，无法据此追溯验证实际回答。

审查版本：`LLMPerf` commit `f137bfa307667112e5763f211e8a10df5c7fe516`。本次复查未修改 LLMPerf 运行代码；报告、复查脚本与精简验证证据归档在本目录。复查使用临时本机 HTTP 服务和明确构造的响应，没有访问生产服务，也没有测试真实模型。下游校验所需的服务状态及启动协议证据为已知正常的模拟数据，用于隔离验证回放响应问题能否被结果校验拦截。

**高优先级：异常响应可以被计为成功，并通过最终校验。**

正式回放的 `ok` 只要求 HTTP 状态为 200 且未捕获异常。`stream_done` 用于退出读取循环，却没有参与成功判断；也没有要求 `finish_reason` 或可观察输出。SSE 中无法解析的 JSON 被静默跳过，响应对象里的 `error` 没有被检查。

本机 HTTP 实测以下情况均得到 CLI 退出码 0、`ok=true`，并通过 `read_attempt` 得到 `VALID`：

- 有正文和 usage，但没有 `[DONE]`；
- 只有部分正文和 usage，既无 `finish_reason` 也无 `[DONE]`；
- 只有正数 usage，没有任何输出；
- 损坏的 JSON 事件之后附带有效 usage；
- 明确的 API error 事件之后附带有效 usage。

这些测试中的 HTTP 传输正常关闭，问题是应用层响应不完整或无效；不能由此推断所有网络断开都会被漏检。HTTP 500 被正确判失败，缺少 usage 的请求也会使整轮退出失败。启动前协议检查不能覆盖后续业务请求。

代码：[成功条件](../../../benchmarks/replay/replay_jsonl_sglang.py#L416)、[JSON 解析失败被忽略](../../../benchmarks/replay/replay_jsonl_sglang.py#L223)、[流结束处理](../../../benchmarks/replay/replay_jsonl_sglang.py#L362)。

**中优先级：非流式 TTFT/TPOT 的含义不成立。**

非流式响应在完整 JSON 到达后才记录一次输出活动，因此首末活动时间相同。实测 `normal_nonstream` 的 `tpot_seconds=0`，TTFT 接近完整响应耗时。这无法反映模型首次生成或逐 token 生成速度，应将非流式 TTFT/TPOT 标记为不可观测。输出只有一个 token 时也没有可计算的 token 间隔，当前实现仍给 0。

对于流式响应，指标基于客户端收到内容事件的时间，且 reasoning 或工具调用也算首次活动；它不等于用户可见正文的首 token 时间。多个事件合并到一次网络读取时，TPOT 可很小；这是观测口径和传输缓冲的限制，不宜视为模型内部生成耗时。

代码：[非流式响应观测](../../../benchmarks/replay/replay_jsonl_sglang.py#L381)、[指标计算](../../../benchmarks/replay/replay_jsonl_sglang.py#L425)。

**中优先级：usage 校验不够严格。**

服务端的 token 数被直接用于计分，没有独立核对。`numeric_token` 接受 float 并调用 `int()`，实测 `completion_tokens=3.9` 被静默记为 3；`prompt_tokens=5`、`completion_tokens=3`、`total_tokens=999` 也通过最终校验。

下游会核对请求数量、usage 是否缺失、输出 token 总数是否为正，以及吞吐率是否等于输出 token 总数除以测量时长。这能保证部分内部一致性，但不能证明服务端的 token 数正确，也不能证明正文完整。

代码：[token 转换](../../../benchmarks/replay/replay_jsonl_sglang.py#L250)、[下游 usage 与吞吐校验](../../../llmperf/adapters.py#L1070)。

**功能缺口：没有验证实际回答的业务正确性。**

问题为 `3 + 4`，可控服务返回 `8`，正式回放仍得到 `VALID`；单独将这个错误答案输入启动协议探测，结果同样为 `PASS`。协议探测原本声明只检查协议，不检查模型质量，所以这里说明的是验证范围，不是将协议通过等同于答案正确。

正式回放只统计工具调用片段，不组装并校验工具参数。测试中 `finish_reason=length`、工具参数只有 `{"value":` 的响应也得到 `VALID`。达到输出上限本身可以是正常性能测试结果，但应单独记录“被截断”，不能据此证明业务输出可用。

由于没有保存正式响应内容，当前落盘的字符数、usage、耗时不足以区分正确答案、乱码、错误答案或不完整工具参数。

代码：[协议探测内容与规则](../../../benchmarks/server/protocol_check.py#L146)、[正式回放对输出只做计数](../../../benchmarks/replay/replay_jsonl_sglang.py#L256)。

**本次通过的核对。**

- 正常流式、中文 UTF-8 按字节分片、多个事件合并：正文字符数与 usage 被正确读取。
- 固定时钟验证：总耗时 0.8 秒、TTFT 0.2 秒、TPOT 0.1 秒、调度延迟 0.1 秒均符合当前公式。
- 闭环与开环各测试 5 条正式请求、2 条预热请求。服务端各收到 7 条，结果文件各只有 5 条正式记录，无重复或遗漏，输出 token 总数均为 15；预热没有混入正式计数。
- 输出吞吐率等于正式输出 token 总数除以测量时长；逐请求、汇总和端点汇总的计数一致。
- 输入 SHA-256 不匹配会被拒绝；测试服务收到的请求与源请求一致。
- 仓库现有相关测试通过 64 项：`test_protocol_compatibility.py` 18 项、`test_adapters.py` 39 项、`test_server_evidence_integration.py` 7 项。

以上覆盖 15 个响应场景、2 个批量调度场景，以及固定时钟和输入哈希检查。测试环境为临时虚拟环境中的 Python 3.12.14、aiohttp 3.14.5；部署镜像版本未在本次复查中验证。

**建议的修复和验证顺序。**

1. 先补正式请求的协议完整性校验：解析错误、API error、终止状态、每个 choice 的结束状态和 usage 合法性。对缺失或无效证据明确标记，不直接计入有效成绩；保留各后端协议差异的明确处理。
2. 修正指标口径：非流式 TTFT/TPOT 置空；区分首次 reasoning、正文及工具调用；无法观测 token 间隔时明确标记。
3. 逐请求保留响应状态、结束原因、usage 和校验结论；保存可配置数量的完整请求/响应样本及异常响应，关联 request ID、trial、candidate。响应哈希只能证明一致性，不能替代可阅读的样本。
4. 使用相同 tokenizer/模板与可获得的服务端 token ID 交叉核对计数；不能仅对 content 重新分词就断言等于 completion_tokens，reasoning、工具编码及特殊 token 也需要对齐。
5. 增加独立质量验收：有标准答案的请求检查答案；工具调用检查 schema 和参数；开放式任务用人工抽样或任务评测。明确区分性能有效、协议完整和业务正确三个结论。

验证证据：[机器可读结果](audit-results.json)、[固定时钟核对](deterministic-metric-check.json)、[错误答案的协议探测结果](wrong-answer-protocol-check.json)、[复查脚本](audit_replay.py)。所有响应及服务状态证据均为审查用合成数据。

从 LLMPerf 仓库根目录运行以下命令，可重新生成完整输入、响应、逐条指标、汇总、模拟服务状态证据和下游校验结果。输出目录必须尚不存在；网络只访问脚本启动的临时本机 HTTP 服务。耗时数值会随机器和调度变化，归档结果记录的是本次测量，不应当作性能基准。

```sh
python3 -m venv /tmp/llmperf-response-audit-venv
/tmp/llmperf-response-audit-venv/bin/python -m pip install aiohttp==3.14.5
audit_output="$(mktemp -d)/results"
/tmp/llmperf-response-audit-venv/bin/python docs/reviews/2026-10-29-response-statistics/audit_replay.py . "$audit_output"
```

脚本用于记录当前行为，不会将本报告中的已知缺陷全部转成断言失败；脚本退出成功表示复查流程及正常路径断言完成，仍需检查各异常场景的 `downstream_status`。本次归档中的 `VALID` 记录正是已确认的缺口，不能视为正确性验收通过。
