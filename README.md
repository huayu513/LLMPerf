# LLMPerf 自动吞吐压测

只维护 `configs/experiment.json`，用一条命令完成模型识别、环境检查、输入索引、参数搜索、重复验证和结果收集。`LLMPerf/` 可单独复制到目标机器，不依赖旁边的 sglang 或其他 benchmarks 目录。

## 场景驱动智能调优（新模式）

`auto` 保留不调用大模型的规则搜索入口，镜像能力探测和原生功能对照与智能调优同步升级。智能调优用同一个 `experiment.json` 中的模型、镜像、GPU 和搜索预算，默认回放 `input_path` 的原始 JSONL；只有显式选择合成负载时才可以省略 `input_path`。管理员先在其中配置隔离区的 OpenAI 兼容大模型接口：

Web 按入口选择模式，不需要业务人员配置 `strategy`：

| 入口 | 搜索策略 | 大模型参与 |
| --- | --- | --- |
| 生成 Auto 计划 → 开始搜索 | `strict`，原有规则流程 | 不调用 |
| 生成智能调优计划 → 开始搜索 | `adaptive`，实测反馈驱动 | 场景解析、镜像能力分析、规划推荐、搜索过程中的结果分析与下一批选点 |

两个入口可以使用同一份配置；Web 智能调优明确选择 `adaptive`，AUTO 明确选择 `strict`。运行已保存计划时，以计划中的策略为准，不因当前页面或配置改变而切换。

生成新计划时自动读取 `model_path` 下的配置、权重索引与 safetensors 张量头，识别存储规模、dtype 和量化证据，无需手填参数量，不加载模型或占用 GPU。按每个模型副本的容量淘汰已证明放不下的基础拓扑（8 卡 8 实例不会合并 8 张卡显存）；量化和加载证据不足时保持未知，不误杀。智能调优把摘要一并交给能力顾问分析，AUTO 只做规则筛选。前端“镜像原生能力 → 模型权重自动识别”和结果目录的 `model-weights.json`、`weight-capacity.json` 可查看依据；详见 `docs/image-native-tuning.md`。

```json
"advisor": {
  "base_url": "http://10.64.39.63:54321/v1/chat/completions",
  "model": "DeepSeek-V4-Flash-Local",
  "api_key_env": "API_KEY",
  "timeout_seconds": 2000,
  "thinking": true,
  "reasoning_effort": "max",
  "task_max_tokens": {
    "scene": 100000,
    "capabilities": 100000,
    "candidates": 100000,
    "explain": 100000,
    "next_experiments": 100000
  }
}
```

启动 CLI 或 Web 服务前执行 `export API_KEY='接口令牌'`，环境变量只放令牌，不带 `Bearer ` 前缀。顾问请求会发送 `Authorization: Bearer <令牌>` 和 `Content-Type: application/json`；只有配置了非空 `model_pool` 才发送 `X-Model-Pool`。也兼容配置中的 `authorization`，该字段优先于环境变量。

请求体带 `temperature: 0` 和 `stream: false`。上述配置开启顾问思考，并在顶层及 `chat_template_kwargs` 中发送 `reasoning_effort: "max"`；镜像或接口需支持该取值，是否实际生效以服务端为准。未配置 `thinking` 时仍关闭思考。上述配置将所有顾问任务的输出上限设为 100000，未覆盖时场景解析、初始推荐和解释默认 64，Adaptive 反馈选点默认 1024。这些是顾问参数，不会改写被测业务请求的思考与输出设置。场景解析、规划和反馈默认等待 2000 秒；反馈同时受更短的配置超时及剩余探索预算限制。业务请求总超时默认 `request_timeout: 6666` 秒，服务启动等待 `ready_timeout` 不变。旧计划保留冻结设置，修改后需重新生成计划。业务人员可以在 Web 输入自然语言并点“解析并写入场景”，或执行：

```bash
python3 benchctl.py scenario --config configs/experiment.json \
  --describe "并发48，时延优先，其他默认"
python3 benchctl.py tune-plan --config configs/experiment.json
```

顾问采用错峰并发请求：本轮第 0、500、1000、1500 秒各发送一次相同任务，最多四次；旧请求继续等待，首个完整且通过任务校验的结果胜出。HTTP 错误、非法 JSON 和无效场景/候选不能抢占有效结果。收到有效结果后停止追加、尽力关闭其余连接，晚到结果不再应用；服务端不一定随连接关闭停止生成，因此最多可能产生四份推理负载。整轮按单调时钟计时，最多 2000 秒，较短的配置超时或剩余探索预算会缩短窗口，不会为凑齐四次而延长。若所有可发送请求已明确失败，则提前记录失败；仅到截止时间仍未完成时记录超时。运行期无有效结果仍按原逻辑降级规则选点，不将未运行候选判为失败。

`search.adaptive.max_advisor_calls` 继续限制反馈顾问轮次，一轮可能包含最多四次实际 HTTP 请求，并非只消耗一次模型推理。标准错误日志输出 `advisor_request_started`、`advisor_request_finished`、`advisor_call_finished`，记录发送时间、各次状态、实际请求数和采用序号，不输出认证或业务请求正文。规划记录和反馈决策的 `request_attempts` 保存同样的审计信息；Web 反馈决策显示实际请求次数和采用第几次返回。已接受的旧决策在 Resume 时仍复用，不会重新请求。

镜像能力分析是候选推荐前的独立阶段，SGLang 与 vLLM 同时支持，最多 3 个逻辑轮次，共享最多 2000 秒（配置更短时使用更短值），不占 `search.adaptive.max_advisor_calls` 的反馈配额。该阶段沿用顾问思考设置；输出预算优先使用 `advisor.task_max_tokens.capabilities`，未配置时沿用 `candidates`，二者都未配置时为 8192。顾问只能补读已经采集的只读证据 ID，不能执行任意命令。完整流程、能力边界与产物见 [镜像原生能力驱动调优](docs/image-native-tuning.md)。

能力顾问的 TP/DP/PP 建议通过既有合法拓扑候选选择生效，不直接改写 GPU 分配。原生解析预检先不暴露 GPU，只有实际设备识别异常才补充选定 GPU 上下文重试，不加载权重或启动服务；全未知明确显示未知，不冒充解析完成。Web 的“顾问建议落地”和“原生解析预检诊断”分别展示建议最终匹配的候选、设备上下文与失败原因。

模型只提取明确指定的字段；省略或“其他默认”的字段由平台确定，不追问缺项，只追问明确要求中的歧义。默认输入保留原始 `messages`，不截断、不填充；默认输出保留原请求的 `max_tokens` / `max_completion_tokens`，原请求没有上限就不新增，沿用服务端默认值和上下文限制。场景快照中 `max_input_tokens` / `max_output_tokens: null` 表示不添加场景上限。默认目标为 `throughput`；默认并发优先使用 `search.start_concurrency`，否则按配置 GPU 数每卡 8 并发估算（未指定 GPU 时按一张卡保守估算），且不超过 `search.concurrency_max`。缺少 `input_path` 会报错，不静默切换合成负载。

明确指定输出上限时，仅在发送请求时收紧原有生成限制；原上限更小时保留原值，没有上限时增加 `max_tokens`。源 JSONL 不修改，结果记录原始和实际发送请求的哈希及输出限制。明确指定输入上限只核验实际 usage，不自动截断或补齐。实际输入/输出超过指定上限或 usage 缺失的试验不能参与选优。

如确需合成负载，明确指定输入、输出两个长度上限，再执行 `tune-plan --use-synthetic`；旧的 `--use-captured` 仍可使用，但已是默认行为。合成负载仍采用输入与输出三个档位的九种组合，并尝试用目标镜像中的 tokenizer 校准，失败时保留近似并标明。原始回放不进行合成或校准。

平台按模型、GPU 和镜像能力建立参数目录。智能调优默认使用 `adaptive`：候选池写入不可变的 `plan.json`，本地大模型在生成计划时推荐候选，并在搜索过程中分析实测结果、选择下一批候选。初始推荐与运行期选点均向顾问传递引擎类型，提示词区分 SGLang 与 vLLM。候选必须经过参数白名单、取值和现有拓扑约束，模型超时、输出无效或重复时由规则候选补足。规划不启动模型服务进行压测。所有候选在指定业务并发下使用同一固定负载；探索可使用同一小样本子集，最终复验使用完整请求集。吞吐优先按输出 token/s，时延优先按请求 P95 时延选优。没有明确 SLO 时不会凭空设时延门槛。报告保存实际生效配置、长度 usage、复现脚本和试验轨迹。

Web 页面在场景信息不足时保留前文，可直接输入补充信息继续解析；“重新描述”会清空这段对话。

检查计划后，在 Web 页面选中该 run 点击“开始搜索”，或执行：

```bash
python3 benchctl.py run --plan /path/to/result-dir/plan.json
```

中断后点击“严格 Resume”，或执行 `python3 benchctl.py run --plan /path/to/result-dir/plan.json --resume`。Strict 执行不调用顾问，也不依赖原配置文件；Adaptive 可调用顾问，但只使用计划冻结的搜索设置，原配置至多用于读取认证字段。两种模式都不在运行中新增候选。运行前核对模型、固定请求集、GPU、镜像和运行时代码。规划阶段保存 `plan.json`、`plan-lock.json`、`catalog.json` 和 `scenario.json`；开始搜索后才生成 `tune-state.json`、trial 与 `best.json`。运行现有吞吐 `auto` 时仍必须提供 `input_path`；新计划增加通用权重容量筛选，吞吐评分沿用原有逻辑。

### Adaptive：实测反馈驱动选点

第一阶段仅接入 `tune-plan → run`，不接管原来的 `auto`。Web 智能调优自动启用 Adaptive；CLI 的 `tune-plan` 在未指定策略时也默认 Adaptive，无需修改配置。需要调整预算时，可在原 `experiment.json` 的 `search` 中合并以下可选设置：

```json
"search": {
  "max_trials": 64,
  "repetitions": 3,
  "adaptive": {
    "candidate_pool_limit": 256,
    "batch_size": 3,
    "max_advisor_calls": 12,
    "final_top_k": 2
  }
}
```

命令仍是 `tune-plan --config ...`，审查计划后执行 `run --plan ...`。CLI 保留显式 `search.strategy` 配置供对比实验，并可通过 `tune-plan --strategy strict` 或 `--strategy adaptive` 覆盖。Strict 智能调优只在规划时调用顾问，候选与顺序固定。Web 的智能调优按钮始终传入 `--strategy adaptive`，AUTO 始终使用原有 Strict 规则流程，即使共用配置中写了 `adaptive` 也不会调用顾问。旧 Strict 计划不会自动升级；需要反馈调优时请重新生成智能调优计划。

- 计划冻结候选池、请求集、业务并发、目标、策略版本和预算，不冻结执行顺序。候选池按基础拓扑与参数族分层生成，并去掉可确定的每实例请求上限等价配置；池大小可以大于实际试验预算。仍只调现有三个参数，不开放任意命令或新增启动参数。
- 先测基线及代表性配置，每批默认三次探索后调用反馈顾问；首次出现新的失败类型可提前结束批次，进行诊断选点。顾问接收环境摘要、吞吐/P95/TTFT/TPOT、实际配置差异、实际长度、脱敏日志尾部和剩余预算，在最多 24 个合法未测候选的 shortlist 中选择下一批。诊断是待验证假设，不改变结果有效性或 backend 封禁规则。
- 超时、凭据缺失、输出越界或重复、调用配额耗尽时，确定性规则接管选点。`max_advisor_calls: 0` 关闭运行期顾问，便于比较规则自适应；它不关闭规划阶段的初始推荐。每批最多八个候选，`final_top_k` 最多五个。
- 规划预留 `(final_top_k + 1) × repetitions` 次试验，分别用于近优候选和基线复验，另外至少允许两次探索。相同候选不重复复验。探索根据已观测耗时估算并预留复验时间；顾问等待计入总时间，运行期单次请求最多等待 300 秒或更短的配置超时、剩余探索时间。原生 Docker trial 到总截止时间会终止并定向清理，容器清理可能产生少量额外耗时。
- 最终赢家只由完整负载复验决定。只有基线也通过复验，`best.json.comparison.relative_improvement` 才给出相对改善；否则为 `null`。所有失败和中断分配都占试验配额，恢复不会无限免费重试。

Adaptive 的 `plan.json` 不保存令牌。运行期优先使用 `advisor.api_key_env` 指定的环境变量，未指定时尝试 `LLMPERF_ADVISOR_KEY`；兼容配置中的 `authorization`，只读取该字段，不重新加载场景、候选或预算。认证不可用时降级规则，不阻止压测。运行期端点、模型和其他请求设置使用计划中的不可变快照。请避免顾问服务和被测服务共享 GPU；顾问只在试验间隙调用，但共享显存或后台 GPU 活动仍可能干扰结果。

每次决策在启动下一批前写入 `decisions/000001.json` 等文件，保存脱敏输入、测量证据哈希、模型响应、校验结果、最终候选和来源。`tune-state.json` 保存配额与待执行批次。Resume 复用已接受决策；调用中断且没有有效决策时记录规则降级，不重复消耗同一模型调用。试验、决策、原始 JSON/JSONL/复现脚本及最终报告被改动后拒绝恢复。Web 概览显示策略、调用配额、选点假设和降级原因。

这一阶段没有加入在线 GPU/KV 指标采集或贝叶斯优化。业务验证默认使用真实请求回放；比较不同策略时固定候选池、请求集和预算，不能把扩大候选池的收益直接当作模型收益。

## 首次部署与启动

机器需要 Python 3.11 或更新版本、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。模型权重、输入 JSONL 和运行镜像放在目标机器上。默认使用 SGLang，镜像需要提供可用的 `sglang serve` 命令，并包含 PyTorch、aiohttp、bash、curl、setsid；也支持通过 `engine: "vllm"` 使用 vLLM 镜像。程序检查本地镜像，不自动拉取或构建镜像。

`configs/experiment.sglang.json` 保留现有 SGLang 配置，并显式设置 `engine: "sglang"`；`configs/experiment.vllm.json` 提供 vLLM 示例。按需修改后，可完整复制到 Web 默认使用的 `configs/experiment.json`，也可在 CLI 中通过 `--config` 直接指定。SGLang 配置中的既有镜像不代表已验证支持 A800 上的 DeepSeek-V4-Flash-0731 原生权重。

vLLM 示例已包含 `advisor`，沿用现有顾问端点、模型及思考和回答预算；令牌通过 `api_key_env: "API_KEY"` 读取，不在示例中重复保存。启动 CLI 或 Web 服务前设置 `export API_KEY='接口令牌'`，令牌不带 `Bearer ` 前缀。使用 `tune-plan → run` 或 Web 智能调优入口调用顾问；普通 `auto` 仍不调用顾问。

### vLLM-backport：A800 与 DeepSeek-V4-Flash-0731 原生权重

新增配置示例 `configs/experiment.vllm.json`；现有 `configs/experiment.json` 不变。优先使用提供 SM80 支持的 vLLM-backport 镜像，而不是假设普通上游 vLLM 镜像支持 A800。backport 的 `latest-sm80` 是 SM80 标签，裸 `latest` 不是。压测应固定已验证的版本和实际 digest：

```bash
IMAGE='lazymio/vllm-backport:latest-sm80'
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{range .RepoDigests}}{{println .}}{{end}}'
python3 benchctl.py doctor --config configs/experiment.vllm.json
python3 benchctl.py plan --config configs/experiment.vllm.json
python3 benchctl.py auto --config configs/experiment.vllm.json
```

先替换示例中的镜像摘要、模型、输入和输出路径。`doctor` 只检查环境与 CLI，不证明模型能加载；`auto` 才启动服务并回放。程序直接只读挂载原始 checkpoint，不执行权重转换。示例固定 TP=8、DP=1、PP=1，避免先枚举明显装不下模型的较小实例；删除三个并行度设置可恢复拓扑搜索。示例开启 smoke，正式结果需设置 `smoke: false`。

`engine` 默认为 `sglang`。`engine_options` 目前仅用于 vLLM，拒绝未知字段和任意 shell/额外参数：

- `tensor_parallel_size`、`data_parallel_size`、`pipeline_parallel_size`：可选的候选拓扑约束，不设置时搜索合法组合。
- `gpu_memory_utilization`：默认 0.9；与 SGLang 的静态显存占比不是严格等价概念。
- `max_num_seqs`：默认 `search.concurrency_max`，每个 trial 按单实例分到的并发收紧；不是客户端总并发的同义词。
- `max_num_batched_tokens`：默认 4096，表示迭代 token 预算，不等同于 SGLang 的 chunked-prefill-size。
- `max_model_len`：默认不设置，不传 `--max-model-len`，由引擎读取模型配置；只有显式配置时才覆盖。不会自动改为 `auto`、固定 32K 或按请求长度估算。默认上下文导致显存不足时明确报错，不缩短业务请求。
- `enforce_eager`：默认关闭。`enable_expert_parallel`：基线优先采用镜像解析器公开的布尔默认值，未知时维持关闭；原生功能对照可尝试镜像公开的 EP 开关及通信策略，但不覆盖显式 `engine_options`，也不映射成 SGLang DP Attention。
- `moe_backend`、`all2all_backend`：可选原生参数，要求镜像广告相应 CLI。`search.backends` 在 vLLM 下表示原生 MoE backend，与 `moe_backend` 不能同时配置；默认交给引擎 AUTO，不调用 SGLang 的解析或去重流程。
- `speculative_config`：基线默认不启用；可显式配置 `{"method":"deepseek_mtp","num_speculative_tokens":5}`，需对应 backport 版本支持。未显式限定时，可按镜像实际结构和顾问建议生成原生方法对照；不把 SGLang DSpark 直接映射成 vLLM 投机解码。

DeepSeek-V4 的自动工具/推理 parser 映射为 vLLM 的 `deepseek_v4`。显式 `model_overrides` 的 parser 名称优先。镜像必须支持默认聊天模板 kwargs，原始回放请求不做额外改写；请求自身的模板参数仍由服务端合并处理。其他模型若 parser 不兼容，需指定该镜像支持的原生 parser。

与 SGLang 一样，Strict 和智能调优默认使用原始请求：不要求 `max_input_tokens`、`max_output_tokens`，不截断或补齐输入；请求自带的 `max_tokens` / `max_completion_tokens` 原样保留，没有则不添加。显式场景输出上限仍按既有规则收紧请求。旧计划保留冻结的上下文及输出限制，修改配置后应重新生成计划。

运行镜像需要 Python、PyTorch/CUDA/NCCL、aiohttp、bash 和可用的 `vllm serve --help=all`。探测、校准和压测均显式覆盖镜像 ENTRYPOINT。单实例和多实例共用控制器，保留原始 JSONL、流式 usage、吞吐、P95 和失败判定。智能调优与 Web 命令预览使用 vLLM 原生参数目录，不生成 SGLang 的 DPA/DSpark 候选。

取证使用 `/server_info?config_format=json` 的原生 `vllm_config`，以及 `/v1/models` 的模型别名，逐实例核对模型路径、TP/DP/PP、调度和缓存配置。返回文本 repr、缺失接口、字段不匹配或 usage 缺失均不能作为有效结果；不会拿启动请求伪造实际配置。工具/推理 parser 和模板默认值保存在启动命令中供审计，不宣称这些前端参数已由 `vllm_config` 独立核验。

由于 backport 将 `/server_info` 放在开发接口中，压测进程设置 `VLLM_SERVER_DEV_MODE=1`，同时强制只监听容器内 `127.0.0.1`，不发布压测端口。生成的独立复现脚本关闭开发接口，再通过 Docker 只向宿主机 localhost 发布正常推理端口。不要把开启开发接口的服务改成公开监听。

本地无 GPU 测试覆盖配置、规划、命令、原生参数证据和复现契约；仍需在目标 A800 上确认具体镜像版本的权重加载、正确性和性能。切换引擎或更新代码后应生成新计划，不能跨引擎复用旧 run。

修改唯一的配置文件：

```json
{
  "model_path": "/data/models/your-model",
  "input_path": "/data/workloads/input.jsonl",
  "image": "registry.example.com/sglang@sha256:<64位镜像摘要>",
  "smoke": false
}
```
IMAGE='mirrors.sangfor.com/lmsysorg/sglang:dev-cu13'
docker pull "$IMAGE"
docker image inspect "$IMAGE" \
  --format '{{range .RepoDigests}}{{println .}}{{end}}'


`image` 要填写实际不可变的仓库 digest，示例占位符不可直接运行。命令行直接跑完整流程使用：

```bash
cd /path/to/LLMPerf
python3 benchctl.py auto --config configs/experiment.json
```

路径支持 `~` 和环境变量，相对路径以配置文件所在目录为基准。无需填写模型名称、parser、TP/DP/PP、profile、workload 或 policy 文件。现有旧格式配置和旧 plan 需要重新生成。

## 自动执行的流程

1. 读取 checkpoint 中的 `config.json` 等元数据，识别已适配的 Qwen、GLM MoE、DeepSeek 模型及量化信息。模型目录可以任意命名；共用架构名称的变体结合静态聊天模板识别。未适配或证据不足时给出具体原因及同文件覆盖方式，不执行模型目录里的 Python 代码进行识别。
2. 验证原始 JSONL，统计请求、读取请求使用的服务名称并计算 SHA-256。单次实验要求一个 `request.model` 名称，服务端使用这个名称；请求对象原样回放。
3. 探测 GPU 数量、型号、显存、拓扑及镜像内的 SGLang 选项、parser、后端列表。按同型号/算力/显存的 GPU 分组生成部署拓扑和 TP/DP/PP 候选，过滤注意力头数等静态不兼容组合。若选择 2 张卡，会真实比较 `2卡1实例` 与 `2卡2实例`；若选择 8 张卡，会比较 `8卡1/2/4/8实例`。MIG 暂不支持。计划中的 `comparison_group` 会把同一部署、TP、PP 下可比较的普通 DP 与 DP Attention 方案放在一起；两者的 DP 数可能不同，这是因为 SGLang 开启 DP Attention 后的进程世界大小是 `TP×PP`，关闭时是 `TP×DP×PP`。
4. 对 MoE 候选，在实际配置的 Docker 镜像中使用同一启动脚本生成参数，调用该镜像的 SGLang 配置解析流程。若 `AUTO` 解析出了初始候选中没有的具体 backend，自动补入同拓扑的显式候选并再次解析；只有两者解析后的有效配置相同，才删除重复的 `AUTO`。比较时保留完整解析哈希用于审计，但会排除每次探测都会变化的端口、IPC 名称、NCCL 端口和实例 ID。显式候选解析失败时保留 `AUTO`；解析不可用或结果不确定时也保留它。此阶段不启动服务、不加载权重，结果、补充候选和去重依据写入 `backend-resolution.json` 及 `plan.json`。
5. 默认不做单独 smoke。探索阶段直接启动候选服务并跑有用的小样本请求集。`search.start_concurrency` 和 `search.concurrency_max` 可以明确控制并发起点和终点；起点省略时按所选 GPU 数估算，例如 2 卡默认从 16 开始，8 卡默认从 64 开始。之后每次增加 16 个并发，吞吐增长不足或请求失败时停止该分支，并测试内存比例和 prefill 大小的少量邻近设置。
6. 探索层和全量层都不使用固定 topK：所有与当前 top1 输出 tokens/s 差距在 `promotion_tolerance` 内的候选点都会晋级或复跑。最终层使用完整请求集独立重复验证，按输出 tokens/s 中位数选赢家。只有请求成功、服务端 completion usage 可用、输入摘要一致且实际服务参数可核验的结果才参与排名。

每个试验点启动新容器和新服务。模型、输入、索引及脚本只读挂载；结果单独写入。容器只接收选中的物理 GPU，服务进程使用从 0 开始的逻辑编号。回放通过容器内 loopback 访问服务，不占用宿主机固定端口。启动前会检查挂载的 `.sh`、`.py` 和 `.env` 文件；如果上传过程把 LF 转成了 CRLF，会在本次 run 的结果目录下生成 LF 副本并使用它，不修改原始 LLMPerf 目录。

自动方案的被测服务默认开启 max thinking：`thinking=true`、`enable_thinking=true`、`reasoning_effort=max`，SGLang 与 vLLM 使用相同默认值；`model_overrides.chat_template_kwargs` 可显式覆盖，包括关闭思考。镜像和模型模板需支持 `max`，实际是否生效应在目标服务验证，不会静默降为 `high`。原始请求对象仍原样回放，请求自身的模板参数按引擎合并规则处理，采样参数和输出长度不参与吞吐调优。这些是被测服务设置，与 `advisor` 的思考和回答预算独立。旧计划中已冻结的 `high` 或显式模板配置保持不变，要使用新默认值需重新生成计划。索引在宿主机使用标准库生成，不加载 tokenizer；其中不生成估算 token 统计，吞吐计量使用服务端返回的 usage。

如果候选服务在启动阶段因为参数不支持、参数不匹配或其他配置错误而失败，当前 candidate 会记录为 `FAILED`；只有服务已经就绪、原生参数证据明确指出所请求的显式 MoE runner backend 不受支持时，后续使用同一 backend 的候选才会记录为 `SKIPPED`。未就绪、缺少证据、OOM、模型加载失败和拓扑相关参数不匹配只影响当前 candidate。引擎自动选择的 `auto` 没有可确认的 backend，不会据此误跳过其他候选。回放阶段的请求失败或 OOM 只停止当前 candidate 的并发分支。失败候选和 backend 跳过原因会写入 `search-state.json`、`results-index.json` 和 `best.json`。需要复查时可在前端对单个 candidate 使用“单独重跑这个参数”，该入口不受自动搜索的 backend 跳过状态影响。

输入沿用现有捕获格式，每行是一条请求，例如：

```json
{"api":"openai_chat_completions","request_id":"1","source_message_id":"1","captured_at":"2026-09-09T00:00:00Z","request":{"model":"your-served-alias","messages":[{"role":"user","content":"你好"}],"max_tokens":128}}
```

`request_id` 和 `source_message_id` 各自需要唯一，`captured_at` 需要有效时间戳。`request.model` 是请求原有的服务名称，程序自动把模型服务设为这个名称。

## 可选参数仍放在同一份 JSON

默认值适用于直接启动。只有需要限制设备、时间或调整测试规模时才增加字段：

```json
{
  "model_path": "/data/models/your-model",
  "input_path": "/data/workloads/input.jsonl",
  "image": "registry.example.com/sglang@sha256:<64位镜像摘要>",
  "smoke": false,
  "output_dir": "/data/benchmark-results",
  "gpu_indexes": [0, 1],
  "search": {
    "concurrency_max": 64,
    "start_concurrency": 16,
    "explore_request_limit": 256,
    "promotion_tolerance": 0.05,
    "max_trials": 64,
    "max_seconds": 14400,
    "repetitions": 3
  },
  "warmup": 0,
  "request_timeout": 3600,
  "ready_timeout": 3600
}
```

| 字段 | 省略时的行为 |
| --- | --- |
| `output_dir` | 配置文件旁的 `results/`；每次在其中创建独立时间戳目录 |
| `gpu_indexes` | 使用检测到的物理 GPU，按同型号等条件分组测试；省略时使用全部检测到的 GPU。配置为 `[6, 7]` 就只会按 2 张卡规划 |
| `smoke` | 默认 `false`，不单独启动服务做冒烟；设为 `true` 时每个候选先跑一次 8 条以内的冒烟 |
| `search.concurrency_max` | 并发探索终点，默认 64，实际不超过请求数量 |
| `search.start_concurrency` | 并发探索起点；省略时自动使用 `min(concurrency_max, max(8, 8 * 所选GPU数))`；必须不大于 `concurrency_max`，后续每次增加 16 |
| `search.explore_request_limit` | 探索和调参阶段默认最多回放 256 条；设为 0 表示探索阶段也使用完整请求集 |
| `search.promotion_tolerance` | 默认 0.05；与 top1 吞吐差距 5% 以内的候选点都会进入最终全量复跑 |
| `search.max_trials` | 最多 64 个试验点，包含失败点和最终重复验证；启用 `smoke` 时也包含冒烟 |
| `search.max_seconds` | 搜索阶段 14400 秒；在试验点之间检查，正在执行的点允许完成 |
| `search.repetitions` | 最佳候选完整输入验证 3 次；试验预算至少为该值加 2 |
| `warmup` | 每个点额外预热 0 条；预热不替代完整正式请求集 |
| `request_timeout` | 单请求超时 3600 秒 |
| `ready_timeout` | 服务就绪等待 3600 秒 |

自动规划会生成 `AUTO` 和初始后端候选。镜像若根据权重布局等信息把 `AUTO` 解析为初始计划之外的 backend，规划会自动补入该显式候选，验证等价后再删除重复的 `AUTO`。因此这类 MXFP4 模型最终通常会保留 `flashinfer_mxfp4`、`triton`、`deep_gemm`、`flashinfer_trtllm` 作为可测的显式实现，同时去掉已被确认等价的 `AUTO`；这些显式后端是吞吐搜索维度，不是手动配置项。`search.backends` 仍可限制搜索范围，不会被自动补入的后端突破。`search.open_loop_scales` 可选，例如 `[1, 2, 4]`，在闭环赢家验证之后按原始时间戳进行开环回放；默认不运行开环。开环结果独立记录，不用于替换闭环吞吐赢家。

特殊模型适配可以通过同文件的 `model_overrides` 覆盖已支持的 parser、模板参数或模型专用环境变量，覆盖会记录来源。Docker 的网络、共享内存、IPC 和内部端口也可通过 `docker` 对象覆盖。内部服务端口默认是 `25080`；不要设置到常见 Linux 临时端口范围 `32768-60999`，因为 SGLang 启动时也会从该范围分配内部通信端口。拼错或不支持的字段会报错。

## 结果与恢复

启动后终端首先打印 `result_dir`。目录内包含：

```text
resolved-config.json       # 完整配置、模型/输入事实与参数来源
environment.json          # GPU、镜像和引擎能力快照
data/replay_index.json    # 绑定原始输入的回放索引
backend-resolution.json   # 镜像内配置解析结果及 AUTO 去重依据（MoE）
plan.json                 # 自动生成的候选和搜索范围
search-state.json         # 已分配的试验、预算和停止原因
trials/.../attempt-001/   # 每个点的命令、服务信息、日志、原始结果与 trial.json
results-index.json        # 汇总所有已结束的试验
leaderboard.json          # 当前有效试验按输出 tokens/s 排名
best-so-far.json          # 当前已完成试验中的临时最佳点
best.json                 # 已验证赢家、重复成绩、配置和搜索停止原因
```

`best.json` 的 `best.configuration` 保存服务端实际生效的配置，`planned_configuration` 保留规划值，`server_command_paths` 指向各次验证的容器内 SGLang 启动命令，`server_reproduce_paths` 指向可从宿主机直接执行的 Docker 复现脚本。引擎可能对 DP Attention 等参数做换算，复跑时应优先使用 `server.reproduce.sh`，或进入相同容器环境后使用记录的 `server.command.sh`。只有各次重复验证的实际配置一致才会发布赢家。

`best.json` 的 `best` 为最终重复验证通过的结果；未完成验证时为 `null`，可能另有 `provisional_best` 和 `promoted_finalists`。`PASS` 表示闭环验证和显式请求的开环检查已完成；`PARTIAL` 表示闭环赢家已验证，但开环未完成或失败；没有已验证赢家时为 `INCONCLUSIVE`。

中断后使用原来的计划恢复：

```bash
python3 benchctl.py run --plan /path/to/result-dir/plan.json --resume
```

恢复会核验输入、模型信息、运行脚本、GPU 和镜像身份，并重新检查已完成试验的原始证据。改变配置、模型或硬件后应重新执行 `auto`。模型权重身份使用文件清单、大小和修改时间，并非逐字节完整权重哈希；不要在实验期间更换权重。

需要分步检查时，仍使用同一配置：

```bash
python3 benchctl.py doctor --config configs/experiment.json
python3 benchctl.py prepare --config configs/experiment.json
python3 benchctl.py plan --config configs/experiment.json
python3 benchctl.py run --plan /path/to/result-dir/plan.json
python3 benchctl.py collect --run /path/to/result-dir
```

每个 `doctor` / `prepare` / `plan` 命令独立执行其所需的前置步骤并创建结果目录；日常使用 `auto` 即可。退出码：成功为 0，未完成验证或运行错误为 1，配置错误为 2，中断为 130。

## Profiles 与搜索范围

自动流程只使用 `benchmarks/server/profiles/AUTO.sh` 接收生成的参数。F/O/Qwen/GLM profiles 保留用于手工参考，不自动遍历。旧的 models/workloads/policies JSON、设备专用 env、旧计划 schemas、重复入口和未使用的旧调度校验代码已移除。

“最佳”是当前候选集合和预算内实测最好的结果，不是全局最优保证。部署拓扑会作为搜索维度记录并实测，但目前不穷举所有 SGLang 启动参数组合；复杂 MoE 通信后端和更大的调参空间由配置、workload 证据和日志证据逐步纳入。受预算限制未筛选的候选数量与失败原因会保留。若预算在最终重复验证完成前耗尽，不会发布已验证赢家。

## 前端调试控制台

如果要先看计划、逐个检查候选参数、查看启动日志，推荐启动本地 Web 控制台：

```bash
cd /data/hjh/LLMPerf
python3 -m web.server --host 0.0.0.0 --port 18080
```

浏览器打开 `http://服务器IP:18080/`。页面默认扫描 `/data/hjh/LLMPerf/results`，也可以在页面顶部修改结果目录。配置文件默认指向 `configs/experiment.json`，也可以改成其他 JSON。

从 0 开始调试推荐按这个顺序操作：

1. 修改 `configs/experiment.json`，至少确认 `model_path`、`input_path`、`image`、`gpu_indexes` 和 `output_dir`。
2. 在前端点击“生成计划”。这一步执行 `python3 benchctl.py plan --config ...`，只创建 result 目录和 `plan.json`，不会开始压测。
3. 在“运行概览 → 镜像原生能力”查看顾问分析、独立调用轮次、未知项和解析预检，并打开证据 JSON。在“候选规划”检查拓扑和引擎原生参数：SGLang 显示 Attention DP、MoE runner/A2A 与投机参数；vLLM 显示 Expert Parallel、MoE/All2All 与 `speculative_config`，不显示虚假的 DPA/DSpark 关闭项。双击 candidate 查看对应引擎的启动参数预览。
4. 确认计划后点击“开始搜索”。这一步执行 `python3 benchctl.py run --plan <result-dir>/plan.json`。
5. 运行过程中可以在 Jobs 页看当前后台命令、stdout 事件和最新输出；在 Trials 页看每个 trial 的状态、吞吐和失败原因；点进 trial 或 candidate 的 artifact 可以看 `server.log`、`docker.log`、`server.command.sh`、`server.reproduce.sh`、`server.evidence.json`、`server.info.json` 和 `*.summary.json`。这些文件位于 replay 输出目录（例如 `trials/<task>/attempt-001/formal/AUTO/.../run_001/`），不一定直接位于 `attempt-001/` 根目录。其中 `server.command.sh` 是容器内真实 SGLang 命令；`server.reproduce.sh` 已把命令嵌入脚本，可以复制到另一台具备相同 Docker 镜像、宿主机模型路径、足够数量的兼容 GPU 和可用端口的服务器后，用 `bash server.reproduce.sh` 单独启动 server，不需要原始 JSONL、replay index 或 benchmark 目录。脚本按 GPU 数量申请设备，不绑定原服务器的物理 GPU 编号。多实例结果还会在 `instances/<id>/` 下保留每个实例的命令快照；应使用和 `server.evidence.json` 同目录的聚合脚本。
6. 如果需要暂停当前搜索或 resume，在 Jobs 页选中正在运行的任务，点击“停止选中任务”。这会向当前 `benchctl.py` 子进程发送中断信号；已经完成并写入的 trial 会保留，之后继续点击“严格 Resume”。
7. 如果某个候选启动失败，先看 `server.log` 和 `server.evidence.json`。修复 LLMPerf 代码或启动脚本后，可以回到该 candidate，点击“单独重跑这个参数”。单独重跑结果会写入 `debug-trials/`，不改自动搜索的 `best.json` 和 `search-state.json`。

如果服务器上已经有一部分实验结果，重新启动控制台后直接使用已有结果：

1. 启动 Web 控制台并确认结果目录。
2. 在左侧 Runs 里选择已有 result 目录。
3. 在 Jobs 页可以查看当前是否有 resume/search 子进程正在运行；选中任务后可以看实时输出，必要时点击“停止选中任务”暂停。
4. 如果 `results-index.json` 缺失或想重新汇总，点击 “Collect”。
5. 如果代码、模型、输入、镜像和 GPU 都没有变化，点击“严格 Resume”。这一步执行 `python3 benchctl.py run --plan <result-dir>/plan.json --resume`。
6. 如果只修改并上传了 LLMPerf 运行代码，严格 Resume 可能提示 `benchmark runtime changed since planning`。确认要用当前代码继续旧计划时，点击“接受当前代码 Resume”。前端会先备份原始 `plan.json` 和 `search-state.json` 到 `runtime-adoptions/`，再刷新运行时代码 fingerprint，然后继续 resume。

“接受当前代码 Resume”只表示沿用旧 `plan.json` 里的候选参数，用当前 LLMPerf 代码继续跑。它不会重新生成候选列表；如果模型、输入、镜像、GPU 或配置发生变化，应重新从配置生成新的 run。

命令行也可以分两步执行同样流程：

```bash
cd /data/hjh/LLMPerf
python3 benchctl.py plan --config configs/experiment.json
python3 benchctl.py run --plan /path/to/result-dir/plan.json
python3 benchctl.py run --plan /path/to/result-dir/plan.json --resume
```

## 本地验证

测试使用合成元数据、捕获请求和替身运行器，不启动真实 GPU 压测。在 LLMPerf 仓库根目录运行：

```bash
python3 -B -m unittest discover -s tests -v
PYTHONPATH=.:tests python3 -B -m unittest discover -s verification -v
```

vLLM、配置校验、环境探测、Docker、模型识别、回放适配、服务默认值及 Web 预览的相关 `tests/` 已纳入 Git，可在新克隆中运行。其他历史 `tests/` 和整个 `verification/` 仍为忽略跟踪的本地自测材料，完整回归命令仅适用于保留这些目录的环境。`verification/` 覆盖自适应配置、候选池、证据、调度、恢复、预算、容器超时与 Web 等回归场景，均不调用真实顾问或启动 GPU 压测。`tests/` 中 macOS 的环境探测自测依赖 Linux `/dev/shm`，该项应在目标 Linux 环境验证。

复制后的目录可以直接运行 `python3 benchctl.py --help`。实际硬件、checkpoint 和镜像的兼容性由目标机器上的环境探测、探索样本和最终完整输入复跑确认。

# 查看运行状态
先选最新一次 run：

cd /data/hjh/LLMPerf
RUN=$(ls -dt results/* | head -1)
echo "$RUN"

汇总所有 trial 的成功/失败数量：

python3 benchctl.py collect --run "$RUN"

jq -r '.rows[].status' "$RUN/results-index.json" | sort | uniq -c

查看每个实验点的状态、原因和吞吐：

jq -r '.rows[] | [
  .status,
  .task_id,
  ((.reasons // []) | join(","))
] | @tsv' "$RUN/results-index.json" | column -t -s $'\t'

查看最终整体状态：

jq '{status,trials,best,provisional_best,stop_reasons}' "$RUN/best.json"

一般重点看这两个文件：

results/.../results-index.json  # 每个 trial 的状态
results/.../best.json           # 整体是否 PASS，以及最终最佳配置
