# 模型兼容与迁移

兼容分成两件事：**平台能理解 checkpoint 和生成正确的服务配置**，以及**指定镜像能在目标 GPU 上加载并正确提供协议**。前者不能替代后者。尤其不能只改 `experiment.json` 的模型路径，再沿用不支持新架构的旧镜像。

## 1. 归一化与画像

平台只读 JSON 和静态模板，不在宿主机导入 checkpoint 的 Python 文件：

- 文本结构优先取 `text_config`，再回落到根配置；视觉层数、头数不混入文本并行约束。
- 保留根、文本、视觉各自的量化配置及字段来源。缺失字段记 UNKNOWN；已声明的非法结构仍报错，避免隐藏坏配置。
- 混合/压缩注意力及线性注意力头数作为事实保留；没有可靠引擎证据时，不按普通 MHA 额外猜测限制。
- `expert_dtype=fp4` 是 dtype，不自动等价于 `routed_experts_quant_method=mxfp4`。布局不明的 MoE 默认先走 AUTO 和镜像原生解析，不盲目枚举不相干的后端；显式 `search.backends` 保持优先。
- 模型画像只描述协议，不代表 GPU 支持。未知模型进入 generic：没有隐含 parser 或思考 kwargs；后续仍需镜像及运行时验证。

截至 2026-10-21，本次新增以下官方配置快照测试。完整 revision 和 SHA-256 在 `tests/fixtures/models/*/source.json`；测试不下载权重，也不执行模型代码。

| 官方模型 | 实际顶层 model_type | 文本层/注意力头 | 协议画像 |
| --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `deepseek_v41` | 40 / 64 | `deepseek-v41` |
| `Qwen/Qwen3.8-Flash-Next`、`Qwen/Qwen3.8-Flash-Next-FP8` | `qwen4_exp` | 48 / 24 | `qwen4-exp` |
| `zai-org/GLM-5.3-Flash`、`zai-org/GLM-5.3-Flash-BF16` | `glm5_next` | 45 / 64 | `glm5-next` |

营销名称不作为架构判断依据。BF16 与 FP8 变体分别读取各自配置，不用模型目录名猜量化。

### 协议与思考默认值

| 画像 | SGLang 工具 / 推理 parser | vLLM 自动候选 | 默认 chat_template_kwargs |
| --- | --- | --- | --- |
| DeepSeek V4.1 | `deepseekv41` / `deepseek-v41` | 工具、推理均 `deepseek_v41` | `thinking=true, reasoning_effort=max` |
| Qwen 新 Flash | 从静态模板判断；XML 模板使用 `qwen3_coder` / 推理 `qwen3` | 工具按识别结果、推理 `qwen3` | `enable_thinking=true, reasoning_effort=max` |
| GLM 新 Flash | `glm47` / `glm45` | 工具 `glm47`，推理按镜像选择 `glm45` 或 `glm47` | `enable_thinking=true, reasoning_effort=max` |
| generic | 不猜测 | 不猜测 | `{}` |

自动候选必须与镜像实际注册项匹配，不按版本号猜支持情况。显式 parser 覆盖保持原名称，不能把 SGLang 名称直接当 vLLM 名称填写。旧画像沿用既有默认值，避免升级改变旧模型的基线。

DeepSeek V4.1 的官方编码使用新原生协议，不能因没有 `chat_template.jinja` 就报模型目录损坏，也不能回退 V4 parser。官方 V4.1 的带空格 DSML 标签与 V4 不同；是否正确编码由支持该协议的引擎负责。其 `reasoning_effort` 支持数值及 `low/high/max` 别名，平台沿用 `max`，不把它误判为非法值。GLM 的历史思考清理策略等可通过显式 kwargs 设置，不自动改变用户的多轮语义。

## 2. 镜像内检查，而非宿主机 transformers 白名单

`doctor`、自动规划及智能调优会采集镜像内的：

1. 引擎版本、CLI/schema 能力与 parser 注册表（包括可探测到的延迟注册项）。
2. 原生模型注册表中的架构。
3. 引擎自己的 config 解析结果，使用 `trust_remote_code=False`。该检查只挂载单个 `config.json`，开启离线模式，不挂载 checkpoint Python 文件或权重。

结果写入 `model-compatibility.json`，并进入计划、顾问证据和 Web 模型兼容性面板：

- `SUPPORTED`：当前子项有明确证据，例如架构已注册或 config 被接受；不代表整个模型已跑通。
- `UNSUPPORTED`：已知注册表缺少所需架构/parser，阻止生成候选，并给出镜像/覆盖提示。
- `UNKNOWN`：导入、设备识别、config 解析或注册表探测失败；也包括需要运行验证的 Transformers fallback。保留诊断原因，不伪装成不支持或已支持。
- `NOT_RUN`：还没有进行真实运行验证。

即使所有可静态验证的项目通过，整体也保持 WARN、`runtime_verified=false`：加载、编码、量化 kernel 和实际请求仍未验证。探测用的是模型原始声明；把 `model_overrides.architectures` 改成老架构不能绕过真实架构核验。

## 3. 真实启动后的非计分协议检查

新 Flash 及 generic 画像的**新计划**会添加 `protocol_validation`。每个实例就绪并核验实际参数后，在计时回放前运行少量独立检查：

- 普通文本 chat；业务输入有流式请求时额外检查 SSE、`[DONE]` 和 usage。
- 业务输入含 tools/functions/tool_calls 时检查生成工具调用的函数名及 JSON 参数；同时使用流式时额外检查分片拼接。
- 已开启且配置了推理 parser 时，检查 `reasoning_content` 或 `reasoning` 通道。
- 内容泄露原始 DSML、工具或思考标签，以及 HTTP 拒绝、格式错误，明确判失败。

检查使用固定的小型合成请求，单请求最多 256 个输出 token、每实例共享约 60 秒检查预算。网络超时受 urllib 的连接/读取超时约束，不是独立进程的强制截止。工具仅为生成协议用的虚拟函数，**绝不执行**；报告只保留状态、计数和有限诊断，不落盘模型回答正文。这些请求不进入原始业务输入、usage 总量或计分耗时，但可能起到预热作用，不能把结果当作完全冷启动基线。

`server.protocol.json` 与 trial 证据指纹绑定，校验实例覆盖范围、所需 case 和状态一致性：

- FAIL：停止本轮回放或使该 trial 无效；缺少/损坏必需报告也不能通过。
- UNKNOWN：例如输出达到探测上限、未观测到 reasoning/工具调用、usage 缺失或超时。允许继续测吞吐，但记录 `protocol_validation_incomplete`，保持 `model_runtime_verified=false`。
- PASS：所要求的文本协议检查通过，trial 可记录 `model_runtime_verified=true`。**不证明答案质量、精度、长上下文、所有工具语义或长期稳定性**。

文本检查无法验证图像/音频/视频；业务输入含多模态时保持 UNKNOWN、`multimodal_verified=false`，应另行验证对应模态。吞吐结果 `VALID` 和模型协议已验证是两个独立维度，前端、trial 及顾问摘要分别显示。旧画像不强制新增检查，旧计划也不会被升级静默改写。

## 4. 将现有实验迁移到 DeepSeek V4.1

1. 把 `model_path` 指向完整的 `DeepSeek-V4.1-Flash-1017` checkpoint；不要修改原始 `config.json`，也不要添加假 `text_config` 或伪装为 V4。
2. 把 `image` 换为实际提供 V4.1 架构、原生编码、parser、量化 kernel 及目标 GPU 支持的镜像，固定真实 digest。平台不能替旧镜像实现这些能力；没有经过验证的镜像摘要时不要照抄旧示例。
3. 保留实际业务 `input_path` 和资源/预算设置，先运行：

   ```bash
   python3 benchctl.py doctor --config configs/experiment.json
   ```

4. 查看本次结果目录的 `model-compatibility.json`。若 parser/架构明确不支持，应先升级镜像；UNKNOWN 要查看具体诊断，不能当成通过。
5. 新建计划或运行 `python3 benchctl.py auto --config configs/experiment.json`。可临时设 `smoke: true` 限制最初验证负载；正式实验使用所需完整负载和独立重复验证。
6. 检查 `server.protocol.json`、真实参数、服务端 usage、健康检查及日志，再确认吞吐成绩。没有目标 GPU/镜像实测，不宣称 V4.1 已完整可用。

`advisor.model` 是规划顾问的服务名，不是被测 checkpoint；更换压测模型无需同步更换顾问模型。业务请求的 `request.model` 仍作为被测服务名称，请求正文不会为了适配被重写。保存计划绑定脚本、元数据和镜像证据；升级后应重新生成计划，不手动删除哈希来继续旧计划。

特殊镜像有不同原生 parser 名时，可在原配置中加入以下覆盖（SGLang 示例；默认映射正确时无需填写）：

```json
"model_overrides": {
  "tool_call_parser": "deepseekv41",
  "reasoning_parser": "deepseek-v41",
  "chat_template_kwargs": {
    "thinking": true,
    "reasoning_effort": "max"
  }
}
```

覆盖不是绕过镜像兼容性检查的开关。vLLM 的对应名称是 `deepseek_v41`；关闭思考应使用该模型实际支持的参数，例如 V4.1 的 `thinking=false`，而不是给所有模型统一添加相同 kwargs。

## 5. 后续新增模型与测试

通用 JSON 归一化在 `llmperf/model_metadata.py`，协议映射在 `llmperf/model_protocols.py`，静态发现入口在 `llmperf/discovery.py`。只在确认新协议后增加画像；不要为了新模型复制 planner 或修改 checkpoint。引擎注册表/config/parser 探测在 `benchmarks/server/model_capabilities.py`，分层判定在 `llmperf/model_compatibility.py`；文本协议检查在 `benchmarks/server/protocol_check.py`。

新增适配时应提交固定 revision 的官方 JSON 快照及哈希，覆盖嵌套结构、量化变体、静态模板、引擎 parser 名、UNKNOWN/明确失败和真实响应形状。目标镜像/GPU 的真实启动验证另行执行，离线单元测试不能替代。

```bash
PYTHONPATH=.:tests python3 -B -m unittest \
  test_model_compatibility test_image_model_support test_protocol_compatibility -v
```

macOS 本地验证建议使用 Python 3.12，并设置 `TMPDIR=/private/tmp` 避免 `/var` 与 `/private/var` 路径别名导致旧测试断言失败。现有容器探测测试依赖 Linux `/dev/shm`，不应把 macOS 缺少它解释为目标 Linux 镜像不支持模型。
