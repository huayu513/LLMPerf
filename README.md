# Automation 自动吞吐压测

只维护 `configs/experiment.json`，用一条命令完成模型识别、环境检查、输入索引、参数搜索、重复验证和结果收集。`Automation/` 可单独复制到目标机器，不依赖旁边的 sglang 或其他 benchmarks 目录。

## 首次部署与启动

机器需要可运行 `python3` 的 Python 环境、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。模型权重、输入 JSONL 和运行镜像放在目标机器上。镜像需要提供可用的 `sglang serve` 命令，并包含 PyTorch、aiohttp、bash、curl、setsid；程序检查本地镜像，不自动拉取或构建镜像。

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
cd /path/to/Automation
python3 benchctl.py auto --config configs/experiment.json
```

路径支持 `~` 和环境变量，相对路径以配置文件所在目录为基准。无需填写模型名称、parser、TP/DP/PP、profile、workload 或 policy 文件。现有旧格式配置和旧 plan 需要重新生成。

## 自动执行的流程

1. 读取 checkpoint 中的 `config.json` 等元数据，识别已适配的 Qwen、GLM MoE、DeepSeek 模型及量化信息。模型目录可以任意命名；共用架构名称的变体结合静态聊天模板识别。未适配或证据不足时给出具体原因及同文件覆盖方式，不执行模型目录里的 Python 代码进行识别。
2. 验证原始 JSONL，统计请求、读取请求使用的服务名称并计算 SHA-256。单次实验要求一个 `request.model` 名称，服务端使用这个名称；请求对象原样回放。
3. 探测 GPU 数量、型号、显存、拓扑及镜像内的 SGLang 选项、parser、后端列表。按同型号/算力/显存的 GPU 分组生成部署拓扑和 TP/DP/PP 候选，过滤注意力头数等静态不兼容组合。若选择 2 张卡，会真实比较 `2卡1实例` 与 `2卡2实例`；若选择 8 张卡，会比较 `8卡1/2/4/8实例`。MIG 暂不支持。
4. 默认不做单独 smoke。探索阶段直接启动候选服务并跑有用的小样本请求集，自动起始并发按所选 GPU 数估算，例如 2 卡默认从 16 开始；之后轮流增加并发，吞吐增长不足或请求失败时停止该分支，在最佳并发附近细查，并测试内存比例和 prefill 大小的少量邻近设置。
5. 探索层和全量层都不使用固定 topK：所有与当前 top1 输出 tokens/s 差距在 `promotion_tolerance` 内的候选点都会晋级或复跑。最终层使用完整请求集独立重复验证，按输出 tokens/s 中位数选赢家。只有请求成功、服务端 completion usage 可用、输入摘要一致且实际服务参数可核验的结果才参与排名。

每个试验点启动新容器和新服务。模型、输入、索引及脚本只读挂载；结果单独写入。容器只接收选中的物理 GPU，服务进程使用从 0 开始的逻辑编号。回放通过容器内 loopback 访问服务，不占用宿主机固定端口。

`thinking`、`reasoning_effort`、采样参数和输出长度不参与吞吐调优；沿用原始请求及模型/镜像的默认语义。索引在宿主机使用标准库生成，不加载 tokenizer；其中不生成估算 token 统计，吞吐计量使用服务端返回的 usage。

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
    "start_concurrency": null,
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
| `gpu_indexes` | 使用检测到的物理 GPU，按同型号等条件分组测试 |
| `smoke` | 默认 `false`，不单独启动服务做冒烟；设为 `true` 时每个候选先跑一次 8 条以内的冒烟 |
| `search.concurrency_max` | 最大并发 64，实际不超过请求数量 |
| `search.start_concurrency` | 默认自动估算：`min(concurrency_max, max(8, 8 * 所选GPU数))`；可显式指定 |
| `search.explore_request_limit` | 探索和调参阶段默认最多回放 256 条；设为 0 表示探索阶段也使用完整请求集 |
| `search.promotion_tolerance` | 默认 0.05；与 top1 吞吐差距 5% 以内的候选点都会进入最终全量复跑 |
| `search.max_trials` | 最多 64 个试验点，包含失败点和最终重复验证；启用 `smoke` 时也包含冒烟 |
| `search.max_seconds` | 搜索阶段 14400 秒；在试验点之间检查，正在执行的点允许完成 |
| `search.repetitions` | 最佳候选完整输入验证 3 次；试验预算至少为该值加 2 |
| `warmup` | 每个点额外预热 0 条；预热不替代完整正式请求集 |
| `request_timeout` | 单请求超时 3600 秒 |
| `ready_timeout` | 服务就绪等待 3600 秒 |

高级使用可以在 `search.backends` 指定镜像实际支持的 MoE runner 名称；默认使用引擎自动选择和适合模型量化的已声明后端。`search.open_loop_scales` 可选，例如 `[1, 2, 4]`，在闭环赢家验证之后按原始时间戳进行开环回放；默认不运行开环。开环结果独立记录，不用于替换闭环吞吐赢家。

特殊模型适配可以通过同文件的 `model_overrides` 覆盖已支持的 parser、模板参数或模型专用环境变量，覆盖会记录来源。Docker 的网络、共享内存、IPC 和内部端口也可通过 `docker` 对象覆盖。内部服务端口默认是 `25080`；不要设置到常见 Linux 临时端口范围 `32768-60999`，因为 SGLang 启动时也会从该范围分配内部通信端口。拼错或不支持的字段会报错。

## 结果与恢复

启动后终端首先打印 `result_dir`。目录内包含：

```text
resolved-config.json       # 完整配置、模型/输入事实与参数来源
environment.json          # GPU、镜像和引擎能力快照
data/replay_index.json    # 绑定原始输入的回放索引
plan.json                 # 自动生成的候选和搜索范围
search-state.json         # 已分配的试验、预算和停止原因
trials/.../attempt-001/   # 每个点的命令、服务信息、日志、原始结果与 trial.json
results-index.json        # 汇总所有已结束的试验
leaderboard.json          # 当前有效试验按输出 tokens/s 排名
best-so-far.json          # 当前已完成试验中的临时最佳点
best.json                 # 已验证赢家、重复成绩、配置和搜索停止原因
```

`best.json` 的 `best.configuration` 保存服务端实际生效的配置，`planned_configuration` 保留规划值，`server_command_paths` 指向各次验证的真实启动命令。引擎可能对 DP Attention 等参数做换算，复跑时应使用记录的启动命令。只有各次重复验证的实际配置一致才会发布赢家。

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
cd /data/hjh/Automation
python3 -m web.server --host 0.0.0.0 --port 18080
```

浏览器打开 `http://服务器IP:18080/`。页面默认扫描 `/data/hjh/Automation/results`，也可以在页面顶部修改结果目录。配置文件默认指向 `configs/experiment.json`，也可以改成其他 JSON。

从 0 开始调试推荐按这个顺序操作：

1. 修改 `configs/experiment.json`，至少确认 `model_path`、`input_path`、`image`、`gpu_indexes` 和 `output_dir`。
2. 在前端点击“生成计划”。这一步执行 `python3 benchctl.py plan --config ...`，只创建 result 目录和 `plan.json`，不会开始压测。
3. 在“候选规划”页检查每个 candidate。点进 candidate 可以看拓扑、GPU 分配、TP/DP/PP、DP Attention、MoE backend、DSpark、内存比例、chunked prefill，以及预计传给 SGLang 的启动参数。
4. 确认计划后点击“开始搜索”。这一步执行 `python3 benchctl.py run --plan <result-dir>/plan.json`。
5. 运行过程中可以在 Jobs 页看当前后台命令、stdout 事件和最新输出；在 Trials 页看每个 trial 的状态、吞吐和失败原因；点进 trial 或 candidate 的 artifact 可以看 `server.log`、`docker.log`、`server.command.sh`、`server.evidence.json`、`server.info.json` 和 `*.summary.json`。
6. 如果需要暂停当前搜索或 resume，在 Jobs 页选中正在运行的任务，点击“停止选中任务”。这会向当前 `benchctl.py` 子进程发送中断信号；已经完成并写入的 trial 会保留，之后继续点击“严格 Resume”。
7. 如果某个候选启动失败，先看 `server.log` 和 `server.evidence.json`。修复 Automation 代码或启动脚本后，可以回到该 candidate，点击“单独重跑这个参数”。单独重跑结果会写入 `debug-trials/`，不改自动搜索的 `best.json` 和 `search-state.json`。

如果服务器上已经有一部分实验结果，重新启动控制台后直接使用已有结果：

1. 启动 Web 控制台并确认结果目录。
2. 在左侧 Runs 里选择已有 result 目录。
3. 在 Jobs 页可以查看当前是否有 resume/search 子进程正在运行；选中任务后可以看实时输出，必要时点击“停止选中任务”暂停。
4. 如果 `results-index.json` 缺失或想重新汇总，点击 “Collect”。
5. 如果代码、模型、输入、镜像和 GPU 都没有变化，点击“严格 Resume”。这一步执行 `python3 benchctl.py run --plan <result-dir>/plan.json --resume`。
6. 如果只修改并上传了 Automation 运行代码，严格 Resume 可能提示 `benchmark runtime changed since planning`。确认要用当前代码继续旧计划时，点击“接受当前代码 Resume”。前端会先备份原始 `plan.json` 和 `search-state.json` 到 `runtime-adoptions/`，再刷新运行时代码 fingerprint，然后继续 resume。

“接受当前代码 Resume”只表示沿用旧 `plan.json` 里的候选参数，用当前 Automation 代码继续跑。它不会重新生成候选列表；如果模型、输入、镜像、GPU 或配置发生变化，应重新从配置生成新的 run。

命令行也可以分两步执行同样流程：

```bash
cd /data/hjh/Automation
python3 benchctl.py plan --config configs/experiment.json
python3 benchctl.py run --plan /path/to/result-dir/plan.json
python3 benchctl.py run --plan /path/to/result-dir/plan.json --resume
```

## 本地验证

测试使用合成元数据、捕获请求和替身运行器，不启动真实 GPU 压测。包内测试可从包含 `s1slow` 的目录运行：

```bash
python3 -B -m unittest discover -s s1slow/Automation/tests -v
```

复制后的目录可以直接运行 `python3 benchctl.py --help`。实际硬件、checkpoint 和镜像的兼容性由目标机器上的环境探测、探索样本和最终完整输入复跑确认。

# 查看运行状态
先选最新一次 run：

cd /data/hjh/Automation
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