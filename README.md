# FlowEvo / Flow AutoTTS

FlowEvo 是一个用于 **flow-matching 采样器 test-time scaling** 的自动搜索库。当前主实验是：

- 模型：Stable Diffusion 3.5 Medium
- 奖励：OCR reward
- 任务：在固定 NFE budget 下，让 Codex 迭代改写 controller，寻找比普通 ODE 更好的文本渲染采样策略

## 核心思路

库里把采样过程包装成一个受限环境，controller 只能调用公开动作：

```text
spawn      创建分支
forward    向前积分一步或多步
preview    解码/打分当前 clean anchor
backward   从某个 anchor 重新加噪，生成局部分支
prune      剪掉弱分支
answer     返回最终结果
```

Workflow 每轮会：

1. 从 `optimal.template.py` 重置 `flow_autotts/controllers/optimal.py`
2. 给 Codex 一个 AutoTTS 风格 prompt、spec、baseline、最近几轮 summary 和 controller snapshot
3. Codex 只改 `optimal.py`
4. harness 在 OCR train prompts 上评估不同 beta/NFE 档
5. 归档 controller、summary、history，进入下一轮

`beta` 控制 compute budget：`0` 接近 10 NFE deterministic ODE，`1` 接近 64 NFE；中间 beta 会分配更多 preview、branch、backward refinement 和 pruning。

## 安装

推荐用 `uv`：

```bash
cd /root/code/FlowEvo
uv sync --group dev --group sd35
```

OCR reward 依赖 PaddleOCR：

```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```

Codex workflow 还需要 OpenAI Codex CLI 和较新的 Node：

```bash
node --version   # 建议 >= 18
npm install -g @openai/codex
codex exec --help
```

如果本机的 sandbox 不可用，可以在运行 workflow 时加：

```bash
CODEX_EXEC_ARGS="--dangerously-bypass-approvals-and-sandbox"
```

## 数据和模型

默认路径：

```text
SD_3.5_med/                   SD3.5 Medium 本地模型
flow_grpo/dataset/ocr/train.txt
flow_grpo/dataset/ocr/test.txt
```

下载模型示例：

```bash
huggingface-cli login
huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
  --local-dir SD_3.5_med
```

数据文件是一行一个 prompt。当前仓库默认读取 `flow_grpo/dataset/ocr/train.txt` 和 `test.txt`；如果新机器缺失，从 Flow-GRPO 的 `dataset/ocr` 拷贝到同一路径即可。也可以用环境变量 `FLOW_TTS_DATASET=/path/to/ocr` 指向自己的数据目录。

OCR reward 默认要求 prompt 里目标文本用英文双引号包起来，因为 reward 会抽取引号中的字符串并用 PaddleOCR 识别结果做编辑距离打分。

## 运行 5 轮搜索

下面命令用 4 张卡并行评估，每轮评估 500 条 train prompts、5 个 beta 档：

```bash
cd /root/code/FlowEvo

RUN_TAG="autotts_ocr_4gpu_r0000_0004_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/flow_autotts/ocr_sd35/manual_runs

nohup env \
  FLOW_TTS_PROMPT_PROFILE=autotts \
  FLOW_TTS_EVAL_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3" \
  WORKFLOW_RESUME=0 \
  WORKFLOW_ROUNDS=5 \
  WORKFLOW_CONTEXT_HISTORY_ROUNDS=5 \
  FLOW_TTS_SPLIT=train \
  FLOW_TTS_SAMPLE_SIZE=500 \
  FLOW_TTS_SAMPLE_SEED=42 \
  FLOW_TTS_BETAS="0 0.25 0.5 0.75 1.0" \
  FLOW_TTS_BUDGET=64 \
  FLOW_TTS_NUM_STEPS=10 \
  WORKFLOW_HISTORY_DIR="logs/flow_autotts/ocr_sd35/history_${RUN_TAG}" \
  WORKFLOW_CODEX_LOG_PARENT="/root/code/FlowEvo/logs/flow_autotts/ocr_sd35/codex_logs_${RUN_TAG}" \
  WORKFLOW_RESULT_DIR="/root/code/FlowEvo/logs/flow_autotts/ocr_sd35/training_results_${RUN_TAG}" \
  CODEX_EXEC_ARGS="--dangerously-bypass-approvals-and-sandbox" \
  uv run --group sd35 bash flow_autotts/experiments/ocr_sd35/run_workflow.sh \
  > "logs/flow_autotts/ocr_sd35/manual_runs/${RUN_TAG}.log" 2>&1 &

echo "${RUN_TAG}"
```

看进度：

```bash
tail -f logs/flow_autotts/ocr_sd35/manual_runs/${RUN_TAG}.log
nvidia-smi
```

## 输出怎么看

每轮结果在：

```text
logs/flow_autotts/ocr_sd35/history_${RUN_TAG}/rXXXX_*/proposal_results/summary.json
logs/flow_autotts/ocr_sd35/history_${RUN_TAG}/rXXXX_*/proposal_results/history.json
logs/flow_autotts/ocr_sd35/history_${RUN_TAG}/rXXXX_*/flow_autotts/controllers/optimal.py
```

`summary.json` 是给下一轮 Codex 的 compact history，包含：

- 每个 beta 的 reward / NFE / reward_per_nfe
- Pareto frontier
- `action_statistics`：平均 spawn、forward、preview、backward、prune、mean_nfe
- `behavior_summary`：一句话概括该 beta 档 controller 行为

`history.json` 是完整评估日志，包含每个 prompt 的 event log，体积会比较大。

## 快速检查

```bash
python -m py_compile flow_autotts/controllers/optimal.py
python -m pytest tests/test_workflow.py tests/test_ocr_sd35_harness.py
```

## 实验逻辑
```text
整体目标

  FlowEvo 的核心目标不是手工写一个固定采样器，而是用一个自动迭代工作流，在固定实验设定下不断改进 flow_autotts/controllers/
  optimal.py，让它在一组固定 beta 上学到更好的 reward/NFE tradeoff。

  每一轮都做三件事：

  1. 基于已有历史和 baseline，给 Codex 组织一份受控 context。
  2. 让 Codex 从模板出发提出一个新的 optimal.py。
  3. 对新 controller 做统一评测，并和当前 incumbent 比较，决定是否接受为新的最好 controller。

  ———

  每轮是如何开始的

  每轮开始前，workflow 都会先把：

  - flow_autotts/controllers/optimal.py

  重置为：

  - flow_autotts/controllers/optimal.template.py

  也就是说，controller 不是在上一轮代码上直接累积 patch，而是每轮都从 template 重新生成。历史信息通过 context 提供，而不是通
  过保留上一轮文件状态提供。

  这样做有两个目的：

  - 防止 controller 代码逐轮漂移、积累无关改动；
  - 让 proposer 真正根据“历史经验”和“当前弱点”重新综合设计，而不是机械微调上一版文件。

  ———

  每轮给 Codex 的 context 是怎么组织的

  workflow 会为每一轮生成一个 context_pack.md。这个文件是 proposer 的主要上下文预算，Codex 被明确要求优先读它，而不是自由扫描
  整个仓库。

  context pack 的内容大致分成几层。

  1. 固定允许先读的核心文件

  - flow_tts_controller_implementation_spec.md
  - 当前待修改的 optimal.py
  - flow_autotts/controllers/baselines.py
  - flow_autotts/core/state.py
  - flow_autotts/core/errors.py
  - pickscore 实验对应的 harness.py 和 env.py
  - 最近几轮 compact summary 和 controller snapshot

  这一步的作用是把 proposer 的注意力限制在 controller API、状态结构、动作语义和近期历史上。

  2. 写边界和读取纪律
     context 里会明确告诉 Codex：

  - 只能编辑 optimal.py
  - 不能改 harness、workflow、dataset、logs、模型目录等
  - 不要做 repo-wide 扫描
  - 不要 bulk-read 原始 logs、dataset、模型目录、.git/ 等

  这保证了 proposer 是在一个窄边界内优化 controller 行为，而不是改实验系统本身。

  3. baseline references
     workflow 会注入 baseline 的 compact summary，通常优先使用显式指定的 baseline summary，再补最近的 baseline 文件。

  这些 baseline 不是只做展示，而是后面所有 frontier 对比、promotion 评分和 proposer 分析的参照。

  4. beta target curve
     workflow 会显式给出固定 beta 对应的 target NFE 曲线。原始 FlowEvo 的 fixed target map 是：

  - beta 0.0 -> target NFE 10
  - beta 0.25 -> 20
  - beta 0.5 -> 36
  - beta 0.75 -> 48
  - beta 1.0 -> 64

  这里的 target NFE 不是优化目标本身，而是“可比性约束”。意思是：

  - reward 仍然是主目标；
  - 但不同 controller 应该大致处在相同 compute regime 下比较；
  - 不能靠明显少花 NFE 或乱超预算来制造不公平优势。

  5. action semantics
     context 会总结各个公开动作的语义和常见作用：

  - spawn
  - forward
  - preview
  - backward
  - prune
  - answer

  不仅说明动作能干什么，还会强调它们通常如何花费 NFE、适合什么场景、常见失败模式是什么。这个部分是让 proposer 做“策略设计”而
  不只是调参数。

  6. recent promoted history
     workflow 默认只把最近被 accepted 的轮次作为主要历史上下文。也就是说，recent history 不是简单最近 N 轮，而是更偏
     向“promoted-only”的有效历史。

  这样做是为了避免 proposer 被大量 rejected 试验噪声带偏。

  7. 从最近历史中提炼出的结构化分析
     这部分是 FlowEvo workflow 最关键的地方。它不只是塞原始 summary，而是从历史里自动提炼几种二级信息：

  - Historical Best Near Beta Target
    看每个 beta target 附近，历史上最好的 reward 是谁。
  - Recent Round Frontier Comparison
    比较最近几轮在各 beta 上的 NFE、reward、相对 baseline 的 delta。
  - Beta Opportunities
    指出当前哪些 beta 区域最弱，是 underuse、overuse 还是 allocation 不好。
  - Regression Ledger
    比较最近两轮，提示哪些 beta 进步了，哪些 beta 回退了。
  - Rejected Round Lessons
    从 rejected round 中总结负例经验，告诉 proposer 某种 action-shift 为什么可能伤害了 frontier。
  - Historical Action Effects
    汇总“增加/减少某类 action”在历史上通常带来什么 reward/NFE 影响。

  所以 proposer 拿到的不是“原始日志堆”，而是一份被 workflow 预加工过的、面向决策的 context。

  ———

  Codex prompt 是怎么约束 proposer 的

  prompt 不是泛泛地说“改好 controller”，而是把 proposer 约束成一个固定角色：

  - 只能改 optimal.py
  - 必须遵守公开环境 API
  - 必须在一组 beta 上同时优化，而不是只优化某一个 beta
  - 要把 controller 设计成一个随 beta 变化的 schedule，而不是硬编码单点逻辑
  - 必须有“行为上实质性的新机制”，不能只做注释、轻微阈值改动或 cosmetic 改动
  - 必须做 target-NFE reflection，检查每个 beta 是否明显 underuse / overuse
  - 最终回复里要说明：
      - controller 思路
      - 看了哪些历史
      - beta schedule 是什么
      - 和最近最好 controller 相比行为上哪里新
      - 风险是什么

  也就是说，prompt + context pack 共同把 proposer 变成一个“带历史记忆、带 frontier 约束、带 beta-schedule discipline”的
  controller designer。

  ———

  每轮评测是怎么做的

  每轮 proposal 生成后，workflow 会跑统一评测命令。原始 FlowEvo 默认是：

  - pickscore 数据
  - train
  - sample_size=500
  - sample_seed=42
  - beta sweep = 0, 0.25, 0.5, 0.75, 1.0
  - budget = 64
  - 多卡并行评测

  每轮只评一个 controller：当前 proposal 的 optimal.py。

  评测输出会先写到 result 目录，再被归档进对应的 history/r000x.../proposal_results/。

  所以每个 round 在 history 里都会保存：

  - 当轮 controller snapshot
  - proposal_results/summary.json
  - proposal_results/history.json 或相关评测产物
  - proposer/eval 的日志

  ———

  incumbent 是怎么维护的

  workflow 维护一个 incumbent controller，也就是“当前被认可的最佳 controller”。

  初始化时：

  - 如果历史里已有 accepted rounds，会把最近一个 accepted round 当作 incumbent；
  - 否则第一轮没有 incumbent，首个可解析 candidate 默认 accepted。

  之后每一轮都不是直接替换，而是做 candidate vs incumbent 的比较。

  ———

  candidate 和 incumbent 是怎么比较的

  这一步不是简单比较平均 reward，也不是只看某一个 beta，而是做 fixed-target frontier scoring。

  workflow 会先把 candidate summary 和 incumbent summary 都映射成一个 score，核心字段包括：

  - missing_betas
  - over_budget_count
  - betas_beating_baseline
  - total_reward_gap
  - min_reward_gap
  - target_gap_abs_sum
  - covered_betas

  其中：

  - reward_gap 指的是相对于 beta-matched baseline reward 的差值；
  - total_reward_gap 看整体 frontier 是否更强；
  - min_reward_gap 看最差 beta 是否更差；
  - target_gap_abs_sum 看是否严重偏离固定 target NFE；
  - betas_beating_baseline 看有多少 beta 已经超过 baseline；
  - over_budget_count 尤其约束高 beta 不要越界。

  然后 workflow 把这些字段按一个固定优先级组成 tuple 比较。只有 candidate tuple 严格优于 incumbent tuple，candidate 才会被
  accepted；否则就是 rejected。

  所以本质上，workflow 追求的是：

  - frontier 整体提升；
  - 最弱 beta 不要恶化太多；
  - 尽量保持 compute comparability；
  - 不能靠违规超预算取胜。

  ———

  accepted / rejected 后会发生什么

  如果 accepted：

  - incumbent 更新为这一轮 candidate
  - 后续 context 里，这一轮会成为 promoted history 的一部分

  如果 rejected：

  - incumbent 不变
  - 这轮结果仍然归档保留
  - workflow 会把这轮写进 workflow_index.jsonl
  - 后续 context 可能把它当作 rejected lesson，总结成负例经验

  所以 rejected round 不是白跑，它会进入“反例知识库”。

  ———

  如何早停

  原始 FlowEvo 有 early stop 逻辑，但不是特别激进。

  主要有两类条件：

  1. 达到最小轮数后，如果连续若干轮都 rejected，可以停。
  2. 比较最近两个 accepted controller，如果最新 accepted 在 total reward gap 和 worst-beta gap 上都没有实质改进，也可以停。

  默认参数大致是：

  - min_rounds_before_stop = 5
  - max_consecutive_rejections_before_stop = 3
  - min_total_reward_gap_improvement = 0.001

  所以它不是“只要没涨就停”，而是：

  - 至少先跑够若干轮；
  - 只有在 accepted frontier 基本停滞或连续被拒很多轮时才停。

  ———

  为什么这个 workflow 不是普通的 prompt engineering

  因为它的迭代不是靠“把上一轮结果贴给大模型让它再想想”，而是系统化做了三层约束：

  1. 状态约束

  - 每轮从 template 重置
  - 只能改 controller 文件
  - 统一评测接口

  2. 信息约束

  - context pack 只给 controller 设计真正相关的信息
  - 不允许自由扫描 repo 和日志海洋

  3. 决策约束

  - 不是主观选最好，而是 fixed-target frontier promotion
  - 通过 accepted/rejected 形成稳定的搜索轨迹
  - rejected rounds 也会作为负例知识反馈到后续轮次

  所以它更像一个“controller program synthesis + constrained evolutionary selection”的自动迭代系统。

  ———

  一句话概括

  FlowEvo 的迭代逻辑可以概括成：

  在固定 beta-target frontier 下，workflow 每轮从 template 重置 controller，向 Codex 提供一个经过结构化整理的 context pack，
  其中包含 baseline、目标 NFE 曲线、近期 promoted 历史、历史 action 效果、回归记录和 rejected 负例；Codex 只改 optimal.py 生
  成新 controller；workflow 对该 controller 做统一评测，并用 fixed-target frontier score 与 incumbent 比较，决定 accepted 或
  rejected，从而逐轮逼近更优的 reward/NFE Pareto frontier。

这里跑了5轮，没有早停。
换成了ocr数据和reward
```

## 对比结果
见logs/flow_autotts/ocr_sd35/r0004_vs_baselines_train_test.md
