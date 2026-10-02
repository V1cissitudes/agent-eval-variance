# Agent 设计说明（2026-10-02）



## 结构
```
run_experiment.py → eval/runner.py ─ 每个 题目 × 条件 × 重复 调一次 ─→ agent/loop.py
                                                            ├─ llm/client.py   发请求（采样参数全记录）
                                                            ├─ agent/parsing.py 解析 Thought / Action / Final Answer
                                                            └─ tools/search.py  BM25 检索本题自带的 10 段落
             └─ eval/metrics.py（EM/F1）→ eval/trajectory.py（每次运行一行 JSONL，写入 runs/）
```

## 关键决定与理由
1. **文本 ReAct，而非原生 function calling / 约束解码。** 解析在本地、透明、跨平台一致；格式错误本身是方差数据；约束解码会改变采样分布。
2. **不搜索直接回答：接受，记录 n_searches=0。** 拒绝等于干预 agent 的自然行为；方差研究要测的是自然行为分布。
3. **格式错误：给出提示性 Observation 后继续，直到 max_steps。** 不立即终止，记录 n_format_errors。
4. **stop=["\nObservation:", "Observation:"]。** 防止模型编造工具结果。
5. **检索工具确定性。** BM25 同分时按段落顺序打破平局；工具不引入额外随机性。top_k=2。
6. **思考内容不回填历史。** 与 Qwen3 官方多轮做法一致；思考文本单独记录在每步 `reasoning`。
7. **种子。** 每次运行 run_seed = sha256(实验种子, 题目 id, 重复序号)，第 t 步请求 seed = run_seed + t。跨机器可复现。
8. **轨迹格式。** 每次运行一行：条件（端点、平台、模型、精度、温度、top_p/top_k、思考、缓存）、元数据（git commit、是否有未提交改动、PROMPT_VERSION、时间）、每步（输出、思考、解析、观察、token、耗时、finish_reason）、汇总（EM/F1、步数、搜索次数、格式错误、截断、最大 prompt token）。第 t 步输入 = 初始消息 + 之前各步（输出、观察），因此不重复存储。
9. **断点续跑。** 同一命令重跑会跳过已成功的 run_id；异常写成 error 记录而不中断整个实验。

## 已知问题 / 开放设计问题
- **top_k 等采样默认值跨平台不一致**：Ollama 模型自带 top_k=20、top_p=0.95；vLLM 默认读取模型 generation_config（Qwen3 同样 top_k=20 等）。配置中 `top_k: null` 表示不发送、由服务端默认决定。正式实验前需决定是否显式固定（例如两平台都发 top_k），否则"平台"因素里混有采样默认值差异。
- **max-model-len**：dev_smoke（思考关、5 题 × 2 次）最大 prompt 881 token；思考开时需另测。
- **提示词中的示例**是手写的，不来自评测集；PROMPT_VERSION=react-v1，改提示词须升版本号。

## 怎么跑
```bash
uv run python scripts/run_experiment.py configs/experiments/dev_smoke.yaml --prefix-cache na            # Mac Ollama
uv run python scripts/run_experiment.py configs/experiments/dev_smoke.yaml --prefix-cache on --endpoint vllm_5070
```
