# CLAUDE.md

## 项目背景

这是一个 LLM agent 评测方法论项目：从零手写一个小型 agent，在 HotpotQA（distractor 设置，用题目自带段落做本地检索）上采集多种条件下的重复运行轨迹，研究两个问题（以 docs/decisions.md 中的最新决定为准）：
1. 同一输入重复运行时，agent 行为与结果的方差来自哪里：题目、模型、权重精度、推理服务配置（如 prefix cache）、采样温度、随机种子各自贡献多少（方差成分分解，混合效应模型）
2. 基于这些方差成分，一次可靠的 agent 评测需要每题重复几次、需要多少道题（功效分析 / 泛化理论），并产出一个可运行的实验设计计算器

本仓库的 agent 和评测代码还会被后续两个项目复用：轨迹 LoRA 蒸馏对一致性的影响（含熵匹配对照），以及 ALFWorld 上的知识增强具身 agent。



## 硬件与运行环境

- 开发机：MacBook（M4 Pro）。本地模型用 Ollama，接口为 `http://localhost:11434/v1`
- 实验机：RTX 5070（12GB）台式机，WSL2 + vLLM，通过 Tailscale 远程访问。BF16 只能跑约 4B 以下模型，7B 只能用 INT8 / INT4
- 同一组精度对比必须在同一平台内完成，避免跨平台混淆
- 所有模型调用统一用 `openai` Python 客户端，`base_url`、`model`、`api_key` 一律从配置文件或环境变量读取，切换机器只改配置不改代码

## 代码规范

- Python 3.11+，用 uv 管理环境和依赖
- 代码、注释、变量名、commit message 用英文
- 加类型标注，函数保持短小，一个函数只做一件事
- agent 部分**不使用** LangChain、LlamaIndex 等框架，全部手写
- 每次运行的完整轨迹（每步输入、输出、工具调用与结果、token 数、耗时）保存为 JSONL，放在 `runs/` 下
- 所有随机过程固定种子，保证可复现
- API 密钥只放 `.env`，确保 `.env` 和 `runs/` 中的大文件在 `.gitignore` 里
- 每完成一个有意义的里程碑，用清楚的 commit message 提交一次

## 目录结构

- `src/agentrig/`：可安装的包（agent/、tools/、eval/、llm/、config.py），后续项目用 `pip install git+<仓库地址>` 复用
- `configs/`：`endpoints.yaml`（在哪跑）、`models.yaml`（跑什么）、`experiments/`（怎么跑）
- `data/frozen/`：冻结的 300 题评测集 + manifest，禁止修改；`data/raw/` 不进仓库
- `runs/`：原始轨迹 JSONL，不进仓库，用 `scripts/sync_runs.sh` 从 5070 同步
- `analysis/`：统计分析脚本（`NN_name.py`）；`make figures` 依次运行，输出到 `results/` 和 `figures/`
- `scripts/`：冻结、连通性检查、vLLM 启动、轨迹同步
- `docs/`：agent_design、decisions、experiments/、literature/

