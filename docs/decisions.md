# 决策日志（只追加，不删改；推翻旧决定时新增一条并注明）

格式：日期 | 决定 | 理由 | 依据

---

2026-10-01 | 放弃原选题 A（复现一致性研究 + 量化维度） | 新颖性低；关闭 prefix cache 时重复运行可能零方差 | 2609.04748、2602.11619；文献梳理 docs/literature/
2026-10-01 | 项目一倾向选题 C+D（方差成分分解 + 重复次数功效分析），待预实验确认 | 不怕零结果；吸收了量化 × 缓存的交互项 | docs/literature/agent_reliability_litreview.csv
2026-10-01 | 项目二保留选题 B，必须加熵匹配对照 | 避免"一致性提升只是分布变尖"的混淆 | 2604.14877、2603.16843
2026-10-01 | 去掉 GSM8K，只用 HotpotQA（distractor） | 与已有工作可比；减少工作量 | —
2026-10-01 | 同一组精度对比必须在同一平台完成；5070 跑 ≤4B 的完整精度阶梯 | 避免跨平台混淆；7B BF16 放不进 12GB | —
2026-10-02 | 评测集冻结：hotpotqa/hotpot_qa distractor validation（revision 1908d6af），按 sha256("{seed}:{id}") 排序取前 300，seed=20261001；含 229 bridge / 71 comparison | 哈希排序不依赖 Python 随机数实现，跨机器可复现；manifest 记录 SHA256，pytest 与 pre-commit 校验 | data/frozen/*.manifest.json
2026-10-02 | 配置分三层：endpoints.yaml（在哪跑）/ models.yaml（跑什么）/ experiments/（怎么跑）；切换平台只改 experiment 的 endpoint | 同一模型在两平台名称不同；预实验要组合缓存 × 温度 | —
2026-10-02 | analysis/ 脚本按 NN_name.py 编号，只读 runs/、data/frozen/、results/，只写 results/、figures/；make figures 一键重建 | 分析过程仅写 results/ 和 figures/；结果可完全由脚本复现 | —
2026-10-02 | 开发阶段关闭 Qwen3 思考模式；正式实验是否开启待预实验后决定 | 减少开发调试的延迟与输出噪声 | —
2026-10-02 | Mac 上的 Ollama 只用于开发调试，不作为正式实验平台 | Ollama 封装了缓存、批处理、量化等细节，难以精确控制和记录 | —
2026-10-02 | 模型选原版混合 Qwen3-4B（Ollama qwen3:4b-q4_K_M，HF Qwen/Qwen3-4B），不用 2507 拆分版 | 同一权重即可切换思考开/关，避免模型与思考模式混淆；官方有 FP8/AWQ 等量化版可做精度阶梯 | Ollama 实测：qwen3:4b = Thinking-2507，思考无法关闭
2026-10-02 | 思考开关实现：Ollama 用 reasoning_effort="none"（实测唯一能干净关闭的方式；think:false、/no_think、chat_template_kwargs 均无效），vLLM 用 chat_template_kwargs.enable_thinking；check_endpoint.py 做回归检查 | 两平台参数不同，由 llm/client.py 统一转换 | Ollama 0.35.0 实测
2026-10-02 | vLLM 启动约定：`--dtype bfloat16 --reasoning-parser qwen3`，前缀缓存 on/off 显式传参；依赖统一用 `uv sync/run --locked`；check_endpoint 以 finish_reason、非空回答、思考字段（reasoning/reasoning_content）做 PASS/FAIL 判定并可保存原始请求与响应 | reasoning parser 只把思考内容移到 reasoning 字段（与 Ollama 形状一致），是否生成思考仍由 enable_thinking 控制；--locked 防止擅自重锁 | scripts/serve_vllm.sh；scripts/check_endpoint.py




2026-10-02 | agent 采用文本 ReAct 格式（Thought / Action: search[...] / Final Answer），本地正则解析，stop=["Observation:"] | 透明、跨平台一致；格式错误本身是方差数据；不用约束解码以免改变采样分布 | docs/agent_design.md
2026-10-02 | 模型不搜索直接回答时接受并记录 n_searches；格式错误给提示后继续至 max_steps | 不干预 agent 自然行为，行为差异作为数据记录 | docs/agent_design.md
2026-10-02 | 轨迹每次运行一行 JSONL，记录全部发送的采样参数、git commit、PROMPT_VERSION；每次运行种子 = sha256(实验种子, 题目, 重复)，第 t 步 seed = 种子 + t | 可复现、可断点续跑、可追溯 | docs/agent_design.md
2026-10-03 | E00 预实验改为 40 题 × 每题 10 次（5 个种子 × 同种子重复 2 次）× 温度 {0, 0.7} × 缓存 {开, 关}；采样参数 top_p/top_k 仍为暂定 | 彩排显示方差集中在少数题目上，增加题数比增加重复更有用；同种子重复用来把服务栈非确定性与采样方差分开 | docs/experiments/E00R_rehearsal.md
2026-10-03 | serve_vllm.sh 固定 VLLM_USE_FLASHINFER_SAMPLER=0（vLLM 原生采样后端） | FlashInfer JIT 采样器要求与 wheel 匹配的 CUDA toolkit；采样后端属于服务配置，固定在脚本里避免手动设置被遗漏 | scripts/serve_vllm.sh
2026-10-03 | 采样参数：正式实验以 Qwen3 官方非思考推荐为主（top_p 0.8、top_k 20，温度为实验因素）；E01 起把"采样方案"作为因素，加入 top_p=1、不设 top_k 的对照 | 贴近真实使用，结论有外部效度；对照用于检验截断是否压低方差（彩排中已有迹象） | docs/experiments/analysis_plan.md
2026-10-03 | 主结局 EM（F1 为次结局）；主模型为线性混合模型（EM 作 0/1 线性概率），logit GLMM 作敏感性分析 | Q2 关心的是观测尺度上准确率的标准误，线性模型的方差成分可直接用于 D-study；GLMM 复核方向 | docs/experiments/analysis_plan.md
2026-10-03 | 统计分析用 R（lme4 / glmmTMB），Mac 上用 Homebrew 安装；make figures 同时运行 analysis/NN_*.R | 交叉 + 嵌套随机效应在 R 中最成熟 | docs/experiments/analysis_plan.md
2026-10-03 | "可靠"的主定义：同一组题上两个条件相差 3 个百分点时功效 0.8（α=0.05）；同时报告标准误随 (题数, 重复次数) 的曲线和排序翻转概率 | 3pp 约为常见"有意义改进"的下限；2pp 所需样本超出单卡算力；排序翻转概率可与 2602.11619 的 29.3% 直接对比 | docs/experiments/analysis_plan.md

2026-10-03 | 对外发布采用审查通过的快照和独立的单一初始提交；开发仓库保留完整历史 | 发布内容与已审查清单逐文件一致，后续开发独立进行 | 发布流程
2026-10-03 | E01 分两阶段：E01a = BF16 × 300 题 × 缓存{开,关} × 实验臂{greedy 2 次, qwen 6 次, topp1 6 次}（8,400 次运行）；E01b = 精度阶梯，每个精度 greedy + qwen × 缓存{开,关}（档位在可行性调研后确定） | E00 显示方差主要在题目间、温度 0.7 下 3–5 次重复能明显提高功效；分阶段让每阶段一晚内跑完 | docs/experiments/E01_design.md
2026-10-03 | 方差成分 CI 改用按题目重抽样的 bootstrap；比较条件的 D-study 改为按具体对比估计交互方差 | EM 为 0/1 且大量题目全对或全错，参数 bootstrap 的正态假设不成立；具体对比的交互方差比合并估计更贴合 Q2 | docs/experiments/analysis_plan.md 第 6 节修订
2026-10-03 | 缓存对照通过配对脚本先运行 on 再 off，成功后恢复 on；任何失败保留现场、不自动重试或恢复 | 开关以真实服务配置核验，避免把未完成或条件不符的结果继续用于下一轮 | scripts/run_cache_pair.sh


2026-10-03 | 精度阶梯定为 BF16 / FP8（Qwen/Qwen3-4B-FP8）/ INT4（Qwen/Qwen3-4B-AWQ），全部在 5070 上；E01b 用与 E01a 相同的种子基数，BF16 复用 E01a 数据 | 官方提供的量化版本只有 FP8 与 AWQ；同种子使精度比较在题目和种子上配对 | docs/experiments/E01_design.md
2026-10-03 | H4 增加 TOST 等价检验（界限 ±3 个百分点，Holm 校正） | "不显著"不能证明"没变化"（参照 2607.27275）；界限与"可靠"的定义（3 pp）一致 | docs/experiments/E01_design.md

2026-10-03 | 官方 FP8 在本机验证使用 VLLM_USE_DEEP_GEMM=0 的 CUTLASS block-FP8 路径，保留 BF16 非量化层和 KV；FP8 模型通过端点验证 | 默认 DeepGEMM 找不到 CUDA toolkit，CUTLASS 两项端点检查 PASS；无需安装工具链 | docs/experiments/E01_design.md 偏离记录

2026-10-03 | E01b 的 FP8 / INT4 服务固定 KV cache 为 1,361 块（21,776 token，--kv-blocks / --num-gpu-blocks-override），与 E01a 的 BF16 服务相同 | 不固定时容量随权重变小而增大（BF16 21.8k、FP8 41.3k、AWQ 58.6k token），前缀缓存的命中与淘汰会随精度变化，与 H3 的精度效应混淆 | 2026-10-03 晚实测；E01_design.md 偏离记录
2026-10-03 | AWQ 服务用 --dtype bfloat16（与 BF16、FP8 一致），Marlin 内核；两项 check_endpoint 通过 | 本机 AutoAWQ 支持 BF16 激活，统一 dtype 避免非量化层精度不同 | 端点验证原始响应（不公开）；docs/experiments/E01_design.md
2026-10-04 | E01a、E01b 结论确认，包括两条探索性发现：缓存开关会改变 15–77% 题目的贪心轨迹（随精度而异，属确定性计算路径效应）；逐字复现需要同一个 vLLM 编译产物 | 跨启动补测：同一编译产物 FP8/AWQ 100/100 相同，BF16 换编译产物后缓存关 10/50 题不同 | docs/experiments/E01a_results.md、E01b_results.md
2026-10-04 | 解析器：项目一保持 react-v1 的严格行首格式（Final Answer / Action 须在行首）作为主结局，不改；宽松解析（接受行内 "Final Answer:"）的反事实 EM 作为每次分析的标准敏感性输出；项目二设计时再定 | 改解析器等于改 agent（回合会提前结束），已有 19,600 次运行不可比；反事实在第一次格式错误处精确成立，已能给出影响大小（AWQ 差距约三成） | E01b_results.md 探索性发现 1
2026-10-04 | 复现约定：同一组对比的所有条件使用同一条启动命令（含 served-model-name、模型路径写法、kv-blocks）；实验记录保存完整启动命令和 vLLM 编译缓存哈希（torch_compile_cache） | 启动命令变化会换编译产物、改变输出，与打开缓存的影响同一量级 | E01b_results.md 探索性发现 3
