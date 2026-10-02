# 文献

- agent_reliability_litreview.csv：2026-10-01 Claude Science 文献梳理
- notes/：精读笔记，每篇一个文件，命名为 arXiv 编号，如 2602.11619.md

注意：2603.25764 在梳理中被误判为"不存在"，实际存在（Consistency Amplifies，SWE-bench，Mehta 2026）。

## 新增相关工作
- 2610.00447 Frozen Scenes, Shifting Winners（Lyu 组，2026-09）：评测配置方差超过模型间方差，与本项目方法最接近。笔记见 notes/2610.00447.md（仅基于摘要）。
- 精读笔记（2026-10-03）：notes/2602.11619.md（HotpotQA 重复运行一致性，最直接前人工作）、notes/2609.04748.md（缓存 × 量化分歧；关缓存贪心位级可复现）。
- 被引检索（Semantic Scholar，2026-10-03）：2602.11619 被 3 篇引用，相关的是 2607.26587 One Run Is Not an Idea（用 ICC 和冠军反转衡量单次运行不可靠）；2607.27275 被 2608.06564 引用（量化线性压缩决策 margin，可解释量化放大分歧）；2609.04748 暂无被引。
