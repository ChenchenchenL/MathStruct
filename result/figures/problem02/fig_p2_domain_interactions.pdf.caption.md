# 图注：跨尺度配比强度校准与领域单纯形 Hessian 互补/替代交互结构 (fig_p2_domain_interactions)

- **数据来源**：附件 A 真实配方实验与独立检验集（A4–A11，含 1M、60M、1B 尺度）。
- **左图说明**：配比引起的损失变异系数（CV）与标准差随训练尺度的收缩演化，采用双纵轴严格区分量纲：左纵轴（海军蓝）为无量纲变异系数 CV (%)，右纵轴（深红）为损失绝对标准差（Nats）。随着参数与 Token 规模由 1M 扩大至 1B，配方敏感度由 5.27% 稳步平滑收敛至 2.35%（收缩比为 0.447）。基于模拟矩法（MSM）推导线性标定 $\tau_{\text{linear}} = 0.869$，非线性精确标定解为 $\tau^* = 0.835 \approx 0.85$。
- **右图说明**：8 个核心知识领域的单纯形 Hessian 二阶交互曲率矩阵 $\mathcal{H}^R_{ij} = \frac{\partial^2 R}{\partial p_i \partial p_j}$（折算至模型损失需乘以外在系数 $T \approx 0.3 \sim 0.6$）。蓝色区块（负曲率）代表两领域具有超加性的协同互补效应；红色区块（正曲率）代表竞争替代效应。
- **核心结论**：
  1. 配比乘子满足 $R(p) = \exp\left\{ \tau \frac{f(p) - f(p_0)}{f(p_0)} \right\}$，在 The Pile 默认配方 $p_0$ 处精确退化为 $R(p_0) = 1.000000$。
  2. 实证判别发现显著的领域互补协同关系：Top-1 为 stackexchange 问答与 pile_cc 通用语料（$\mathcal{H}^R_{ij} = -9.00$），Top-2 为 arxiv 科技文献与 uspto_backgrounds 专利背景（$\mathcal{H}^R_{ij} = -6.60$）；arxiv 与 github 代码呈现温和协同（$\mathcal{H}^R_{ij} = -1.47$，排全域第 28 位）。
  3. 显著的竞争替代关系包括：pile_cc 与 uspto_backgrounds（Top-1 竞争，$\mathcal{H}^R_{ij} = 8.78$）、enron_emails 与 hackernews（Top-2 竞争，$\mathcal{H}^R_{ij} = 4.96$）以及 arxiv 与 pile_cc（$\mathcal{H}^R_{ij} = +4.28$）；而 pile_cc 与 gutenberg 书籍语料交互接近中性独立（$\mathcal{H}^R_{ij} = -0.16$）。
