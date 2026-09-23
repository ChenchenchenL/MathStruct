# 图注：跨尺度配比强度校准与领域单纯形 Hessian 互补/替代交互结构 (fig_p2_domain_interactions)

- **数据来源**：附件 A 真实配方实验与独立检验集（A4–A11，含 1M、60M、1B 尺度）。
- **左图说明**：配比引起的损失变异系数（CV）与标准差随训练尺度的比较。左纵轴为 CV (%)，右纵轴为损失标准差（Nats）；三组数据分别对应 1M、60M 和 1B 检验集。
- **右图说明**：8 个核心知识领域的单纯形 Hessian 二阶交互曲率矩阵 $\mathcal{{H}}^R_{{ij}} = \frac{{\partial^2 R}}{{\partial p_i \partial p_j}}$（折算至模型损失需乘以外在系数 $T \approx 0.3 \sim 0.6$）。横轴缩写 StackEx、Pile-CC、USPTO、HN 分别对应 `stackexchange`、`pile_cc`、`uspto_backgrounds`、`hackernews`；蓝色区块（负曲率）表示协同互补效应，红色区块（正曲率）表示竞争替代效应。
- **说明**：配比标定参数、Hessian 数值和领域对排序见 `EXP-20260923-P2-03` 记录与配套结果表。
