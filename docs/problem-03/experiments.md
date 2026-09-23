# experiments.md — 问题三实验设计与正式运行规程

> 本文档依据《数学建模项目总规范》建立，详述问题三的全部 5 项正式实验（EXP-301 ~ EXP-305）的输入数据、模型公式、实验配置、评价指标与产出物规范。

---

## 实验清单概览

| 实验编号 | 实验名称 | 对应模型公式 | 核心考察内容 | 产出物文件 |
|---|---|---|---|---|
| **EXP-301** | 三档典型预算基准联合优化 | M3-EQ01, M3-EQ02, M3-EQ03 | $C \in \{10^{19}, 10^{22}, 10^{24}\}$ FLOPs 下，在 3 类成本函数及两种固定配比情景下的最优 $(n^*, d^*, Q^*)$、算力支出拆解及最优 Loss；并在 $\tau \in [0.45,1.50]$ 的五点情景网格上分析敏感性 | `tab_p3_budget_allocation.csv`<br>`tab_p3_tau_sensitivity.csv`<br>`fig_p3_pareto_frontiers.pdf/png` |
| **EXP-302** | 连续对数预算谱扫描与结构性转移识别 | M3-EQ02, M3-EQ03, M3-EQ05 | $\bar{C} \in [1, 10^7]$（50 个对数采样点）连续扫描，追踪 $Q^*(C)$ 区制跃迁、支出份额演化曲线、弹性漂移及扫描点重抽样稳定性 | `tab_p3_structural_shift_metrics.csv`<br>`fig_p3_structural_shift.pdf/png` |
| **EXP-303** | 上下文长度与长文本注意力敏感性分析 | M3-EQ04 | 依托 C7 提取的 5 类典型架构窗口（2048, 4096, 8192, 32768, 131072），围绕解析临界值 $L_{\text{ctx}}^{\text{crit}}=30000$ 评估注意力算力挤占效应与配置重构 | `tab_p3_context_sensitivity.csv`<br>`fig_p3_context_scaling.pdf/png` |
| **EXP-304** | 成本函数几何形态深度对比与基准对齐消融 | M3-EQ01, M3-EQ03 | 原始量纲对比 vs 在参考提升 $\Delta Q = 0.1$ 处单位成本归一化（$\bar{h}(0.684)=\text{const}$）的“纯函数形态（凸/凹/线性）”消融实验，剥离量纲假象与本征几何差异 | `tab_p3_cost_comparison.csv`<br>`fig_p3_cost_geometry.pdf/png` |
| **EXP-305** | 假设参数扰动情景分析与相对数值参考解后悔度 | M3-EQ01, M3-EQ02 | 根据问题二边际标准误独立生成 1000 组参数情景；每档预算评估固定策略损失，并对前 100 个情景求数值参考解后计算子样本后悔度 | `tab_p3_regret_analysis.csv`<br>`fig_p3_regret_robustness.pdf/png` |

---

## 详细实验设计

### EXP-301：三档典型预算基准联合优化
- **实验目标**：精确求解赛题指定的 $C = 10^{19}, 10^{22}, 10^{24}$ FLOPs（归一化预算 $\bar{C} = 10, 10^4, 10^6$ EFLOPs）在默认上下文窗口 $L_{\text{ctx}} = 2048$ 下的最优配置。
- **实验矩阵**：
  - 算力预算 3 档 $\times$ 成本函数 3 种（指数型、幂函数型、对数型）$\times$ 配比设置 2 类（基准配方 $p_0$ vs 固定最优推荐配方 $p^*_{\text{opt}}$）= 18 组正式运行。
  - **配比条件说明**：本实验评估的是**两种固定配比情景**（$p_0$ 与固定推荐配比 $p^*_{\text{opt}}$），而非连续联合寻优 $p$。其中 $p^*_{\text{opt}}$ 的效果以问题二标定的 $\tau=0.85$ 为名义校准值（$R(p^*_{\text{opt}}) \approx 0.9602$），属于跨数据源迁移假设下的条件结果，而非独立识别出的普适最优配比。
  - **$\tau$ 敏感性子实验**：在情景范围 $\tau \in [0.45, 1.50]$ 上采用五点网格 $\{0.45,0.65,0.85,1.15,1.50\}$，重算 $R(p;\tau)$ 及最优配置；该范围不是置信区间，也未对网格间所有连续值求解。
- **求解算法**：
  - 主求解器：多起点 L-BFGS-B（50 组对数均匀初始网格，容差 $10^{-15}$）；
  - 全局复核：差分进化（Differential Evolution，种群 20，迭代 2000，种子 42）。
- **记录指标**：最优参数量 $n^*$ (B)、最优 Token 数 $d^*$ (B)、最优质量 $Q^*$、最优损失 $L^*$、基础训练支出 $s_{\text{train}}$、注意力支出 $s_{\text{attn}}$、清洗支出 $s_Q$、预算残差 $|C_{\text{spent}} - C|/C$。
- **交付表格**：`result/tables/problem03/tab_p3_budget_allocation.csv` 与 `result/tables/problem03/tab_p3_tau_sensitivity.csv`。

---

### EXP-302：连续对数预算谱扫描与结构性转移识别
- **实验目标**：对算力预算在 $\bar{C} \in [1, 10^7]$（$C \in [10^{18}, 10^{25}]$ FLOPs）范围内生成 50 个对数均匀间隔的采样点，追踪最优轨迹的连续演变。
- **核心检验**：
  1. **边界区制判别**：以容差 $\varepsilon_Q = 0.005$ 标记 $Q^*$ 处于区制 I（粗放）、区制 II（集约）、区制 III（饱和）的分界预算 $C_{\text{trans1}}, C_{\text{trans2}}$；
  2. **支出份额变点分析**：对 $s_Q(\ln \bar{C})$ 执行两相分段折线回归（Broken-stick Model），网格搜索折点，并对 50 个确定性扫描点作 1000 次有放回重抽样，报告探索性重抽样区间。因扫描曲线不是独立随机样本，该区间不解释为总体置信区间，也不据此作显著性检验；
  3. **对数弹性偏离度**：数值微分计算 $\kappa_n(\bar{C}) = \frac{d \ln n^*}{d \ln \bar{C}}$ 与 $\kappa_d(\bar{C}) = \frac{d \ln d^*}{d \ln \bar{C}}$，与 Chinchilla 常数弹性（0.4515, 0.5485）对比。
- **交付文件**：`result/tables/problem03/tab_p3_structural_shift_metrics.csv` 与 `result/figures/problem03/fig_p3_structural_shift.pdf/png`。

---

### EXP-303：上下文长度与长文本注意力敏感性分析
- **实验目标**：全面评估附件 C7 提取的 5 档典型架构上下文窗口（2048, 4096, 8192, 32768, 131072）对最优资源配置与模型能力的重构效应。
- **关键物理指标**：
  - 相对于基础训练的注意力比率：$\psi(L_{\text{ctx}}) = \frac{\eta L_{\text{ctx}}}{6} = \frac{L_{\text{ctx}}}{30000}$；
  - 有效训练算力折扣因子：$\phi(L_{\text{ctx}}) = \frac{1}{1 + \psi(L_{\text{ctx}})}$；
  - 考察在不同窗口下，各预算档位的最优参数量 $n^*$ 与最优数据量 $d^*$ 的相对缩水幅度与损失抬升量 $\Delta L(L_{\text{ctx}})$。
- **交付文件**：`result/tables/problem03/tab_p3_context_sensitivity.csv` 与 `result/figures/problem03/fig_p3_context_scaling.pdf/png`。

---

### EXP-304：成本函数几何形态深度对比与基准对齐消融
- **实验目标**：解耦附录 B.1 中三类成本函数的“参数量纲尺度”与“函数本征凸凹性”。
- **实验设置**：
  - 组别 A（原始量纲）：直接运行默认参数 $\gamma, \lambda$；
  - 组别 B（$\Delta Q = 0.1$ 形态对齐）：调整幂函数与对数函数的比例系数 $\gamma'$，使得在基准提升点 $Q = Q_0 + 0.1 = 0.684$ 处的增量清洗成本完全等于指数型增量成本：
    $$\bar{h}_{\text{pow}}(0.684) = \bar{h}_{\text{log}}(0.684) = \bar{h}_{\text{exp}}(0.684) \approx 0.02738\text{ GFLOPs/Token}$$
  - 在对齐后对比三者的边际成本曲线 $\bar{h}'(Q)$、各预算档位下的 $Q^*$ 演变及最优资源投入比例。
- **交付文件**：`result/tables/problem03/tab_p3_cost_comparison.csv` 与 `result/figures/problem03/fig_p3_cost_geometry.pdf/png`。

---

### EXP-305：假设参数扰动情景分析与相对数值参考解后悔度
- **实验目标**：评估在给定参数扰动设定下名义资源配置的损失变化和相对数值参考解的后悔度。本实验是情景敏感性分析，不是经验后验协方差传播。
- **算法流程**：
  1. 围绕 $\theta_{\text{nom}}=(A,\alpha,B,\beta,\rho)$，以问题二报告的边际标准误为标准差，按独立正态假设生成 1000 组扰动情景；参数非正时拒绝并重抽。未使用参数协方差，因此忽略参数间相关性。
  2. 对 1000 组情景下的名义策略和保守数据偏重策略计算损失；经验 2.5%–97.5% 区间是情景分布分位区间，不是置信区间。
  3. 仅对前 100 个情景调用 `verify_global=False` 的多起点局部优化器求数值参考损失 $\widehat L^*(\theta^{(m)})$，并计算子样本后悔度 $\widehat{\mathcal R}^{(m)}=L(x_{\text{policy}};\theta^{(m)})-\widehat L^*(\theta^{(m)})$。
  4. 报告该 100 个子样本的均值、95% 分位数及样本最大值。由于参考解不是经过全局最优性证明的 oracle，最大值不称为 Minimax 上界或全样本最优后悔度。
- **交付文件**：`result/tables/problem03/tab_p3_regret_analysis.csv` 与 `result/figures/problem03/fig_p3_regret_robustness.pdf/png`。
