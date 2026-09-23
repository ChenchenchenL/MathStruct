# experiments.md — 问题四实验设计与执行规程

> 本文档依据《数学建模项目总规范》建立，详述问题四（技术演进分析与前沿预测）的 5 项正式实验（EXP-401 ~ EXP-405）的输入数据、模型公式、评价指标及产出文件对应关系。

---

## 实验清单概览

| 实验 ID | 实验名称 | 对应核心设问 | 核心模型与算法 | 核心输出图表 |
|---|---|---|---|---|
| **EXP-401** | 数据清理、C8 逐任务聚合复核与综合能力度量 | 样本对象定义、许可证筛选、BBH 官方规则微观复算、C3 历史 0 分诊断 | 数据清洗管道、加权/宏平均比对、基线扣除回归 | `tab_p4_c8_bbh_reconstruction.csv`<br>`tab_p4_sample_filtering_summary.csv` |
| **EXP-402** | Loss–Benchmark 连续单调饱和桥接建模 | 损失到能力映射、反制 T20 假相关、误差分层与泛化检验 | 有界 Logistic 饱和曲线、LOFOCV、加权最小二乘 | `tab_p4_bridge_parameters.csv`<br>`fig_p4_loss_benchmark_bridge.pdf/png` |
| **EXP-403** | 有界动态随机前沿估计与反事实 Shapley 贡献分解 | 动力学前沿面估计、规模与非规模技术进步解耦及贡献占比 | 动态随机生产前沿 (SFA)、状态空间平滑、反事实 Shapley 分解 | `tab_p4_shapley_contribution.csv`<br>`fig_p4_dynamic_frontier_history.pdf/png` |
| **EXP-404** | 算力放缓下未来 12/24 个月前沿预测与双路径闭环 | 三档算力放缓情景外推、Pretrained/Chat 分层预测、与前三问闭环 | 几何算力增速情景、路径 A (动态外推) vs 路径 B (标度律+桥接) | `tab_p4_frontier_forecast_12_24m.csv`<br>`fig_p4_forecast_scenarios_dual_path.pdf/png` |
| **EXP-405** | 多层级不确定性传递、时间外滚动回测与敏感性检验 | 误差传播置信区间、时间外泛化检验、许可证/时间轴敏感性 | 族群聚类 Bootstrap (1000 次)、滚动时间窗口交叉检验 | `tab_p4_oot_validation_metrics.csv`<br>`fig_p4_uncertainty_decomposition.pdf/png` |

---

## 详细实验规程

### EXP-401：数据清理、C8 逐任务微观复核与综合能力度量
- **目的**：
  1. 严格落实开源权重与研究许可证筛选口径，建立包含完整算力与评测分的高置信样本池；
  2. 针对 C8 中 1863 个模型原始评测 JSON，自底向上读取 BBH 24 项子任务，验证样本量加权平均，并推导从 raw accuracy 到 C1 normalized score 的精确转换公式；
  3. 诊断 C3 中 2019–2022 年历史 0 分的结构性成因，构建科学屏蔽规则；
  4. 计算 6 项基准的宏平均综合分并完成各年份分布检验。
- **输入数据**：`leaderboard_cleaned.csv` (C1), `leaderboard_enhanced.csv` (C2), `leaderboard_extended_timeseries.csv` (C3), `epoch_all_ai_models.csv` (C4), `detailed_results/` (C8)。
- **评价指标**：BBH 重建判定系数 $R^2$、均方根误差 $\text{RMSE}$、过滤样本保留率。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_c8_bbh_reconstruction.csv`；
  - 表格：`result/tables/problem04/tab_p4_sample_filtering_summary.csv`。

---

### EXP-402：Loss–Benchmark 连续单调饱和桥接建模
- **目的**：
  1. 基于 C6/C5 桥接数据，拟合交叉熵验证损失 $L$ 到综合能力 $S$ 及 6 项单维 Benchmark 的有界 Logistic 饱和曲线；
  2. 严格核验 $b_k > 0$（单调递减性），生成彻底戳穿出题组 T20 假相关（$r=+0.68$）的实证证据链；
  3. 对高可比（Pythia 7 模型）与中可比（68 模型）样本分层拟合并报告留一家族交叉验证（LOFOCV）均方误差。
- **核心模型**：$S_k(L) = \frac{100}{1 + \exp(-(a_k - b_k L))}$。
- **评价指标**：$R^2$、RMSE、Spearman 秩相关系数 $\rho$、LOFOCV MAE。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_bridge_parameters.csv`；
  - 图表：`result/figures/problem04/fig_p4_loss_benchmark_bridge.pdf/png`。

---

### EXP-403：有界动态随机前沿估计与反事实 Shapley 贡献分解
- **目的**：
  1. 在连续 Logit 空间中估计有界动态随机前沿模型（SFA），估计算力规模弹性 $b$、月度非规模技术状态序列 $\{z_t\}$、对齐抬升效应 $v_r$ 与模型家族效应 $u_f$；
  2. 采用反事实 Shapley 排序平均分解法，分别在基座组与对齐微调组中精确量化规模扩张与非规模技术进步的绝对贡献点数与相对占比；
  3. 对比 95% 高分位数时间平滑面板模型，检验前沿轨迹稳健性。
- **核心模型**：M4-EQ05 ~ M4-EQ08。
- **评价指标**：对数似然值 $\ln \mathcal{L}$、残差方差拆解比 $\gamma = \sigma_d^2 / (\sigma_e^2 + \sigma_d^2)$、Shapley 贡献百分点与置信区间。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_shapley_contribution.csv`；
  - 图表：`result/figures/problem04/fig_p4_dynamic_frontier_history.pdf/png`。

---

### EXP-404：算力放缓下未来 12/24 个月前沿预测与双路径闭环
- **目的**：
  1. 测定历史前沿算力年化增速 $g_C$；设定惯性、减半、停滞三档条件情景；
  2. 以 2025-03-13 为起点，预测至 2026-03（+12m）与 2027-03（+24m）前沿上限；
  3. 执行双路径闭环比对：路径 A（动态前沿直接预测）vs 路径 B（将情景算力输入第三问模型求出最优损失 $L^*(C)$ 后经式 (10) 桥接转化为 Benchmark 得分）；
  4. 回测两条路径的历史吻合度并量化技术溢出差。
- **评价指标**：情景预测分、路径 A 与路径 B 残差 $\Delta_{A-B}$、各 Benchmark 细项突破值。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_frontier_forecast_12_24m.csv`；
  - 图表：`result/figures/problem04/fig_p4_forecast_scenarios_dual_path.pdf/png`。

---

### EXP-405：多层级不确定性传递、时间外滚动回测与敏感性检验
- **目的**：
  1. 采用族群聚类 Bootstrap（1000 次重采样），层层传递参数估计方差、评测度量噪声、桥接映射残差与算力情景差异，输出 80% 与 95% 条件前沿与单模型观测区间；
  2. 采用滚动历史截断点（如 2024-10-01 截断）进行真实时间外（Out-of-Time, OOT）回测，报告 MAE 与区间覆盖率；
  3. 完成许可证宽松度、时间轴（提交 vs 发布）、样本子集的敏感性检验。
- **评价指标**：OOT MAE、95% 区间覆盖率、敏感性参数漂移幅度。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_oot_validation_metrics.csv`；
  - 图表：`result/figures/problem04/fig_p4_uncertainty_decomposition.pdf/png`。
