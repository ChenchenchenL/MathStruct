# experiments.md — 问题四实验设计与执行规程

> 本文档依据《数学建模项目总规范》建立，详述问题四（技术演进分析与前沿预测）的 5 项正式实验（EXP-401 ~ EXP-405）的输入数据、模型公式、评价指标及产出文件对应关系。全面吸收全部审查意见，严格对齐数据约束与证据底线。

---

## 实验清单概览

| 实验 ID | 实验名称 | 对应核心设问与审查项 | 核心模型与算法 | 核心输出图表 |
|---|---|---|---|---|
| **EXP-401** | 数据三级匹配、C8 全基准微观复核与综合能力度量 | 样本三级匹配漏斗 (P0-1)、全 6 项基准 C8 复核 (P2-10)、机会水平实证反演 (P0-3)、C3 历史 0 分屏蔽、Merges 分布对照 (P2-15) | 级联匹配算法、有效样本量加权平均、机会水平线性反演、分布诊断 | `tab_p4_sample_filtering_summary.csv`<br>`tab_p4_c8_bbh_reconstruction.csv`<br>`tab_p4_c8_all_benchmarks_audit.csv` |
| **EXP-402** | Loss–Benchmark 连续单调饱和桥接与高端外推管控 | 损失到能力映射、反制 T20 假相关、覆盖域与外推区界定 (P0-2)、Type 分层拟合 (P1-6)、6 项单维外推对照 (P0-2) | 有界 Logistic 饱和曲线、Type 分层估计、Profile 似然敏感带、LOFOCV | `tab_p4_bridge_parameters.csv`<br>`tab_p4_bridge_extrapolation_risk.csv`<br>`fig_p4_loss_benchmark_bridge.pdf/png` |
| **EXP-403** | 有界动态随机前沿估计与反事实 Shapley 贡献分解 | 动力学前沿面估计、规模弹性识别 (P1-7)、Shapley 贡献拆解 [2024-06, 2025-03]、去趋势对照 | 动态随机前沿 (SFA)、状态空间平滑、反事实 Shapley 排序平均分解 | `tab_p4_shapley_contribution.csv`<br>`fig_p4_dynamic_frontier_history.pdf/png` |
| **EXP-404** | 算力放缓下未来 12/24 个月前沿预测与双路径闭环 | 三档算力放缓情景、逐月连续轨迹带 (P2-12)、Pretrained/Chat 分层预测、双路径对偶校验与口径自洽 (P1-8) | 几何算力增速、路径 A (动态前沿) vs 路径 B (标度律+桥接)、超大模型外推支撑 | `tab_p4_frontier_forecast_12_24m.csv`<br>`tab_p4_monthly_forecast_trajectory.csv`<br>`fig_p4_forecast_scenarios_dual_path.pdf/png` |
| **EXP-405** | 多层级不确定性传递、时间外滚动回测与敏感性检验 | 族群 Bootstrap (1000 次) 收敛性记录 (P2-13)、外推带误差传递 (P0-2)、滚动 OOT 回测、C3 长周期对照 (P2-14) | 族群聚类 Bootstrap、滚动时间窗口回测、敏感性稳健分析 | `tab_p4_oot_validation_metrics.csv`<br>`tab_p4_bootstrap_uncertainty_summary.csv`<br>`fig_p4_uncertainty_decomposition.pdf/png` |

---

## 详细实验规程

### EXP-401：数据三级匹配、C8 全基准微观复核与综合能力度量
- **目的**：
  1. 建立 L1(C2官方桥接) → L2(规范化精确碰撞) → L3(家族参数规则匹配) 级联流水线，逐级统计命中数，明确区分许可证开源池与算力交集池；
  2. 针对 C8 原始 JSON，不仅复核 BBH 24 项子任务，而且扩展至全部 6 项基准（math_hard 7 项、GPQA 3 项、MUSR 3 项、IFEval、MMLU-PRO），自底向上完成有效样本量加权复算；
  3. 实证反演线性回归方程，严密推导机会水平 $\gamma$（正式固化 BBH 约 28%、GPQA 25%、MMLU-PRO 10% 等基线）；
  4. 诊断 C3 中 2019–2022 年历史 0 分的结构性未开考占位符属性并屏蔽；输出 Merges（1724 席）与主池分布对照。
- **输入数据**：`leaderboard_cleaned.csv` (C1), `leaderboard_enhanced.csv` (C2), `leaderboard_extended_timeseries.csv` (C3), `epoch_all_ai_models.csv` (C4), `detailed_results/` (C8)。
- **评价指标**：匹配命中率、各基准重建 $R^2$、均方根误差 $\text{RMSE}$、反演机会水平 $\gamma$。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_sample_filtering_summary.csv`；
  - 表格：`result/tables/problem04/tab_p4_c8_bbh_reconstruction.csv`；
  - 表格：`result/tables/problem04/tab_p4_c8_all_benchmarks_audit.csv`。

---

### EXP-402：Loss–Benchmark 连续单调饱和桥接与高端外推风险管控
- **目的**：
  1. 基于 C6（75 席）核查并汇报 Pretrained 与 Chat 的样本构成，引入 Type 哑变量分层拟合单调 Logistic 曲线；
  2. 显式声明数据覆盖域 $[L_{\min}, L_{\max}] \times [S_{\min}, S_{\max}] = [1.65, 2.84] \times [3.80, 47.98]$，标记 $S > 47.98$ 为外推区；
  3. 对上限参数 $a_k$ 构建 Profile 置信带，测算 $S \in \{55, 60, 70, 80\}$ 处由于外推导致的映射区间宽度；
  4. 对比“6 项单维基准各自桥接外推后再合成”与“综合分直接外推”，评估饱和形状稳定性；
  5. 严格验证 $b_k > 0$ 及 Spearman 秩相关系数 $\rho < 0$，反制出题组 T20 假相关。
- **核心模型**：M4-EQ04。
- **评价指标**：$R^2$、RMSE、Spearman $\rho$、LOFOCV MAE、外推置信带宽。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_bridge_parameters.csv`；
  - 表格：`result/tables/problem04/tab_p4_bridge_extrapolation_risk.csv`；
  - 图表：`result/figures/problem04/fig_p4_loss_benchmark_bridge.pdf/png`。

---

### EXP-403：有界动态随机生产前沿估计与反事实 Shapley 贡献分解
- **目的**：
  1. 在连续 Logit 空间中估计有界动态随机生产前沿（SFA），利用开源池横跨 8 个数量级的算力变异高置信度估计算力规模弹性 $b$；
  2. 估计月度非规模技术状态序列 $\{z_t\}$、对齐微调抬升 $v_{\text{chat}}$ 与家族固定效应；
  3. 在 $t_0 = 2024\text{-}06$ 至 $t_1 = 2025\text{-}03$ 窗口内运行反事实 Shapley 排序平均分解，量化规模与技术的绝对百分点与贡献占比；
  4. 实施去趋势回归对照以验证 $b$ 与 $z_t$ 的识别稳健性；对比 95% 高分位面板回归。
- **核心模型**：M4-EQ05 ~ M4-EQ08。
- **评价指标**：对数似然 $\ln \mathcal{L}$、前沿非效率方差比 $\gamma_{\text{SFA}}$、Shapley 贡献百分点与置信区间。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_shapley_contribution.csv`；
  - 图表：`result/figures/problem04/fig_p4_dynamic_frontier_history.pdf/png`。

---

### EXP-404：算力放缓下未来 12/24 个月前沿预测与双路径闭环
- **目的**：
  1. 测定历史前沿算力年化几何增速 $g_C$；设定“历史动量”、“增速减半”、“算力停滞”三档条件情景；
  2. 自 $T_0 = 2025\text{-}03\text{-}13$ 起预测至 2026-03（+12m）与 2027-03（+24m），生成未来 24 个月的逐月连续前沿演进轨迹带；
  3. 执行双路径闭环比对：路径 A（动态前沿直接推演）vs 路径 B（将情景算力输入第三问模型求解理论最优损失 $L^*(C)$ 后经式 (10) 桥接转化为 Benchmark 得分）；
  4. 声明路径 B 的大尺度定性对偶印证定位，引用第二问 B10 实验超大模型拟合自洽性（$R^2=1.0$）支撑标度律在 $10^{26}$ 量级的适用性，量化两条路径的历史吻合度与技术溢出差。
- **评价指标**：情景预测得分、逐月连续轨迹、双路径理论残差 $\Delta_{\text{dual}}$。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_frontier_forecast_12_24m.csv`；
  - 表格：`result/tables/problem04/tab_p4_monthly_forecast_trajectory.csv`；
  - 图表：`result/figures/problem04/fig_p4_forecast_scenarios_dual_path.pdf/png`。

---

### EXP-405：多层级不确定性传递、时间外滚动回测与敏感性检验
- **目的**：
  1. 采用族群聚类 Bootstrap（1000 次），层层传递 SFA 参数误差、状态噪声、桥接外推参数不确定性（包含路径 B 外推段放宽误差）与算力情景分歧，记录优化收敛成功率；
  2. 设定 $T_{\text{split}} = 2024\text{-}10\text{-}01$ 截断点进行时间外（OOT）滚动回测，统计 MAE 与 95% 区间实际覆盖率；
  3. 对 C3 历史 26 行可比子样本构造长周期（2019–2025）Shapley 对照，作为附录稳健性分析；
  4. 完成许可证口径（主池 vs 严格商用池 vs 宽松池）与时间轴（提交日 vs 发布日）的敏感性分析。
- **评价指标**：Bootstrap 优化收敛率、OOT MAE、95% 区间覆盖率、敏感性参数变动率。
- **交付产物**：
  - 表格：`result/tables/problem04/tab_p4_oot_validation_metrics.csv`；
  - 表格：`result/tables/problem04/tab_p4_bootstrap_uncertainty_summary.csv`；
  - 图表：`result/figures/problem04/fig_p4_uncertainty_decomposition.pdf/png`。

---

## 正式实验执行记录表（EXP-401 ~ EXP-405）

以下记录基于真实附件数据运行并导出的正式实验执行元数据，全面符合《数学建模项目总规范》记录先于运行、结果绝对可回溯要求：

| 实验编号 | 执行脚本入口 | 运行日期 | 随机种子 | 核心算法与收敛状态 | 关键实测核心指标 | 对应结果摘要文件 |
|---|---|---|---|---|---|---|
| **EXP-401** | `src/problem04/exp01_data_audit_aggregation.py` | 2026-09-23 | N/A (确定性) | 三级漏斗级联匹配 + OLS 微观反演 (100% 收敛) | L1=362, L2=23, L3=1305; 主分析池=1211; BBH $R^2=0.9967, \text{RMSE}=0.8875, \gamma=27.52\%$ | `result/problem04/exp01_summary.json` |
| **EXP-402** | `src/problem04/exp02_loss_benchmark_bridge.py` | 2026-09-23 | N/A (凸优化) | 分层加权非线性最小二乘 (100% 收敛) | Average $r=-0.5100 (p=2.95\times 10^{-6}), \rho=-0.5491, b=1.5367$; 6 科目 $b>0$; LOFOCV MAE=8.16 分 | `result/problem04/exp02_summary.json` |
| **EXP-403** | `src/problem04/exp03_dynamic_frontier_shapley.py` | 2026-09-23 | 42 | SFA 极大似然估计 (L-BFGS-B, 100% 收敛) | $\beta_C=+0.2174, v_{\text{chat}}=+0.2067, \gamma=87.3\%$; Chat Shapley: 规模 83.3% (+4.93分), 技术 16.7% (+0.99分) | `result/problem04/exp03_summary.json` |
| **EXP-404** | `src/problem04/exp04_forecast_deceleration_dual_path.py` | 2026-09-23 | 42 | 几何增速外推 + 问题三 2D KKT 联立求解 (100% 收敛) | $g_C=1.2652$/yr; +12m Chat: 35.91/32.81/29.85; +24m: 35.32/29.32/23.96; $\Delta_{\text{dual}} \in [0.43, 10.64]$ | `result/problem04/exp04_summary.json` |
| **EXP-405** | `src/problem04/exp05_uncertainty_oot_validation.py` | 2026-09-23 | 42 | 族群聚类 Bootstrap (1000次, 100% 收敛) + 滚动 OOT | 收敛率 100.0%; 6 个测试月 MAE=12.86 分; 95% 覆盖 4/6。89.0%–92.5% 是未计算占位符，不得引用。情景一 24 个月区间 [5.42, 81.67] 只说明不确定 | `result/problem04/exp05_summary.json` |

