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

**2026-09-25 版本说明**：上述 EXP-404/405 数值由 M4-v1.0 代码产生，保留为历史运行记录。`DynamicSFAModel.predict_frontier` 的状态斜率已按 M4-EQ12 修正；本次只做了只读诊断，没有重跑正式 EXP-404/405，也没有生成可供论文引用的新预测值。旧预测结果不能当作 M4-v1.1 输出。

---

## EXP-407：M4-v2 重构实验预登记（2026-09-25，首次正式运行前）

- **模型与数据**：M4-v2 / M4V2-EQ01 至 EQ05；`D4V2-C1/C2/C4`，C3 仅作不可比性审计。输入哈希见 `data-dictionary.md`；不读取《数据说明》的建模文字。
- **问题类别与比较**：历史规模/时间关联分解 + 连续纪录预测。主基线为上一月累计纪录，替代为训练期平均纪录增量，候选为条件算力位移。预测对象、许可、类型、月度时间轴和去重规则在 M4-v2 中冻结。
- **代码入口**：拟建 `src/problem04/exp07_frontier_v2.py`；只写 `result/problem04/v2/`、`result/tables/problem04/exp407_*`、`result/figures/problem04/exp407_*`，不覆盖 EXP-401 至 EXP-405。
- **代码版本**：运行时记录入口脚本 SHA256 与 Git HEAD；开发基准 HEAD `26757d768a19ee12e8ad6214cb53290a3386aef0`。模型超参：$\lambda_C=\lambda_t=\lambda_r=0.1,\lambda_f=5$；Theil–Sen 算力增长截在 $[0,\ln4]$；随机种子 20260925，仅用于参数敏感性抽样（若执行）。
- **环境**：Windows，Python 3.11.15，numpy 2.4.6，pandas 3.0.6，scipy 1.17.1，scikit-learn 1.9.1，matplotlib 3.11.2。
- **验证与指标**：先验证 C1/C2 身份、匹配日期、纪录单调与小样例 Shapley 加和；再在 2024-10 至 2025-03 的六个单月滚动起报点比较 MAE（分）和偏差，补报 $h=3$ 可评价窗口、纪录更新次数及失败月。若候选不胜基线，未来主点预测回退到胜出的基线；12/24 个月不报伪覆盖率。
- **预定输出**：逐行匹配审计、月度纪录、模型系数与历史分解、滚动回测、12/24 月情景表、摘要 JSON、运行清单、一个无图内总标题的矢量/PDF 图及配套元数据。每份结果标注 EXP-407、M4-v2、数据哈希和脚本哈希。
- **停止/失败条件**：C1/C2 身份不一致、未来日期匹配非零、有效模型不足 30 行、任一预测小于已知纪录、回测目标不一致或优化器报告失败时终止，不发布正式预测。旧实验结果保持原样。

### EXP-407 实际执行与失败判定（不可改写）

- **执行**：2026-09-25，`D:\Program Files\CondaEnvs\AIOPS\python.exe src/problem04/exp07_frontier_v2.py`；代码 SHA256 `1d5014c0c2dcb63be4b38705aa9b4b268591ad74ab825783a0f12835d9329a1d`，随机种子 20260925。完整依赖及输入哈希见 `result/problem04/v2/exp407_manifest.json`。
- **输入审计**：C1 4,576 行，入选 1,850 行，同名首次提交去重后 1,805 行；可信时间匹配 113 行、64 个 C4 身份，未来发布日期匹配数 0。
- **历史关联**：均值模型 Logit 空间加权 $R^2=0.6842$；Shapley 规模 7.47 分、时间关联 1.76 分，属于普通模型条件均值面的分解，**不得称为前沿贡献**。
- **同目标回测**：2024-10 至 2025-03 六个单月起报，累计纪录无更新；持平 MAE 0.00 分，平均增量 1.55 分，条件均值模型 0.58 分。三个月的四个可评价起报点也无更新，持平 MAE 0.00 分。候选未胜基线，未来主点预测回退到持平。12/24 月条件情景与重采样 10–90% 范围**没有预测覆盖率验证**。
- **失败原因**：条件均值不等同于前沿；后期纪录无更新使持平基线在所测窗口达到零误差。保留 EXP-407 和全部原结果，另立 M4-v2.1 / EXP-408 评估上分位替代模型。

## EXP-408：上分位前沿与 C3 审计预登记（首次运行前）

- **输入、目标和数据版本**：继承 EXP-407 的 D4V2-C1/C2/C4 同一哈希、C1 累计纪录目标、许可/类型/日期规则和六个月滚动起报；增加 D4V2-C3 的逐年零分/缺失审计，不用于回归。
- **模型与基线**：M4-v2.1 的 $\tau=0.90$ 平滑 pinball 上分位面；同一滚动窗口的持平、平均增量及 EXP-407 均值条件模型是比较对象。家族截距、重复 C4 身份权重和非负算力系数保持一致。
- **入口与输出**：拟建 `src/problem04/exp08_frontier_quantile.py`，输出新文件 `result/problem04/v21/exp408_*`、`result/tables/problem04/exp408_*`、`result/figures/problem04/exp408_*`；不覆盖 EXP-407。运行时记录脚本 SHA256、Git HEAD、依赖版本、200 次家族身份重采样成功率与所有输出哈希。
- **检验门槛**：小样例验证 pinball 方向、梯度、边界与单调纪录；正式回测报告同目标 $h=1,3$ MAE、更新次数，另报按家族留出 pinball 损失。只有 MAE 严格优于持平时，情景点数才允许称为经短期回测选择；无 12/24 月同口径测试，绝不称长期预测已验证。
- **冻结细节**：家族逐一留出，训练集同时剔除测试家族涉及的所有 C4 身份；未见家族效应取零。模型级比较使用相同测试行、C4 身份权重、原始 0.90 pinball 损失与加权覆盖率。滚动回测重新拟合均值和上分位模型，优先以 $h=1$ MAE 选持平、平均增量、均值条件或上分位条件预测；误差并列时按上述顺序选。200 次 C4 身份 Bootstrap 只传播系数估计敏感性，起报纪录与算力路径保持固定。

### EXP-408 实际执行与判定

- **执行与代码版本**：2026-09-25，`D:\Program Files\CondaEnvs\AIOPS\python.exe src/problem04/exp08_frontier_quantile.py`；入口和复用 EXP-407 源文件 SHA256、Git HEAD、Python/依赖版本、输入与输出文件 SHA256 均由 `result/problem04/v21/exp408_manifest.json` 固化。随机种子 20260925 只控制 200 次 C4 身份重采样；优化器采用确定性加权分位截距初始化。输出表为 `result/tables/problem04/exp408_*`，图及元数据为 `result/figures/problem04/exp408_frontier_quantile.*`。
- **小样例验证**：解析梯度与中心差分最大绝对差 $5.66\times10^{-10}$；上分位截距高于中位数、非负算力系数下界与累计纪录单调均通过；早期滚动一步预测仍在当期纪录至 100 分内。C1/C2 逐行身份、C4 匹配先后、113 行可信匹配与 64 个 C4 身份复核通过。
- **C3 审计**：C3 4,599 行中历史来源 26 行，23 行含至少一个基准零分，零个缺失单元；零分视为可能的未评测占位而不强行改值。C3 不进入主拟合与回测。逐年与逐项基准计数见 `result/tables/problem04/exp408_c3_year_source_audit.csv`。
- **模型级评价**：九个家族逐一留出，113 条测试行；按 C4 身份加权的 $\rho_{0.90}$ 平均损失为上分位 0.1017、均值 0.3274（Logit 单位）；加权覆盖率分别为 0.783、0.407。此指标仅比较条件能力面，不是月度纪录误差。
- **同目标时间外评价**：2024-10 至 2025-03 六个 $h=1$ 点全无纪录更新；持平、平均增量、均值条件、上分位条件 MAE 依次为 0、1.550、0.577、0.923 分。$h=3$ 四个点依次为 0、5.442、1.658、2.626 分。选持平作为 12/24 个月主点预测（51.2313 分）；200/200 次重采样成功，但 10–90% 参数范围不是经长期覆盖率验证的预测区间。
- **历史条件分解**：上分位得分面中规模关联位移 9.927 分、时间关联位移 4.943 分、总位移 14.871 分；加和误差为 0。该分解没有识别纯技术进步的因果份额。上分位面改善模型级 pinball，却未胜纪录预测持平基线；EXP-408 不通过将条件情景升级为已验证长期点预测的门槛。
