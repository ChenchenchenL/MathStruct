# experiments.md — 问题一实验记录

> 每次重要运行一条不可变记录（EXP-YYYYMMDD-序号），格式见 AGENTS.md。
> 勘察脚本 `.workbuddy/p1_recon.py` 为数据勘察，不属于正式实验，其结论记入 data-dictionary.md 质量检查记录。

---

## EXP-20260923-01 质量评价链小规模验证（方向/权重/冲突/双口径）

- **对应模型**：M1-v3（EQ07–11、EQ19'、EQ18、EQ19、EQ20、EQ21、验证协议第 6 条）
- **数据版本**：A1/A2/A3 原始附件 v1（未修改，流式读取）
- **代码版本**：`src/problem01/exp01_quality.py`；SHA256=`9F6F5CE3E6293599D8CC7607553234A2DA51B5CDF8C5D3032ACC02020761EB06`
- **运行入口**：`python exp01_quality.py`
- **环境**：Python 3.13.12（venv `~/.workbuddy/binaries/python/envs/default`）、numpy 2.5.3、pandas 3.0.5、scipy 1.18.1、scikit-learn 1.9.1、Windows
- **随机性**：seed=20260923（本实验无随机步骤，记录备案）
- **参数**：ρ_min=0.2、判定需 ≥4/7 域；熵权/等权两方案；秩归一 1001 分位格点（A1 拟合冻结，AS-11）；冲突度 C1–C4 按 M1-EQ19；τ 候选 P90/P95/P97.5
- **输出**：`result/problem01/exp01_quality/{exp01_summary.json, eq21_direction_table.csv, a1_domain_q.csv, a2a3_dual_protocol.csv}`
- **指标**：
  - 方向校验（EQ21，7 域 ρ vs 明确正向基准）：判正向 7 项——modernbert_professionalism(6/7 域)、rps_doc_word_count(4/7)、rps_doc_unigram_entropy(5/7)、qur_ws(7/7)、**qur_re(6/7)**、qur_ft(7/7)、qur_ev(7/7)；判负向 5 项——dsir_books/math/wiki(各 4/7)、rps_doc_frac_unique_words(4/7)、rps_doc_frac_no_alph_words(5/7)；方向不定 8 项——adfree(3/7)、num_sentences、top_2gram、top_3gram、uppercase、terminal_punct、numerical_chars、mean_word_length。入选集 |S|=17（base5 + 7 正向 + 5 负向翻转）。
  - 权重方案对比：等权 vs 熵权样本级 Spearman = **1.0000**（秩归一输入下熵权退化为等权，区分度消失）。
  - 冲突阈值扫描：c̃ 分位数 P90/P95/P97.5 = 0.835/0.874/0.904；代理抽查（C1 主导样本）——被标样本广告强度均值 0.87–0.91（全体 0.5）、教育分均值 0.90–0.94（全体 0.5），双高特征成立。
  - 域级 Q（等权中位数）：arxiv 0.884 > book 0.719 > commoncrawl 0.626 > stackexchange 0.521 > c4 0.477 > wikipedia 0.417 > github 0.393。
  - 双口径（A 全量/B 去重）：arxiv 0.8851/0.8851（重叠 1,419/17,523=8.1%）；github 0.3908/0.3907（重叠 10,000/203,752=4.9%）；冻结标尺迁移一致性：A2 arxiv 0.8851 ≈ A1 arxiv 0.8844、A3 github 0.3908 ≈ A1 github 0.3926。
  - A1 域构成实测：c4/github/stackexchange/wikipedia 各 10,000、commoncrawl 9,640、arxiv 1,419、book 171；**A1 的 arxiv 子集与 A2 重叠 100%（1,419）、github 子集与 A3 重叠 100%（10,000）**——证实 A1 arxiv/github 部分即 A2/A3 的子样本。
- **耗时**：26.1 s
- **结论**：
  1. 方向链闭合可行，22 指标中 12 项实证判定方向、8 项剔除待定；adfree 不入权重集但**仍用于冲突对 C1**（冲突机制独立于权重合成）；
  2. 熵权在秩归一输入下无区分度 → 主方案定为**等权**（熵权降级为形式对照）；
  3. 冲突定义代理抽查有效；**τ 的工作点选择仍需原文人工抽查**（本轮为代理证据）；
  4. 双口径差异可忽略（≤0.0002），独立复验与全量义务同时满足；
  5. **dsir_* 的"负向"判定标 provisional**：其与基准的相关被长度混杂污染（AS-03 未关闭，长文 dsir 更负而 word_count 判正向）——正式版须先长度残差化再重跑 EQ21。

## EXP-20260923-02 配比回归基线与组合影响（A4/A5 内部 + est 秩检验）

- **对应模型**：M1-v3（EQ14–17、EQ22/23、验证协议 v2 第 1/3 条）
- **数据版本**：A4/A5（512 组）+ A12/A13/A14/A15（各 63 组，外推表仅秩检验）
- **代码版本**：`src/problem01/exp02_mixture.py`；SHA256=`BC02B5B09B3E1F3BBD7096AECFCE4854A176542E47C8D657618D89B68E3D5EC0`
- **运行入口**：`python exp02_mixture.py`
- **环境**：同 EXP-01
- **随机性**：seed=20260923（KFold shuffle、HistGB、bootstrap）
- **参数**：目标 y=13 域等权平均；伪计数 ε=1e-3（敏感性 1e-4/1e-2）；ALR ref=pile_cc（敏感性 wikipedia_en/arxiv）；5 折 CV；转移 r=0.5；bootstrap B=200
- **输出**：`result/problem01/exp02_mixture/{transfer_matrix_m05.csv, exp02_summary.json}`
- **指标**：
  - 行和（AS-08）：0.996–1.003，重归一化后使用，检查项关闭。
  - CV（A4/A5 内部 5 折，out-of-sample R² 相对训练均值）：均值基线 RMSE=0.3245；OLS(ALR) RMSE=0.2922±0.0182、**R²=0.1904**；HistGB RMSE=0.1735±0.0174、**R²=0.7141**。全样本拟合 OLS R²=0.2461。**数据说明宣称"OLS R²=0.93"被实测否定。**
  - OLS-ALR 转移效应矩阵（r=0.5，p⁰=A4 均值配比）：top-5 负效应（有利）——从 pubmed_central/pile_cc/arxiv/github/freelaw 各转出 50% 至 ubuntu_irc，Δy≈−0.036~−0.038；top-5 正效应（不利）——ubuntu_irc→hackernews +0.0222、dm_mathematics→hackernews +0.0219、stackexchange→hackernews +0.0212 等。10 个最强对 bootstrap 95% CI 均不含 0（B=200，OLS-ALR 解析效应）。
  - est 秩检验（无校准，协议第 3 条）：10B Spearman=**−0.516**、70B=**−0.568**（n=63）。
  - 敏感性：ε=1e-4/1e-3/1e-2 → R²=0.162/0.190/0.247（ε 越大，小域 log 杠杆被压缩，R² 越高——论文须报告）；ref 更换（pile_cc/wikipedia_en/arxiv）→ R² 完全相同（0.1904），验证 ALR 预测的 ref 不变性（换 ref 仅线性重参数化）。
- **耗时**：1.4 s
- **结论**：
  1. **模型选型（协议第 1 条）**：HistGradientBoosting（16 维 ALR 输入）显著优于 OLS-ALR（CV R² 0.71 vs 0.19）——HistGB 为主预测模型；OLS-ALR 保留用于**系数解释与转移效应解析表达**。本实现不宣称使用 LightGBM。
  2. 协议第 2 条（A6/A7 独立检验）与第 4 条（校准后分析）留待下一轮运行。
  3. **est 表负秩相关**：1M 排序与 est 外推排序反相——est 表的排序由外推机制主导，不能支持"1M 配比排序可外推至 10B/70B"的结论；此为 A12–A15 局限性的实证证据，论文如实报告（不与真实实验的 rank invariance 混淆）。
  4. 转移效应分析可行且 bootstrap 显著；极小配比域（enron 0.22%、philpapers 0.55%、nih 0.59%）对数杠杆大，比例转移口径（EQ22 参数化）已回写 model.md。

## EXP-20260923-03 质量链关闭（dsir 残差化重判 + 三口径敏感性 + EQ10 对照 + λ 扫描）

- **对应模型**：M1-v3（AS-03、EQ21 重跑、EQ11/13 口径、EQ18 λ、AS-13 敏感性）
- **数据版本**：A1 v1（含 content 用于抽查材料）
- **代码版本**：`src/problem01/exp03_quality_close.py`；SHA256=`071EFABB3F422FB4121B3EF03ECFC63C377F022F2005F850720064B832FCA1CC`；公共模块 `p1_common.py`
- **环境/随机性**：同 EXP-01；seed=20260923
- **输出**：`result/problem01/exp03_quality_close/{eq21_rerun_after_dsir_residual.csv, conflict_spotcheck_top50.csv, exp03_summary.json}`
- **指标与结论**：
  1. **AS-03 证实**：dsir_books/math/wiki 与 log(word_count) 的 Spearman = −0.889/−0.913/−0.893（斜率 ≈ −5000/log-word）——长度混杂极强，EXP-01 的"负向"判定确系长度伪影。
  2. **残差化后 EQ21 重判**：dsir 三项翻转为**正向 6/7 域**（与 Meta-rater importance 语义一致）→ 方向表定稿：正向 12 项（base5 + dsir×3 + professionalism + word_count + entropy + qur_ws/re/ft/ev），负向 2 项（frac_unique_words、frac_no_alph_words），不定 8 项剔除。|S|=17。
  3. **三口径敏感性**：qur_re 与 professionalism 的 drop/keep 域级排序完全一致（ρ=1.0），flip 仍 0.964——**域级结论对该二维方向选择稳健**（AS-13 关闭）。
  4. **EQ10 双口径**：期望等级 vs argmax 的 round-match 率 0.81–0.96；argmax 在 A1 内部分辨力低（reasoning/professionalism 的 argmax 近常数导致 Spearman 未定义）→ 主口径定 softmax 期望等级（平滑、保留序信息）。
  5. **λ 扫描**（τ=P95=0.874）：域级排序对所有 λ∈{0,…,1} 完全不变（ρ=1.0）；冲突样本 Q 平均下降 λ×0.167；Q_min>0——**λ 不影响域级结论**，默认 λ=0.5，敏感性入报告。
  6. 抽查材料：top-50 冲突样本导出（构成 C1×33、C2×16、C4×1；域：stackexchange 19、c4 14、github 6……），供人工原文复核；**人工复核结论待录**。
- **耗时**：20.7 s

## EXP-20260923-04 A6–A11 全量检验与配比收尾

- **对应模型**：M1-v3（验证协议 v2 第 2/3/4 条、AS-05、EQ23b/c、AS-09、AS-04）
- **数据版本**：A4/A5（512）+ A6–A11（256/256/64）
- **代码版本**：`src/problem01/exp04_validation.py`；SHA256=`B136F1F86EC1EF899D69C1FA2E7AAB4D233EF63BBD6A6F406BB42E039D0982E9`
- **环境/随机性**：同上；seed=20260923（KFold/bootstrap/HistGB）
- **输出**：`result/problem01/exp04_validation/{bootstrap_ci_B1000.csv, exp04_summary.json}`
- **指标与结论**：
  1. **协议第 2 条（A6/A7 独立同尺度）**：HistGB RMSE=0.1374（较均值基线降 57.6%）、R²=0.767、Spearman=0.867；OLS RMSE=0.2517（降 22.4%）、R²=0.218、Spearman=0.464。1M 配比效应在独立检验上成立，非线性模型显著占优。
  2. **协议第 3 条（60M/1B 无校准 Spearman）**：HistGB 0.833（60M）/0.665（1B）；OLS 0.414/0.670——**秩不变性在真实检验集上成立**；与 est 表的 −0.52/−0.57 对照，证实 est 负相关为外推机制伪影。
  3. **协议第 4 条（分半校准）**：无校准 RMSE 60M≈1.51、1B≈2.85（规模基线差 α=−1.47/−2.83）；奇半拟合截距后偶半 RMSE 0.10–0.21。规模混淆被定量演示，该口径仅标注为"经目标尺度校准的分析"。
  4. **AS-05 系数漂移**：Spearman(β_60M, β_1M)=0.774、Spearman(β_1B, β_1M)=0.621；最大漂移集中在 enron_emails、ubuntu_irc、philpapers 等小域——配比效应方向大体保持、小域系数不稳。
  5. **EQ23b 转移对照**：HistGB 与线性版方向大量不一致（10 对中 6 对不一致，集中于 ubuntu_irc 相关对）——**线性解析式的 ubuntu_irc 效应疑为小域对数杠杆外推伪影**；转移效应的稳健结论以树模型在可行配比范围内的预测为主口径，不一致域对已在结果中点名。
  6. **EQ23c 三域组合**（pile_cc +0.05、github +0.05、arxiv −0.10）：OLS +0.004 / HistGB −0.065，均不等于两单域转移之和（+0.001/+0.012）——组合效应≠单域效应之和（非线性交互实证）。
  7. **AS-09**：17 列（保留 4 个无 Loss 域配比列）CV R²=0.190 vs 13 列 0.068——**保留方案被数据强支持**（4 列携带其他域的信息）。
  8. **AS-04 目标敏感性**：等权均值 y R²=0.190；配比加权 y R²=0.406；**逐域多输出平均 R²=0.724（0.599–0.834）**——逐域 Loss 的可预测性远高于 13 域均值（均值把域间异质效应抵消）。主目标维持题目口径（13 域均值，供问题三标量优化），逐域多输出升级为并列分析视角写入报告。
  9. **B=1000 bootstrap**：top-10 转移对 95% CI 全部显著（与 B=200 一致）。
- **耗时**：1.8 s

## EXP-20260923-05 问题一终版复验（A1+A2/A3扩展集全量复验/双口径/冲突原文抽查/A16映射实证/论文图表交付）

- **对应模型**：M1-v3 当前口径（全量质量评价 EQ07–11/EQ19'/EQ18/EQ19/EQ20/EQ21、验证协议第 6 条双口径、A16 映射分析、EQ22/23 组合转移）
- **数据版本**：A1–A3 原始附件 v1 + A4–A15 原始附件 v1
- **代码版本**：`src/problem01/exp05_final_closure.py`；SHA256=`82F75B73CE00B78E3150AD2B047F7767B0E226726C5C96644D1A9E8A8F53FD55`
- **运行入口**：`D:\Program Files\CondaEnvs\AIOPS\python.exe src/problem01/exp05_final_closure.py`
- **环境**：AIOPS 虚拟环境，Windows；Python 3.11.15、numpy 2.4.6、pandas 3.0.6、scipy 1.17.1、scikit-learn 1.9.1、matplotlib 3.11.2
- **随机性**：seed=20260923
- **参数**：DSIR 对 log1p(wc) 残差化；冻结 A1 分位标尺（1001 格点）；入选特征集 |S|=17；τ=0.8736（P95）；λ=0.5；ALR ref=pile_cc；ε=1e-3
- **输出**：
  - 数据表：`result/tables/problem01/table_p1_domain_q_a1.csv`、`result/tables/problem01/table_p1_dual_protocol_a2a3.csv`、`result/tables/problem01/table_p1_q_vs_beta.csv`
  - 交付图表：`result/figures/problem01/fig_p1_quality_conflict.{png,pdf,json}`、`result/figures/problem01/fig_p1_conflict_distribution.{png,pdf,json}`、`result/figures/problem01/fig_p1_model_validation.{png,pdf,json}`、`result/figures/problem01/fig_p1_scale_rank.{png,pdf,json}`；四张图均为独立结果图，图内无总标题和图注，另有同名 `.caption.md` 外置图注文件，PNG 按 600 dpi 输出。
  - 更新后新增：`result/problem01/exp05_final_closure/exp05_summary.json`
- **指标与结论**：
  1. **A1 终版域级 Q（中位数/IQR）**：arxiv 0.7271 (0.0918) > commoncrawl 0.6498 (0.1793) > book 0.5639 (0.1418) > stackexchange 0.5106 (0.1644) > c4 0.5096 (0.2673) > wikipedia 0.4330 (0.2873) > github 0.3371 (0.1797)。
  2. **A2/A3 全量复验与双口径（履行全量义务+独立性）**：
     - arxiv（A2 扩展集 17,523 条，与 A1 重叠 1,419 条 / 8.10%）：全量 Protocol A 中位数 **0.7277**（IQR 0.0839）；去重 Protocol B 中位数 **0.7278**（IQR 0.0827）；与 A1 基准中位数 0.7271 偏差仅 0.0006。
     - github（A3 扩展集 203,752 条，与 A1 重叠 10,000 条 / 4.91%）：全量 Protocol A 中位数 **0.3362**（IQR 0.1821）；去重 Protocol B 中位数 **0.3361**（IQR 0.1822）；与 A1 基准中位数 0.3371 偏差仅 0.0009。
      - 结论：全量质量信号（272,505 条）评分完成；冻结标尺下两种扩展集口径的域级中位数差异 <0.001，支持本次样本范围内的统计稳定性，不外推为确定性稳定性。
  3. **扩展集冲突分布复验（主导机制得到复现）**：
     - A1 全域冲突率 4.88%（C1=31.8%, C2=59.1%, C3=0.4%, C4=8.7%）；
     - arxiv 域冲突类型在 A1 与 A2 几乎完全一致：A1_arxiv 为 C1 19.4% / C2 79.4% / C4 1.2%；A2_arxiv 为 C1 18.2% / C2 78.9% / C4 2.9%（LaTeX 公式/代码语法标记被统计为非字母符号，与高教育价值构成 C2 冲突）；
     - github 域冲突类型在 A1 与 A3 几乎完全一致：A1_github 为 C1 5.1% / C2 90.4% / C4 4.5%；A3_github 为 C1 4.3% / C2 90.3% / C4 5.3%（程序代码的特殊字符与语法结构被判定为非字母词高占比，主导 C2 冲突）。
      - 结论：抽样集发现的冲突类别及主导机制在扩展集上得到复现，A1 与 A2/A3 的类别比例总体接近；该结果支持领域异质性解释，但不等于逐样本或比例 100% 一致。
  4. **τ 原文抽查定性复核（top-50 冲突样本核验）**：
     - C1 主导样本（如 c4 中的 Ready, Set, Go! 数学练习册宣传、Puget Sound 儿童牙科科普宣传、烟农太阳能农场改造报告）：高教育价值（>0.96）与商业营销推广/广告语料（ad_strength>0.95）高度共存，完全契合题目"教育价值高×广告含量高"的冲突定义；
     - C2 主导样本（如富勒烯分子结构的学术论文 LaTeX、Java 算法与 R 语言数据处理）：高学术教育价值（>0.97）与 LaTeX 标记、代码字符（noalph>0.96）共存；
     - C4 主导样本（如 SO(3) 表现群理论证明）：高逻辑推理价值（>0.62）与数学符号重复共存。
      - 定性抽查结论：top-50 原文中可观察到与 C1/C2/C4 定义相符的典型文本结构；由于未保存结构化人工标签，不能据此报告准确率、召回率或人工最优阈值。hinge 降权的作用由 λ 敏感性结果支持。
  5. **A16 跨体系映射与质量 Q 引入配比论证（关闭任务 3 关键问答）**：
     - 6 个可映射域中，域级质量分 $Q_d$ 与 ALR 配比边际效应 $\beta_d$ 的 Spearman 秩相关为 **0.2571**（$p=0.6228$，统计不显著）；
     - 配方加权质量 $Q_{\text{recipe}} = \sum_{d \in \text{mapped}} p_{jd} Q_d$ 与验证集 Loss 的秩相关仅为 **0.0859**（$p=0.0591$，接近正相关且无解释力）；
     - 论证边界：当前 A16 映射分析未发现通用质量分 $Q$ 与配比效应的稳定单调关联；加之固定 $Q_d$ 与配比 $p_d$ 存在严格代数共线性（不可识别），因此在配比建模中**不引入伪交互项**，而将 $Q$ 留作问题二中、在存在质量变异的数据上进一步检验的候选输入。
- **耗时**：26.4 s
