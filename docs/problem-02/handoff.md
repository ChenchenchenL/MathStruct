# handoff.md — 问题二论文交接报告

> 本文档面向论文写作者与后续问题研究者，汇总问题二（跨维度数据融合与广义标度律）的全部核心数学模型、实证参数估计、图表路径及交接接口。所有结论均可回溯至具体公式编号、实验 ID 与数据文件。

---

## 1. 问题定位与核心数学模型体系

### 1.1 广义标度律模型与退化性（M2-EQ01）
基于现代大语言模型“有效信息量假说（Effective Token Volume Hypothesis）”，将数据质量 $Q$ 与领域配比 $p$ 统一映射为单位 Token 的有效知识密度，推导出结构自洽的广义标度律：
$$L(N, D, Q, p) = E + A N^{-\alpha} + B D^{-\beta} \cdot \exp\left\{ -\rho (Q - Q_{\text{anchor}}) \right\} \cdot R(p)$$

- **双概念分离与退化性检查**：
  1. **理论退化锚点（$Q_{\text{anchor}} \equiv 1.0$）**：
     设定 $Q_{\text{anchor}}\equiv1.0$ 为质量评分参考上界。在基准配方 $p_0$ 与该参考质量下，$R(p_0)=1$、$\exp\{0\}=1$，广义标度律按定义退化为经典双变量形式 $L=E+A N^{-\alpha}+B D^{-\beta}$。
  2. **经济成本底线（$Q_{\text{base}} \equiv 0.584$ 严密溯源与聚合定义）**：
     - **题面授权依据**：赛题附录 B 正文显式说明：“$Q_0$ 表示未经额外处理的基准数据质量（可由附件 A 质量评分或合理假设给出）”；
     - **聚合公式与成本基线假设依据**：问题一中各领域多维质量中位数显示：arXiv 0.727、Common Crawl (cc) 0.650、Books 0.564、StackExchange 0.511、C4 0.510、Wikipedia 0.433、GitHub 0.337（全样本未加权中位数 $\approx 0.500$，The Pile 官方配比加权中位数 $\approx 0.543$）。由于大语言模型预训练语料中超过 80%~85% 来源于开放网络抓取（Web Scraping），在工业界未投入昂贵额外清洗的语料直接采用开放网络原始抓取；因此定义 $Q_{\text{base}}$ 为**开放网络抓取域（Web-scraped Baseline: Common Crawl + C4）的合成中位数质量**：
       $$Q_{\text{base}} \equiv \text{Median}(Q_{\text{cc}} \cup Q_{\text{c4}}) \approx 0.584$$
       （A1 集记录的 Common Crawl 0.650 与 C4 0.510 按规模混合的中位数恰为 $0.580 \sim 0.584$）；
     - **物理与成本功能**：$Q_{\text{base}}$ 严格代表“直接采用公网爬虫原始语料时的天然零成本底线”（在 $Q \le Q_{\text{base}}$ 时 $\bar{h}(Q) = 0$），仅用于附录 B 增量清洗算力方程 $C_Q = D [g(Q) - g(Q_{\text{base}})]_+$，不偏移物理标度律退化常数。敏感性检验发现若将退化锚点错设为 0.584，拟合 $R^2$ 骤降至 0.7827。

### 1.2 跨尺度配比响应乘子（M2-EQ02）
$$R(p) = \exp\left\{ \tau \cdot \frac{f(p) - f(p_0)}{f(p_0)} \right\}$$
- **模拟矩法（MSM）标定**：
  问题一独立检验集（A6–A11）中 1M、60M、1B 的损失 CV 分别为 5.27%、5.74% 和 2.35%；1B/1M 比值为 0.4466。1B 尺度下配方预测相对波动标准差 $\text{std}\left(\frac{f(p)-f(p_0)}{f(p_0)}\right)=0.02708$。
  一阶线性矩匹配解为：
  $$\tau_{\text{linear}} = \frac{\text{CV}_{\text{emp}}(1\text{B})}{\text{std}(\text{rel\_diff}_{1\text{B}})} = \frac{0.02352}{0.02708} = 0.8686$$
  指数非线性精确标定解为：
  $$\tau^* = \arg\min_\tau |\text{std}(\exp(\tau \cdot \text{rel\_diff}_{1\text{B}})) - \text{CV}_{\text{emp}}(1\text{B})| = 0.8348 \approx 0.85$$
  采用 $\tau=0.85$ 时，模拟分布标准差为 0.02397，与 1B 实测 CV 的绝对误差为 0.00045。该值是当前工作模型下的经验校准，不等同于跨数据源的独立识别。

### 1.3 无量纲弹性与边际性价比（M2-EQ03）
- **无量纲点弹性**：$\epsilon_N=-\alpha \frac{A N^{-\alpha}}{L}<0$，$\epsilon_D=-\beta \frac{T}{L}<0$，$\epsilon_Q=-\rho Q \frac{T}{L}<0$。负号是当前模型在正参数与正变量定义域内的解析性质。
- **边际技术替代率（MRTS）**：
  $$\text{MRTS}_{N_B, Q} = -\left. \frac{dN_B}{dQ} \right|_{L = \text{const}} = \frac{\rho B D^{-\beta} e^{-\rho(Q-Q_{\text{anchor}})} R(p)}{\alpha A N_B^{-\alpha - 1}}$$
  （单位：十亿参数 / 单位质量评分）。
- **“算一笔账：堆参数 vs 买教材更划算？”（严格对接赛题附录 B.1）**：
  总算力成本 $C = C_{\text{train}} + C_Q = 6 N_{\text{raw}} D_{\text{raw}} + D_{\text{raw}} [g(Q) - g(Q_{\text{base}})]_+$。
  边际投资回报率（单位 FLOP 带来的损失下降量）之比严格推导为规模不变量（$D_{\text{raw}}$ 严格对消）：
  $$\frac{\text{ROI}_Q}{\text{ROI}_N} = \frac{-\partial L / \partial Q / \partial C / \partial Q}{-\partial L / \partial N_{\text{raw}} / \partial C / \partial N_{\text{raw}}} = \text{MRTS}_{N_B, Q} \cdot \frac{6 \times 10^9}{g'(Q)}$$
  附录 B.1 三类清洗成本函数的均衡质量临界点 $Q^*$（以 $N=1.04\text{B}, D=300\text{B}$ 为基准）：
  1. **指数型成本**（$g_{\text{exp}} = 10^7 e^{6Q}$）：$Q^* = 0.8480 \approx 0.85$；当 $Q < 0.85$ 时买教材更划算（$Q=0.70$ 时比值为 **2.681**），当 $Q > 0.85$ 时边际清洗成本指数激增，优势翻转为堆参数更划算；
  2. **幂函数型成本**（$g_{\text{pow}} = 5 \times 10^9 Q^4$）：$Q^* = 0.7955 \approx 0.80$；当 $Q < 0.80$ 时买教材更划算（$Q=0.70$ 时比值为 **1.564**），当 $Q > 0.80$ 时优势翻转为堆参数；
  3. **对数型成本**（$g_{\text{log}} = 2 \times 10^9 \ln(1 + 10Q)$）：在可行域内始终有 $\text{ROI}_Q / \text{ROI}_N > 1$（$Q=0.70$ 时比值为 **4.290**），提升质量始终比堆参数划算。
  - **重要理论价值**：该翻转阈值揭示了为何不能盲目拉满数据质量，为问题三预算优化提供了直接的边际切换依据。

### 1.4 等效替代与“标度律天花板饱和”奇点理论（M2-EQ04）
在**固定数据量 $D$ 与领域配比 $p$** 的前提下，提升质量 0.1 的降损量为 $\Delta L_{\text{quality}} = T (1 - e^{-0.1 \rho})$：
- **语义一（等效模型缩减，Cost-saving）**：
  $$N_{\text{new}} = \left[ N^{-\alpha} + \frac{T}{A}(1 - e^{-0.1 \rho}) \right]^{-1/\alpha} < N \implies \Delta N_{\text{save}} = N - N_{\text{new}} > 0$$
  该解在定义域内**恒存在唯一正实数解**，质量提升永远能实现模型缩容。
- **语义二（等效规模飞跃，Scaling Equivalent Gain）**：
  在原质量下依靠堆参数达到同等效果：$A N_{\text{equiv}}^{-\alpha} = A N^{-\alpha} - \Delta L_{\text{quality}}$。
  - **充要可解条件**：$N < N_{\text{crit}} \equiv \left[ \frac{A}{T(1 - e^{-0.1 \rho})} \right]^{1/\alpha}$；
  - **等效扩容边界（Scaling Ceiling Saturation）**：
    当 $N\ge N_{\text{crit}}$ 时，当前等效扩容方程无有限实数解。该结论只适用于给定广义标度律、固定 $D,p$ 和质量增量 0.1 的比较场景。

### 1.5 领域配比互补与替代（M2-EQ05）
- 单纯形乘子 Hessian 交叉偏导：$\mathcal{H}^R_{ij} = \frac{\partial^2 R(p)}{\partial p_i \partial p_j}$（对应模型损失曲率为 $\mathcal{H}^L_{ij} = T \cdot \mathcal{H}^R_{ij}$，外在缩放因子 $T \approx 0.3 \sim 0.6$）。
- 实证发现：
  - **强协同互补领域（$\mathcal{H}^R < 0$）**：Top-1 为 `stackexchange` 问答与 `pile_cc` 通用网络语料（$\mathcal{H}^R = -8.9955$），Top-2 为 `arxiv` 科技文献与 `uspto_backgrounds` 专利背景（$\mathcal{H}^R = -6.5963$）；`arxiv` 与 `github` 代码呈现温和协同（$\mathcal{H}^R = -1.4724$）；
  - **强竞争替代领域（$\mathcal{H}^R > 0$）**：Top-1 为 `pile_cc` 与 `uspto_backgrounds`（$\mathcal{H}^R = +8.7823$），Top-2 为 `enron_emails` 与 `hackernews`（$\mathcal{H}^R = +4.9609$），Top-3 为 `arxiv` 与 `pile_cc`（$\mathcal{H}^R = +4.2833$）；而 `pile_cc` 与 `gutenberg` 书籍语料交互接近中性独立（$\mathcal{H}^R = -0.1631$）。
- 多步长扰动鲁棒性检验（$h \in [0.010, 0.030]$）证实核心协同与竞争对符号保持一致。

---

## 2. 数据处理规则清单（可直接作为论文预处理小节）
- **P2-01**：变量量纲统一与十亿级无量纲化变换；
- **P2-02**：质量方向诊断与 B8 隔离规则（B6/B7 总体负相关；B8 Pearson $r=+0.9132$、Spearman $\rho=+0.9208$，仅作诊断且原因未确定）；
- **P2-03**：完整模型时间序列轨迹隔离与留一验证规则（LOMOCV，杜绝步间泄漏）；
- **P2-04**：早期训练 Warmup（step $\ge 500$）截断三口径稳健对照；
- **P2-05**：跨尺度配比方差收缩比校准规则（1B / 1M 收缩比 0.4466，校准 $\tau^* = 0.85$）；
- **P2-06**：跨族模型与文献基线截距对齐规则；
- **P2-07**：奇点饱和数值保护与存在性截断规则。

---

## 3. 主要定量结果与证据链（对应 EXP-P2-01 ~ EXP-P2-05）

### 3.1 经典与广义标度律参数估计（EXP-P2-01, EXP-P2-02）
- **数据表路径**：`result/tables/problem02/table_p2_scaling_parameters.csv`、`result/tables/problem02/table_p2_quality_models.csv`
- **对应图表**：`result/figures/problem02/fig_p2_classical_scaling.pdf`、`result/figures/problem02/fig_p2_quality_effect.pdf`
- **正式参数估计表**：
  - 不可约损失下限：$E = 1.6898 \pm 0.0001$ Nats（95% CI: $[1.6897, 1.6899]$）
  - 模型规模系数：$A = 0.3540 \pm 0.0001$（95% CI: $[0.3539, 0.3541]$）
  - 模型规模幂指数：$\alpha = 0.3400 \pm 0.0001$（95% CI: $[0.3398, 0.3401]$）
  - 数据量系数：$B = 1.2403 \pm 0.0001$（95% CI: $[1.2402, 1.2404]$）
  - 数据量幂指数：$\beta = 0.2799 \pm 0.0001$（95% CI: $[0.2798, 0.2799]$）
  - 数据质量效率系数：$\rho = 0.6646 \pm 0.0135$（95% CI: $[0.6275, 0.7013]$）
  - 经典拟合优度：$R^2 = 1.00000$, $\text{RMSE} = 0.00015$ Nats；留一泛化：LOOCV $R^2 = 1.00000$
  - 质量模型比选：指数型 $\text{AIC} = -2082.21, \text{MAE} = 0.07507$ 显著优于幂律型 $\text{AIC} = -1947.08$；Model 1b（强行设 $Q_0=0.584$）因破坏退化基准使 $R^2$ 骤降至 0.78273；Model 3（6 参数自由拟合）由于合成切片共线性导致 $\beta$ 塌陷至 0.1090，证实冻结经典底座参数的必要性。

### 3.2 质量提升 0.1 等效参数计算（EXP-P2-04）
- **数据表路径**：`result/tables/problem02/table_p2_substitution_01.csv`、`result/tables/problem02/table_p2_marginal_roi.csv`
- **对应图表**：`result/figures/problem02/fig_p2_substitution_ceiling.pdf`
- **基准场景（$D=300$B Tokens, 起始质量 $Q=0.7$）代表性结论**：
  - **70M 模型（0.0705B）**：提升 0.1 质量等价于节省 0.0045B (6.4%) 参数；或等价于扩大参数量至 0.0754B (+7.0%)；
  - **1B 模型（1.0409B）**：提升 0.1 质量等价于节省 0.1554B (14.9%) 参数；或等价于扩大参数量至 1.2351B (+18.7%)；
  - **7B 模型（6.8610B）**：提升 0.1 质量等价于节省 1.7766B (25.9%) 参数；或等价于扩大参数量至 9.5789B (+39.6%)；
  - **13B 模型（11.9658B）**：提升 0.1 质量等价于节省 3.6041B (30.1%) 参数；或等价于扩大参数量至 17.9978B (+50.4%)；
  - **70B 模型（70.0000B）**：提升 0.1 质量等价于节省 32.4861B (46.4%) 参数；或等价于扩大参数量至 154.6540B (+120.9%)！

### 3.3 全景多尺度、跨族与文献泛化验证（EXP-P2-05）
- **数据表路径**：`result/tables/problem02/table_p2_multiscale_validation.csv`
- **对应图表**：`result/figures/problem02/fig_p2_multiscale_validation.pdf`
- **跨数据集验证指标汇总**：
  1. **B3（Pythia 轨迹内插一致性检验，4,000点）**：$\text{RMSE} = 0.0038$ Nats, $R^2 = 1.0000$, Spearman $\rho = 1.0000$，验证了模型在训练中间步的光滑连续一致性；
  2. **B4（12 族现代开源模型收敛点，57点）**：校准后 $\text{RMSE} = 0.1927$ Nats, **$R^2 = 0.8286$**, **Spearman $\rho = 0.9830$**；
  3. **B5（学术经典文献数据，44点）**：校准后 $\text{RMSE} = 0.1997$ Nats, **$R^2 = 0.7249$**, **Spearman $\rho = 0.9588$**；
  4. **B2（Cerebras 族跨架构截距校准，非独立验证，1,029 点）**：校准后 $\text{RMSE}=0.4185$ Nats，**$R^2=0.3116$**，Spearman $\rho=0.7881$。点预测离散较大但排序仍有一致性；本实验不能识别架构、训练调度或数据构成的因果贡献；
  5. **B9（132 组工业界 Frontier 模型算力核验）**：有 FLOPs 记录的样本中，实测 FLOPs 与 $6ND$ 理论算力比值中位数为 **0.99998**，与 $C\approx6ND$ 近似一致；
  6. **B10（超大模型外推估算基准，全集 128 点 / 119 点稠密语言模型绘图）**：校准后 $\text{RMSE}=0.0009$ Nats，**$R^2=1.00000$**，**Spearman $\rho=1.0000$**。B10 仅用于外推一致性检查，不作为独立实证验证。

---

## 4. 向问题三与问题四传递的接口清单

1. **广义标度律全参数表**：
   $$L(N, D, Q, p) = 1.6898 + 0.3540 N^{-0.3400} + 1.2403 D^{-0.2799} \cdot \exp\{-0.6646 (Q - 1.0)\} \cdot R(p)$$
   直接作为问题三算力预算优化模型的目标函数。
2. **多维边际替代率（MRTS）与附录 B 成本回报均衡阈值 $Q^*$**：
   用于问题三求解 KKT 条件下参数、数据量与质量提升的最优点。在预算跨越时，指数成本（$Q^* \approx 0.85$）与幂函数成本（$Q^* \approx 0.80$）将触发资源倾斜的**拐点转移机制**。
3. **标度律天花板饱和临界值 $N_{\text{crit}}$**：
   用于识别数据受限下追求参数规模向提升教材质量的不可逾越边界。

---

## 5. 完成状态与已知限制

- **完成状态**：EXP-20260923-P2-01 至 EXP-20260923-P2-05 已运行完成；主模型、方向诊断、跨尺度校准、领域交互、弹性与等效替代、跨来源检查及问题三接口均已交付。
- **B8 限制**：质量-损失方向与 B6/B7 不同，具体原因未确定；不参与主模型参数估计。
- **B10 限制**：属于估算外推基准，不是独立实验观测，只支持一致性讨论。
- **B2 限制**：$R^2=0.3116$，点预测离散较大；应与 Spearman $\rho=0.7881$ 同时报告。；论文中称为「跨架构截距校准性分析」，不称为独立验证。
- **进入问题三**：问题二结果生产阶段已完成，可将 M2-EQ01、成本函数、MRTS、$Q^*$ 和 $N_{\text{crit}}$ 作为问题三输入，但须保留上述适用边界。
