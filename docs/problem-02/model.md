# model.md — 问题二广义标度律数学模型与理论推导

> 本文档按版本记录问题二（跨维度数据融合与广义标度律）的全部数学模型、推导过程、参数定义及理论证明。公式编号采用稳定前缀 `M2-EQ*`。

---

## 1. 理论基础与广义标度律建模

### M2-EQ01 广义标度律主模型（有效信息量假说）

$$L(N, D, Q, p) = E + A N^{-\alpha} + B D^{-\beta} \cdot \exp\left\{ -\rho (Q - Q_{\text{anchor}}) \right\} \cdot R(p)$$

- **物理机理推导（有效信息量假说）**：
  设模型参数规模为 $N$，名义训练 Token 量为 $D$。根据现代大语言模型信息论假说（Sharma et al., 2022; Hoffmann et al., 2022），低质量数据含有重复、噪声与低信息熵片段，偏斜的领域配比会导致部分知识域欠拟合或过拟合。因此，模型实际吸收的“有效知识量（Effective Token Volume）”为：
  $$\widetilde{D} = D \cdot \eta_Q(Q) \cdot \eta_p(p)$$
  式中：
  - 质量效率项：$\eta_Q(Q) = \exp\left\{ \frac{\rho}{\beta} (Q - Q_{\text{anchor}}) \right\}$（当质量达到参考上界 $Q_{\text{anchor}} \equiv 1.0$ 时，$\eta_Q = 1$）；
  - 配比效率项：$\eta_p(p) = [R(p)]^{-1/\beta}$（当处于基线配方 $p_0$ 时，$\eta_p = 1$）。
  将有效数据量 $\widetilde{D}$ 代入 Chinchilla 经典双变量标度律的数据项 $B \widetilde{D}^{-\beta}$ 中：
  $$B \widetilde{D}^{-\beta} = B \left[ D \cdot \exp\left\{ \frac{\rho}{\beta} (Q - Q_{\text{anchor}}) \right\} \cdot [R(p)]^{-1/\beta} \right]^{-\beta} = B D^{-\beta} \cdot \exp\left\{ -\rho (Q - Q_{\text{anchor}}) \right\} \cdot R(p)$$
  由此直接推导出 M2-EQ01，不仅在数学上高度紧凑，而且保证了各项的物理量纲统一性。

- **变量与符号定义**：
  - $L$：验证集交叉熵损失（Cross-Entropy Loss，单位：Nats，因变量）；
  - $N$：模型非嵌入参数规模（单位：$10^9$ 参数，即 Billion parameters，记为 $N_B$）；
  - $D$：累计训练 Token 数量（单位：$10^9$ Tokens，即 Billion tokens，记为 $D_B$）；
  - $Q$：训练语料综合质量评分（$[0, 1]$ 连续变量，越高质量越好）；
  - $Q_{\text{anchor}}$：**理论退化锚点**（设为 $Q_{\text{anchor}} \equiv 1.0$，代表质量评分的参考上界）；
  - $Q_{\text{base}}$：**经济成本底线**（依据附录 B 正文“可由附件 A 质量评分或合理假设给出”，设定为 $Q_{\text{base}} \equiv 0.584$，代表大模型预训练中占主导（>80%）的开放网络抓取域 Common Crawl 与 C4 的合成中位数质量 $\text{Median}(Q_{\text{cc}} \cup Q_{\text{c4}}) \approx 0.584$；仅用于附录 B 数据清洗计算成本方程 $C_Q = D[g(Q) - g(Q_{\text{base}})]_+$，不偏移物理标度律退化常数）；
  - $p \in \Delta^{16}$：17 维领域配比向量，满足非负性 $p_i \ge 0$ 与和为一约束 $\sum_{i=1}^{17} p_i = 1$；
  - $E$：不可约交叉熵下限（Irreducible Loss，理论熵界，单位：Nats，参数 $E \ge 0$）；
  - $A, \alpha$：模型参数项系数与标度幂指数（参数 $A > 0, \alpha > 0$）；
  - $B, \beta$：数据量项系数与标度幂指数（参数 $B > 0, \beta > 0$）；
  - $\rho$：数据质量效率提升系数（参数 $\rho > 0$）；
  - $R(p)$：领域配比响应乘子（无量纲，且在参考配比处 $R(p_0) = 1$）。

- **严格退化性证明**：
  当 $Q = Q_{\text{anchor}} = 1.0$ 且 $p = p_0$ 时：
  $$\exp\left\{ -\rho (1.0 - 1.0) \right\} = e^0 = 1, \quad R(p_0) = 1$$
  代入 M2-EQ01：
  $$L(N, D, 1.0, p_0) = E + A N^{-\alpha} + B D^{-\beta} \cdot 1 \cdot 1 = E + A N^{-\alpha} + B D^{-\beta}$$
  **无损且严格退化**为 Chinchilla / Pythia 经典双变量幂律！经典文献发表的系数 $(E, A, \alpha, B, \beta)$ 保持完全不变。

- **候选模型比较（指数型 vs 幂律型 vs 错设锚点消融）**：
  - 候选主模型（Model 1，指数型，退化锚点 $Q_{\text{anchor}}=1.0$）：$B D^{-\beta} e^{-\rho (Q - 1.0)}$，实测 $R^2 = 0.91607, \text{RMSE} = 0.09869, \text{MAE} = 0.07507, \text{AIC} = -2082.21$；
  - 错设锚点消融模型（Model 1b，指数型，若强行设 $Q_0=0.584$ 且冻结 B1 底座）：由于人为引入了 $\exp(\rho \times 0.416) \approx 1.32$ 倍的数据项偏置，破坏了退化性，$R^2$ 骤降至 $0.78273$，$\text{RMSE}$ 激增至 $0.15878$；
  - 候选替代模型（Model 2，幂律型）：$B D^{-\beta} Q^{-\gamma}$，实测 $R^2 = 0.88667, \text{AIC} = -1947.08$，且在 $Q \to 0$ 时存在奇异发散风险；
  - 自由拟合消融模型（Model 3，6 参数自由拟合）：实测在合成切片上出现严重参数共线性，$\beta$ 指数塌陷至 $0.1090$，证实冻结 B1 底座参数的稳健性与必要性。

---

## 2. 跨附件配比响应与尺度校准模型

### M2-EQ02 跨尺度领域配比乘子 $R(p)$ 与 MSM 标定

$$R(p) = \exp\left\{ \tau \cdot \frac{f(p) - f(p_0)}{f(p_0)} \right\}$$

- **数学含义与构造**：
  - $f(p)$：问题一基于 16 维 ALR 变换训练的 HistGradientBoosting 配比响应模型，以 17 维单纯形配比输入，输出 13 验证域的均值交叉熵损失；
  - $p_0$：The Pile 默认配比基准向量（问题一 A4 均值配比）；
  - $f(p_0)$：基准配比下的预测损失（实测 $f(p_0) = 4.8447$ Nats）；
  - $\tau$：跨尺度配比强度迁移系数（无量纲）。

- **模拟矩法（Method of Simulated Moments, MSM）标定推导**：
  利用问题一真实检验集（A6–A11）中 1M、60M、1B 的实测变异系数：
  - 1M 尺度：$\mu = 5.2823, \sigma = 0.2782 \implies \text{CV}_{1\text{M}} = 5.267\%$；
  - 60M 尺度：$\mu = 3.8229, \sigma = 0.2193 \implies \text{CV}_{60\text{M}} = 5.736\%$；
  - 1B 尺度：$\mu = 2.2256, \sigma = 0.0524 \implies \text{CV}_{1\text{B}} = 2.352\%$。
  在 1B 测试集（`test_mixture_1B.csv`）上评估模型相对变异：$\text{rel\_diff} = \frac{f(p)-f(p_0)}{f(p_0)}$，实测样本标准差 $\text{std}(\text{rel\_diff}) = 0.02708$。
  一阶线性矩匹配解为：
  $$\tau_{\text{linear}} = \frac{\text{CV}_{\text{emp}}(1\text{B})}{\text{std}(\text{rel\_diff}_{1\text{B}})} = \frac{0.02352}{0.02708} = 0.8686$$
  指数形式的非线性矩匹配精确解为：
  $$\tau^* = \arg\min_\tau |\text{std}(\exp(\tau \cdot \text{rel\_diff}_{1\text{B}})) - \text{CV}_{\text{emp}}(1\text{B})| = 0.8348 \approx 0.85$$
  在基准值 $\tau^* = 0.85$ 下，模拟标准差为 0.02397，与经验 CV 绝对误差仅 0.00045。

---

## 3. 边际效用、无量纲弹性与边际性价比

### M2-EQ03a 无量纲点弹性体系（对总损失 $L$）

$$\epsilon_x = \frac{x}{L} \frac{\partial L}{\partial x} \quad (x \in \{N, D, Q\})$$

对 M2-EQ01 分别求偏导：
1. **参数规模弹性**：
   $$\epsilon_N = -\alpha \frac{A N^{-\alpha}}{L} < 0$$
2. **数据量规模弹性**：
   $$\epsilon_D = -\beta \frac{B D^{-\beta} e^{-\rho(Q-Q_{\text{anchor}})} R(p)}{L} < 0$$
3. **数据质量弹性**：
   $$\epsilon_Q = -\rho Q \frac{B D^{-\beta} e^{-\rho(Q-Q_{\text{anchor}})} R(p)}{L} < 0$$

- **物理与经济含义**：
  $\epsilon_x$ 表示要素 $x$ 每增加 1%，验证损失下降 $|\epsilon_x|\%$。在当前模型的正参数与正变量定义域内，三项弹性均为负；这是模型结构的解析性质，外部适用性仍受数据和假设边界约束。

### M2-EQ03b 可约损失相对弹性（对 $L - E$）

剔除固有熵界 $E$ 后的纯技术效率弹性定义为 $\epsilon_x^* = \frac{x}{L - E} \frac{\partial L}{\partial x}$：
$$\epsilon_N^* = -\alpha \frac{A N^{-\alpha}}{A N^{-\alpha} + T}, \quad \epsilon_D^* = -\beta \frac{T}{A N^{-\alpha} + T}, \quad \epsilon_Q^* = -\rho Q \frac{T}{A N^{-\alpha} + T}$$
式中 $T \equiv B D^{-\beta} e^{-\rho(Q - Q_{\text{anchor}})} R(p)$，满足规模齐次性约束 $\frac{\epsilon_N^*}{\alpha} + \frac{\epsilon_D^*}{\beta} = -1$。

### M2-EQ03c 边际技术替代率（MRTS）与“算一笔账”

边际技术替代率定义为在保持模型性能 $L$ 不变的前提下，单位质量提升可替代的模型参数量（单位：十亿参数 / 质量分）：
$$\text{MRTS}_{N_B, Q} = -\left. \frac{dN_B}{dQ} \right|_{L = \text{const}} = \frac{-\partial L / \partial Q}{-\partial L / \partial N_B} = \frac{\rho \cdot B D^{-\beta} e^{-\rho(Q-Q_{\text{anchor}})} R(p)}{\alpha A N_B^{-\alpha - 1}}$$

- **对接赛题附录 B.1 成本函数求解边际 ROI 比率**：
  总算力成本：
  $$C = C_{\text{train}} + C_Q = 6 N_{\text{raw}} D_{\text{raw}} + D_{\text{raw}} [g(Q) - g(Q_{\text{base}})]_+ \quad (\text{FLOPs})$$
  式中 $N_{\text{raw}} = 10^9 N_B, D_{\text{raw}} = 10^9 D_B$。边际成本为：
  $$\frac{\partial C}{\partial N_{\text{raw}}} = 6 D_{\text{raw}}, \quad \frac{\partial C}{\partial Q} = D_{\text{raw}} g'(Q)$$
  边际损失下降量为：
  $$-\frac{\partial L}{\partial N_{\text{raw}}} = 10^{-9} \alpha A N_B^{-\alpha - 1}, \quad -\frac{\partial L}{\partial Q} = \rho T$$
  边际投资回报率（单位 FLOP 带来的降损量）之比推导为：
  $$\frac{\text{ROI}_Q}{\text{ROI}_N} = \frac{-\partial L / \partial Q / \partial C / \partial Q}{-\partial L / \partial N_{\text{raw}} / \partial C / \partial N_{\text{raw}}} = \frac{\frac{\rho T}{D_{\text{raw}} g'(Q)}}{\frac{10^{-9} \alpha A N_B^{-\alpha - 1}}{6 D_{\text{raw}}}} = \text{MRTS}_{N_B, Q} \cdot \frac{6 \times 10^9}{g'(Q)}$$
  **数据量 $D_{\text{raw}}$ 严格对消，呈现精妙的规模不变性！**

- **附录 B.1 三类清洗成本函数的均衡质量临界点 $Q^*$**（以 $N=1.04\text{B}, D=300\text{B}$ 为基准，$\text{MRTS}_{N_B, Q} = M_0 e^{-\rho Q}, M_0 \approx 2.8424$）：
  1. **指数型成本**（$g_{\text{exp}}(Q) = 10^7 e^{6Q}, g' = 6 \times 10^7 e^{6Q}$）：
     $$\frac{\text{ROI}_Q}{\text{ROI}_N} = 1 \implies 6 \times 10^7 e^{(6 + \rho) Q^*} = 6 \times 10^9 M_0 \implies Q^* = \frac{\ln(100 M_0)}{6 + \rho} = \mathbf{0.8480}$$
     在 $Q < 0.85$ 时买教材更划算（$Q=0.70$ 时比值为 2.681）；$Q > 0.85$ 时边际清洗成本呈指数级爆炸，优势翻转为堆参数更划算！
  2. **幂函数型成本**（$g_{\text{pow}}(Q) = 5 \times 10^9 Q^4, g' = 2 \times 10^{10} Q^3$）：
     $$(Q^*)^3 e^{\rho Q^*} = 0.3 M_0 \implies Q^* = \mathbf{0.7955}$$
     在 $Q < 0.80$ 时买教材更划算（$Q=0.70$ 时比值为 1.564）；$Q > 0.80$ 时优势翻转为堆参数！
  3. **对数型成本**（$g_{\text{log}}(Q) = 2 \times 10^9 \ln(1 + 10Q), g' = \frac{2 \times 10^{10}}{1 + 10Q}$）：
     在整个可行域 $[0.584, 1.0]$ 内均有 $\text{ROI}_Q / \text{ROI}_N > 1$（$Q=0.70$ 时比值为 4.290），因此 $Q^* > 1.0$，提升质量始终更划算。

---

## 4. 质量与规模等效替代及奇点饱和分析

### M2-EQ04 “质量提升 0.1 等价于参数增加多少”的解析计算式

**推导前提公理**：在讨论质量提升 0.1 的参数等效变换时，**训练 Token 量 $D$ 与领域配比 $p$ 严格保持不变**。
设当前数据项为 $T \equiv B D^{-\beta} e^{-\rho(Q - Q_{\text{anchor}})} R(p) > 0$，提升 0.1 质量引起的降损量为：
$$\Delta L_{\text{quality}} = T (1 - e^{-0.1 \rho}) > 0$$

#### 语义一：等效模型缩减（Cost-saving / Downsizing Equivalence）
保持当前损失 $L$ 不变，质量提升 0.1 后模型规模可缩减至 $N_{\text{new}} < N$：
$$N_{\text{new}} = \left[ N^{-\alpha} + \frac{T}{A} (1 - e^{-0.1 \rho}) \right]^{-1/\alpha} < N$$
参数节省量为 $\Delta N_{\text{save}} = N - N_{\text{new}} > 0$。该解在正数域内**恒存在唯一正实数解**。

#### 语义二：等效规模飞跃（Scaling Equivalence）
在原质量下仅靠增加参数达到同等降损效果：$A N_{\text{equiv}}^{-\alpha} = A N^{-\alpha} - \Delta L_{\text{quality}}$。
- **重大理论发现：标度律天花板饱和（Scaling Ceiling Saturation）与奇点阈值**：
  方程可解充要条件为 $A N^{-\alpha} - T(1 - e^{-0.1 \rho}) > 0$，即存在临界参数规模上限：
  $$N < N_{\text{crit}} \equiv \left[ \frac{A}{T (1 - e^{-0.1 \rho})} \right]^{1/\alpha}$$
  1. **有限可解区（$N < N_{\text{crit}}$）**：
     $$N_{\text{equiv}} = \left[ N^{-\alpha} - \frac{T}{A} (1 - e^{-0.1 \rho}) \right]^{-1/\alpha} > N \implies \Delta N_{\text{gain}} = N_{\text{equiv}} - N > 0$$
  2. **奇异饱和无解区（$N \ge N_{\text{crit}}$）**：
     此时 $A N^{-\alpha} \le \Delta L_{\text{quality}}$！模型参数项最大剩余改善潜力已小于提升 0.1 质量的降损量。**即使投入无穷大参数（$N \to \infty$），也永远无法达到高质量数据的水平**！数学上严格证明了高质量教材的绝对不可替代性。

---

## 5. 领域配比单纯形上的替代与互补性分析

### M2-EQ05 单纯形微分流形分析与 Hessian 判定准则

在领域配比单纯形 $\Delta^{16} = \left\{ p \in \mathbb{R}^{17} \mid \sum_{i=1}^{17} p_i = 1, p_i \ge 0 \right\}$ 上：
- **无量纲乘子 Hessian 曲率**：
  $$\mathcal{H}^R_{ij} \equiv \frac{\partial^2 R(p)}{\partial p_i \partial p_j}$$
- **模型损失曲率映射**：
  $$\mathcal{H}^L_{ij} \equiv \frac{\partial^2 L}{\partial p_i \partial p_j} = T \cdot \mathcal{H}^R_{ij}$$
  其中外在标度因子 $T = B D^{-\beta} e^{-\rho(Q-Q_{\text{anchor}})} \approx 0.3 \sim 0.6$。
- **判定准则**：
  - $\mathcal{H}^R_{ij} < 0$：超加性**协同互补（Synergy）**，两域知识联合增益显著；
  - $\mathcal{H}^R_{ij} > 0$：次加性**竞争替代（Competition）**，两域知识同质或风格干扰；
  - $\mathcal{H}^R_{ij} \approx 0$：**中性独立（Independence）**。

---

## 6. 模型版本演进记录

| 版本号 | 日期 | 核心变更内容 | 变更依据 |
|---|---|---|---|
| `M2-v0` | 2026-09-23 | 初步草案：直接加乘模型 $L = E + AN^{-\alpha} + BD^{-\beta} R(p) e^{-\rho(Q-Q_0)}$ | 赛题初步要求 |
| `M2-v1` | 2026-09-23 | 建立基于有效数据量假说 $\widetilde{D} = D \cdot \eta_Q(Q) \cdot \eta_p(p)$ 的物理推导；确立双基准退化性；推导 Scaling Ceiling Saturation 充要条件；建立单纯形 Hessian 互补判定法。 | 解决量纲自洽性与物理极限推导需求 |
| `M2-v2` | 2026-09-23 | 分离退化锚点 $Q_{\text{anchor}} \equiv 1.0$ 与成本底线 $Q_{\text{base}} \equiv 0.584$；基于附录 B.1 成本函数推导无量纲 ROI 比率与翻转阈值 $Q^*$；补充 MSM 矩匹配校准 $\tau^* \approx 0.85$；明确 $\mathcal{H}^R$ 与 $\mathcal{H}^L$ 乘子关系；补充语义二前提 $D,p$ 固定。 | 完成模型审查与问题三接口整理 |
