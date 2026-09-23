# model.md — 问题三核心数学模型与优化理论推导

> 本文档依据《数学建模项目总规范》建立，详述问题三（算力约束下的多维资源联合优化与结构性转移）的完整数学模型、量纲闭合证明、一阶 KKT 均衡条件、上下文临界解析式与结构性转移统计判据。

---

## 1. 统一量纲系与变量定义

为彻底杜绝原始量纲混淆（FLOPs、Tokens、参数量之间的数个量级跨度），统一建立以“十亿 / EFLOPs”为基准的归一化变量体系：

| 物理量 | 原始符号与单位 | 归一化符号 | 定义式与基准单位 | 典型数值范围 |
|---|---|---|---|---|
| 模型参数量 | $N$ (个) | $n$ | $n = N / 10^9$ ($\text{Billion params}$) | $n \in [0.001, 1000.0]$ ($1\text{M} \sim 1000\text{B}$) |
| 训练数据量 | $D$ (Tokens) | $d$ | $d = D / 10^9$ ($\text{Billion tokens}$) | $d \in [0.1, 100000.0]$ ($100\text{M} \sim 100\text{T}$) |
| 总算力预算 | $C$ (FLOPs) | $\bar{C}$ | $\bar{C} = C / 10^{18}$ ($\text{EFLOPs}$) | $\bar{C} \in [1, 10^7]$ ($10^{18} \sim 10^{25}\text{ FLOPs}$) |
| 数据质量评分 | $Q$ (分) | $Q$ | 无量纲连续评分，基线 $Q_0 = 0.584$ | $Q \in [0.584, 1.000]$ |
| 上下文长度 | $L_{\text{ctx}}$ (Tokens) | $L_{\text{ctx}}$ | 外生窗口长度 (Tokens)，依据 C7 | $L_{\text{ctx}} \in \{2048, 4096, 8192, 32768, 131072\}$ |
| 领域配比 | $p \in \mathbb{R}^{17}$ | $p$ | 17 领域单纯形配比，$\sum p_i = 1$ | $p_i \ge \varepsilon_p = 0.005$ |
| 增量清洗成本 | $h(Q)$ (FLOPs/Token) | $\bar{h}(Q)$ | $\bar{h}(Q) = [g(Q) - g(Q_0)]_+ / 10^9$ ($\text{GFLOPs/Token}$) | $\bar{h} \in [0, 4.2]$ ($\text{GFLOPs/Token}$) |

### 量纲闭合推导证明
算力消耗由三部分组成：
1. **基础训练开销**：$C_{\text{train}} = 6 N D = 6 (10^9 n)(10^9 d) = 6 \times 10^{18} n d$ (FLOPs)；
2. **长文本注意力开销**：$C_{\text{attn}} = \eta N D L_{\text{ctx}} = \eta (10^9 n)(10^9 d) L_{\text{ctx}} = (\eta L_{\text{ctx}}) \times 10^{18} n d$ (FLOPs)，其中 $\eta = 2 \times 10^{-4}$；
3. **数据质量清洗开销**：$C_Q = D \cdot h(Q) = (10^9 d) \cdot h(Q) = 10^{18} d \left( \frac{h(Q)}{10^9} \right) = 10^{18} d \bar{h}(Q)$ (FLOPs)。

总算力开销方程为：
$$C_{\text{total}} = C_{\text{train}} + C_{\text{attn}} + C_Q = 10^{18} d \left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right]$$
两端除以 $10^{18}$，严格得到归一化算力约束：
$$d \left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right] \le \bar{C}$$
- **量纲校验**：括号内 $(6 + \eta L_{\text{ctx}})n$ 与 $\bar{h}(Q)$ 的物理单位均为 $\text{GFLOPs/Token}$（即 $10^9\text{ FLOPs/Token}$）；乘以 $d$（单位为 $10^9\text{ Tokens}$），得到 $10^9 \times 10^9\text{ FLOPs} = 10^{18}\text{ FLOPs} = 1\text{ EFLOP}$，与右侧 $\bar{C}$ 的单位严格相等。量纲实现绝对自洽闭合！

---

## 2. 核心数学模型体系

### 2.1 统一有界非线性规划模型（M3-EQ01）
$$\begin{aligned}
\min_{n, d, Q} \quad & L(n, d, Q; p) = E + A n^{-\alpha} + B d^{-\beta} \exp\{-\rho(Q - Q_{\text{anchor}})\} R(p) \\
\text{s.t.} \quad & d \left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right] \le \bar{C} \\
& n \in [n_{\min}, n_{\max}] = [0.001, 1000.0] \\
& d \in [d_{\min}, d_{\max}] = [0.1, 100000.0] \\
& Q \in [Q_0, 1.000] = [0.584, 1.000] \\
\end{aligned}$$
其中 $p$ 不是本问的连续决策变量，而是情景参数，分别取 $p_0$ 与 $p_{\text{opt}}$ 两个固定值。
其中：
- 广义标度律参数（来自问题二实证标定）：
  $$E = 1.6898, \quad A = 0.3540, \quad \alpha = 0.3400, \quad B = 1.2403, \quad \beta = 0.2799, \quad \rho = 0.6646, \quad \tau^* = 0.85$$
- **质量项模型条件性**：质量项采用单调指数衰减形式 $\exp\{-\rho(Q - Q_{\text{anchor}})\}$。问题二对半合成数据 B6/B7 的分析显示，固定切片中相邻离散质量点的损失下降比例分别约为 74.6% 与 74.8%。因此所得最优解 $Q^*$ 是在该**单调指数质量项模型假定**下的条件最优解；该半合成数据结果不等同于对真实清洗过程的验证；
- 理论退化锚点 $Q_{\text{anchor}} \equiv 1.0$；
- 成本基线起点 $Q_0 \equiv Q_{\text{base}} = 0.584$（依据附录 B.1 正文“可由附件 A 质量评分或合理假设给出”，标定为预训练语料主体开放网络抓取域 Common Crawl 与 C4 的合成中位数 $\text{Median}(Q_{\text{cc}} \cup Q_{\text{c4}}) \approx 0.584$，代表未经额外处理的公网爬虫原始语料质量基线，由此起算增量清洗成本）；
- 附录 B.1 数据质量成本函数 $g(Q)$：
  1. 指数型：$g_{\text{exp}}(Q) = \gamma e^{\lambda Q}$，$\gamma = 10^7, \lambda = 6.0$；
  2. 幂函数型：$g_{\text{pow}}(Q) = \gamma Q^\lambda$，$\gamma = 5 \times 10^9, \lambda = 4.0$；
  3. 对数型：$g_{\text{log}}(Q) = \gamma \ln(1 + \lambda Q)$，$\gamma = 2 \times 10^9, \lambda = 10.0$；
  - 归一化增量清洗成本：$\bar{h}(Q) = [g(Q) - g(Q_0)]_+ / 10^9$。

---

### 2.2 严格单调性解析降维（M3-EQ02）

- **有效预算吸收区间**：
  $$\begin{aligned}
  \bar{C}_{\min} &= d_{\min} \left[ (6 + \eta L_{\text{ctx}}) n_{\min} + \bar{h}(Q_0) \right] \\
  \bar{C}_{\max} &= d_{\max} \left[ (6 + \eta L_{\text{ctx}}) n_{\max} + \bar{h}(1.0) \right]
  \end{aligned}$$
  在赛题设定的预算谱 $\bar{C} \in [10, 10^6]$（即 $10^{19} \sim 10^{24}$ FLOPs）内，$\bar{C}_{\min} \approx 0.1 \times 6 \times 0.001 = 0.0006 \ll 10$，$\bar{C}_{\max} \approx 100000 \times (6.41 \times 1000 + 4.2) \approx 6.4 \times 10^8 \gg 10^6$。预算完全落在有效吸收区间内。
- **活跃约束与解析消元**：
  在变量未触及硬物理上界时，对任意给定的 $(n, Q, p)$，因 $\beta > 0$ 且数据项乘子严格为正，有：
  $$\frac{\partial L}{\partial d} = -\beta B d^{-\beta - 1} \exp\{-\rho(Q - 1.0)\} R(p) < 0$$
  目标函数关于数据量 $d$ 严格单调递减。因此，最优资源配置必将全部算力预算耗尽，预算不等式约束必在边界严格取等（$\lambda > 0$）。
  由此可完全解析求解最优数据量闭式解：
  $$d^*(n, Q; \bar{C}, L_{\text{ctx}}) = \frac{\bar{C}}{(6 + \eta L_{\text{ctx}}) n + \bar{h}(Q)}$$

- **两种固定配比情景分析**：
  为保持结论的稳健与实验可追溯性，本问考察**两种固定配比情景**：
  1. **基准配比情景（$p_0$）**：$p_0$ 是 The Pile 原始配比在附件 A4 中的平均向量，不是 17 个领域等比例的均匀配比；$R(p_0) = 1.0$，广义标度律在该基准配比下保留质量调整项；
  2. **优质推荐配比情景（$p^*_{\text{opt}}$）**：固定为问题一在多尺度验证下表现最优的推荐配方，以名义校准值 $\tau=0.85$ 代入计算得 $R(p^*_{\text{opt}}) \approx 0.9602$。该方案属于跨来源迁移假设下的条件最优解；
  3. **$\tau$ 情景敏感性分析**：在 $\tau \in [0.45, 1.50]$ 上选取 $\{0.45, 0.65, 0.85, 1.15, 1.50\}$ 五个网格点重新计算 $R(p;\tau) = \exp\left(\tau \frac{f(p)-f(p_0)}{f(p_0)}\right)$。该区间是情景扫描范围，不是置信区间，结果不代表对连续区间所有取值均已求解。
  引入对数参数变换 $u = \ln n \in [\ln n_{\min}, \ln n_{\max}]$：
  $$\min_{u \in [\ln 0.001, \ln 1000.0], \, Q \in [0.584, 1.000]} \tilde{L}(u, Q; p) = E + A e^{-\alpha u} + B \left[ \frac{\bar{C}}{(6 + \eta L_{\text{ctx}}) e^u + \bar{h}(Q)} \right]^{-\beta} e^{-\rho(Q - 1.0)} R(p), \quad p \in \{p_0, p_{\text{opt}}\}$$
  对每个固定配比情景，单调消元将原约束问题精确化为二维连续问题；数值解由多起点局部优化求得，并以差分进化交叉核验。由于目标函数未被证明为全局凸函数，数值交叉核验不构成全局最优性的解析证明。

---

### 2.3 完整 Kuhn-Tucker (KKT) 一阶均衡条件（M3-EQ03）

构建拉格朗日函数：
$$\begin{aligned}
\mathcal{L}(n, d, Q, \lambda, \mu) = & L(n, d, Q) + \lambda \left\{ d \left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right] - \bar{C} \right\} \\
& - \mu_n^- (n - n_{\min}) + \mu_n^+ (n - n_{\max}) \\
& - \mu_d^- (d - d_{\min}) + \mu_d^+ (d - d_{\max}) \\
& - \mu_Q^- (Q - Q_0) + \mu_Q^+ (Q - 1.0)
\end{aligned}$$
其中 $\lambda \ge 0$ 为预算约束乘子，$\mu \ge 0$ 为各变量边界的互补松弛乘子。

在内点最优解处（$\mu = 0, \lambda > 0$）：
1. **关于模型参数量 $n$ 的一阶导数**：
   $$\frac{\partial \mathcal{L}}{\partial n} = -\alpha A n^{-\alpha - 1} + \lambda d (6 + \eta L_{\text{ctx}}) = 0 \implies \alpha A n^{-\alpha} = \lambda d (6 + \eta L_{\text{ctx}}) n$$
2. **关于训练数据量 $d$ 的一阶导数**：
   $$\frac{\partial \mathcal{L}}{\partial d} = -\beta B d^{-\beta - 1} e^{-\rho(Q - 1.0)} R(p) + \lambda \left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right] = 0$$
   代入 $\left[ (6 + \eta L_{\text{ctx}}) n + \bar{h}(Q) \right] = \frac{\bar{C}}{d}$，两端同乘 $d$：
   $$\beta B d^{-\beta} e^{-\rho(Q - 1.0)} R(p) = \lambda \bar{C}$$
   联立消除拉格朗日乘子 $\lambda = \frac{\beta T(d, Q)}{\bar{C}}$（其中 $T(d, Q) = B d^{-\beta} e^{-\rho(Q-1.0)} R(p)$），得到**参数-数据配置的规模边际均衡方程**：
   $$\alpha A n^{-\alpha} = \beta T(d, Q) \cdot \frac{d (6 + \eta L_{\text{ctx}}) n}{\bar{C}} = \beta T(d, Q) \cdot s_{\text{train+attn}}$$
   其中 $s_{\text{train+attn}} = \frac{d(6 + \eta L_{\text{ctx}})n}{\bar{C}} = 1 - s_Q$ 为基础训练加注意力的算力支出份额。
3. **关于数据质量 $Q$ 的一阶导数**：
   $$\frac{\partial \mathcal{L}}{\partial Q} = -\rho T(d, Q) + \lambda d \bar{h}'(Q) = 0$$
   代入 $\lambda = \frac{\beta T(d, Q)}{\bar{C}} = \frac{\beta T(d, Q)}{d [(6 + \eta L_{\text{ctx}}) n + \bar{h}(Q)]}$，$T(d, Q)$ 与 $d$ 精妙对消，得到**质量投资边际替代率（MRTS）均衡公式**：
   $$\rho = \beta \frac{\bar{h}'(Q)}{(6 + \eta L_{\text{ctx}}) n + \bar{h}(Q)} = \beta \frac{g'(Q) / 10^9}{(6 + \eta L_{\text{ctx}}) n + \bar{h}(Q)}$$

- **边界区制解析判据**：
  - **角点区制 I（质量停留在基线 $Q^* = Q_0$）**：
    若在 $Q = Q_0^+$ 处边际清洗成本过高，使得 $-\rho + \beta \frac{g'(Q_0)/10^9}{(6 + \eta L_{\text{ctx}}) n} > 0 \iff \mu_Q^- > 0$。此时继续清洗数据带来的降损不足以弥补挤占 Token 带来的损失，最优决策为 $Q^* = Q_0$；
  - **饱和区制 III（质量触碰完美天花板 $Q^* = 1.0$）**：
    若在 $Q = 1.0^-$ 处边际清洗成本仍然低廉，使得 $-\rho + \beta \frac{g'(1.0)/10^9}{(6 + \eta L_{\text{ctx}}) n + \bar{h}(1.0)} < 0 \iff \mu_Q^+ > 0$。此时质量投资始终具备正向超额边际效益，最优决策为 $Q^* = 1.0$；
  - **集约区制 II（内点平滑均衡 $Q^* \in (Q_0, 1.0)$）**：
    当边际收益与边际成本在可行域内部取得平衡时，由上述均衡式隐函数确定唯一的 $Q^*$。

---

### 2.4 上下文长度解析临界值与算力折扣定理（M3-EQ04）

令长文本注意力开销等于基础训练开销：
$$C_{\text{attn}} = C_{\text{train}} \iff (\eta L_{\text{ctx}}) \times 10^{18} n d = 6 \times 10^{18} n d \iff L_{\text{ctx}}^{\text{crit}} = \frac{6}{\eta} = \frac{6}{2 \times 10^{-4}} = 30,000 \text{ Tokens}$$

- **定义“相对于基础训练的注意力开销比率”**：
  $$\psi(L_{\text{ctx}}) = \frac{C_{\text{attn}}}{C_{\text{train}}} = \frac{\eta L_{\text{ctx}}}{6} = \frac{L_{\text{ctx}}}{30000}$$
- **在附件 C7 实测 5 档模型窗口中的定量评估**：
  1. $L_{\text{ctx}} = 2048$（Pythia, Cerebras-GPT, Falcon）：$\psi = 2048 / 30000 = 6.83\%$；
  2. $L_{\text{ctx}} = 4096$（Llama-2, Yi, DeepSeek-LLM）：$\psi = 4096 / 30000 = 13.65\%$；
  3. $L_{\text{ctx}} = 8192$（Meta-Llama-3, Gemma-2）：$\psi = 8192 / 30000 = 27.31\%$；
  4. $L_{\text{ctx}} = 32768$（Mistral-7B, Qwen2, Qwen2.5）：$\psi = 32768 / 30000 = 109.23\%$（注意力开销超过基础训练）；
  5. $L_{\text{ctx}} = 131072$（Llama-3.1-8B）：$\psi = 131072 / 30000 = 436.91\%$（注意力开销是基础训练的 $4.37$ 倍！）。
- **有效训练算力折扣因子（Effective Compute Discount Factor）**：
  在不考虑质量清洗时，可用于基础训练的算力占比为：
  $$\phi(L_{\text{ctx}}) = \frac{C_{\text{train}}}{C_{\text{train}} + C_{\text{attn}}} = \frac{1}{1 + \psi(L_{\text{ctx}})} = \frac{1}{1 + \frac{L_{\text{ctx}}}{30000}}$$
  在 128k 超长上下文中，$\phi = \frac{1}{1 + 4.369} = 18.63\%$。这意味着超过 $81.37\%$ 的前向算力被长序列注意力开销吞噬，使得模型参数与训练 Token 的规模配置被严重压缩。

---

### 2.5 结构性转移（Structural Shift）的三重科学判据（M3-EQ05）

为使“结构性转移”具备可复核的模型内定义，设立明确的数值容差与识别流程：

1. **准则一：最优质量的边界区制跃迁（Regime Transition with Tolerance $\varepsilon_Q = 0.005$）**：
   - 区制 I（粗放扩展期）：$Q^*(\bar{C}) - Q_0 \le \varepsilon_Q$（质量停留在基线，无清洗投入）；
   - 区制 II（集约清洗期）：$Q_0 + \varepsilon_Q < Q^*(\bar{C}) < 1.0 - \varepsilon_Q$（质量进入内点动态调整）；
   - 区制 III（质量饱和期）：$1.0 - Q^*(\bar{C}) \le \varepsilon_Q$（质量达到理论上限）。
2. **准则二：支出份额的分段回归与扫描点稳定性分析（Segmented Regression & Changepoint）**：
   定义三类算力支出份额：
   $$s_N(\bar{C}) = \frac{6 n d}{\bar{C}}, \quad s_Q(\bar{C}) = \frac{d \bar{h}(Q)}{\bar{C}}, \quad s_{\text{attn}}(\bar{C}) = \frac{(\eta L_{\text{ctx}}) n d}{\bar{C}}$$
   对 50 点对数扫描所得的支出份额序列 $s_Q(\ln \bar{C})$，拟合两相分段折线回归模型（Broken-stick Model）：
   $$s_Q(\ln \bar{C}) = \beta_0 + \beta_1 \ln \bar{C} + \beta_2 (\ln \bar{C} - \kappa)_+ + \epsilon$$
   通过候选折点网格搜索，以残差平方和最小确定拐点 $\kappa^* = \ln \bar{C}_{\text{crit}}$。随后对 50 个确定性扫描点作 1000 次有放回重抽样，报告折点及斜率变化量的 2.5%–97.5% 经验分位范围，用于描述固定网格重加权下的数值稳定性。由于扫描点并非独立随机观测，该范围不解释为总体置信区间，也不据此计算显著性 $p$ 值。
3. **准则三：对数缩放弹性偏离检验（Log-Scaling Elasticity Drift）**：
   计算数值最优扩张弹性：
   $$\kappa_n(\bar{C}) = \frac{d \ln n^*}{d \ln \bar{C}}, \quad \kappa_d(\bar{C}) = \frac{d \ln d^*}{d \ln \bar{C}}$$
   在经典 Chinchilla 无质量清洗情形下，$\kappa_n = \frac{\beta}{\alpha+\beta} \approx 0.4515, \kappa_d = \frac{\alpha}{\alpha+\beta} \approx 0.5485$ 为恒定常数。比较质量投资介入前后 $\kappa_n$ 与 $\kappa_d$ 的模型内偏离与回流，不作样本总体层面的显著性推断。

---

## 3. 求解算法与数值实现规范

1. **主求解器（多起点有界连续局部优化）**：
   - 算法：`scipy.optimize.minimize(method='L-BFGS-B')`；
   - 搜索空间：$u = \ln n \in [\ln 0.001, \ln 1000.0], Q \in [0.584, 1.000]$；
   - 多起点策略：10 个对数均匀分布的 $n_0 \in [0.01, 100]$ 与 5 个均匀分布的 $Q_0 \in [0.584, 0.95]$ 笛卡尔积生成 50 个初值；
   - 停止容差：`ftol = 1e-15`, `gtol = 1e-12`, `maxiter = 1000`；
2. **全局验证求解器（差分进化全局复核）**：
   - 算法：`scipy.optimize.differential_evolution`；
   - 参数：`seed = 42`, `popsize = 20`, `maxiter = 2000`, `tol = 1e-8`, `mutation = (0.5, 1.0)`, `recombination = 0.7`；
   - 检验指标：主求解器与全局求解器的最优 Loss 相对误差要求 $< 10^{-6}$；
3. **预算残差与约束活跃性校验**：
   - 校验：$|C_{\text{spent}} - C| / C < 10^{-12}$，确保预算完全用尽无浪费。
