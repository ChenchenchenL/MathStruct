# model.md — 问题四数学模型建立与解析推导

> 本文档依据《数学建模项目总规范》建立，详述问题四（技术演进分析与前沿预测）的全部数学推导、状态空间方程、反事实因果贡献分解、桥接映射及闭环验证体系。所有公式遵循 `M4-EQ*` 唯一定义编号。

---

## 1. 建模总体思路与模块数据流

问题四要求在宏观技术演进视角下，分离大语言模型能力增长中“规模扩张”与“非规模技术进步”的贡献，并在算力放缓情景下预测未来 12/24 个月的能力前沿。整体建模体系包含五大模块：

```text
C1/C2 (Leaderboard) + C4 (Epoch AI) + C8 (Subtasks)
                     │
                     ▼
       [模块一：样本筛选、C8微观复核与综合能力度量]
       (M4-EQ01 ~ M4-EQ03, 排除未授权, 扣除机会水平)
                     │
                     ├────────────────────────────────┐
                     ▼                                ▼
       [模块二：Loss-to-Benchmark 桥接]      [模块三：有界动态随机前沿 (SFA)]
       (M4-EQ04, C6 负单调 Logistic,        (M4-EQ05 ~ M4-EQ07, 状态空间动力学,
        彻底隔离 T20 反向假相关)               反事实 Shapley 分解 M4-EQ08)
                     │                                │
                     │                                ▼
                     │              [模块四：算力放缓外推与双路径闭环]
                     └─────────────► 路径 A: 动态前沿外推
                                     路径 B: 问题三最优损失映射 M4-EQ10
                                                      │
                                                      ▼
                                     [模块五：多源不确定性与时间外检验]
                                     (M4-EQ11, 聚类 Bootstrap 与 OOT 回测)
```

---

## 2. 详细数学模型构建

### 2.1 样本筛选准则与分层映射方程（M4-EQ01）
定义原始模型样本空间 $\Omega$。建立指示函数进行多重筛选：
$$\mathbb{I}_{\text{valid}}(i) = \mathbb{I}_{\text{open}}(i) \times \mathbb{I}_{\text{license}}(i) \times \mathbb{I}_{\text{eval}}(i)$$
- **权重开源指示**：$\mathbb{I}_{\text{open}}(i) = 1$ 当且仅当 C4 中 `Open model weights? == 'Yes'` 或属于 `Open weights` 范畴；
- **许可证指示**：$\mathbb{I}_{\text{license}}(i) = 1$ 当且仅当许可证属于学术可复现范围（Apache, MIT, BSD, Llama, Gemma, Qwen 等）；
- **分层划分（Pretrained vs Chat/Finetuned）**：
  $$r_i = \begin{cases} 0 & (\text{Pretrained 基座组}) \\ 1 & (\text{Chat / Finetuned 对齐组}) \end{cases}$$
- **时间轴与起报点**：
  以评测记录日 $t_i \in [T_{\min}, T_0]$ 为主时间轴，预测起报基准严格锚定在最后有效记录日 $T_0 = 2025\text{-}03\text{-}13$。

---

### 2.2 综合能力度量与 C8 逐任务微观加权复核方程（M4-EQ02 ~ M4-EQ03）

#### 2.2.1 综合能力宏平均度量（M4-EQ02）
针对 Open LLM Leaderboard v2 的 6 项基准（IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO），定义固定权重宏平均综合能力得分 $S_i$：
$$S_i = \frac{1}{6} \sum_{k=1}^6 S_{k, i} \in [0, 100]$$
同时在分析中保留每一维度的独立度量。

#### 2.2.2 C8 逐任务底层聚合与基线归一化复核（M4-EQ03）
对 C8 中 BBH 的 24 项独立子任务 $j=1,\dots,24$，自底向上提取有效测试样本量 $w_{i, j}$ 与原始准确率 $\text{acc}_{i, j}$：
$$S_{\text{BBH}, i}^{\text{raw}} = \frac{\sum_{j=1}^{24} w_{i, j} \cdot \text{acc}_{i, j}}{\sum_{j=1}^{24} w_{i, j}} \times 100$$
官方榜单采用扣除随机猜测机会水平（Chance level baseline $\gamma_{\text{BBH}} \approx 27.96\%$）后的归一化得分：
$$S_{\text{BBH}, i}^{\text{norm}} = \frac{S_{\text{BBH}, i}^{\text{raw}} - \gamma_{\text{BBH}}}{100 - \gamma_{\text{BBH}}} \times 100$$
经实测回归校验，$R^2 = 0.9982$，完美解释了 C8 原始准确率与 C1 报告分数之间的转换关系，排除了版本损坏与双重汇总干扰。

---

### 2.3 Loss–Benchmark 连续单调饱和桥接模型（M4-EQ04）
依据真实物理规律，验证交叉熵损失 $L$ 越低，语言模型在下游多维基准上的表现越好。构建连续、光滑、有界的负单调 Logistic 桥接模型：
$$S_k(L) = \frac{100}{1 + \exp(-(a_k - b_k L))} \quad (b_k > 0)$$
- **导数与单调性**：
  $$\frac{\partial S_k}{\partial L} = - \frac{100 b_k \exp(-(a_k - b_k L))}{\left[ 1 + \exp(-(a_k - b_k L)) \right]^2} < 0$$
  严格保证负相关性（$L \downarrow \implies S_k \uparrow$），彻底推翻出题组在 T20 植入的 $r = +0.68$ 伪结论！
- **渐近极限**：
  $$\lim_{L \to 0^+} S_k(L) = \frac{100}{1 + e^{-a_k}} \approx 100, \quad \lim_{L \to \infty} S_k(L) = 0$$

---

### 2.4 有界动态随机生产前沿（SFA）动力学模型（M4-EQ05 ~ M4-EQ07）

#### 2.4.1 有界 Logit 空间映射（M4-EQ05）
为避免直接在线性空间外推导致预测得分突破 100 分天花板，对综合得分引入平滑 Logit 变换：
$$y_i = \text{logit}\left(\frac{S_i}{100}\right) = \ln \left( \frac{S_i / 100 + \varepsilon}{1 - S_i / 100 + \varepsilon} \right) \quad (\varepsilon = 10^{-4})$$

#### 2.4.2 动态随机前沿观测方程（M4-EQ06）
在连续时间与算力空间中，模型能力被分解为规模弹性、非规模技术状态、类型效应、家族效应与复合误差：
$$y_i = a + b \ln C_i + z_{t_i} + v_{\text{chat}} \cdot r_i + u_{f_i} + e_i - d_i$$
- $C_i$：模型累计训练算力（$\text{FLOPs}$），$b \ge 0$ 为规模弹性系数；
- $z_{t}$：月份 $t$ 的宏观非规模技术状态（架构、算法、数据质量累积创新）；
- $v_{\text{chat}}$：Post-training 对齐微调相较于纯基座模型的结构性抬升效应；
- $u_{f}$：模型所属家族（如 Llama, Qwen, Yi, Gemma）的固定/随机效应；
- $e_i \sim \mathcal{N}(0, \sigma_e^2)$：对称度量/评测噪声；
- $d_i \ge 0, \, d_i \sim \mathcal{HN}(0, \sigma_d^2)$：单侧半正态分布的非负技术差距（距前沿的距离）。

#### 2.4.3 状态转移方程（带阻尼局部线性趋势，M4-EQ07）
技术状态的动态演进服从局部趋势系统：
$$z_{t+1} = z_t + v_t + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma_\xi^2)$$
$$v_{t+1} = \delta v_t + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2) \quad (\delta \in [0.90, 1.0])$$
其中 $v_t$ 为技术进步的瞬时动量速度，$\delta$ 为长期阻尼系数，防止 24 个月外推出现超指数发散。

#### 2.4.4 前沿生产边界定义
固定类型 $r$ 与标杆领先家族 $u^*$，消除非效率项 $d_i = 0$ 与对称噪声 $e_i = 0$，得到前沿预测函数：
$$F(C, t; r) = \text{logit}^{-1} \left( a + b \ln C + z_t + v_{\text{chat}} \cdot r + u^* \right) \times 100$$

---

### 2.5 反事实 Shapley 排序平均因果贡献分解（M4-EQ08）
在历史考察区间 $[t_0, t_1]$ 内，前沿算力自 $C_0$ 扩张至 $C_1$，技术状态自 $z_0$ 演进至 $z_1$。总前沿能力增量为：
$$\Delta_{\text{total}} = F(C_1, t_1) - F(C_0, t_0)$$
为消除非线性 Logit 映射下先后调整自变量的路径依赖偏差，构建 4 个反事实状态，采用 Shapley 平均法进行严格拆解：
- **算力规模扩张贡献**：
  $$\Delta_C = \frac{1}{2} \left\{ \left[ F(C_1, t_0) - F(C_0, t_0) \right] + \left[ F(C_1, t_1) - F(C_0, t_1) \right] \right\}$$
- **非规模技术进步贡献**：
  $$\Delta_{\text{tech}} = \frac{1}{2} \left\{ \left[ F(C_0, t_1) - F(C_0, t_0) \right] + \left[ F(C_1, t_1) - F(C_1, t_0) \right] \right\}$$
- **公理化恒等性与贡献占比**：
  $$\Delta_C + \Delta_{\text{tech}} \equiv \Delta_{\text{total}}$$
  $$\text{Share}_C = \frac{\Delta_C}{\Delta_{\text{total}}} \times 100\%, \quad \text{Share}_{\text{tech}} = \frac{\Delta_{\text{tech}}}{\Delta_{\text{total}}} \times 100\%$$

---

### 2.6 算力放缓条件情景设定与未来 12/24 个月前沿外推（M4-EQ09）

#### 2.6.1 历史前沿算力年化增速
从可比历史窗口 $[t_0, T_0]$ 拟合前沿算力对数轨迹：
$$\ln C_{\text{frontier}}(t) = \ln C_0 + g_C \cdot (t - t_0)$$
由此测得基准年化几何增速 $g_C$。

#### 2.6.2 三档条件放缓情景
- **情景 1（历史动量）**：$g_1 = g_C$；
- **情景 2（增速腰斩）**：$g_2 = 0.5 g_C$；
- **情景 3（算力停滞）**：$g_3 = 0$（算力保持在 $T_0$ 水平不变）。

在时间步长 $\Delta t \in \{12\text{ 个月}, 24\text{ 个月}\}$ 下，未来算力为 $C(T_0 + \Delta t) = C(T_0) \exp(g_s \cdot \Delta t)$。结合外推的技术状态 $z(T_0 + \Delta t)$，输出对应的前沿预测值。

---

### 2.7 与前三问形成闭环：路径 A 与路径 B 双证据链对偶校验（M4-EQ10）
- **路径 A（宏观动态前沿直接推演）**：
  $$S_{\text{PathA}}(t) = F(C_s(t), t; r=0)$$
- **路径 B（标度律最优配置 + 桥接映射）**：
  将情景算力 $C_s(t)$ 输入第三问 KKT 优化模型求解理论最优损失 $L^*(C_s(t))$，再代入式 (10) 桥接模型：
  $$S_{\text{PathB}}(t) = S_{\text{bridge}}\left( L^*(C_s(t)) \right) = \frac{100}{1 + \exp\left( -(a - b L^*(C_s(t))) \right)}$$
- **对偶残差与技术溢出**：
  在历史重叠期计算两者的吻合度与残差：
  $$\Delta_{\text{dual}}(t) = S_{\text{PathA}}(t) - S_{\text{PathB}}(t)$$
  该差值客观反映了下游多维评测基准所捕获的、未被单一自回归交叉熵损失完全涵盖的架构与对齐技术溢出。

---

### 2.8 多源误差传播、置信区间与滚动时间外检验（M4-EQ11）
- **族群聚类 Bootstrap（1000 次重采样）**：
  每次重采样均重新估计：SFA 参数 $(a, b, \sigma_e, \sigma_d)$、动态状态 $\{z_t\}$、桥接参数 $(a_k, b_k)$ 与情景外推轨迹。
- **输出区间**：
  1. **条件前沿区间（Frontier Interval）**：反映理论技术上限的 $80\%$ 与 $95\%$ 不确定性带；
  2. **新模型观测区间（Observation Interval）**：在条件前沿基础上叠加技术差距 $d_i$ 与度量噪声 $e_i$。
- **滚动时间外回测（OOT Validation）**：
  将历史样本在截断日 $T_{\text{split}} = 2024\text{-}10\text{-}01$ 处截断，用历史训练前沿模型，预测后续 5 个月的前沿表现，报告平均绝对误差 $\text{MAE}$ 与 95% 置信区间实际覆盖率。
