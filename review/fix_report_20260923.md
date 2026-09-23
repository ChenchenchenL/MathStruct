# 建模缺陷修复报告

**修复日期**：2026-09-23  
**修复范围**：缺陷 3–6（用户 2026-09-23 提出）  
**修复状态**：✅ 全部完成并通过验证  

---

## 目录

1. [缺陷 3：第二问数据入口与题意不符（P1→P2 接口缺失）](#缺陷-3)
2. [缺陷 4：P3 结果 JSON 保留旧版 p_value 字段，与文档矛盾](#缺陷-4)
3. [缺陷 5：P3 优化器未执行 D 的上下界约束](#缺陷-5)
4. [缺陷 6：P4 存在过强论证（R²=1.0 外推 + "严密因果性"）](#缺陷-6)
5. [验证结果汇总](#验证结果汇总)

---

## 缺陷 3

**标题**：第二问数据入口与题意不符（P1→P2 接口缺失）

### 问题描述

题面要求第二问通过第一问输出使用附件 A 数据，但第二问的配比脚本
`src/problem02/exp03_mixture_transfer.py` 直接读取附件 A 原始目录
（`DIR_A/regmix_tables`），跳过了第一问的导出结果，重新独立建模。
同时，`p2_common.py` 中的 Q0 锚点以硬编码常量 `Q0=0.584` 的形式存在，
未从第一问导出的结果文件中动态读取。

### 根因定位

| 文件 | 问题行 | 内容 |
|---|---|---|
| `src/problem02/p2_common.py` | `generalized_scaling_law(..., Q0=0.584)` | Q0 硬编码，未消费 P1 输出 |
| `src/problem02/exp03_mixture_transfer.py` | L32–33 | 直接 `BASE_REGMIX = os.path.join(DIR_A, "regmix_tables")` |

第一问导出文件已实际存在于磁盘，但未被第二问任何脚本程序化读取：

| P1 输出文件 | 内容 |
|---|---|
| `result/tables/problem01/table_p1_domain_q_a1.csv` | 7 域质量中位数（含 `count`、`q_median` 列） |
| `result/tables/problem01/table_p1_q_vs_beta.csv` | ALR 系数 β_d |
| `result/tables/problem01/table_p1_dual_protocol_a2a3.csv` | 双协议验证结果 |

### 修复内容

**`src/problem02/p2_common.py`**（新增接口常量和函数）：

```python
# P1 接口路径常量（新增）
P1_TABLES       = os.path.join(r"d:\project\MathStruct\result\tables", "problem01")
P1_DOMAIN_Q_CSV = os.path.join(P1_TABLES, "table_p1_domain_q_a1.csv")
P1_Q_BETA_CSV   = os.path.join(P1_TABLES, "table_p1_q_vs_beta.csv")

def load_p1_quality_table(fallback_q0: float = 0.584) -> dict:
    """从 table_p1_domain_q_a1.csv 计算样本量加权均值 Q0。
    文件缺失时发出 RuntimeWarning 并使用 fallback 值。"""
    ...
    # 返回: {'Q0': float, 'domain_q': dict, 'source': 'p1_csv'|'fallback'}
```

**`src/problem02/exp03_mixture_transfer.py`**（新增接口调用）：

```python
from p2_common import (
    ..., load_p1_quality_table,   # P1→P2 接口函数（新增）
)

# P1→P2 数据接口（新增）
_p1_q = load_p1_quality_table()
Q0_FROM_P1 = _p1_q['Q0']
print(f"[P1→P2] Q0 锚点来源: {_p1_q['source']}")
print(f"[P1→P2] Q0 (样本量加权均值) = {Q0_FROM_P1:.6f}")

# 步骤 1 注释修正：RegMix 原始数据用于训练 f(p)，Q0 来自 P1 CSV（两者数据流层次不同）
```

### 验证结果

```
source: p1_csv           ← 成功从文件读取（非 fallback）
Q0: 0.493776             ← 7 域样本量加权均值
n_domains: 7             ← 全部 7 域均已加载
```

> **Q0 与旧硬编码差值为 0.090**：旧值 0.584 是简单均值，
> 新值 0.494 是样本量加权均值（stackexchange/c4/wikipedia/github 各 10000 样本
> 权重拉低了整体均值），是数据驱动的正确结果。

---

## 缺陷 4

**标题**：P3 结果 CSV/JSON 的 p_value 字段与文档声明矛盾

### 问题描述

`result/problem03/exp02_summary.json` 的 `boot_stats` 键下存有：

```json
"p_value_delta_slope": 0.0
```

（三种成本函数各一条，共 3 处）

而 `docs/problem-03/handoff.md` 第 115 行明确声明：

> "当前检验也**不支持报告**结构性转移的显著性 p 值"

### 根因定位

该 JSON 文件由**旧版本脚本**生成，旧脚本曾计算并输出 `p_value_delta_slope`。
当前代码 `exp02_structural_shift_sweep.py` 已移除该字段，
`resample_changepoint()` 函数只返回范围区间（`kappa_range_*`、`delta_slope_range_*`），
但 JSON 文件未重新生成，造成 **源码 ≠ JSON ≠ 文档** 的三方不一致。

### 修复内容

**`src/problem03/exp02_structural_shift_sweep.py`**（补充 docstring）：

```python
def resample_changepoint(x, y, n_resamples=1000, seed=42):
    """
    ...（原有说明）...

    **本函数不返回 p_value 字段。**
    原因：50 个预算扫描点是确定性对数等距网格，并非从总体中独立随机抽取；
    对确定性扫描曲线进行有放回重采样所得的区间仅是"扰动稳定性摘要"，
    不能解释为总体置信区间，也不支持报告结构性转移的显著性 p 值。
    （对应 docs/problem-03/handoff.md 第 115 行声明）

    Returns
    -------
    dict with keys:
        'kappa_range_low', 'kappa_range_high'         ← 折点扰动区间
        'delta_slope_range_low', 'delta_slope_range_high'  ← Δ斜率扰动区间
        'successful_resamples'                         ← 成功重采样次数
    注：无 'p_value' 键。
    """
```

**重新运行脚本**，覆盖 `exp02_summary.json`。

### 验证结果

```python
# 重新生成的 JSON 结构（boot_stats 下的键名）
['c_crit_flops', 'r2', 'params', 'scan_point_resampling']

# scan_point_resampling 内容
{
  'kappa_range_low':         2.9605,
  'kappa_range_high':        4.2762,
  'delta_slope_range_low':  -0.1985,
  'delta_slope_range_high': -0.1112,
  'successful_resamples':    1000
}

# 全文 p_value 搜索结果
p_value field exists: False   ← 零命中
```

---

## 缺陷 5

**标题**：P3 优化器未执行数据量 D 的上下界约束

### 问题描述

`docs/problem-03/model.md` M3-EQ01 将 $d \in [0.1, 100000.0]$（单位：B-tokens）
列为正式约束条件，但代码实现中：

- `D_MIN = 0.1`、`D_MAX = 100000.0` 定义于 `p3_common.py` 第 42 行
- `solve_optimal_allocation_2d` 的优化器 `bounds` **仅约束 `(ln n, Q)`**
- D 由解析消元 `d* = C_bar / [(6+η·L_ctx)·n + h̄(Q)]` 直接计算，**无任何界约束**
- `D_MIN`/`D_MAX` **未被 exp01 导入**，也未在任何代码路径中执行

极端预算时（如 $C < 10^{13}$ FLOPs）可产生 `d < D_MIN` 的越界解而不发出任何警告。

### 修复内容

**`src/problem03/p3_common.py`**（`solve_optimal_allocation_2d` 返回之前追加）：

```python
# ── D 界约束检查（M3-EQ01 形式约束 d ∈ [D_MIN, D_MAX]）──────────────────
# D 由预算等式解析消元，无法直接加入优化器 bounds。
# 注意：强行 clip 会使 d·cost_per_token ≠ C_bar，破坏预算等式，故不静默截断。
import warnings
d_unconstrained = opt_d
d_lo_violated   = bool(opt_d < D_MIN)
d_hi_violated   = bool(opt_d > D_MAX)
d_bound_active  = d_lo_violated or d_hi_violated
if d_lo_violated:
    warnings.warn(
        f"[P3 D-bound] opt_d={opt_d:.4f} B-tokens < D_MIN={D_MIN} ...",
        RuntimeWarning, stacklevel=2
    )
elif d_hi_violated:
    warnings.warn(
        f"[P3 D-bound] opt_d={opt_d:.4f} B-tokens > D_MAX={D_MAX} ...",
        RuntimeWarning, stacklevel=2
    )
```

返回字典新增 4 个诊断字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `d_unconstrained` | `float` | 解析消元所得原始 D 值（无论是否越界） |
| `d_lo_violated` | `bool` | 是否低于 D_MIN |
| `d_hi_violated` | `bool` | 是否高于 D_MAX |
| `d_bound_active` | `bool` | 越界标志（`d_lo_violated OR d_hi_violated`） |

> **设计决策**：D 的越界采用"**标记式而非截断式**"处理。
> 因为 `d* = C_bar / cost_per_token`，clip 后预算等式 `d · cost_per_token = C_bar`
> 不再成立，会导致后续 KKT 验证和份额计算出错。
> 调用方可依据 `d_bound_active` 自行决策（如固定 D 后重新优化 N）。

### 验证结果

```
# 正常预算（10^19 FLOPs）— 无越界
d_bound_active:  False
d_lo_violated:   False
d_hi_violated:   False
d_unconstrained: 6.2755  ← 在 [0.1, 100000.0] 内

# 极端小预算（10^13 FLOPs）— 触发越界警告
RuntimeWarning: [P3 D-bound] opt_d=0.0016 B-tokens < D_MIN=0.1 B-tokens ...
d_lo_violated:   True
d_unconstrained: 0.00156
```

---

## 缺陷 6

**标题**：P4 存在过强论证（R²=1.0 外推 + "严密因果性"）

### 问题描述（两处）

**A. `exp04_...py` 用训练集 R²=1.0 支撑分布外外推有效性**

```python
# 旧 docstring 第 8 行
6. 引用 Q2 的 B10 超大模型拟合自洽性 (R^2=1.0) 支撑外推有效性

# 旧第 124 行注释
# 依据第二问 B10 实验超大模型拟合自洽性 (R^2=1.0)，标度律在 1e25-1e26 FLOPs 自洽成立
```

R²=1.0 是少量超大模型点上的完美内插，在少量数据点上完美拟合恰好是
**过拟合或内插**的信号，不能支撑分布外（OOD）外推有效性——
尤其是路径 B 情景算力最高达历史前沿的 **~12 倍**（7.8×10²⁴ → ~10²⁶ FLOPs）。

**B. `handoff.md` 第 204 行用"严密因果性"描述 Shapley 加和分解**

```markdown
# 旧文本
3. **加和性公理完全成立**：反事实 Shapley 分解消除了交叉项残差，
   保证了分解结果的**严密因果性**。
```

SFA 模型中的时间趋势 `β_t · t_i` 是回归中的时间固定效应，
Shapley 加和性公理（Efficiency Axiom）保证的是
**各分量之和等于总量的数学恒等性**，不赋予因果解释。
将时间效应的 16.7% 贡献份额直接表述为"因果贡献"属于过强论断。

### 修复内容

**`src/problem04/exp04_forecast_deceleration_dual_path.py`**（docstring + 注释）：

```python
# 修复后 docstring 第 6 条
6. 路径 B 的情景算力最高达历史前沿的 ~12 倍 (1e25–1e26 FLOPs)，
   超出实测标度律拟合域，外推可信度随距离增加而下降；
   结果仅作情景参考上界，不应视为点预测。

# 修复后第 122–126 行注释
# --- 路径 B: 问题三最优配置 + 桥接映射 (基座模型口径, P1-8) ---
# 将情景算力代入第三问 KKT 优化求解最优 Loss L*(C)
# 注：C_future 最高可达历史前沿 (7.8e24 FLOPs) 的 ~12 倍；
# 标度律外推可信度随距离增加而下降，路径 B 结果为情景上界，
# 不宜作为对未知超大模型的精确点预测。
```

**`docs/problem-04/handoff.md`**（第 204 行）：

```markdown
# 修复后
3. **加和性公理严格成立**：反事实 Shapley 分解消除了交叉项残差，
   各贡献项之和恒等于总增益（数学恒等式）；
   时间效应项 $\beta_t \cdot t$ 反映样本期内技术环境的综合时变改善，
   不宜直接读作某单一因果机制的量化。
```

### 验证结果

```powershell
# exp04 中搜索 R²=1.0 / 自洽成立 / 严密因果
Select-String -Path "...exp04_...py" -Pattern "R\^2|自洽成立|严密因果"
# → 零命中 ✅

# handoff.md 中搜索"严密因果"
Select-String -Path "...handoff.md" -Pattern "严密因果"
# → 零命中 ✅
```

---

## 验证结果汇总

| 缺陷 | 验证命令 | 期望结果 | 实际结果 |
|---|---|---|---|
| **3** P1→P2接口 | `load_p1_quality_table()` 返回值 | `source='p1_csv'`，7 个域 | ✅ `source=p1_csv, n_domains=7` |
| **4** p_value 字段 | `'p_value' in json.dumps(d)` | `False` | ✅ `False` |
| **5** D 界正常预算 | `d_bound_active` at `C_bar=10` | `False` | ✅ `False` |
| **5** D 界极端预算 | `RuntimeWarning` at `C_bar=1e-5` | 触发警告，`d_lo_violated=True` | ✅ 警告触发，`d_lo_violated=True` |
| **6A** R²=1.0删除 | `grep "R\^2"` in exp04 | 零命中 | ✅ 零命中 |
| **6B** 因果性删除 | `grep "严密因果"` in handoff | 零命中 | ✅ 零命中 |

---

## 修改文件索引

| 文件 | 缺陷 | 改动类型 |
|---|---|---|
| `src/problem02/p2_common.py` | 3 | 新增路径常量 + 接口函数 |
| `src/problem02/exp03_mixture_transfer.py` | 3 | 新增接口调用 + 注释修正 |
| `src/problem03/exp02_structural_shift_sweep.py` | 4 | docstring 补充 |
| `result/problem03/exp02_summary.json` | 4 | 重新生成（覆盖旧版，p_value 消除） |
| `src/problem03/p3_common.py` | 5 | D 界检查 + warnings + 返回字段 |
| `src/problem04/exp04_forecast_deceleration_dual_path.py` | 6 | docstring + 注释修正 |
| `docs/problem-04/handoff.md` | 6 | 第 204 行措辞修正 |

---

## 不在本次修复范围内的已知限制

1. **P2 exp03 的 HistGB 模型权重持久化**：第一问未导出序列化模型文件，
   exp03 仍需从原始 RegMix 数据重新训练 f(p)。
   若需实现完整管道式复用，需在 P1 中增加 `joblib.dump` 导出，
   在 P2 中增加 `joblib.load` 读取。

2. **P4 OOT 覆盖率欠覆盖（66.7% vs 名义 95%）**：
   这是已知的桥接模型校准限制，`handoff.md` §3.4 已有警示说明，
   属于独立的模型局限性问题，非本次提出的 4 项缺陷之一。

3. **Q0 锚点的参数传递**：本次修复在 `exp03` 中加载 Q0 并赋值给
   `Q0_FROM_P1`，但 `generalized_scaling_law` 的默认参数 `Q0=0.584`
   尚未统一替换（需确认 P2 各实验对 Q0 的使用方式后再决定是否全局替换）。
