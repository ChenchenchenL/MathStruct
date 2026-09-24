# 10 项建模问题修改记录

**修改日期**：2026-09-24  
**基于审查**：`review/audit_report_20260924.md`  
**修改范围**：7 项代码/文档改动（问题 1–6、8–9）；问题 7、10 维持现状  

---

## 修改汇总

| # | 问题 | 改动类型 | 文件 | 状态 |
|---|---|---|---|---|
| 1 | A12-A15 Spearman 负值未持久化 | 代码 + 重新运行 | `src/problem01/exp02_mixture.py` + JSON | ✅ |
| 2 | P3 图 caption 缺 p_opt 条件说明 | 文档 | `result/figures/problem03/fig_p3_pareto_frontiers.caption.md` | ✅ |
| 3 | P4 handoff §2.4 未披露 Path B 参数约束 | 文档 | `docs/problem-04/handoff.md` | ✅ |
| 4 | `verify_global=False` 无代码注释 | 代码 | `src/problem04/exp04_forecast_deceleration_dual_path.py` | ✅ |
| 5 | Q_A→Q_B 映射缺口无声明 | 文档 | `docs/problem-01/handoff.md` | ✅ |
| 6 | Q_base=0.584 措辞偏强（"实测溯源"） | 文档 | `docs/problem-02/handoff.md` | ✅ |
| 7 | B8 冲突（维持现状） | — | — | 无需修改 |
| 8 | B2 "验证"措辞偏强 | 文档 | `docs/problem-02/handoff.md` | ✅ |
| 9 | exp04_summary.json 缺 Bootstrap 汇总 | 数据 | `result/problem04/exp04_summary.json` | ✅ |
| 10 | OOT 66.7%（维持现状） | — | — | 无需修改 |

---

## 逐项改动详情

### 问题 1：A12-A15 Spearman 负值持久化

**文件**：[`src/problem01/exp02_mixture.py`](file:///d:/project/MathStruct/src/problem01/exp02_mixture.py)

**改动**：在 JSON 输出块之前新增循环，重新计算 10B/70B 外推表的 Spearman 值并存入 JSON；
原 `"est_spearman": "see stdout"` 替换为 `"est_spearman_extrapolated": {...}`。

**验证结果**（重新运行 exp02 后）：
```json
{
  "10B": { "spearman_rho": -0.5157, "n": 63, "note": "extrapolated table..." },
  "70B": { "spearman_rho": -0.5684, "n": 63, "note": "extrapolated table..." }
}
```

---

### 问题 2：P3 帕累托图 caption 补注 p_opt 条件性

**文件**：[`result/figures/problem03/fig_p3_pareto_frontiers.caption.md`](file:///d:/project/MathStruct/result/figures/problem03/fig_p3_pareto_frontiers.caption.md)

**新增内容（引用块）**：
> 本图仅展示**基准配比情景**（$p_0$，$R(p_0)=1.0$）下的帕累托前沿。**优化配比情景**（$p_{\text{opt}}$，$R=0.9602$）未纳入本图，其数值结果详见 `tab_p3_optimal_allocation_results.csv` 中的 `Fixed_p_opt` 行。$p_{\text{opt}}$ 是在跨数据源迁移假设下的条件配方，不是独立识别的普适最优配比，两种情景下的 $N^*,D^*,Q^*,L^*$ 均为固定配比假设下的条件解。

---

### 问题 3：P4 handoff §2.4 补充 Path B 参数约束

**文件**：[`docs/problem-04/handoff.md`](file:///d:/project/MathStruct/docs/problem-04/handoff.md) §2.4

**新增引用块**（里程碑表之前）：
- `p_multiplier=0.9602`：来源链 P1跨尺度秩检验 → P2 τ标定 → P3 Fixed_p_opt → P4 Path B；条件假设，非独立识别
- `verify_global=False`：72次调用，关闭差分进化控制计算开销；局部解与DE残差 < 1e-5
- Path B 定性为"条件双路径对照，不作为独立预测"

---

### 问题 4：`verify_global=False` 补充注释

**文件**：[`src/problem04/exp04_forecast_deceleration_dual_path.py`](file:///d:/project/MathStruct/src/problem04/exp04_forecast_deceleration_dual_path.py) L128–134

```python
p_multiplier=0.9602,  # 继承自问题三 Fixed_p_opt 情景（跨数据源迁移假设下的名义校准值）
verify_global=False   # 共 24月×3情景=72 次调用，关闭差分进化以控制计算开销；
                      # 多起点 L-BFGS-B（50×5=250 起点）与 DE 的目标函数残差 <1e-5（见 P3 自测）
```

---

### 问题 5：Q_A→Q_B 映射缺口声明

**文件**：[`docs/problem-01/handoff.md`](file:///d:/project/MathStruct/docs/problem-01/handoff.md) §4 接口 1

**新增引用块**：Q_d（A1 绝对质量分）与 Q_B（成本增量参数）量纲不完全等同，关系是"锚点估计"而非等量映射；**缺口显式保留，不建立强制等值换算**；论文中两种标尺分开定义。

---

### 问题 6：Q_base=0.584 措辞修正

**文件**：[`docs/problem-02/handoff.md`](file:///d:/project/MathStruct/docs/problem-02/handoff.md) §1.1

| 位置 | 改前 | 改后 |
|---|---|---|
| §1.1 子标题 | `严密溯源与聚合定义` | `聚合定义与成本基线假设依据` |
| 子标题二级 | `聚合公式与实测溯源` | `聚合公式与成本基线假设依据` |
| 行内（L20） | `实测 Common Crawl 0.650` | `A1 集记录的 Common Crawl 0.650` + "成本基线假设，非独立测定值" |
| L21 末尾 | `消融实验证实` | `敏感性检验发现` |

---

### 问题 8：B2 措辞修正

**文件**：[`docs/problem-02/handoff.md`](file:///d:/project/MathStruct/docs/problem-02/handoff.md)

| 位置 | 改前 | 改后 |
|---|---|---|
| L111 条目标题 | `B2（Cerebras 族外半合成，1,029点）` | `B2（Cerebras 族跨架构截距校准，非独立验证，1,029点）` |
| L134 已知限制条目 | `...应与 Spearman ρ=0.7881 同时报告。` | `...同时报告；论文中称为"跨架构截距校准性分析"，不称为独立验证。` |

---

### 问题 9：exp04_summary.json 补充 Bootstrap 汇总值

**文件**：[`result/problem04/exp04_summary.json`](file:///d:/project/MathStruct/result/problem04/exp04_summary.json)

**新增字段** `bootstrap_summary_from_exp405`（来源标注为 EXP-405）：
- 六组情景的 Bootstrap 95% CI（从 exp05_summary.json 引入）
- OOT 覆盖率（66.7%）、敏感性商业口径（92.5%）
- 历史前沿锚点：SFA包络均值 42.60，单体峰值 52.08
- 情景一 +12m Bootstrap 中位数近似值 40.82
- 三口径对应说明：SFA 条件期望 35.91 / Bootstrap 中位数 40.82 / 历史峰值 52.08

---

## 维持现状的问题

| # | 原因 |
|---|---|
| **7** B8 方向冲突 | 已正确隔离，"原因未确定"诚实标注，论文不编造原因 |
| **10** OOT 66.7% | 已充分披露，三组敏感性分析（89%~92.5%），结论为条件情景 |
