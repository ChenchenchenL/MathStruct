# data-dictionary.md — 问题二数据集与字段记录

> 本文档记录问题二使用的所有数据源（附件 B 数据集 B1–B12 及问题一成果表）、字段定义、物理单位、样本量、可信度分级与质量检查。

---

## 1. 数据集概览与可信度分级

| 数据集 ID | 文件路径（`data/real_attachments/`） | 样本量 | 性质属性 | 绑定任务与角色 | 可信度等级 |
|---|---|---|---|---|---|
| **B1** | `B_scaling_laws/pythia_training_log_existing.csv` | 1,176 行 | 真实实验观测 | 经典标度律主拟合（8 条完整模型训练轨迹） | ★★★★★（高） |
| **B2** | `B_scaling_laws/cerebras_training_log.csv` | 1,029 行 | 半合成校准日志 | 族外泛化验证（Cerebras 架构，7 种规格） | ★★★☆☆（中） |
| **B3** | `B_scaling_laws/training_trajectories/*.csv` (8 个) | 8×500 行 | 插值平滑轨迹 | 连续轨迹内插一致性验证 | ★★★★☆（良好） |
| **B4** | `B_scaling_laws/scaling_baseline.csv` | 57 行 | 真实收敛点 | 跨族验证（涵盖 LLaMA, Qwen2, Gemma 等 12 族） | ★★★★★（高） |
| **B5** | `B_scaling_laws/published_scaling_data.csv` | 44 行 | 文献正式数据 | 学术标杆比对（Kaplan, Chinchilla, PaLM 等） | ★★★★★（高） |
| **B6** | `B_scaling_laws/supplementary_NQ_experiment.csv` | 360 点 | 半合成实验 | 质量 $Q$ 效率系数主拟合（$Q \in [0.1, 1.0]$） | ★★★★☆（良好） |
| **B7** | `B_scaling_laws/supplementary_NQ_experiment_expanded.csv` | 450 点 | 半合成实验 | 质量 $Q$ 扩展拟合（补充 $Q=0.5, 0.7$ 网格） | ★★★★☆（良好） |
| **B8** | `B_scaling_laws/supplementary_NQ_experiment_large.csv` | 1,704 点 | 方向诊断数据 | 与 B6/B7 方向不同的质量-损失关系（实测 $r=+0.913$），不参与主拟合 | ★☆☆☆☆（诊断） |
| **B9** | `B_scaling_laws/supplementary_large_models.csv` | 132 行 | 真实模型元数据 | 100B–10000B 大模型规模与训练量信息 | ★★★★★（高） |
| **B10** | `B_scaling_laws/supplementary_large_baseline.csv` | 128 行 | 估算外推数据 | 超大规模标度律外推一致性检查 | ★★☆☆☆（参考） |
| **B11** | `B_scaling_laws/open_model_family_metadata.csv` | 18 行 | 真实辅助信息 | 模型仓库元数据与文件规模辅助 | ★★★★★（高） |
| **B12** | `B_scaling_laws/pythia_checkpoint_index.csv` | 1,386 行 | 真实辅助信息 | Pythia 检查点 commit 映射索引 | ★★★★★（高） |
| **P1-Q** | `result/tables/problem01/table_p1_domain_q_a1.csv` | 7 行 | 问题一正式产物 | 7 大质量域稳健综合质量分 $Q_d$（A1 标尺） | ★★★★★（高） |
| **P1-Beta** | `result/tables/problem01/table_p1_q_vs_beta.csv` | 6 行 | 问题一正式产物 | 6 大映射领域 ALR 配比回归边际系数 $\beta_d$ | ★★★★★（高） |
| **P1-Mix** | `A_data_value/regmix_tables/train_mixture_1m.csv` | 512 行 | 问题一正式产物 | 17 领域基线配比 $p_0$ 均值向量 | ★★★★★（高） |
| **P1-Loss** | `A_data_value/regmix_tables/test_pile_loss_*.csv` | 576 行 | 问题一正式产物 | 1M (256)、60M (256)、1B (64) 真实检验集损失 | ★★★★★（高） |

---

## 2. 核心字段与物理单位

| 字段名称 | 物理与业务含义 | 数据类型 | 允许范围 | 原始单位 | 建模统一无量纲/标准化单位 |
|---|---|---|---|---|---|
| `N_params_B` | 模型参数量（非嵌入核心参数） | float | $[0.07, 10000]$ | 十亿参数（Billion params） | $N$（以 $10^9$ 为基准单位） |
| `D_tokens_B` | 训练所消耗的数据 Token 总量 | float | $[0.1, 36000]$ | 十亿 Token（Billion tokens） | $D$（以 $10^9$ 为基准单位） |
| `val_loss` | 验证集交叉熵损失（自回归预测） | float | $[1.0, 7.0]$ | Nats / Token | $L$（自然对数交叉熵，越低越好） |
| `Q_score` | 数据质量评分（综合多维质量特征） | float | $[0.0, 1.0]$ | 无量纲连续得分 | $Q$（越高越好；附录 B 清洗成本底线 $Q_{\text{base}} \approx 0.584$ 标定为开放网络抓取域 CC+C4 合成中位数） |
| `p_d` | 领域 $d$ 在训练语料中所占比例 | float | $[0.0, 1.0]$ | 比例（和为 1） | 单纯形向量 $p \in \Delta^{16}$ |
| `steps` | 累计训练优化迭代步数 | int | $[64, 143000]$ | 步数（Steps） | 辅助诊断（对应 $D$ 的物理流） |
| `C_FLOPs_1e21` | 累计计算浮点运算次数（Chinchilla 6ND） | float | $\ge 0$ | $10^{21}$ FLOPs（ZettaFLOPs） | 验证计算量约束 |

---

## 3. 质量检查与异常诊断记录

1. **B1 检查**：
   - 8 个模型参数分别为：0.0705B (70M), 0.1624B (160M), 0.4090B (410M), 1.0409B (1B), 1.4162B (1.4B), 2.7828B (2.8B), 6.8610B (6.9B), 11.9658B (12B)。
   - 每个模型均严格记录了从 step 64 至 step 143,000 的 147 个连续检查点，Token 量从 0.134B 扩展至 299.893B。
   - 数据完整，无缺失值，整条轨迹损失总体递减，呈现典型的幂律曲线。
2. **B6/B7 检查**：
   - 覆盖 9 种模型规格（0.07B 至 11.97B），5 种数据量（10B, 50B, 150B, 300B, 600B），B6 含 8 种 $Q$、B7 补充至 10 种 $Q$。
   - B6 与 B7 在重叠的 360 点上损失数值差异为 0.0000，B7 增补了 $Q=0.5$ 与 $0.7$。
   - 在固定 $N, D$ 的切片汇总中，B6/B7 的质量-损失总体相关为负，相邻质量点下降比例分别为 74.6% 和 74.8%；在 $Q=1.0$ 处，其损失值与 B1 经典标度律预测值的比率均值为 1.014。
3. **B8 方向诊断**：
   - 在 1,704 个样本中，`corr(Q_score, val_loss) = +0.9132`，Spearman $\rho=+0.9208$，与 B6/B7 的总体方向不同。
   - 例如 $N=0.7B, D=10B$ 时，$Q=0.1$ 损失为 0.6150，而 $Q=1.0$ 损失为 2.6504。
   - B8 同时包含与主数据尺度不同的低损失观测。该现象提示需要检查数据生成或字段定义，但仅凭当前数据不能确定具体原因。
   - 诊断结论：B8 按规则 P2-02 保留为对照诊断集，不参与质量参数主拟合；不据此推断其生成原因。
