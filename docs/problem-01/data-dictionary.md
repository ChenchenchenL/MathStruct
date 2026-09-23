# data-dictionary.md — 问题一数据字典

数据版本规则：数据集 ID = 附件编号 + 文件名（稳定），版本一律 v1（原始附件未做任何修改）。来源均为题目附件 `data/real_attachments/A_data_value/`，下表路径相对该目录。时区/采集时间：不适用（静态附件）。

## 数据集清单

| 数据集 ID | 文件路径 | 规模 | 性质 | 角色 |
|---|---|---|---|---|
| A1 | slimpajama_quality_signal_sample.jsonl.xz | 51,230 条 × 27 字段 | 真实 | 质量主样（抽样集，7 域） |
| A2 | slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz | 17,523 条 × 24 字段 | 真实 | arxiv 扩展（必用） |
| A3 | slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz | 203,752 条 × 24 字段 | 真实 | github 扩展（必用） |
| A4 | regmix_tables/train_mixture_1m.csv | 512 行 × 18 列 | 真实 | 训练配比（1M 模型） |
| A5 | regmix_tables/train_pile_loss_1m.csv | 512 行 × 14 列 | 真实 | 训练 Loss（13 域） |
| A6/A7 | regmix_tables/test_mixture_1m.csv / test_pile_loss_1m.csv | 256 行 × 18/14 列 | 真实 | 检验（1M） |
| A8/A9 | regmix_tables/test_mixture_60m.csv / test_pile_loss_60m.csv | 256 行 × 18/14 列 | 真实 | 检验（60M） |
| A10/A11 | regmix_tables/test_mixture_1B.csv / test_pile_loss_1B.csv | 64 行 × 18/14 列 | 真实 | 检验（1B） |
| A12/A13 | regmix_tables/est_mixture_10b.csv / est_pile_loss_10b.csv | 63 行 × 18/14 列 | 子集 / 外推 | 外推（10B） |
| A14/A15 | regmix_tables/est_mixture_70b.csv / est_pile_loss_70b.csv | 63 行 × 18/14 列 | 子集 / 外推 | 外推（70B） |
| A16 | domain_mapping_guide.csv | 17 行 × 4 列 | 参考（人工整理） | 域映射 |
| A17 | regmix_domain_summary.csv | 17 行 × 5 列 | 真实 | 域文本统计（辅助） |
| A18 | regmix_domain_sample.jsonl.xz | 138,034 条 | 真实 | 域原始文本（可选验算） |

## 字段定义

### A1–A3 质量信号（JSONL，xz 压缩，流式读取）

辅助字段 5 个（A1）：`id`（文本 ID）、`content`（原文，仅统计用）、`sub_path`、`_source_domain`（域标签）、`_source_path`。A2/A3 无 content/_source_domain/_source_path，域由文件名推断。

22 个原始质量字段（14 个标量字段、8 个列表字段；经 P02 压缩后形成 25 个标量候选特征）→ 类型 / 量纲现状 / 方向 / 压缩口径。**官方分量语义来源**：HF opendatalab/SlimPajama-Meta-rater 数据卡 + Meta-rater 论文（arXiv:2504.14194，2026-09-23 核实）。

| # | 字段 | 类型 | 量纲现状 | 方向 | 压缩/使用口径（官方定义） |
|---|---|---|---|---|---|
| 1 | fineweb_edu | 列表[1] | 无量纲分 | 正向（教育价值） | 取 x[0] |
| 2 | fluency_en | 列表[2] | logits，**[not_fluent, fluent]** | 正向 | z2−z1（fluent−not_fluent）；argmax 对照 |
| 3 | ad_en | 列表[2] | logits，**[has_ad, no_ad]** | 转化为正向"无广告度" | z2−z1（no_ad−has_ad）；argmax 对照 |
| 4 | qurater | 列表[4] | 分×4，**[Writing Style, Required Expertise, Facts&Trivia, Educational Value]** | 四维分别经 M1-EQ21 判定方向；负向维补转换，方向不定维剔除 | 解包为 `qur_ws`、`qur_re`、`qur_ft`、`qur_ev` 四个独立候选特征，不预先合成为单一分数 |
| 5 | modernbert_professionalism | 列表[6] | logits×6，levels 0–5 | **文本属性维（理解所需专业程度），方向域依赖，经 M1-EQ21 判定（AS-13）** | softmax 期望等级 [0,5]；argmax（官方用法）对照 |
| 6 | modernbert_readability | 列表[6] | 同上 | 正向 | 同上 |
| 7 | modernbert_reasoning | 列表[6] | 同上 | 正向 | 同上 |
| 8 | modernbert_cleanliness | 列表[6] | 同上 | 正向（格式完整、无噪声） | 同上 |
| 9 | dsir_books | 标量 | 大负数（1e3~1e5 量级，DSIR importance，books 目标域） | 正向（长度残差化后，AS-03） | P04：先对 `log1p(word_count)` 残差化，再按 A1 冻结分位数标尺 |
| 10 | dsir_wiki | 标量 | 同上（wikipedia 目标域） | 正向（长度残差化后，AS-03） | P04：先对 `log1p(word_count)` 残差化，再按 A1 冻结分位数标尺 |
| 11 | dsir_math | 标量 | 同上（AutoMathText 目标域） | 正向（长度残差化后，AS-03） | P04：先对 `log1p(word_count)` 残差化，再按 A1 冻结分位数标尺 |
| 12 | rps_doc_word_count | 标量 | 词数（计数） | 正向（EQ21，4/7 域） | P03：A1 域内秩相关判定；P05 分位数映射 |
| 13 | rps_doc_num_sentences | 标量 | 句数（计数） | 方向不定 | P03 剔除出主合成集，单列报告 |
| 14 | rps_doc_unigram_entropy | 标量 | 熵（无量纲，原始底数沿用附件） | 正向（EQ21，5/7 域） | P03：A1 域内秩相关判定；P05 分位数映射 |
| 15 | rps_doc_frac_unique_words | 标量 | 百分数（0–100，RedPajama 释义：词汇多样性） | 负向（EQ21，4/7 域） | P03 判为负向后补转换 $x\mapsto1-x$ |
| 16 | rps_doc_frac_no_alph_words | 标量 | 百分数 | **负向**（非字母词占比） | 语义明确，补转换 x→1−x/100 |
| 17 | rps_doc_frac_chars_top_2gram | 标量 | 百分数 | **负向**（重复度） | 同上 |
| 18 | rps_doc_frac_chars_top_3gram | 标量 | 百分数 | **负向**（重复度） | 同上 |
| 19 | rps_lines_uppercase_letter_fraction | 标量 | 百分数 | 方向不定（3/7 域） | P03 剔除出主合成集，单列报告 |
| 20 | rps_lines_ending_with_terminal_punctution_mark | 标量 | 百分数（字段名疑原始拼写 punct**u**tion，保留原名引用） | 方向不定 | P03 剔除出主合成集，单列报告 |
| 21 | rps_lines_numerical_chars_fraction | 标量 | 百分数 | 方向不定（域依赖：代码/公式域高） | P03 剔除出主合成集，单列报告 |
| 22 | rps_doc_mean_word_length | 标量 | 字符/词 | 方向不定 | P03 剔除出主合成集，单列报告 |

注：①"压缩/使用口径"列为官方定义绑定（M1-EQ07~11），不得另行发明分量顺序；② 方向列含两轮信息——官方/题目给定语义 + **EXP-01/03 实测判定（EQ21 终版，dsir 经长度残差化）**：正向 12 项（base5 + dsir_books/math/wiki【残差化后 6/7 域】+ professionalism【6/7】+ word_count【4/7】+ unigram_entropy【5/7】+ qur_ws/re/ft/ev【7/7/6/7】）；负向 2 项补转换入 S：frac_unique_words（4/7）、frac_no_alph_words（5/7）；方向不定 8 项剔出权重集 S、单列报告：adfree（仍用于冲突对 C1）、num_sentences、top_2gram、top_3gram、uppercase、terminal_punct、numerical_chars、mean_word_length；③ Meta-rater 官方 25-rater 学习权重仅作对照（decisions.md D-09），不直接采用。

### A4–A15 配比与 Loss（CSV）

- `index`：配方编号；A4↔A5、A6↔A7 等配比表与 Loss 表按 index 对应。
- 配比列：`train_the_pile_*` 共 17 列（arxiv, freelaw, nih_exporter, pubmed_central, wikipedia_en, dm_mathematics, github, philpapers, stackexchange, enron_emails, gutenberg_pg_19, pile_cc, ubuntu_irc, europarl, hackernews, pubmed_abstracts, uspto_backgrounds），单位：**无量纲比例**，行和≈1（千分位舍入，见 AS-08）。
- Loss 列：`metric/the_pile_*_val_loss` 共 13 列，单位：**交叉熵损失（nat，自然对数底，与验证集底数一致——待与 Pythia/RegMix 口径核对后确认单位标注）**。无 Loss 的 4 域：nih_exporter、enron_emails、europarl、philpapers（AS-09）。
- 规模参数背景：A4/A6–A11 分别对应 1M/1M/60M/1B 参数规模的训练实验；A12–A15 对应 10B/70B 的**外推**值（est Loss 非观测，AS-06）。

### A16 域映射

`mixture_domain`（17 配方域）→ `quality_domain`（7 质量域：arxiv, book, c4, commoncrawl, github, stackexchange, wikipedia），`mapping_type` ∈ {direct: 3 域, near_direct: 3 域, inferred: 11 域}。c4 无配方域对应。单位：不适用（分类映射）。

### A17 域文本统计

`domain, source_path, sample_rows（条）, sample_bytes_requested（字节，=83,886,080 恒定）, avg_text_chars（字符/条）`。仅作 inferred 域关联的辅助证据。

## 质量检查记录（勘察阶段实测）

| 检查项 | 结果 | 日期 |
|---|---|---|
| A1 首两条记录 22 指标字段完整性 | 22/22 存在，列表长度与类型表一致 | 2026-09-23 |
| A2 与 A1 同 id 记录（id=BkiUdvE4eIfiUWiFdbiw）字段值 | 逐字段一致（重叠样本），证实 A2 与 A1 同源同口径 | 2026-09-23 |
| A4 行和 | 行 1 = 1.000，行 2 = 0.999（千分位舍入，AS-08 成立） | 2026-09-23 |
| A10 与 A4 的 index 体系 | A10 从 0 起、A4 从 1 起，无跨表对应关系（独立实验） | 2026-09-23 |
| A12 配比 vs A4 配比 | index=1,2 逐列相同，证实 est 配比为 train 子集 | 2026-09-23 |
| A1 域构成（EXP-01 全量读取） | c4/github/stackexchange/wikipedia 各 10,000、commoncrawl 9,640、arxiv 1,419、book 171，共 51,230 | 2026-09-23 |
| A1∩A2 重叠 id | 1,419（= A1 arxiv 子集全量，占 A2 的 8.1%） | 2026-09-23 |
| A1∩A3 重叠 id | 10,000（= A1 github 子集全量，占 A3 的 4.9%） | 2026-09-23 |
| A1 指标缺失率 | 最高 modernbert_reasoning 0.025%，其余≈0；A2/A3 同量级；质量分按行对可用入选指标重新归一化，冲突度按可用冲突分量取最大值 | 2026-09-23 |
| A4 全量行和（EXP-02） | 0.996–1.003，重归一化后使用 | 2026-09-23 |

派生变量记录见 `data_procee.md` 的 P01–P10 及 P05a；最终入选集为 17 个特征，缺失值按行可用指标数重新归一化。

## 划分规则（已确认）

- 任务 1/2：A1 为定义/校准集，A2/A3 为复验集（AS-10），无随机划分（按题目既定角色）。
- 任务 3：A4–A5 拟合；A6–A11 检验（逐尺度报告）；A12–A15 仅稳健性讨论（AS-06）。不引入随机划分，划分由题目数据角色给定。
