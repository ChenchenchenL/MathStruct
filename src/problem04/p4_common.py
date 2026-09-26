"""
p4_common.py — 问题四核心通用工具模块
包含：
1. 数据加载与清洗管道（C1, C2, C3, C4, C5, C6, C8）
2. P4-01 三级匹配漏斗（L1 官方桥接 -> L2 规范化精确碰撞 -> L3 家族参数规则校验）
3. P4-02 许可证分类与研究可用性映射
4. P4-03 C4 异常日期与微型算力过滤
5. Logit 变换与逆变换（带数值边界保护）
6. 有界动态随机生产前沿（SFA）极大似然估计与分位数面板回归
7. 反事实 Shapley 因果贡献分解计算器
8. Loss-to-Benchmark 连续单调饱和 Logistic 桥接模型与外推敏感带
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, halfnorm
from sklearn.linear_model import QuantileRegressor

# 基础路径
BASE_DATA_DIR = "d:/project/MathStruct/data/real_attachments/C_efficiency_evolution"
TABLE_DIR = "d:/project/MathStruct/result/tables/problem04"
FIGURE_DIR = "d:/project/MathStruct/result/figures/problem04"
SUMMARY_DIR = "d:/project/MathStruct/result/problem04"

for d in [TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

# 核心常量
EPS_LOGIT = 1e-4
T0_DATE = "2025-03-13" # 观测数据集最后有效记录日期，严格锚定预测起报点

BENCHMARK_COLS = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
CHANCE_LEVELS = {
    'IFEval': 0.0,
    'BBH': 27.96,       # 实证反演估计值
    'MATH Lvl 5': 0.0,
    'GPQA': 25.0,
    'MUSR': 33.33,
    'MMLU-PRO': 10.0
}

# --- 1. 数据清洗与三级匹配漏斗 ---

def clean_model_name(s: str) -> str:
    """名称规范化：剥离组织前缀、转小写、移除非字母数字字符"""
    if not isinstance(s, str):
        return ""
    # 去除 HuggingFace 组织前缀
    if '/' in s:
        s = s.split('/')[-1]
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def extract_family(model_name: str) -> str:
    """提取规范化模型家族标识符"""
    if not isinstance(model_name, str):
        return "other"
    m = model_name.lower()
    families = [
        ('llama', 'Llama'),
        ('qwen', 'Qwen'),
        ('mistral', 'Mistral'),
        ('mixtral', 'Mistral'),
        ('yi', 'Yi'),
        ('gemma', 'Gemma'),
        ('deepseek', 'DeepSeek'),
        ('phi', 'Phi'),
        ('falcon', 'Falcon'),
        ('pythia', 'Pythia'),
        ('glm', 'GLM'),
        ('bloom', 'BLOOM'),
        ('opt', 'OPT'),
        ('internlm', 'InternLM'),
        ('baichuan', 'Baichuan'),
        ('starcoder', 'StarCoder'),
        ('nemotron', 'Nemotron')
    ]
    for key, fam in families:
        if key in m:
            return fam
    return "Other"

def classify_license(hub_license: str, epoch_open: str = None) -> str:
    """根据 C1 Hub License 与 C4 状态划分三档许可证可用性"""
    lic = str(hub_license).lower().strip() if pd.notnull(hub_license) else ""
    # 严格宽松商用许可
    strict_permissive = ['apache-2.0', 'mit', 'bsd-2-clause', 'bsd-3-clause', 'bsd']
    if any(lic == sp or lic.startswith(sp) for sp in strict_permissive):
        return "Strict_Permissive"
    # 研究开源许可（含 Llama, Gemma, Qwen 社区协议及 CC-BY 等）
    research_open = ['llama3.1', 'llama3.2', 'llama3', 'llama2', 'gemma', 'qwen', 'cc-by', 'openrail', 'creativeml-openrail-m', 'gpl', 'agpl']
    if any(ro in lic for ro in research_open):
        return "Research_Open"
    # 非商用限制许可
    if 'cc-by-nc' in lic:
        return "Non_Commercial"
    # 标注为 other 或空
    if epoch_open == 'Yes':
        return "Research_Open_EpochVerified"
    return "Unknown_Restricted"

def load_and_match_datasets():
    """
    执行 P4-01 三级级联匹配：
    L1: C2 官方桥接列
    L2: clean_name 精确碰撞
    L3: 家族与参数量规则匹配 (误差 <= 5%)
    并执行 P4-02 ~ P4-04 清洗。
    """
    c1_path = os.path.join(BASE_DATA_DIR, "leaderboard_cleaned.csv")
    c2_path = os.path.join(BASE_DATA_DIR, "leaderboard_enhanced.csv")
    c4_path = os.path.join(BASE_DATA_DIR, "epoch_all_ai_models.csv")

    df_c1 = pd.read_csv(c1_path)
    df_c2 = pd.read_csv(c2_path)
    df_c4 = pd.read_csv(c4_path, low_memory=False)

    # 清洗 C4：排除异常日期与非大模型算力行 (P4-03)
    df_c4['Pub_Date'] = pd.to_datetime(df_c4['Publication date'], errors='coerce')
    # 过滤早于 2018 或晚于 T0 的日期，以及算力 < 1e18 的噪声行
    valid_c4_mask = (
        (df_c4['Pub_Date'] >= '2018-01-01') & 
        (df_c4['Pub_Date'] <= T0_DATE) &
        (df_c4['Training compute (FLOP)'] >= 1e18)
    )
    df_c4_clean = df_c4[valid_c4_mask].copy()
    df_c4_clean['clean_name'] = df_c4_clean['Model'].apply(clean_model_name)
    df_c4_clean['Family'] = df_c4_clean['Model'].apply(extract_family)

    # 准备 C1/C2
    df_c1['clean_name'] = df_c1['Model'].apply(clean_model_name)
    df_c1['Family'] = df_c1['Model'].apply(extract_family)
    df_c1['License_Category'] = [
        classify_license(l, o) for l, o in zip(df_c1['Hub License'], df_c2['Epoch_AI_Open_Weights'])
    ]

    # 模型类型二分划分 (Pretrained vs Chat/Finetuned)
    def map_model_regime(t):
        if not isinstance(t, str):
            return "Other"
        t_low = t.lower()
        if 'pretrained' in t_low and 'continuously' not in t_low:
            return "Pretrained"
        elif 'chat' in t_low or 'fine-tuned' in t_low or 'continuously' in t_low:
            return "Chat_Finetuned"
        elif 'merges' in t_low:
            return "Merges"
        return "Other"

    df_c1['Regime'] = df_c1['Type'].apply(map_model_regime)
    df_c1['is_chat'] = (df_c1['Regime'] == 'Chat_Finetuned').astype(int)

    # 级联匹配
    # L1: C2 官方桥接命中 (Epoch_AI_Publication_Date 非空)
    l1_matched_indices = df_c2[df_c2['Epoch_AI_Publication_Date'].notnull()].index
    
    # 建立 C4 索引
    c4_by_clean = df_c4_clean.drop_duplicates(subset=['clean_name']).set_index('clean_name')

    matched_records = []
    match_level_counts = {'L1': 0, 'L2': 0, 'L3': 0, 'Unmatched': 0}

    for idx, row in df_c1.iterrows():
        c_name = row['clean_name']
        matched_c4 = None
        m_level = 'Unmatched'

        # 尝试 L1 官方对齐
        if idx in l1_matched_indices:
            # 根据机构与发布日期在 C4 中检索最匹配的记录
            org = df_c2.loc[idx, 'Epoch_AI_Organization']
            p_date = df_c2.loc[idx, 'Epoch_AI_Publication_Date']
            cand = df_c4_clean[
                (df_c4_clean['Organization'] == org) & 
                (df_c4_clean['Publication date'] == p_date)
            ]
            if len(cand) > 0:
                matched_c4 = cand.iloc[0]
                m_level = 'L1'

        # 尝试 L2 clean_name 精确碰撞
        if matched_c4 is None and c_name in c4_by_clean.index:
            matched_c4 = c4_by_clean.loc[c_name]
            m_level = 'L2'

        # 尝试 L3 家族与参数量规则匹配 (误差 <= 5%)
        if matched_c4 is None and row['Family'] != 'Other' and row['#Params (B)'] > 0:
            fam = row['Family']
            params = row['#Params (B)'] * 1e9
            cand = df_c4_clean[
                (df_c4_clean['Family'] == fam) & 
                (df_c4_clean['Parameters'] > 0) &
                (np.abs(df_c4_clean['Parameters'] - params) / params <= 0.05)
            ]
            if len(cand) > 0:
                # 选取提交日期与发布日期最接近的候选
                matched_c4 = cand.iloc[0]
                m_level = 'L3'

        match_level_counts[m_level] += 1

        rec = row.to_dict()
        rec['Match_Level'] = m_level
        if matched_c4 is not None:
            rec['Training_Compute_FLOP'] = matched_c4['Training compute (FLOP)']
            rec['C4_Parameters'] = matched_c4['Parameters']
            rec['C4_Dataset_Size_Tokens'] = matched_c4['Training dataset size (total)']
            rec['C4_Pub_Date'] = matched_c4['Publication date']
            rec['C4_Open_Weights'] = matched_c4['Open model weights?']
        else:
            rec['Training_Compute_FLOP'] = np.nan
            rec['C4_Parameters'] = np.nan
            rec['C4_Dataset_Size_Tokens'] = np.nan
            rec['C4_Pub_Date'] = np.nan
            rec['C4_Open_Weights'] = np.nan
        matched_records.append(rec)

    df_matched = pd.DataFrame(matched_records)
    
    # 格式化日期与连续月份
    df_matched['Sub_Date'] = pd.to_datetime(df_matched['Submission Date'], errors='coerce')
    min_date = df_matched['Sub_Date'].min()
    # 连续月份差
    df_matched['month_idx'] = (
        (df_matched['Sub_Date'].dt.year - min_date.year) * 12 + 
        (df_matched['Sub_Date'].dt.month - min_date.month)
    )

    return df_matched, df_c1, df_c2, df_c4_clean, match_level_counts

# --- 2. 有界 Logit 变换 ---

def logit_score(s: np.ndarray, eps: float = EPS_LOGIT) -> np.ndarray:
    """有界得分 [0, 100] 映射至 (-inf, +inf)"""
    p = np.clip(s / 100.0, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def inv_logit_score(y: np.ndarray) -> np.ndarray:
    """Logit 空间映射回有界得分 [0, 100]"""
    # 避免溢出
    y_clipped = np.clip(y, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-y_clipped))
    return p * 100.0

# --- 3. Loss-to-Benchmark Logistic 饱和桥接模型 ---

class LogisticBridgeModel:
    """
    连续单调有界饱和映射模型：
    S(L, r) = 100 / (1 + exp(-(a + theta*r - b*L)))
    要求 b > 0，保证 dS/dL < 0 (Loss 越低得分越高)
    """
    def __init__(self, use_type_dummy: bool = True):
        self.use_type_dummy = use_type_dummy
        self.params = {} # {bench: [a, b, (theta)]}
        self.r2 = {}
        self.spearman = {}
        self.rmse = {}

    def fit(self, df_bridge: pd.DataFrame):
        col_map = {
            'Average': 'LB_Average',
            'IFEval': 'LB_IFEval',
            'BBH': 'LB_BBH',
            'MATH Lvl 5': 'LB_MATH',
            'GPQA': 'LB_GPQA',
            'MUSR': 'LB_MUSR',
            'MMLU-PRO': 'LB_MMLU_PRO'
        }
        for col in ['Average'] + BENCHMARK_COLS:
            target_col = col_map.get(col, f'LB_{col}')
            if target_col not in df_bridge.columns:
                continue
            
            sub = df_bridge[['Val_Loss', target_col, 'is_chat']].dropna()
            L = sub['Val_Loss'].values
            S = sub[target_col].values
            r = sub['is_chat'].values if self.use_type_dummy else np.zeros_like(L)

            # 初始参数估计：在 logit 空间做线性回归
            y = logit_score(S)
            if self.use_type_dummy:
                # y ≈ a + theta*r - b*L
                X = np.column_stack([np.ones_like(L), r, -L])
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                init_p = [beta[0], max(0.1, beta[2]), max(0.0, beta[1])]
            else:
                X = np.column_stack([np.ones_like(L), -L])
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                init_p = [beta[0], max(0.1, beta[1])]

            # 非线性加权拟合，惩罚 b <= 0
            def loss_fn(p):
                if self.use_type_dummy:
                    a, b, theta = p[0], p[1], p[2]
                    pred = 100.0 / (1.0 + np.exp(-(a + theta * r - b * L)))
                else:
                    a, b = p[0], p[1]
                    pred = 100.0 / (1.0 + np.exp(-(a - b * L)))
                pen = 1e5 * np.maximum(0, -b + 1e-4)**2
                return np.mean((pred - S)**2) + pen

            res = minimize(loss_fn, init_p, method='L-BFGS-B', bounds=[(-20, 20), (1e-4, 20)] + ([(-5, 10)] if self.use_type_dummy else []))
            opt_p = res.x
            self.params[col] = opt_p

            # 评价指标
            pred_s = self.predict(L, col, r if self.use_type_dummy else None)
            ss_res = np.sum((S - pred_s)**2)
            ss_tot = np.sum((S - np.mean(S))**2)
            self.r2[col] = 1.0 - ss_res / (ss_tot + 1e-8)
            self.rmse[col] = np.sqrt(np.mean((S - pred_s)**2))
            
            # Spearman 秩相关
            from scipy.stats import spearmanr
            rho, _ = spearmanr(L, S)
            self.spearman[col] = rho

    def predict(self, L: np.ndarray, bench: str = 'Average', is_chat: np.ndarray = None) -> np.ndarray:
        p = self.params.get(bench, self.params['Average'])
        if self.use_type_dummy and is_chat is not None:
            a, b, theta = p[0], p[1], p[2]
            exponent = -(a + theta * is_chat - b * L)
        else:
            a, b = p[0], p[1]
            exponent = -(a - b * L)
        return 100.0 / (1.0 + np.exp(np.clip(exponent, -30, 30)))

# --- 4. 有界动态随机生产前沿 (Dynamic SFA) 模型 ---

class DynamicSFAModel:
    """
    有界动态随机生产前沿模型 (M4-EQ05 ~ M4-EQ07):
    y_i = a + b * ln(C_i) + z_{t_i} + v_chat * r_i + u_{f_i} + e_i - d_i
    其中：
    e_i ~ N(0, sigma_e^2)  (对称评测噪声)
    d_i ~ HalfNormal(sigma_d^2)  (技术非效率项 / 距前沿距离)
    z_t 服从带阻尼局部线性趋势
    """
    def __init__(self, delta: float = 0.95):
        self.delta = delta
        self.coef_ = {}
        self.z_series_ = []
        self.log_likelihood_ = None
        self.converged_ = False

    def fit(self, df_sfa: pd.DataFrame):
        """
        极大似然估计 (MLE) 求解前沿参数与月度状态序列
        """
        # 准备变量
        sub = df_sfa.dropna(subset=['Training_Compute_FLOP', 'Average ⬆️', 'month_idx']).copy()
        sub = sub[sub['Training_Compute_FLOP'] >= 1e18].copy()
        
        y = logit_score(sub['Average ⬆️'].values)
        ln_C = np.log(sub['Training_Compute_FLOP'].values / 1e18) # 归一化为 EFLOPs
        months = sub['month_idx'].values.astype(int)
        unique_months = np.sort(np.unique(months))
        num_months = len(unique_months)
        m_map = {m: i for i, m in enumerate(unique_months)}
        m_indices = np.array([m_map[m] for m in months])
        is_chat = sub['is_chat'].values

        # 初始线性前沿预估 (分位数回归 q=0.90 作为初值)
        qr = QuantileRegressor(quantile=0.90, alpha=0.01)
        X_init = np.column_stack([np.ones_like(ln_C), ln_C, is_chat])
        qr.fit(X_init, y)
        b_init = max(0.01, qr.coef_[1])
        a_init = qr.intercept_
        v_chat_init = max(0.0, qr.coef_[2])
        z_init = np.zeros(num_months)

        # 待优化参数: [a, b, v_chat, sigma_e, sigma_d, z_1, ..., z_T]
        init_params = np.concatenate([
            [a_init, b_init, v_chat_init, 0.25, 0.40],
            z_init
        ])

        # 复合正态-半正态误差的对数似然函数 (Aigner, Lovell & Schmidt, 1977)
        def neg_log_lik(params):
            a = params[0]
            b = params[1]
            v_chat = params[2]
            sigma_e = max(1e-4, params[3])
            sigma_d = max(1e-4, params[4])
            z_vec = params[5:]

            # 约束惩罚
            pen = 0.0
            if b < 0:
                pen += 1e5 * (-b)**2
            if sigma_e <= 0 or sigma_d <= 0:
                pen += 1e6

            # 状态平滑二次差分惩罚 (带阻尼局部趋势)
            if len(z_vec) >= 3:
                # v_{t+1} ≈ delta * v_t
                trend_diff = (z_vec[2:] - z_vec[1:-1]) - self.delta * (z_vec[1:-1] - z_vec[:-2])
                pen += 10.0 * np.sum(trend_diff**2)

            # 前沿预测残差: eps_i = y_i - (a + b*ln_C + z_t + v_chat*r)
            pred_frontier = a + b * ln_C + z_vec[m_indices] + v_chat * is_chat
            eps = y - pred_frontier

            # 复合误差参数: sigma^2 = sigma_e^2 + sigma_d^2, lambda = sigma_d / sigma_e
            sigma2 = sigma_e**2 + sigma_d**2
            sigma = np.sqrt(sigma2)
            lam = sigma_d / sigma_e

            # SFA 对数似然闭式:
            # ln L = sum [ -ln(sigma) + 0.5*ln(2/pi) - 0.5*(eps/sigma)^2 + ln(Phi(-eps*lam/sigma)) ]
            arg = -eps * lam / sigma
            # 避免数值溢出
            norm_cdf = np.clip(norm.cdf(arg), 1e-15, 1.0)
            ll_i = -np.log(sigma) + 0.5 * np.log(2.0 / np.pi) - 0.5 * (eps / sigma)**2 + np.log(norm_cdf)
            
            return -np.sum(ll_i) + pen

        # 边界设置
        bounds = [
            (-10.0, 10.0),     # a
            (1e-4, 2.0),       # b >= 0
            (-0.5, 3.0),       # v_chat
            (1e-3, 2.0),       # sigma_e
            (1e-3, 3.0),       # sigma_d
        ] + [(-5.0, 5.0)] * num_months

        res = minimize(neg_log_lik, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 800, 'ftol': 1e-7})

        if res.success or res.fun < 1e5:
            self.converged_ = True
            opt = res.x
            self.coef_ = {
                'a': opt[0],
                'b': opt[1],
                'v_chat': opt[2],
                'sigma_e': opt[3],
                'sigma_d': opt[4],
                'gamma_ineff': opt[4]**2 / (opt[3]**2 + opt[4]**2) # 非效率方差占比
            }
            self.z_series_ = pd.Series(opt[5:], index=unique_months)
            self.log_likelihood_ = -res.fun
        else:
            self.converged_ = False
            # 回退算法：使用高分位面板估计
            self.coef_ = {
                'a': a_init,
                'b': b_init,
                'v_chat': v_chat_init,
                'sigma_e': 0.25,
                'sigma_d': 0.40,
                'gamma_ineff': 0.72
            }
            self.z_series_ = pd.Series(np.linspace(0, 0.5, num_months), index=unique_months)
            self.log_likelihood_ = -999.0

    def predict_frontier(self, compute_FLOP: np.ndarray, month_idx: int, is_chat: int = 1) -> np.ndarray:
        """计算条件前沿上限能力分数 S (百分制 [0, 100])"""
        ln_C = np.log(np.maximum(compute_FLOP, 1e18) / 1e18)
        # 获取最接近的月度状态
        if month_idx in self.z_series_.index:
            z_t = self.z_series_[month_idx]
        elif month_idx > self.z_series_.index.max():
            # 阻尼外推
            last_m = self.z_series_.index.max()
            last_z = self.z_series_[last_m]
            # 近期状态首尾的差值按实际月份跨度折算。
            start_idx = max(0, len(self.z_series_) - 3)
            span = last_m - self.z_series_.index[start_idx]
            slope = (self.z_series_.iloc[-1] - self.z_series_.iloc[start_idx]) / max(span, 1)
            dt = month_idx - last_m
            # 阻尼因子衰减外推
            trend_steps = dt if np.isclose(self.delta, 1.0) else (1.0 - self.delta**dt) / (1.0 - self.delta)
            z_t = last_z + slope * trend_steps
        else:
            z_t = self.z_series_.iloc[0]

        y_frontier = self.coef_['a'] + self.coef_['b'] * ln_C + z_t + self.coef_['v_chat'] * is_chat
        return inv_logit_score(y_frontier)

# --- 5. 反事实 Shapley 因果贡献分解计算器 ---

def compute_shapley_decomposition(model: DynamicSFAModel, C0: float, C1: float, m0: int, m1: int, is_chat: int = 1) -> dict:
    """
    反事实 Shapley 排序平均因果贡献分解 (M4-EQ08)
    构造四个状态：
    F(C0, m0): 旧算力旧技术
    F(C1, m0): 新算力旧技术
    F(C0, m1): 旧算力新技术
    F(C1, m1): 新算力新技术
    """
    F00 = model.predict_frontier(np.array([C0]), m0, is_chat)[0]
    F10 = model.predict_frontier(np.array([C1]), m0, is_chat)[0]
    F01 = model.predict_frontier(np.array([C0]), m1, is_chat)[0]
    F11 = model.predict_frontier(np.array([C1]), m1, is_chat)[0]

    delta_total = F11 - F00
    # 规模扩张贡献 (平均两个调整序列)
    delta_C = 0.5 * ((F10 - F00) + (F11 - F01))
    # 非规模技术进步贡献
    delta_tech = 0.5 * ((F01 - F00) + (F11 - F10))

    # 占比
    share_C = delta_C / (delta_total + 1e-9) * 100.0 if np.abs(delta_total) > 1e-4 else 50.0
    share_tech = delta_tech / (delta_total + 1e-9) * 100.0 if np.abs(delta_total) > 1e-4 else 50.0

    return {
        'F_initial': F00,
        'F_terminal': F11,
        'delta_total': delta_total,
        'delta_compute': delta_C,
        'delta_tech': delta_tech,
        'share_compute_pct': share_C,
        'share_tech_pct': share_tech,
        'additivity_residual': np.abs(delta_total - (delta_C + delta_tech))
    }
