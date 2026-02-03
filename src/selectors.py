import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
from .utils import get_logger

logger = get_logger()

def filter_by_pcc(X, y_series, config):
    """
    PCC 筛选 (向量化加速版)
    """
    # 1. 直接计算向量化相关性 (Series.corrwith 或 np.corrcoef)
    # X 是 (Samples, Genes), y_series 是 (Samples,)
    
    # 计算相关系数 (R值)
    # 注意：X.corrwith(y_series) 会对齐索引，非常快
    correlations = X.corrwith(y_series)
    
    # 2. 计算 P值 (基于 t 分布统计量)
    n = len(X)
    # 避免除以0
    t_stat = correlations * np.sqrt((n - 2) / (1 - correlations**2 + 1e-10))
    from scipy.stats import t
    # 双尾检验 P值
    p_values = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))
    
    # 将结果对齐回 DataFrame 以便处理
    stats_df = pd.DataFrame({
        'R': correlations,
        'P': p_values
    }, index=X.columns)
    
    # 处理可能的 NaN (方差为0的情况)
    stats_df = stats_df.dropna()

    # 3. 筛选逻辑 (保持原有的 FDR 或 P-value 逻辑)
    if config.use_fdr:
        from statsmodels.stats.multitest import multipletests
        try:
            reject, pvals_corrected, _, _ = multipletests(stats_df['P'].values, alpha=config.fdr_alpha, method='fdr_bh')
            mask = (reject) & (stats_df['R'].abs() > config.pcc_r_threshold)
        except:
            mask = (stats_df['P'] < config.pcc_p_threshold) & (stats_df['R'].abs() > config.pcc_r_threshold)
    else:
        mask = (stats_df['P'] < config.pcc_p_threshold) & (stats_df['R'].abs() > config.pcc_r_threshold)
        
    return stats_df.index[mask].tolist()

def run_lasso(X, y, alpha, random_state):
    """
    运行 Lasso 回归。
    修改点：增加了 max_iter=10000 以解决 ConvergenceWarning。
    """
    # 设置较高的迭代次数
    MAX_ITER = 10000
    
    # 如果特征太少，直接返回
    if X.shape[1] == 0:
        return []

    try:
        # 优先尝试 LassoCV (自动选择 Alpha)
        # n_jobs=1: 因为我们在 core.py 已经对代谢物进行了多进程并行，
        # 这里如果再开启并行会导致资源争抢和死锁风险。
        model = LassoCV(
            cv=5, 
            random_state=random_state, 
            n_jobs=1, 
            max_iter=MAX_ITER  # <--- 关键修改
        )
        model.fit(X, y)
        
    except Exception as e:
        # 如果 CV 失败（例如样本极少导致无法划分Fold），回退到固定 Alpha 的 Lasso
        # logger.warning(f"LassoCV 失败，回退到固定 Alpha Lasso: {e}")
        model = Lasso(
            alpha=alpha, 
            random_state=random_state, 
            max_iter=MAX_ITER  # <--- 关键修改
        )
        model.fit(X, y)
    
    # 返回系数不为0的特征
    return X.columns[model.coef_ != 0].tolist()

def run_svm_rfe(X, y, n_features):
    """
    运行 SVM-RFE。
    """
    if X.shape[1] <= n_features: 
        return X.columns.tolist()
        
    try:
        estimator = SVR(kernel="linear")
        # step=0.1: 每次移除 10% 的特征，加快速度
        selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        selector.fit(X, y)
        return X.columns[selector.support_].tolist()
    except Exception as e:
        logger.error(f"SVM-RFE 运行出错: {e}")
        # 出错时为了程序不中断，返回空列表或所有基因，这里选择返回空
        return []

def run_rf(X, y, n_estimators, top_k, random_state):
    """
    运行随机森林特征选择。
    """
    if X.shape[1] <= top_k: 
        return X.columns.tolist()
        
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=1  # 同样设为1，避免嵌套并行
        )
        model.fit(X, y)
        
        # 按重要性排序
        indices = np.argsort(model.feature_importances_)[::-1]
        # 取前 K 个
        selected = X.columns[indices[:top_k]].tolist()
        return selected
    except Exception as e:
        logger.error(f"Random Forest 运行出错: {e}")
        return []
