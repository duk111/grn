import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger()

def load_and_clean_data(gene_path, metab_path):
    """加载原始数据，转置并对齐样本"""
    try:
        df_gene = pd.read_csv(gene_path, index_col=0)
        df_metab = pd.read_csv(metab_path, index_col=0)
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return None, None

    # 转置：(样本 x 特征)
    X = df_gene.T
    y = df_metab.T
    
    # 统一索引格式
    X.index = X.index.astype(str).str.strip()
    y.index = y.index.astype(str).str.strip()
    
    # 样本对齐
    common_samples = X.index.intersection(y.index)
    if len(common_samples) < len(X.index):
        logger.info(f"样本对齐: 从 {len(X.index)} 缩减为 {len(common_samples)}")
    
    X = X.loc[common_samples]
    y = y.loc[common_samples]
    
    # 基础清洗：去除全为0或方差为0的基因
    X = X.loc[:, X.var() > 0]
    
    return X, y

def standardize_data(df):
    """Z-score 标准化"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    return pd.DataFrame(data_scaled, index=df.index, columns=df.columns)
