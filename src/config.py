from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """分析参数配置类"""
    output_dir: str = "results"
    
    # --- 1. 预筛选参数 ---
    pcc_p_threshold: float = 0.05
    use_fdr: bool = True
    fdr_alpha: float = 0.05
    pcc_r_threshold: float = 0.3
    
    # --- 2. 机器学习与稳定性选择 ---
    enable_stability: bool = True      
    n_bootstraps: int = 20
    stability_threshold: float = 0.6
    
    lasso_alpha: float = 0.01          
    rf_n_estimators: int = 100
    selection_ratio: float = 1.0       
    min_selected_features: int = 5     
    
    # --- 3. WGCNA 高级配置 (修改部分) ---
    # 初筛: 使用多少个高变基因构建网络 (标准流程通常取前 3000-5000 或前 25%)
    wgcna_top_n_genes: int = 3000      
    wgcna_soft_power: int = 6          # 软阈值 (可以通过可视化图表辅助选择)
    wgcna_min_module_size: int = 30    # 最小模块大小 (WGCNA 默认通常是 30)
    wgcna_merge_cut_height: float = 0.25 # 模块合并阈值 (0.25 对应相关性 0.75)
    
    random_state: int = 42
    n_jobs: int = -1

