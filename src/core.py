import pandas as pd
import numpy as np
import math
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from . import loaders, selectors, plotting
from .utils import get_logger

logger = get_logger()

# --- Worker Function (必须在类外定义以便并行) ---
def _process_metabolite_task(metabolite_name, X, y_series, config, target_k):
    """
    单个代谢物的处理任务。
    包含：PCC预筛选 -> (可选: Bootstrap稳定性选择) -> Lasso/SVM/RF
    """
    # 1. PCC 预筛选 (始终基于全量数据，作为第一道防线)
    pcc_genes = selectors.filter_by_pcc(X, y_series, config)
    n_pcc = len(pcc_genes)
    
    if n_pcc == 0:
        return (metabolite_name, [], [], [], n_pcc)

    X_sub = X[pcc_genes]
    
    # 定义单次运行的函数
    def run_single_pass(X_curr, y_curr):
        l_g = selectors.run_lasso(X_curr, y_curr, config.lasso_alpha, config.random_state)
        s_g = selectors.run_svm_rfe(X_curr, y_curr, target_k)
        r_g = selectors.run_rf(X_curr, y_curr, config.rf_n_estimators, target_k, config.random_state)
        return set(l_g), set(s_g), set(r_g)

    # 2. 特征筛选 (根据配置决定是否使用稳定性选择)
    final_lasso, final_svm, final_rf = set(), set(), set()
    
    if config.enable_stability:
        # === 稳定性选择 (Stability Selection) ===
        counts_lasso = {}
        counts_svm = {}
        counts_rf = {}
        
        # 初始化计数
        for g in X_sub.columns:
            counts_lasso[g] = 0
            counts_svm[g] = 0
            counts_rf[g] = 0
            
        # Bootstrap 循环
        for i in range(config.n_bootstraps):
            X_boot, y_boot = resample(X_sub, y_series, random_state=config.random_state + i)
            l_set, s_set, r_set = run_single_pass(X_boot, y_boot)
            
            for g in l_set: counts_lasso[g] += 1
            for g in s_set: counts_svm[g] += 1
            for g in r_set: counts_rf[g] += 1
        
        # 过滤：保留出现频率 > 阈值的基因
        threshold_count = config.n_bootstraps * config.stability_threshold
        final_lasso = [g for g, c in counts_lasso.items() if c >= threshold_count]
        final_svm = [g for g, c in counts_svm.items() if c >= threshold_count]
        final_rf = [g for g, c in counts_rf.items() if c >= threshold_count]
        
    else:
        # === 传统模式 (运行一次) ===
        l_set, s_set, r_set = run_single_pass(X_sub, y_series)
        final_lasso, final_svm, final_rf = list(l_set), list(s_set), list(r_set)

    return (metabolite_name, final_lasso, final_svm, final_rf, n_pcc)


class MultiOmicsEngine:
    def __init__(self, gene_path, metab_path, config):
        self.config = config
        self.X_raw, self.y_raw = loaders.load_and_clean_data(gene_path, metab_path)
        self.X_scaled = None
        
        # 初始化结果容器
        self.results = {
            "grn_edges": [],
            "key_genes": set(),
            "stats": [],
            "module_trait_corr": None,
            "module_trait_pval": None,
            "wgcna_stats": None, # 存放软阈值分析结果
            "wgcna_Z": None,     # 存放基因聚类树
            "ME_Z": None         # 存放ME聚类树
        }
        self.modules_df = None   # 基因-模块映射表
        self.ME_df = None        # 模块特征基因矩阵
        self.sets_data = {}      # 集合数据用于画 Venn/UpSet
        
    def preprocess(self):
        if self.X_raw is None: raise ValueError("数据未加载")
        logger.info("正在标准化数据...")
        self.X_scaled = loaders.standardize_data(self.X_raw)
        
    def run(self):
        """执行主分析流程：集成筛选 + 独立WGCNA"""
        if self.X_scaled is None: self.preprocess()
        
        # --- Part 1: 机器学习集成筛选 (Parallel Ensemble) ---
        n_samples = self.X_scaled.shape[0]
        dynamic_k = math.ceil(n_samples * self.config.selection_ratio)
        dynamic_k = max(dynamic_k, self.config.min_selected_features)
        
        msg = f"开始机器学习分析 (Target K={dynamic_k})"
        if self.config.enable_stability:
            msg += f" [Stability: Bootstraps={self.config.n_bootstraps}]"
        logger.info(msg)
        
        tasks = Parallel(n_jobs=self.config.n_jobs)(
            delayed(_process_metabolite_task)(
                m, self.X_scaled, self.y_raw[m], self.config, dynamic_k
            ) for m in tqdm(self.y_raw.columns, desc="Processing ML")
        )
        
        # 汇总 ML 结果
        logger.info("正在汇总机器学习结果...")
        all_lasso = set()
        all_svm = set()
        all_rf = set()
        
        for m_name, l_g, s_g, r_g, n_left in tasks:
            self.results['stats'].append({
                'Metabolite': m_name, 
                'PCC_Genes_Left': n_left,
                'Lasso_Selected': len(l_g),
                'SVM_Selected': len(s_g),
                'RF_Selected': len(r_g)
            })
            
            if n_left == 0: continue
            
            all_lasso.update(l_g)
            all_svm.update(s_g)
            all_rf.update(r_g)
            
            # 构建 GRN 边 (Evidence Score)
            union_genes = set(l_g) | set(s_g) | set(r_g)
            for gene in union_genes:
                methods = []
                if gene in l_g: methods.append('Lasso')
                if gene in s_g: methods.append('SVM')
                if gene in r_g: methods.append('RF')
                
                self.results['grn_edges'].append({
                    'Metabolite': m_name,
                    'Gene': gene,
                    'Evidence_Score': len(methods),
                    'Methods': "|".join(methods)
                })

        # 计算共识基因
        self.key_genes = list(all_lasso.intersection(all_svm).intersection(all_rf))
        # 如果太少，放宽到并集 (防止后续绘图报错)
        if len(self.key_genes) < 2:
            self.key_genes = list(all_lasso | all_svm | all_rf)
            
        self.key_genes.sort()
        self.sets_data = {'Lasso': all_lasso, 'SVM': all_svm, 'RF': all_rf}
        logger.info(f"机器学习筛选完成。Key Genes: {len(self.key_genes)}")
        
        # --- Part 2: 独立 WGCNA 流程 ---
        # 这里的调用名称必须与下面定义的方法名一致
        self.analyze_wgcna_pipeline()

    def analyze_wgcna_pipeline(self):
        """学术级 WGCNA 完整流程 (修改版：保留初始模块标签)"""
        logger.info("开始 WGCNA 分析流程...")
        
        # --- Step 1: 基因初筛 ---
        variances = self.X_scaled.var().sort_values(ascending=False)
        top_n = min(self.config.wgcna_top_n_genes, len(variances))
        wgcna_genes = variances.index[:top_n].tolist()
        X_wgcna = self.X_scaled[wgcna_genes]
        
        # --- Step 2: 计算 TOM & 聚类 ---
        logger.info(f"计算邻接矩阵 (Power={self.config.wgcna_soft_power})...")
        cor_mat = np.corrcoef(X_wgcna.T)
        adj_mat = np.power(np.abs(np.nan_to_num(cor_mat)), self.config.wgcna_soft_power)
        
        dissim = 1 - adj_mat
        np.fill_diagonal(dissim, 0)
        
        dist_vec = squareform(np.clip(dissim, 0, 1), checks=False)
        Z = linkage(dist_vec, method='average')
        self.results['wgcna_Z'] = Z 
        
        # --- Step 3: 动态剪切 (初始模块) ---
        max_d = np.max(Z[:, 2])
        labels_initial = fcluster(Z, t=max_d * 0.9, criterion='distance')
        
        # 初始化 modules_df，保存初始标签
        # 注意：为了绘图好看，我们将数字标签转为字符串 "M1", "M2" 等
        self.modules_df = pd.DataFrame({
            'Gene': wgcna_genes,
            'Module_Initial': [f"M{x}" for x in labels_initial]
        })
        
        # 计算初始 ME
        temp_df = pd.DataFrame({'Gene': wgcna_genes, 'Module': labels_initial})
        ME_df_initial = self._calculate_MEs(X_wgcna, temp_df)
        
        # --- Step 4: 模块合并 (Merged Modules) ---
        labels_merged = []
        if ME_df_initial.shape[1] > 1:
            logger.info("正在合并相似模块...")
            me_corr = ME_df_initial.corr()
            me_dist = 1 - me_corr
            Z_me = linkage(squareform(np.clip(me_dist, 0, 1), checks=False), method='average')
            self.results['ME_Z'] = Z_me 
            
            # 聚类切割
            merge_labels = fcluster(Z_me, t=self.config.wgcna_merge_cut_height, criterion='distance')
            
            # 映射: 旧模块 -> 新模块
            # ME 列名例如 "ME_1", merge_labels 对应 ME_df 的列顺序
            merge_map = {old_col: f"Merged{new_id}" for old_col, new_id in zip(me_corr.columns, merge_labels)}
            
            # 更新每个基因的标签
            for _, row in temp_df.iterrows():
                old_m = f"ME_{row['Module']}"
                # 处理可能的命名差异
                if old_m not in merge_map and str(row['Module']) in merge_map:
                    old_m = str(row['Module'])
                
                if old_m in merge_map:
                    labels_merged.append(merge_map[old_m])
                else:
                    labels_merged.append(f"M{row['Module']}") # 未合并
        else:
            labels_merged = self.modules_df['Module_Initial'].tolist()
            
        self.modules_df['Module_Merged'] = labels_merged
        
        # 重新计算合并后的 ME
        # _calculate_MEs 需要 'Module' 列
        df_for_calc = self.modules_df[['Gene', 'Module_Merged']].rename(columns={'Module_Merged': 'Module'})
        self.ME_df = self._calculate_MEs(X_wgcna, df_for_calc)
        
        logger.info(f"WGCNA 完成。初始模块数: {len(np.unique(labels_initial))}, 合并后: {self.ME_df.shape[1]}")

        # --- Step 5: 模块-性状关联 ---
        self._calculate_module_trait_correlation()

    def _calculate_MEs(self, X_data, module_assign_df):
        """计算模块特征基因 (PCA PC1)"""
        me_dict = {}
        pca = PCA(n_components=1)
        for m in module_assign_df['Module'].unique():
            genes = module_assign_df[module_assign_df['Module'] == m]['Gene']
            if len(genes) < self.config.wgcna_min_module_size: continue
            
            dat = X_data[genes]
            if dat.shape[1] < 1: continue
            
            pc1 = pca.fit_transform(dat).flatten()
            
            # 符号校正
            mean_expr = dat.mean(axis=1)
            # 处理 constant input 警告
            if np.std(pc1) > 0 and np.std(mean_expr) > 0:
                if np.corrcoef(pc1, mean_expr)[0,1] < 0:
                    pc1 = -pc1
            
            key_name = m if str(m).startswith("M") else f"ME_{m}"
            me_dict[key_name] = pc1
            
        return pd.DataFrame(me_dict, index=X_data.index)

    def _calculate_module_trait_correlation(self):
        """计算 ME 与 代谢物 的相关性"""
        if self.ME_df is None or self.ME_df.empty: return
        
        common = self.ME_df.index.intersection(self.y_raw.index)
        me_sub = self.ME_df.loc[common]
        y_sub = self.y_raw.loc[common]
        
        corr_df = pd.DataFrame(index=me_sub.columns, columns=y_sub.columns)
        pval_df = pd.DataFrame(index=me_sub.columns, columns=y_sub.columns)
        
        for m in me_sub.columns:
            for t in y_sub.columns:
                r, p = pearsonr(me_sub[m], y_sub[t])
                corr_df.loc[m, t] = r
                pval_df.loc[m, t] = p
                
        self.results['module_trait_corr'] = corr_df.astype(float)
        self.results['module_trait_pval'] = pval_df.astype(float)

    def save_results(self):
        """保存所有结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # GRN
        pd.DataFrame(self.results['grn_edges']).to_csv(
            os.path.join(self.config.output_dir, "GRN_network.csv"), index=False
        )
        
        # Key Genes
        pd.DataFrame(self.key_genes, columns=['Key_Genes']).to_csv(
            os.path.join(self.config.output_dir, "key_genes.csv"), index=False
        )
        
        # WGCNA Modules
        if self.modules_df is not None:
            self.modules_df.to_csv(os.path.join(self.config.output_dir, "wgcna_modules.csv"), index=False)
            
        # Module-Trait
        if self.results['module_trait_corr'] is not None:
            self.results['module_trait_corr'].to_csv(os.path.join(self.config.output_dir, "module_trait_corr.csv"))
            self.results['module_trait_pval'].to_csv(os.path.join(self.config.output_dir, "module_trait_pval.csv"))
            
    def generate_plots(self):
        """生成并保存图表 (Batch Mode)"""
        plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 这里只生成最关键的图保存到磁盘，GUI 中会实时生成更多
        if self.sets_data:
            fig = plotting.plot_upset({'Lasso': self.sets_data['Lasso'], 'SVM': self.sets_data['SVM'], 'RF': self.sets_data['RF']})
            if fig: fig.savefig(os.path.join(plots_dir, "upset.png"))
            
        if self.results['module_trait_corr'] is not None:
            fig = plotting.plot_module_trait_heatmap(self.results['module_trait_corr'], self.results['module_trait_pval'])
            if fig: fig.savefig(os.path.join(plots_dir, "module_trait_heatmap.png"))

