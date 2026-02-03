import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from upsetplot import from_contents, UpSet

# --- 1. WGCNA: 软阈值筛选图 (Scale Free Topology) ---
def plot_soft_threshold(stats_df):
    """
    绘制软阈值选择图：
    左图: Scale Independence (R^2)
    右图: Mean Connectivity
    """
    if stats_df is None or stats_df.empty: return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Scale Independence
    ax1.plot(stats_df['Power'], stats_df['SFT_R_sq'], 'o-', color='red', markersize=5)
    ax1.set_xlabel('Soft Threshold (power)')
    ax1.set_ylabel('Scale Free Topology Model Fit, $R^2$')
    ax1.set_title('Scale Independence')
    ax1.axhline(0.85, color='gray', linestyle='--', alpha=0.7) # 常见阈值线
    for x, y in zip(stats_df['Power'], stats_df['SFT_R_sq']):
        ax1.text(x, y, str(int(x)), fontsize=8)
    
    # Right: Mean Connectivity
    ax2.plot(stats_df['Power'], stats_df['Mean_Connectivity'], 'o-', color='blue', markersize=5)
    ax2.set_xlabel('Soft Threshold (power)')
    ax2.set_ylabel('Mean Connectivity')
    ax2.set_title('Mean Connectivity')
    
    plt.tight_layout()
    return fig

# --- 2. WGCNA: 模块-性状显著性热图 (支持过滤) ---
def plot_module_trait_heatmap(corr_df, pval_df, sig_threshold=0.05, filter_sig=True):
    """
    绘制 Module-Trait 关联热图
    filter_sig: True 时，剔除没有任何显著关联的行(Module)和列(Trait)
    """
    if corr_df is None or corr_df.empty: return None
    
    disp_corr = corr_df.copy()
    disp_pval = pval_df.copy()
    
    # 过滤逻辑: 只保留至少有一个 p < 0.05 的行或列
    if filter_sig:
        sig_mask = disp_pval < sig_threshold
        # 保留显著的列 (Trait)
        cols_to_keep = sig_mask.any(axis=0) 
        # 保留显著的行 (Module)
        rows_to_keep = sig_mask.any(axis=1)
        
        disp_corr = disp_corr.loc[rows_to_keep, cols_to_keep]
        disp_pval = disp_pval.loc[rows_to_keep, cols_to_keep]
    
    if disp_corr.empty: return None

    # 自动调整画布大小
    n_rows, n_cols = disp_corr.shape
    w = max(8, n_cols * 0.6)
    h = max(6, n_rows * 0.6)
    
    fig, ax = plt.subplots(figsize=(w, h))
    
    # 构造标签: R\n(p)
    annot_mat = []
    for r, p in zip(disp_corr.values.flatten(), disp_pval.values.flatten()):
        txt = f"{r:.2f}"
        if p < 0.05: txt += f"\n({p:.1e})" # 显著的加 P 值
        if p < 0.01: txt = "*" + txt # 高显著加星号
        annot_mat.append(txt)
    annot_mat = np.array(annot_mat).reshape(disp_corr.shape)
    
    sns.heatmap(
        disp_corr, 
        annot=annot_mat, 
        fmt="", 
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor='lightgray',
        annot_kws={"size": 8},
        ax=ax
    )
    ax.set_title(f"Module-Trait Relationships (Filtered: {filter_sig})")
    ax.set_xlabel("Traits (Metabolites)")
    ax.set_ylabel("Modules")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return fig

# --- 3. WGCNA: 模块合并聚类图 (ME Clustering) ---
def plot_module_clustering(Z_me, cut_height):
    """绘制 ME 聚类树及切割线"""
    if Z_me is None: return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z_me, ax=ax, orientation='top', color_threshold=0, above_threshold_color='k')
    ax.axhline(cut_height, color='r', linestyle='--', label=f'Cut Height={cut_height}')
    ax.set_title("Clustering of Module Eigengenes (ME)")
    ax.set_ylabel("Dissimilarity (1-Cor)")
    ax.set_xlabel("Modules")
    ax.legend()
    sns.despine()
    return fig

# --- 4. WGCNA: 基因聚类树 + 模块颜色条 ---
def plot_gene_dendro_dual_bars(Z_gene, initial_series, merged_series, title="Gene Dendrogram"):
    """
    绘制基因聚类树，下方配两行颜色条：
    1. 初始模块 (Initial Modules)
    2. 合并后模块 (Merged Modules)
    线条更细 (linewidth=0.5)
    """
    if Z_gene is None: return None
    
    # 设置线条细度
    with plt.rc_context({'lines.linewidth': 0.5}):
        fig = plt.figure(figsize=(12, 8))
        # 布局: 树(85%), 初始条(5%), 合并条(5%), 间距微调
        gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 0.05, 0.05], hspace=0.02)
        
        ax_tree = fig.add_subplot(gs[0])
        ax_bar1 = fig.add_subplot(gs[1])
        ax_bar2 = fig.add_subplot(gs[2])
        
        # 1. 画树
        # no_plot=False, no_labels=True
        ddata = dendrogram(Z_gene, no_labels=True, above_threshold_color='gray', ax=ax_tree)
        ax_tree.set_title(title)
        ax_tree.set_ylabel("TOM Dissimilarity")
        ax_tree.set_xticks([])
        sns.despine(ax=ax_tree, bottom=True)
        
        # 2. 准备颜色映射
        leaves = ddata['leaves'] # 聚类后的基因索引顺序
        
        # 辅助函数：生成颜色列表
        def get_colors(series):
            # 获取按照树叶子顺序排列的模块标签
            # 假设 series 的索引与构建 Z_gene 的矩阵行顺序一致 (即 0..N)
            # 如果 series index 是基因名，且 Z_gene 是基于隐式索引，这里需要小心
            # 我们假设 series 是按原始数据顺序排列的
            ordered_labels = [series.iloc[i] for i in leaves]
            
            unique_labs = sorted(list(set(ordered_labels)))
            # 使用 tab20 调色板，不够循环使用
            palette = sns.color_palette("tab20", len(unique_labs))
            pal_map = {lab: palette[i] for i, lab in enumerate(unique_labs)}
            
            # 特殊处理灰色
            if 'M0' in pal_map: pal_map['M0'] = (0.8, 0.8, 0.8)
            
            return [pal_map[l] for l in ordered_labels], unique_labs, pal_map

        # 生成初始条颜色
        colors1, labs1, map1 = get_colors(initial_series)
        ax_bar1.imshow([colors1], aspect='auto')
        ax_bar1.set_yticks([])
        ax_bar1.set_xticks([])
        ax_bar1.set_ylabel("Initial", rotation=0, ha='right', va='center', fontsize=9)
        
        # 生成合并条颜色
        colors2, labs2, map2 = get_colors(merged_series)
        ax_bar2.imshow([colors2], aspect='auto')
        ax_bar2.set_yticks([])
        ax_bar2.set_xticks([])
        ax_bar2.set_ylabel("Merged", rotation=0, ha='right', va='center', fontsize=9)
        ax_bar2.set_xlabel("Genes")

        # (可选) 添加图例，仅展示合并后的模块颜色
        patches = [plt.Rectangle((0,0),1,1, color=map2[m]) for m in labs2]
        # 放在树的右上角
        ax_tree.legend(patches, labs2, loc='upper right', ncol=2, fontsize='x-small', title="Merged Modules")
        
    return fig

# --- 5. 基础绘图: 样本聚类树 ---
def plot_sample_tree(X, title="Sample Clustering (Outlier Detection)"):
    """仅绘制样本树"""
    try:
        Z = linkage(X, method='average', metric='euclidean')
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, labels=X.index, leaf_rotation=90, ax=ax, above_threshold_color='black')
        ax.set_title(title)
        ax.set_ylabel("Euclidean Distance")
        sns.despine()
        return fig
    except Exception as e:
        print(f"Sample tree error: {e}")
        return None

# --- 6. 基础绘图: PCA ---
def plot_pca(X, title="PCA Plot"):
    from sklearn.decomposition import PCA
    try:
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        df = pd.DataFrame(data=components, columns=['PC1', 'PC2'], index=X.index)
        var = pca.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='PC1', y='PC2', s=100, edgecolor='k', alpha=0.8, ax=ax)
        
        # 标注样本名
        for i, txt in enumerate(df.index):
            ax.annotate(txt, (df.iloc[i].PC1, df.iloc[i].PC2), fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
            
        ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
        ax.set_title(title)
        sns.despine()
        return fig
    except Exception: return None

# --- 7. UpSet Plot ---
def plot_upset(sets_dict):
    if not sets_dict: return None
    try:
        data = from_contents(sets_dict)
        fig = plt.figure(figsize=(10, 6))
        upset = UpSet(data, subset_size='count', show_counts=True, sort_by='cardinality')
        upset.plot(fig=fig)
        return fig
    except Exception: return None

# --- 8. Heatmap ---
def plot_heatmap(X_subset):
    if X_subset.empty: return None
    data = X_subset.T
    h = max(6, len(data) * 0.25)
    # 使用 clustermap
    g = sns.clustermap(data, z_score=0, cmap="vlag", figsize=(10, h), method='average')
    return g.fig

# --- 9. Regression ---
def plot_gene_metabolite_correlation(X, y_series, gene_name, metabolite_name):
    data = pd.DataFrame({'Gene Expression': X[gene_name], 'Metabolite Level': y_series})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(data=data, x='Gene Expression', y='Metabolite Level', ax=ax, scatter_kws={'s':50, 'alpha':0.6}, line_kws={'color':'red'})
    r = data.corr().iloc[0, 1]
    ax.set_title(f"{gene_name} vs {metabolite_name}\nR = {r:.3f}")
    sns.despine()
    return fig

# --- 10. Enrichment (Placeholder) ---
def plot_enrichment_dot(gene_list, database):
    # 此处需要 gseapy 库，为保持简洁暂略，保留接口
    return None
