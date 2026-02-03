import streamlit as st
import pandas as pd
import os
import shutil
import time
import io
import math

# 引入动态网络图库
from streamlit_agraph import agraph, Node, Edge, Config

# 导入 DeepOmics 核心包
from deepomics.config import AnalysisConfig
from deepomics.core import MultiOmicsEngine
from deepomics import plotting

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="多组学数据分析",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 辅助函数 ---
def save_uploaded_file(uploaded_file, target_folder):
    if not os.path.exists(target_folder): os.makedirs(target_folder)
    file_path = os.path.join(target_folder, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def display_fig_with_download(fig, filename_prefix, container_width=False):
    if fig is None: return
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format='svg', bbox_inches='tight')
    buf_svg.seek(0)
    
    # 布局: 图居中或拉伸，按钮在下方
    st.pyplot(fig, width="stretch" if container_width else "content")
    
    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
    
    with c2:
        st.download_button(
            label="📥 Download PNG", 
            data=buf_png, 
            file_name=f"{filename_prefix}.png", 
            mime="image/png",
            use_container_width=True
        )
    with c3:
        st.download_button(
            label="📥 Download SVG", 
            data=buf_svg, 
            file_name=f"{filename_prefix}.svg", 
            mime="image/svg",
            use_container_width=True
        )

def calculate_concentric_positions(nodes_data, center_radius=0, layer_gap=200):
    positions = {}
    groups = {}
    for n in nodes_data: 
        g = n.get('group', 'score1')
        if g not in groups: groups[g] = []
        groups[g].append(n['id'])
        
    radii = {'metabolite': center_radius, 'score3': center_radius+layer_gap, 'score2': center_radius+layer_gap*2, 'score1': center_radius+layer_gap*3}
    for group_name, node_ids in groups.items():
        count = len(node_ids)
        if count == 0: continue
        if group_name == 'metabolite' and count == 1:
            positions[node_ids[0]] = {'x': 0, 'y': 0}
            continue
        radius = radii.get(group_name, center_radius+layer_gap*3)
        angle_step = 2 * math.pi / count
        for i, node_id in enumerate(node_ids):
            angle = i * angle_step
            positions[node_id] = {'x': radius * math.cos(angle), 'y': radius * math.sin(angle)}
    return positions

# --- 3. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数配置")
    
    with st.expander("1. 预筛选 (PCC/FDR)", expanded=False):
        p_threshold = st.number_input("P-value 阈值", 0.05, step=0.01, format="%.3f")
        r_threshold = st.slider("相关系数 (|R|) 阈值", 0.0, 1.0, 0.3, 0.05)
        use_fdr = st.checkbox("启用 FDR (BH) 校正", True)
    
    with st.expander("2. 稳定性选择 (Bootstrap)", expanded=True):
        st.caption("学术增强：通过重抽样提高结果鲁棒性。")
        enable_stability = st.checkbox("启用稳定性选择", True)
        n_bootstraps = st.number_input("重抽样次数", 10, 100, 20)
        stability_threshold = st.slider("保留频率阈值", 0.5, 1.0, 0.6)
    
    with st.expander("3. 机器学习模型", expanded=False):
        lasso_alpha = st.number_input("Lasso Alpha", 0.01, step=0.001, format="%.4f")
        selection_ratio = st.slider("筛选特征比例", 0.1, 2.0, 1.0)
        rf_estimators = st.number_input("RF 树数量", 100)
    
    # --- 新增 WGCNA 高级配置 ---
    with st.expander("4. WGCNA 设置", expanded=True):
        st.markdown("**网络构建:**")
        wgcna_top_n = st.number_input("初筛高变基因数", 1000, 20000, 3000, help="使用方差最大的前N个基因构建网络")
        wgcna_power = st.slider("软阈值 (Power)", 1, 20, 6)
        
        st.markdown("**模块识别:**")
        min_mod_size = st.number_input("最小模块大小", 10, 200, 30)
        merge_cut = st.slider("模块合并阈值 (Cut Height)", 0.0, 1.0, 0.25, step=0.05, help="0.25 表示相关性 > 0.75 时合并")

    with st.expander("5. 系统设置", expanded=False):
        n_jobs = st.radio("并行核心数", [-1, 1, 2, 4], 0)

# --- 4. 主页面 ---
st.title("🧬 多组学数据分析")
st.divider()

c1, c2 = st.columns(2)
with c1:
    gene_file = st.file_uploader("Gene Matrix (基因表达量)", ["csv"], key="gene")
    if gene_file: st.success(f"已加载: {gene_file.name}")
with c2:
    metab_file = st.file_uploader("Metabolome Matrix (代谢物丰度)", ["csv"], key="metab")
    if metab_file: st.success(f"已加载: {metab_file.name}")

st.divider()
run_btn_col, _ = st.columns([1, 4])
with run_btn_col:
    run_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# --- 5. 核心逻辑 ---
if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
if 'engine_result' not in st.session_state: st.session_state['engine_result'] = None

if run_button:
    if not gene_file or not metab_file:
        st.error("❌ 请先上传数据！")
    else:
        temp_dir = "temp_uploads"
        output_dir = f"results_{int(time.time())}"
        try:
            gene_path = save_uploaded_file(gene_file, temp_dir)
            metab_path = save_uploaded_file(metab_file, temp_dir)
            
            # --- 初始化配置 ---
            config = AnalysisConfig(
                output_dir=output_dir, 
                pcc_p_threshold=p_threshold, 
                pcc_r_threshold=r_threshold,
                use_fdr=use_fdr, 
                # Stability
                enable_stability=enable_stability,
                n_bootstraps=n_bootstraps,
                stability_threshold=stability_threshold,
                # ML
                lasso_alpha=lasso_alpha, 
                selection_ratio=selection_ratio,
                rf_n_estimators=rf_estimators, 
                # WGCNA
                wgcna_top_n_genes=wgcna_top_n,
                wgcna_soft_power=wgcna_power,
                wgcna_min_module_size=min_mod_size,
                wgcna_merge_cut_height=merge_cut,
                # Sys
                n_jobs=n_jobs
            )
            
            engine = MultiOmicsEngine(gene_path, metab_path, config)
            
            with st.status("正在进行深度分析...", expanded=True) as status:
                st.write("📂 数据标准化与质控...")
                engine.preprocess()
                
                st.write(f"⚙️ 运行多算法集成筛选 (Stability={enable_stability})...")
                # 这里假设 engine.run() 内部已经包含了 ML 和 WGCNA 的逻辑 (见上一轮 core.py 修改)
                engine.run() 
                
                st.write("💾 保存结果文件...")
                engine.save_results()
                status.update(label="✅ 分析完成!", state="complete", expanded=False)
            
            st.session_state['engine_result'] = engine
            st.session_state['analysis_done'] = True
            
            st.success(f"✅ 完成！共识关键基因: {len(engine.key_genes)} 个。")
            if hasattr(engine, 'ME_df') and engine.ME_df is not None:
                st.info(f"WGCNA: 识别出 {engine.ME_df.shape[1]} 个基因模块。")

        except Exception as e:
            st.error("分析错误！")
            st.exception(e)
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

# --- 6. 结果展示 ---
if st.session_state['analysis_done']:
    engine = st.session_state['engine_result']
    
    st.divider()
    st.header("📊 结果可视化")
    
    tabs = st.tabs([
        "🔬 数据质控", "🧩 WGCNA 分析", "🕸️ 关键基因网络", 
        "📊 算法交集", "🔥 关键基因热图", "🧪 功能富集"
    ])
    
    # --- Tab 1: 数据质控 (布局优化: 垂直排列) ---
    with tabs[0]:
        st.subheader("1. 样本聚类 (离群点检测)")
        fig_tree = plotting.plot_sample_tree(engine.X_scaled)
        display_fig_with_download(fig_tree, "qc_sample_tree", container_width=True)
        
        st.divider()
        
        st.subheader("2. PCA 主成分分析")
        fig_pca = plotting.plot_pca(engine.X_scaled)
        display_fig_with_download(fig_pca, "qc_pca", container_width=True)
        
        # 如果计算了软阈值 stats
        if 'wgcna_stats' in engine.results and engine.results['wgcna_stats'] is not None:
            st.divider()
            st.subheader("3. 网络拓扑分析 (Soft Thresholding)")
            fig_soft = plotting.plot_soft_threshold(engine.results['wgcna_stats'])
            display_fig_with_download(fig_soft, "wgcna_soft_threshold", container_width=True)

    # --- Tab 2: WGCNA (完整流程) ---
    with tabs[1]:
        st.subheader("加权基因共表达网络分析 (WGCNA)")
        
        # 1. 基因聚类树
        st.markdown("#### 1. 基因聚类与模块划分")
        st.caption(f"基于 Top {wgcna_top_n} 高变基因构建 TOM 矩阵。")
        if 'wgcna_Z' in engine.results and engine.modules_df is not None:
            # 必须确保 engine.modules_df 包含 Initial 和 Merged 列
            if 'Module_Initial' in engine.modules_df.columns:
                fig_gene = plotting.plot_gene_dendro_dual_bars(
                    engine.results['wgcna_Z'], 
                    engine.modules_df['Module_Initial'],
                    engine.modules_df['Module_Merged'],
                    title=f"Gene Dendrogram (Lines: Thin)"
                )
                display_fig_with_download(fig_gene, "wgcna_gene_dendro_dual", container_width=True)
            else:
                st.warning("旧版数据结构，请重新运行分析以生成双层模块图。")
        else:
            st.warning("未生成基因聚类数据。")
            
        st.divider()
        
        # 2. ME 聚类
        st.markdown("#### 2. 模块特征基因 (ME) 聚类与合并")
        if 'ME_Z' in engine.results:
            fig_me = plotting.plot_module_clustering(engine.results['ME_Z'], merge_cut)
            display_fig_with_download(fig_me, "wgcna_me_clustering")
        else:
            st.info("模块数量较少，无需合并或未进行合并。")
            
        st.divider()
        
        # 3. 关联热图
        st.markdown("#### 3. 模块-代谢物关联热图")
        filter_sig = st.checkbox("仅展示显著结果 (p < 0.05)", value=True)
        
        mt_corr = engine.results.get('module_trait_corr')
        mt_pval = engine.results.get('module_trait_pval')
        
        if mt_corr is not None:
            fig_mt = plotting.plot_module_trait_heatmap(mt_corr, mt_pval, filter_sig=filter_sig)
            if fig_mt:
                display_fig_with_download(fig_mt, "wgcna_module_trait_heatmap", container_width=True)
            else:
                st.warning("筛选后无显著关联模块。")
        else:
            st.warning("未进行关联分析。")

    # --- Tab 3: 网络图 (保持原有逻辑) ---
    with tabs[2]:
        st.subheader("Gene-Metabolite Regulatory Network")
        grn_data = engine.results.get('grn_edges', [])
        if grn_data:
            df_edges = pd.DataFrame(grn_data)
            nc1, nc2, nc3 = st.columns(3)
            with nc1: min_score = st.slider("Evidence Score", 1, 3, 2)
            with nc2: physics = st.checkbox("启用物理引擎", True)
            
            filtered_df = df_edges[df_edges['Evidence_Score'] >= min_score]
            if not filtered_df.empty:
                nodes = []
                edges = []
                added = set()
                node_groups = []
                for _, r in filtered_df.iterrows():
                    g, m, s = r['Gene'], r['Metabolite'], r['Evidence_Score']
                    if m not in added:
                        node_groups.append({'id': m, 'group': 'metabolite'})
                        added.add(m)
                    if g not in added:
                        node_groups.append({'id': g, 'group': f'score{s}'})
                        added.add(g)
                    edges.append(Edge(source=g, target=m, label=str(s), width=s, color="#D5D8DC"))
                
                pos_map = {}
                if not physics: pos_map = calculate_concentric_positions(node_groups)
                
                for info in node_groups:
                    nid = info['id']
                    grp = info['group']
                    kw = {'id': nid, 'label': nid, 
                          'shape': 'square' if grp=='metabolite' else 'dot', 
                          'color': '#E74C3C' if grp=='metabolite' else '#3498DB',
                          'size': 25 if grp=='metabolite' else 15}
                    
                    if not physics and nid in pos_map:
                        kw['x'] = pos_map[nid]['x']
                        kw['y'] = pos_map[nid]['y']
                        kw['fixed'] = True
                    nodes.append(Node(**kw))
                
                config_graph = Config(width=1000, height=800, directed=False, physics=physics, collapsible=False)
                agraph(nodes=nodes, edges=edges, config=config_graph)
            else: st.warning("当前筛选条件下无数据。")
        else: st.warning("无网络数据。")

    # --- Tab 4: UpSet ---
    with tabs[3]:
        st.subheader("特征选择算法交集 (UpSet Plot)")
        if hasattr(engine, 'sets_data') and engine.sets_data:
            upset_data = {
                'Lasso': engine.sets_data['Lasso'],
                'SVM-RFE': engine.sets_data['SVM'],
                'Random Forest': engine.sets_data['RF']
            }
            fig_upset = plotting.plot_upset(upset_data)
            display_fig_with_download(fig_upset, "upset_intersection")
        else: st.info("无数据。")

    # --- Tab 5: Heatmap ---
    with tabs[4]:
        st.subheader("共识关键基因表达热图")
        if engine.key_genes:
            fig_heat = plotting.plot_heatmap(engine.X_raw[engine.key_genes])
            display_fig_with_download(fig_heat, "heatmap")
        else: st.warning("未筛选出关键基因。")
        
        if engine.results['grn_edges']:
            st.divider()
            st.subheader("单基因回归验证")
            df_e = pd.DataFrame(engine.results['grn_edges'])
            c1, c2 = st.columns(2)
            mets = sorted(df_e['Metabolite'].unique())
            m = c1.selectbox("选择代谢物", mets)
            gs = sorted(df_e[df_e['Metabolite']==m]['Gene'].unique())
            if len(gs)>0: 
                g = c2.selectbox("选择关联基因", gs)
                fig_reg = plotting.plot_gene_metabolite_correlation(engine.X_raw, engine.y_raw[m], g, m)
                display_fig_with_download(fig_reg, "regression_plot")

    # --- Tab 6: Enrichment ---
    with tabs[5]:
        st.subheader("功能富集 (Enrichr)")
        st.info("需要联网访问 Enrichr API。")
        db = st.selectbox("选择数据库", ["GO_Biological_Process_2021", "KEGG_2021_Human"])
        if st.button("Run Enrichment"):
            if len(engine.key_genes)>=3:
                fig_go = plotting.plot_enrichment_dot(engine.key_genes, db)
                if fig_go: display_fig_with_download(fig_go, "enrichment_dotplot")
                else: st.warning("无显著富集结果或连接超时。")
            else: st.warning("基因数过少。")
