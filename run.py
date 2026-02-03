import os
from deepomics.config import AnalysisConfig
from deepomics.core import MultiOmicsEngine

def main():
    # 1. 准备配置
    # 在这里可以灵活修改参数，而不需要去改源代码
    config = AnalysisConfig(
        output_dir="results_project_A",
        use_fdr=True,           # 开启 FDR 校正
        n_jobs=-1,              # 全速并行
        selection_ratio=0.5     # 调整筛选比例
    )
    
    # 2. 指定数据路径
    gene_file = "data/ym_transcriptome_avg.csv"
    metab_file = "data/ym_metabolome_avg.csv"
    
    if not os.path.exists(gene_file):
        print("请确保 data 目录下有 csv 文件")
        return

    # 3. 初始化并运行引擎
    engine = MultiOmicsEngine(gene_file, metab_file, config)
    
    try:
        engine.run()
        engine.save_results()
        engine.generate_plots()
        print("分析全部完成！结果已保存在 results_project_A 目录。")
        
    except Exception as e:
        print(f"运行时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
