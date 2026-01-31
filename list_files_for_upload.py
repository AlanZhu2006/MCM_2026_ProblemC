"""
列出所有需要上传到Overleaf的文件
方便用户逐个上传
"""
import os
from pathlib import Path

def list_files_for_upload():
    """列出所有需要上传的文件"""
    
    print("=" * 70)
    print("Overleaf上传文件清单")
    print("=" * 70)
    print()
    
    # 文件清单
    files = {
        "根目录（最重要！）": [
            ("MCM_Latex2026/mcmthesis.cls", "mcmthesis.cls", "模板文件，必须上传到根目录"),
            ("main.tex", "main.tex", "主文件，必须上传到根目录"),
        ],
        "sections文件夹（创建文件夹后上传）": [
            ("sections/summary_sheet.tex", "sections/summary_sheet.tex", ""),
            ("sections/introduction.tex", "sections/introduction.tex", ""),
            ("sections/stage1_preprocessing.tex", "sections/stage1_preprocessing.tex", ""),
            ("sections/stage2_fan_vote_estimation.tex", "sections/stage2_fan_vote_estimation.tex", ""),
            ("sections/stage3_voting_comparison.tex", "sections/stage3_voting_comparison.tex", ""),
            ("sections/stage4_factor_impact.tex", "sections/stage4_factor_impact.tex", ""),
            ("sections/stage5_new_system.tex", "sections/stage5_new_system.tex", ""),
            ("sections/conclusions.tex", "sections/conclusions.tex", ""),
            ("sections/memo_to_producers.tex", "sections/memo_to_producers.tex", ""),
            ("sections/references.tex", "sections/references.tex", ""),
            ("sections/ai_use_report.tex", "sections/ai_use_report.tex", ""),
        ],
        "visualizations文件夹（创建文件夹后上传）": [
            ("visualizations/stage2_fan_vote_estimation.png", "visualizations/stage2_fan_vote_estimation.png", ""),
            ("visualizations/stage3_voting_comparison.png", "visualizations/stage3_voting_comparison.png", ""),
            ("visualizations/stage4_factor_impact.png", "visualizations/stage4_factor_impact.png", ""),
            ("visualizations/stage5_ml_system.png", "visualizations/stage5_ml_system.png", ""),
            ("visualizations/overall_summary.png", "visualizations/overall_summary.png", ""),
            ("visualizations/uncertainty_analysis.png", "visualizations/uncertainty_analysis.png", ""),
            ("visualizations/confidence_intervals.png", "visualizations/confidence_intervals.png", ""),
            ("visualizations/controversial_cases_detailed.png", "visualizations/controversial_cases_detailed.png", ""),
            ("visualizations/parameter_sensitivity.png", "visualizations/parameter_sensitivity.png", ""),
            ("visualizations/data_sensitivity.png", "visualizations/data_sensitivity.png", ""),
        ],
    }
    
    total_files = 0
    missing_files = []
    
    for category, file_list in files.items():
        print(f"\n{category}:")
        print("-" * 70)
        for i, (source_path, target_path, note) in enumerate(file_list, 1):
            source = Path(source_path)
            if source.exists():
                size = source.stat().st_size / 1024  # KB
                status = "[OK]"
                total_files += 1
            else:
                status = "[缺失]"
                missing_files.append(source_path)
            
            print(f"{i:2d}. {status} {source_path}")
            if note:
                print(f"     -> {target_path} ({note})")
            elif status == "[OK]":
                print(f"     -> {target_path} ({size:.1f} KB)")
            print()
    
    print("=" * 70)
    print(f"总计: {total_files} 个文件")
    if missing_files:
        print(f"缺失: {len(missing_files)} 个文件")
        print("\n缺失的文件:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("所有文件都存在！")
    print("=" * 70)
    print()
    print("上传顺序建议:")
    print("1. 先上传 mcmthesis.cls 到根目录（最重要！）")
    print("2. 上传 main.tex 到根目录")
    print("3. 创建 sections/ 文件夹，上传所有 .tex 文件")
    print("4. 创建 visualizations/ 文件夹，上传所有 .png 文件")
    print("5. 设置 main.tex 为主文档")
    print("6. 编译")

if __name__ == '__main__':
    list_files_for_upload()
