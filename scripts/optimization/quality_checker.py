"""
最终质量检查脚本
检查数据一致性、数学公式、交叉引用等
"""

import json
from pathlib import Path
import re

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: pandas未安装，部分检查将跳过")

def check_data_consistency():
    """检查数据一致性"""
    print("=" * 70)
    print("1. 数据一致性检查")
    print("=" * 70)
    
    issues = []
    
    # 检查验证结果
    if Path('validation_results.json').exists():
        with open('validation_results.json', 'r', encoding='utf-8') as f:
            validation = json.load(f)
        
        total_weeks = validation.get('total_weeks', 0)
        correct = validation.get('correct_predictions', 0)
        accuracy = correct / total_weeks * 100 if total_weeks > 0 else 0
        
        print(f"\n✓ Stage 2 验证结果:")
        print(f"  总周数: {total_weeks}")
        print(f"  正确预测: {correct}")
        print(f"  准确率: {accuracy:.2f}%")
        
        if total_weeks != 299:
            issues.append(f"总周数应该是299，实际为{total_weeks}")
        if abs(accuracy - 90.0) > 0.1:
            issues.append(f"准确率应该是90.0%，实际为{accuracy:.2f}%")
    
    # 检查Stage 3结果
    if Path('voting_method_comparison.csv').exists() and HAS_PANDAS:
        df = pd.read_csv('voting_method_comparison.csv')
        rank_correct = df['rank_method_correct'].sum()
        percent_correct = df['percent_method_correct'].sum()
        total = len(df)
        
        rank_acc = rank_correct / total * 100
        percent_acc = percent_correct / total * 100
        
        print(f"\n✓ Stage 3 投票方法比较:")
        print(f"  排名法准确率: {rank_acc:.2f}% ({rank_correct}/{total})")
        print(f"  百分比法准确率: {percent_acc:.2f}% ({percent_correct}/{total})")
        
        if abs(rank_acc - 60.5) > 0.5:
            issues.append(f"排名法准确率应该是60.5%，实际为{rank_acc:.2f}%")
        if abs(percent_acc - 97.0) > 0.5:
            issues.append(f"百分比法准确率应该是97.0%，实际为{percent_acc:.2f}%")
    
    # 检查Stage 4结果
    if Path('factor_impact_analysis.json').exists():
        with open('factor_impact_analysis.json', 'r', encoding='utf-8') as f:
            factors = json.load(f)
        
        age_judge = factors.get('age', {}).get('judge_correlation', 0)
        age_fan = factors.get('age', {}).get('fan_correlation', 0)
        
        print(f"\n✓ Stage 4 影响因素分析:")
        print(f"  年龄-评委评分相关性: {age_judge:.2f}")
        print(f"  年龄-粉丝投票相关性: {age_fan:.2f}")
        
        if abs(age_judge - (-0.24)) > 0.05:
            issues.append(f"年龄-评委评分相关性应该是-0.24，实际为{age_judge:.2f}")
        if abs(age_fan - (-0.26)) > 0.05:
            issues.append(f"年龄-粉丝投票相关性应该是-0.26，实际为{age_fan:.2f}")
    
    # 检查Stage 5结果
    if Path('stage5_model_analysis_lgb.txt').exists():
        with open('stage5_model_analysis_lgb.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取准确率
        acc_match = re.search(r'预测准确率:\s*(\d+\.\d+)%', content)
        if acc_match:
            acc = float(acc_match.group(1))
            print(f"\n✓ Stage 5 ML系统:")
            print(f"  预测准确率: {acc:.2f}%")
            
            if abs(acc - 97.99) > 0.5:
                issues.append(f"ML系统准确率应该是97.99%，实际为{acc:.2f}%")
    
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ 所有数据一致性检查通过！")
    
    return len(issues) == 0

def check_latex_files():
    """检查LaTeX文件"""
    print("\n" + "=" * 70)
    print("2. LaTeX文件检查")
    print("=" * 70)
    
    issues = []
    
    # 检查主文件
    if Path('main.tex').exists():
        with open('main.tex', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查团队编号
        if '[Your Team Number]' in content or '[2603215]' in content:
            print("  ⚠️  团队编号需要确认")
        
        # 检查所有章节是否被引用
        sections = [
            'summary_sheet', 'introduction', 'stage1_preprocessing',
            'stage2_fan_vote_estimation', 'stage3_voting_comparison',
            'stage4_factor_impact', 'stage5_new_system', 'conclusions',
            'memo_to_producers', 'references', 'ai_use_report'
        ]
        
        for section in sections:
            if f'\\input{{sections/{section}}}' not in content:
                issues.append(f"缺少章节引用: {section}")
        
        print(f"\n✓ 主文件检查完成")
        if issues:
            print(f"  ⚠️  发现 {len(issues)} 个问题")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ 所有章节引用正确")
    
    # 检查图表引用
    sections_dir = Path('sections')
    if sections_dir.exists():
        figure_refs = []
        for tex_file in sections_dir.glob('*.tex'):
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找图表引用
            refs = re.findall(r'\\ref\{fig:(\w+)\}', content)
            figure_refs.extend(refs)
            
            # 查找图表定义
            labels = re.findall(r'\\label\{fig:(\w+)\}', content)
            
            # 检查引用是否都有对应的label
            for ref in refs:
                if ref not in labels:
                    issues.append(f"{tex_file.name}: 引用 fig:{ref} 但没有对应的 \\label")
        
        print(f"\n✓ 图表引用检查完成")
        if issues:
            print(f"  ⚠️  发现 {len(issues)} 个问题")
        else:
            print("  ✓ 所有图表引用正确")
    
    return len(issues) == 0

def check_visualizations():
    """检查可视化文件"""
    print("\n" + "=" * 70)
    print("3. 可视化文件检查")
    print("=" * 70)
    
    required_figs = [
        'stage2_fan_vote_estimation.png',
        'stage3_voting_comparison.png',
        'stage4_factor_impact.png',
        'stage5_ml_system.png',
        'overall_summary.png'
    ]
    
    vis_dir = Path('visualizations')
    missing = []
    
    if vis_dir.exists():
        for fig in required_figs:
            if (vis_dir / fig).exists():
                print(f"  ✓ {fig}")
            else:
                missing.append(fig)
                print(f"  ✗ {fig} (缺失)")
    else:
        print("  ✗ visualizations 目录不存在")
        return False
    
    if missing:
        print(f"\n⚠️  缺失 {len(missing)} 个图表文件")
        return False
    else:
        print("\n✓ 所有必需图表文件存在")
        return True

def generate_quality_report():
    """生成质量检查报告"""
    print("\n" + "=" * 70)
    print("生成质量检查报告")
    print("=" * 70)
    
    results = {
        'data_consistency': check_data_consistency(),
        'latex_files': check_latex_files(),
        'visualizations': check_visualizations()
    }
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    print("质量检查总结")
    print("=" * 70)
    print(f"数据一致性: {'✓ 通过' if results['data_consistency'] else '✗ 失败'}")
    print(f"LaTeX文件: {'✓ 通过' if results['latex_files'] else '✗ 失败'}")
    print(f"可视化文件: {'✓ 通过' if results['visualizations'] else '✗ 失败'}")
    print(f"\n总体结果: {'✓ 所有检查通过' if all_passed else '⚠️  发现问题，请修复'}")
    
    # 保存报告
    report_path = Path('quality_check_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("质量检查报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"数据一致性: {'通过' if results['data_consistency'] else '失败'}\n")
        f.write(f"LaTeX文件: {'通过' if results['latex_files'] else '失败'}\n")
        f.write(f"可视化文件: {'通过' if results['visualizations'] else '失败'}\n")
        f.write(f"\n总体结果: {'所有检查通过' if all_passed else '发现问题，请修复'}\n")
    
    print(f"\n报告已保存到: {report_path}")
    
    return all_passed

if __name__ == '__main__':
    generate_quality_report()
