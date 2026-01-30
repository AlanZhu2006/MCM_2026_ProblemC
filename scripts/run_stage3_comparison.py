"""
阶段3：投票方法比较分析 - 运行脚本
"""

from __future__ import annotations
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

from voting_method_comparator import VotingMethodComparator
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段3的所有任务"""
    
    print("=" * 70)
    print("阶段3：投票方法比较分析")
    print("=" * 70)
    print()
    
    try:
        # 1. 加载数据
        print("步骤1: 加载数据...")
        try:
            estimates_df = pd.read_csv('fan_vote_estimates.csv')
            print(f"✓ 加载粉丝投票估计数据成功")
            print(f"  数据形状: {estimates_df.shape[0]} 行 × {estimates_df.shape[1]} 列")
        except FileNotFoundError:
            print("❌ 错误: 未找到 fan_vote_estimates.csv")
            print("   请先运行阶段2生成粉丝投票估计数据")
            sys.exit(1)
        
        try:
            processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
            print(f"✓ 加载预处理数据成功")
            print(f"  数据形状: {processed_df.shape[0]} 行 × {processed_df.shape[1]} 列")
        except FileNotFoundError:
            print("❌ 错误: 未找到 2026_MCM_Problem_C_Data_processed.csv")
            print("   请先运行阶段1进行数据预处理")
            sys.exit(1)
        
        print()
        
        # 2. 创建比较器
        print("步骤2: 创建投票方法比较器...")
        comparator = VotingMethodComparator(estimates_df, processed_df)
        print("✓ 比较器创建成功")
        print()
        
        # 3. 执行比较
        print("步骤3: 比较所有周次的两种方法...")
        print("  这可能需要一些时间，请耐心等待...")
        print()
        
        # 可以选择只处理前几季进行测试
        # comparison_df = comparator.compare_all_weeks(seasons=[1, 2, 3])
        comparison_df = comparator.compare_all_weeks()
        
        print()
        print(f"✓ 比较完成，共处理 {len(comparison_df)} 周")
        
        # 保存比较结果
        output_path = 'voting_method_comparison.csv'
        comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 比较结果已保存到: {output_path}")
        print()
        
        # 4. 分析差异
        print("步骤4: 分析差异...")
        analysis = comparator.analyze_differences(comparison_df)
        
        print(f"\n差异分析结果:")
        print(f"  总周数: {analysis['total_weeks']}")
        print(f"  两种方法一致: {analysis['methods_agree_count']} 周 "
              f"({analysis['methods_agree_count']/analysis['total_weeks']*100:.1f}%)")
        print(f"  两种方法不一致: {analysis['methods_disagree_count']} 周 "
              f"({analysis['methods_disagree_count']/analysis['total_weeks']*100:.1f}%)")
        print(f"  排名法准确率: {analysis['rank_method']['accuracy']:.2%}")
        print(f"  百分比法准确率: {analysis['percent_method']['accuracy']:.2%}")
        
        # 保存差异分析（需要转换numpy/pandas类型为Python原生类型）
        def convert_to_native(obj):
            """将numpy/pandas类型转换为Python原生类型"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        analysis_path = 'method_differences_analysis.json'
        analysis_native = convert_to_native(analysis)
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 差异分析已保存到: {analysis_path}")
        print()
        
        # 5. 分析争议案例
        print("步骤5: 分析争议案例...")
        controversial_df = comparator.analyze_controversial_cases(comparison_df)
        
        if len(controversial_df) > 0:
            controversial_path = 'controversial_cases_analysis.csv'
            controversial_df.to_csv(controversial_path, index=False, encoding='utf-8-sig')
            print(f"✓ 争议案例分析已保存到: {controversial_path}")
            print(f"  找到 {len(controversial_df)} 条争议案例记录")
        else:
            print("⚠️  未找到争议案例数据")
        print()
        
        # 6. 生成报告
        print("步骤6: 生成摘要报告...")
        report_path = 'stage3_comparison_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("阶段3：投票方法比较分析 - 摘要报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. 总体统计\n")
            f.write(f"   总周数: {analysis['total_weeks']}\n")
            f.write(f"   两种方法一致: {analysis['methods_agree_count']} 周\n")
            f.write(f"   两种方法不一致: {analysis['methods_disagree_count']} 周\n")
            f.write(f"   不一致率: {analysis['disagree_rate']:.2%}\n\n")
            
            f.write("2. 方法准确率\n")
            f.write(f"   排名法准确率: {analysis['rank_method']['accuracy']:.2%}\n")
            f.write(f"     正确预测: {analysis['rank_method']['correct_predictions']}/{analysis['rank_method']['total_predictions']}\n")
            f.write(f"   百分比法准确率: {analysis['percent_method']['accuracy']:.2%}\n")
            f.write(f"     正确预测: {analysis['percent_method']['correct_predictions']}/{analysis['percent_method']['total_predictions']}\n\n")
            
            f.write("3. 按季统计差异\n")
            for stat in analysis['season_stats']:
                f.write(f"   第{stat['season']}季: {stat['disagree_count']}/{stat['total_weeks']} 周不一致 "
                       f"({stat['disagree_rate']:.2%})\n")
            
            f.write("\n4. 按选手数量统计差异\n")
            for stat in analysis['n_contestants_stats']:
                f.write(f"   {stat['n_contestants']}位选手: {stat['disagree_count']}/{stat['total_weeks']} 周不一致 "
                       f"({stat['disagree_rate']:.2%})\n")
            
            if len(controversial_df) > 0:
                f.write("\n5. 争议案例摘要\n")
                for _, row in controversial_df.iterrows():
                    f.write(f"   第{row['season']}季 {row['celebrity']} (第{row['week']}周):\n")
                    f.write(f"     评委排名: {row['judge_rank']}, 粉丝排名: {row['fan_rank']}\n")
                    f.write(f"     排名法会淘汰: {'是' if row['rank_method_would_eliminate'] else '否'}\n")
                    f.write(f"     百分比法会淘汰: {'是' if row['percent_method_would_eliminate'] else '否'}\n")
                    f.write(f"     实际结果: {row['actual_result']}\n\n")
        
        print(f"✓ 摘要报告已保存到: {report_path}")
        
        print("\n" + "=" * 70)
        print("阶段3完成！所有任务已成功执行。")
        print("=" * 70)
        print("\n生成的文件:")
        print(f"  - {output_path}")
        print(f"  - {analysis_path}")
        if len(controversial_df) > 0:
            print(f"  - {controversial_path}")
        print(f"  - {report_path}")
        
        return comparator, comparison_df, analysis, controversial_df
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件")
        print(f"   请确保已完成阶段1和阶段2")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    comparator, comparison_df, analysis, controversial_df = main()
