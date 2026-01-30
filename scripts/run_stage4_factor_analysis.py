"""
阶段4：影响因素分析 - 运行脚本
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

from factor_impact_analyzer import FactorImpactAnalyzer
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段4的所有任务"""
    
    print("=" * 70)
    print("阶段4：影响因素分析")
    print("=" * 70)
    print()
    
    try:
        # 1. 加载数据
        print("步骤1: 加载数据...")
        try:
            processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
            print(f"✓ 加载预处理数据成功")
            print(f"  数据形状: {processed_df.shape[0]} 行 × {processed_df.shape[1]} 列")
        except FileNotFoundError:
            print("❌ 错误: 未找到 2026_MCM_Problem_C_Data_processed.csv")
            print("   请先运行阶段1进行数据预处理")
            sys.exit(1)
        
        try:
            estimates_df = pd.read_csv('fan_vote_estimates.csv')
            print(f"✓ 加载粉丝投票估计数据成功")
            print(f"  数据形状: {estimates_df.shape[0]} 行 × {estimates_df.shape[1]} 列")
        except FileNotFoundError:
            print("❌ 错误: 未找到 fan_vote_estimates.csv")
            print("   请先运行阶段2生成粉丝投票估计数据")
            sys.exit(1)
        
        print()
        
        # 2. 创建分析器
        print("步骤2: 创建影响因素分析器...")
        analyzer = FactorImpactAnalyzer(processed_df, estimates_df)
        print("✓ 分析器创建成功")
        print()
        
        # 3. 执行综合分析
        print("步骤3: 执行综合分析...")
        print("  这包括：")
        print("    - 专业舞者影响分析")
        print("    - 选手特征影响分析（年龄、行业、地区）")
        print("    - 比较对评委评分和粉丝投票的不同影响")
        print()
        
        analysis = analyzer.generate_comprehensive_analysis()
        
        print()
        print("✓ 分析完成")
        
        # 4. 保存分析结果
        print("\n步骤4: 保存分析结果...")
        
        # 转换numpy类型为Python原生类型
        def convert_to_native(obj):
            """将numpy/pandas类型转换为Python原生类型"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, tuple):
                # 将tuple转换为字符串（JSON不支持tuple键）
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # 处理字典，确保键是字符串
                result = {}
                for key, value in obj.items():
                    # 如果键是tuple，转换为字符串
                    if isinstance(key, tuple):
                        new_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                    else:
                        new_key = str(key) if not isinstance(key, (str, int, float, bool)) else key
                    result[new_key] = convert_to_native(value)
                return result
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        analysis_native = convert_to_native(analysis)
        
        analysis_path = 'factor_impact_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 分析结果已保存到: {analysis_path}")
        
        # 保存专业舞者统计
        if 'pro_dancer_impact' in analysis and 'pro_dancer_stats' in analysis['pro_dancer_impact']:
            pro_dancer_df = pd.DataFrame(analysis['pro_dancer_impact']['pro_dancer_stats'])
            pro_dancer_path = 'pro_dancer_impact.csv'
            pro_dancer_df.to_csv(pro_dancer_path, index=False, encoding='utf-8-sig')
            print(f"✓ 专业舞者影响统计已保存到: {pro_dancer_path}")
        
        # 保存行业影响统计
        if 'celebrity_features_impact' in analysis and 'industry' in analysis['celebrity_features_impact']:
            if 'industry_performance' in analysis['celebrity_features_impact']['industry']:
                industry_df = pd.DataFrame(analysis['celebrity_features_impact']['industry']['industry_performance'])
                industry_path = 'industry_impact.csv'
                industry_df.to_csv(industry_path, index=False, encoding='utf-8-sig')
                print(f"✓ 行业影响统计已保存到: {industry_path}")
        
        # 保存地区影响统计
        if 'celebrity_features_impact' in analysis and 'region' in analysis['celebrity_features_impact']:
            if 'region_performance' in analysis['celebrity_features_impact']['region']:
                region_df = pd.DataFrame(analysis['celebrity_features_impact']['region']['region_performance'])
                region_path = 'region_impact.csv'
                region_df.to_csv(region_path, index=False, encoding='utf-8-sig')
                print(f"✓ 地区影响统计已保存到: {region_path}")
        
        print()
        
        # 5. 生成文本报告
        print("步骤5: 生成文本报告...")
        from factor_impact_analyzer import generate_text_report
        report_path = 'stage4_factor_impact_report.txt'
        generate_text_report(analysis, report_path)
        print(f"✓ 文本报告已保存到: {report_path}")
        
        # 6. 打印关键发现
        print("\n步骤6: 关键发现摘要...")
        print_key_findings(analysis)
        
        print("\n" + "=" * 70)
        print("阶段4完成！所有任务已成功执行。")
        print("=" * 70)
        print("\n生成的文件:")
        print(f"  - {analysis_path}")
        if 'pro_dancer_impact' in analysis:
            print(f"  - pro_dancer_impact.csv")
        if 'celebrity_features_impact' in analysis and 'industry' in analysis['celebrity_features_impact']:
            print(f"  - industry_impact.csv")
        if 'celebrity_features_impact' in analysis and 'region' in analysis['celebrity_features_impact']:
            print(f"  - region_impact.csv")
        print(f"  - {report_path}")
        
        return analyzer, analysis
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件")
        print(f"   请确保已完成阶段1和阶段2")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_key_findings(analysis: dict):
    """打印关键发现"""
    print("\n关键发现:")
    print("-" * 70)
    
    # 专业舞者影响
    if 'pro_dancer_impact' in analysis:
        pro_dancer = analysis['pro_dancer_impact']
        if 'summary' in pro_dancer and 'top_5_dancers_by_placement' in pro_dancer['summary']:
            print("\n1. 表现最好的专业舞者（前3名）:")
            for i, dancer in enumerate(pro_dancer['summary']['top_5_dancers_by_placement'][:3], 1):
                print(f"   {i}. {dancer.get('pro_dancer', 'N/A')}: "
                     f"平均排名 {dancer.get('avg_placement', 0):.2f}")
    
    # 年龄影响
    if 'celebrity_features_impact' in analysis and 'age' in analysis['celebrity_features_impact']:
        age = analysis['celebrity_features_impact']['age']
        if 'correlation_with_judge_score' in age and 'correlation_with_fan_votes' in age:
            judge_corr = age['correlation_with_judge_score'].get('correlation', 0)
            fan_corr = age['correlation_with_fan_votes'].get('correlation', 0)
            print(f"\n2. 年龄影响:")
            print(f"   与评委评分的相关性: {judge_corr:.4f}")
            print(f"   与粉丝投票的相关性: {fan_corr:.4f}")
    
    # 行业影响
    if 'celebrity_features_impact' in analysis and 'industry' in analysis['celebrity_features_impact']:
        industry = analysis['celebrity_features_impact']['industry']
        if 'top_industries_by_placement' in industry:
            print(f"\n3. 表现最好的行业（前3名）:")
            for i, ind in enumerate(industry['top_industries_by_placement'][:3], 1):
                print(f"   {i}. {ind.get('industry', 'N/A')}: "
                     f"平均排名 {ind.get('avg_placement', 0):.2f}")
    
    # 影响差异
    if 'judge_vs_fan_comparison' in analysis:
        comparison = analysis['judge_vs_fan_comparison']
        if 'age' in comparison:
            age_comp = comparison['age']
            if 'correlation_comparison' in age_comp:
                corr_comp = age_comp['correlation_comparison']
                diff = corr_comp.get('correlation_difference', 0)
                print(f"\n4. 年龄对两种投票的影响差异: {diff:.4f}")
                if abs(corr_comp.get('judge_score_correlation', 0)) > abs(corr_comp.get('fan_votes_correlation', 0)):
                    print("   结论: 年龄对评委评分的影响更大")
                else:
                    print("   结论: 年龄对粉丝投票的影响更大")


if __name__ == "__main__":
    analyzer, analysis = main()
