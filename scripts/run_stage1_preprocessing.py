"""
阶段1：数据探索与预处理 - 运行脚本
"""

from __future__ import annotations
import sys
import os

# 添加项目根目录到路径（确保优先使用项目本地模块）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录插入到sys.path的最前面，确保优先导入本地模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # 如果已经在路径中，移除后重新插入到最前面
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

# 现在导入项目模块（应该优先使用项目本地的loader.py）
from preprocess_dwts import DWTSDataPreprocessor
from loader import load_data
import pandas as pd


def main():
    """运行阶段1的所有预处理任务"""
    
    print("=" * 70)
    print("阶段1：数据探索与预处理")
    print("=" * 70)
    print()
    
    try:
        # 1. 加载数据
        print("步骤1: 加载数据...")
        df = load_data()
        print(f"✓ 数据加载成功")
        print(f"  数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print()
        
        # 2. 创建预处理器并执行所有任务
        print("步骤2: 创建预处理器...")
        preprocessor = DWTSDataPreprocessor(df)
        print("✓ 预处理器创建成功")
        print()
        
        # 执行所有预处理任务
        print("开始执行预处理任务...")
        print()
        
        # 任务1: 检查数据完整性
        integrity_report = preprocessor.check_data_integrity()
        
        # 任务2: 处理缺失值
        missing_report = preprocessor.handle_missing_values()
        
        # 任务3: 处理被淘汰选手的0分
        eliminated_report = preprocessor.handle_eliminated_contestants()
        
        # 任务4: 计算每周的评委总分和排名
        processed_df = preprocessor.calculate_weekly_scores_and_ranks()
        
        # 任务5: 识别每季的周数和选手数量
        season_info = preprocessor.identify_season_info()
        
        # 生成完整报告
        summary_report = preprocessor.generate_summary_report()
        
        # 保存处理后的数据
        print("\n" + "=" * 70)
        print("保存处理后的数据...")
        output_path = '2026_MCM_Problem_C_Data_processed.csv'
        processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 处理后的数据已保存到: {output_path}")
        
        # 保存摘要报告（可选，保存为文本文件）
        report_path = 'stage1_preprocessing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("阶段1：数据探索与预处理报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. 数据完整性检查\n")
            f.write(f"   总行数: {integrity_report['total_rows']}\n")
            f.write(f"   总列数: {integrity_report['total_columns']}\n")
            f.write(f"   季数范围: {integrity_report.get('season_range', 'N/A')}\n")
            f.write(f"   总季数: {integrity_report.get('unique_seasons', 'N/A')}\n\n")
            
            f.write("2. 缺失值处理\n")
            f.write(f"   总缺失值数: {missing_report['total_missing']}\n")
            f.write(f"   评分列数量: {missing_report['score_columns_count']}\n\n")
            
            f.write("3. 被淘汰选手处理\n")
            f.write(f"   总0分数量: {eliminated_report['total_zero_scores']}\n")
            f.write(f"   有0分的选手数: {len(eliminated_report['contestants_with_zeros'])}\n\n")
            
            f.write("4. 每周评分计算\n")
            f.write(f"   已计算周数: {summary_report['weekly_scores'].get('weeks_calculated', 0)}\n\n")
            
            f.write("5. 各季信息\n")
            for season in sorted(season_info.keys()):
                info = season_info[season]
                f.write(f"   第{season}季: {info['contestant_count']}位选手, {info['week_count']}周\n")
        
        print(f"✓ 摘要报告已保存到: {report_path}")
        
        print("\n" + "=" * 70)
        print("阶段1完成！所有预处理任务已成功执行。")
        print("=" * 70)
        
        return preprocessor, processed_df, summary_report
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件")
        print(f"   请确保 2026_MCM_Problem_C_Data.csv 文件在正确的位置")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    preprocessor, processed_df, report = main()
