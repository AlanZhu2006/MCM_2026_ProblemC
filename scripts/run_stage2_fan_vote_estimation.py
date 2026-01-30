"""
阶段2：粉丝投票估计模型 - 运行脚本
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

from fan_vote_estimator import FanVoteEstimator
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段2的所有任务"""
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型")
    print("=" * 70)
    print()
    
    try:
        # 1. 加载处理后的数据
        print("步骤1: 加载数据...")
        try:
            df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
            print(f"✓ 从处理后的数据文件加载成功")
            print(f"  数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        except FileNotFoundError:
            print("⚠️  未找到处理后的数据文件")
            print("  正在运行阶段1预处理...")
            from preprocess_dwts import DWTSDataPreprocessor
            from loader import load_data
            
            raw_df = load_data()
            preprocessor = DWTSDataPreprocessor(raw_df)
            preprocessor.check_data_integrity()
            preprocessor.handle_missing_values()
            preprocessor.handle_eliminated_contestants()
            df = preprocessor.calculate_weekly_scores_and_ranks()
            preprocessor.identify_season_info()
            
            # 保存处理后的数据
            df.to_csv('2026_MCM_Problem_C_Data_processed.csv', index=False, encoding='utf-8-sig')
            print("✓ 阶段1预处理完成，数据已保存")
        
        print()
        
        # 2. 创建粉丝投票估计器
        print("步骤2: 创建粉丝投票估计器...")
        estimator = FanVoteEstimator(df)
        print("✓ 估计器创建成功")
        print()
        
        # 3. 估计粉丝投票
        print("步骤3: 估计所有周次的粉丝投票...")
        print("  这可能需要一些时间，请耐心等待...")
        print()
        
        # 可以选择只处理前几季进行测试
        # estimates_df = estimator.estimate_all_weeks(seasons=[1, 2, 3])
        estimates_df = estimator.estimate_all_weeks()
        
        print()
        print(f"✓ 估计完成，共处理 {len(estimates_df)} 条记录")
        
        # 保存估计结果
        output_path = 'fan_vote_estimates.csv'
        estimates_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 估计结果已保存到: {output_path}")
        print()
        
        # 4. 验证模型
        print("步骤4: 验证估计值是否与已知淘汰结果一致...")
        validation_results = estimator.validate_estimates(estimates_df)
        
        # 保存验证结果（需要转换numpy/pandas类型为Python原生类型）
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
        
        validation_path = 'validation_results.json'
        validation_results_native = convert_to_native(validation_results)
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 验证结果已保存到: {validation_path}")
        print()
        
        # 5. 不确定性量化
        print("步骤5: 使用蒙特卡洛方法量化不确定性...")
        print("  进行500次模拟（这可能需要一些时间）...")
        uncertainty_df = estimator.quantify_uncertainty_monte_carlo(
            estimates_df, 
            n_simulations=500
        )
        
        # 保存不确定性分析结果
        uncertainty_path = 'fan_vote_uncertainty.csv'
        uncertainty_df.to_csv(uncertainty_path, index=False, encoding='utf-8-sig')
        print(f"✓ 不确定性分析结果已保存到: {uncertainty_path}")
        print()
        
        # 6. 生成摘要报告
        print("步骤6: 生成摘要报告...")
        report_path = 'stage2_fan_vote_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("阶段2：粉丝投票估计模型 - 摘要报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. 估计结果统计\n")
            f.write(f"   总记录数: {len(estimates_df)}\n")
            f.write(f"   涉及季数: {estimates_df['season'].nunique()}\n")
            f.write(f"   涉及周数: {estimates_df.groupby('season')['week'].nunique().sum()}\n")
            f.write(f"   平均粉丝投票数: {estimates_df['fan_votes'].mean():.2f}\n")
            f.write(f"   粉丝投票数范围: {estimates_df['fan_votes'].min():.2f} - {estimates_df['fan_votes'].max():.2f}\n\n")
            
            f.write("2. 模型验证结果\n")
            f.write(f"   总周数: {validation_results['total_weeks']}\n")
            f.write(f"   正确预测: {validation_results['correct_predictions']}\n")
            f.write(f"   错误预测: {validation_results['incorrect_predictions']}\n")
            f.write(f"   准确率: {validation_results['accuracy']:.2%}\n\n")
            
            f.write("3. 不确定性分析\n")
            f.write(f"   平均标准差: {uncertainty_df['fan_votes_std'].mean():.2f}\n")
            f.write(f"   平均95%置信区间宽度: {(uncertainty_df['fan_votes_ci_upper'] - uncertainty_df['fan_votes_ci_lower']).mean():.2f}\n\n")
            
            f.write("4. 按季统计\n")
            for season in sorted(estimates_df['season'].unique()):
                season_data = estimates_df[estimates_df['season'] == season]
                f.write(f"   第{season}季: {len(season_data)}条记录, "
                       f"平均投票数: {season_data['fan_votes'].mean():.2f}\n")
        
        print(f"✓ 摘要报告已保存到: {report_path}")
        
        print("\n" + "=" * 70)
        print("阶段2完成！所有任务已成功执行。")
        print("=" * 70)
        print("\n生成的文件:")
        print(f"  - {output_path}")
        print(f"  - {uncertainty_path}")
        print(f"  - {validation_path}")
        print(f"  - {report_path}")
        
        return estimator, estimates_df, uncertainty_df, validation_results
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件")
        print(f"   请确保已完成阶段1预处理")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    estimator, estimates_df, uncertainty_df, validation_results = main()
