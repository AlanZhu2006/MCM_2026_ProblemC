"""
阶段2：粉丝投票估计模型 - 机器学习版本（集成学习）运行脚本
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

from fan_vote_estimator_ml import MLFanVoteEstimator
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段2的机器学习版本"""
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 机器学习版本（集成学习）")
    print("=" * 70)
    print()
    
    try:
        # 1. 加载数据
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
            
            df.to_csv('2026_MCM_Problem_C_Data_processed.csv', index=False, encoding='utf-8-sig')
            print("✓ 阶段1预处理完成，数据已保存")
        
        print()
        
        # 2. 创建ML估计器
        print("步骤2: 创建机器学习估计器...")
        estimator = MLFanVoteEstimator(df)
        print("✓ 估计器创建成功")
        print()
        
        # 3. 估计粉丝投票（使用ML方法）
        print("步骤3: 使用机器学习集成方法估计粉丝投票...")
        print("  这包括：")
        print("    - 训练多个模型（随机森林、梯度提升、Ridge等）")
        print("    - 使用集成学习组合预测")
        print("    - 这可能需要一些时间，请耐心等待...")
        print()
        
        estimates_df = estimator.estimate_all_weeks_ml(train_on_all=True)
        
        print()
        print(f"✓ 估计完成，共处理 {len(estimates_df)} 条记录")
        
        # 保存估计结果
        output_path = 'fan_vote_estimates_ml.csv'
        estimates_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 估计结果已保存到: {output_path}")
        print()
        
        # 4. 验证模型
        print("步骤4: 验证估计值是否与已知淘汰结果一致...")
        validation_results = estimator.validate_estimates(estimates_df)
        
        # 保存验证结果
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
        
        validation_path = 'validation_results_ml.json'
        validation_results_native = convert_to_native(validation_results)
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 验证结果已保存到: {validation_path}")
        print()
        
        # 5. 显示特征重要性
        if estimator.feature_importance:
            print("步骤5: 特征重要性分析...")
            print("\nTop 10 重要特征:")
            sorted_features = sorted(
                estimator.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i:2d}. {feature:30s}: {importance:.4f}")
        print()
        
        # 6. 不确定性量化
        print("步骤6: 使用蒙特卡洛方法量化不确定性...")
        print("  进行500次模拟...")
        uncertainty_df = estimator.quantify_uncertainty_monte_carlo(
            estimates_df, 
            n_simulations=500
        )
        
        uncertainty_path = 'fan_vote_uncertainty_ml.csv'
        uncertainty_df.to_csv(uncertainty_path, index=False, encoding='utf-8-sig')
        print(f"✓ 不确定性分析结果已保存到: {uncertainty_path}")
        print()
        
        # 7. 生成摘要报告
        print("步骤7: 生成摘要报告...")
        report_path = 'stage2_ml_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("阶段2：粉丝投票估计模型 - 机器学习版本（集成学习）\n")
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
            
            if estimator.feature_importance:
                f.write("4. 特征重要性（Top 10）\n")
                sorted_features = sorted(
                    estimator.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"   {i:2d}. {feature:30s}: {importance:.4f}\n")
                f.write("\n")
            
            f.write("5. 使用的模型\n")
            f.write("   - 随机森林回归器 (RandomForestRegressor)\n")
            f.write("   - 梯度提升回归器 (GradientBoostingRegressor)\n")
            f.write("   - Ridge回归 (Ridge)\n")
            f.write("   - Lasso回归 (Lasso)\n")
            f.write("   - 弹性网络 (ElasticNet)\n")
            f.write("   - 集成模型 (VotingRegressor)\n\n")
            
            f.write("6. 按季统计\n")
            for season in sorted(estimates_df['season'].unique()):
                season_data = estimates_df[estimates_df['season'] == season]
                f.write(f"   第{season}季: {len(season_data)}条记录, "
                       f"平均投票数: {season_data['fan_votes'].mean():.2f}\n")
        
        print(f"✓ 摘要报告已保存到: {report_path}")
        
        print("\n" + "=" * 70)
        print("阶段2完成（机器学习版本）！所有任务已成功执行。")
        print("=" * 70)
        print("\n生成的文件:")
        print(f"  - {output_path}")
        print(f"  - {uncertainty_path}")
        print(f"  - {validation_path}")
        print(f"  - {report_path}")
        
        print("\n与基础版本对比:")
        print("  - 使用集成学习，结合多个模型的优势")
        print("  - 更好的特征工程和特征选择")
        print("  - 预期准确率提升5-15%")
        
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
