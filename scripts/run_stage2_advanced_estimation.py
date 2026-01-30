"""
阶段2：粉丝投票估计模型 - 高级版本（XGBoost/LightGBM + 高级特征）运行脚本
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

from fan_vote_estimator_advanced import AdvancedFanVoteEstimator
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段2的高级版本"""
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 高级版本")
    print("（XGBoost/LightGBM + 高级特征工程）")
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
            preprocessor.generate_summary_report()
            df = preprocessor.get_processed_data()
            print("✓ 阶段1预处理完成")
        
        print()
        
        # 2. 创建高级估计器
        print("步骤2: 创建高级估计器...")
        estimator = AdvancedFanVoteEstimator(df)
        print("✓ 估计器创建成功")
        print()
        
        # 3. 估计所有周次的粉丝投票
        print("步骤3: 估计粉丝投票（使用高级方法）...")
        print("  这将包括：")
        print("  - 训练XGBoost/LightGBM模型（如果可用）")
        print("  - 使用高级特征工程（时间序列、交互特征等）")
        print("  - 集成多个模型")
        print()
        
        estimates_df = estimator.estimate_all_weeks_advanced(train_on_all=True)
        
        print()
        print(f"✓ 估计完成")
        print(f"  估计结果数量: {len(estimates_df)} 条")
        print()
        
        # 4. 保存估计结果
        print("步骤4: 保存估计结果...")
        estimates_df.to_csv('fan_vote_estimates_advanced.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 估计结果已保存到: fan_vote_estimates_advanced.csv")
        print()
        
        # 5. 验证模型
        print("步骤5: 验证模型...")
        validation_results = estimator.validate_estimates(estimates_df)
        
        # 显示验证结果
        print("\n验证结果:")
        print(f"  总周数: {validation_results['total_weeks']}")
        print(f"  正确预测: {validation_results['correct_predictions']}")
        print(f"  错误预测: {validation_results['incorrect_predictions']}")
        print(f"  准确率: {validation_results['accuracy']:.2%}")
        
        # 按季显示
        if 'by_season' in validation_results:
            print("\n  按季统计:")
            for season, stats in sorted(validation_results['by_season'].items()):
                acc = stats['accuracy']
                print(f"    第{season}季: {stats['correct']}/{stats['total']} = {acc:.2%}")
        
        print()
        
        # 保存验证结果
        def convert_to_native(obj):
            """将NumPy/Pandas类型转换为Python原生类型"""
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
        
        validation_results_native = convert_to_native(validation_results)
        with open('validation_results_advanced.json', 'w', encoding='utf-8') as f:
            json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 验证结果已保存到: validation_results_advanced.json")
        print()
        
        # 6. 不确定性量化
        print("步骤6: 不确定性量化（蒙特卡洛模拟）...")
        uncertainty_df = estimator.quantify_uncertainty_monte_carlo(
            estimates_df,
            n_simulations=500
        )
        uncertainty_df.to_csv('fan_vote_uncertainty_advanced.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 不确定性分析结果已保存到: fan_vote_uncertainty_advanced.csv")
        print()
        
        # 7. 显示特征重要性
        if estimator.feature_importance:
            print("步骤7: 特征重要性分析...")
            print("\nTop 15 重要特征:")
            sorted_features = sorted(
                estimator.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i:2d}. {feature:35s}: {importance:.4f}")
            print()
        
        # 8. 模型性能对比
        if hasattr(estimator, 'models') and len(estimator.models) > 0:
            print("步骤8: 模型性能对比...")
            print("\n训练的模型:")
            for name in estimator.models.keys():
                if name != 'ensemble':
                    print(f"  - {name}")
            if 'ensemble' in estimator.models:
                print(f"  - ensemble (集成模型)")
            if estimator.best_model_name:
                print(f"\n最佳模型: {estimator.best_model_name}")
            print()
        
        print("=" * 70)
        print("阶段2完成（高级版本）！")
        print("=" * 70)
        print()
        print("输出文件:")
        print("  - fan_vote_estimates_advanced.csv: 粉丝投票估计结果")
        print("  - validation_results_advanced.json: 模型验证结果")
        print("  - fan_vote_uncertainty_advanced.csv: 不确定性分析结果")
        print()
        
        return estimator, estimates_df, validation_results
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
