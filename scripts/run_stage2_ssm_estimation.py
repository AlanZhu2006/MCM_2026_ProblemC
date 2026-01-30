"""
阶段2：粉丝投票估计模型 - 状态空间模型版本运行脚本
借鉴2024年MCM C题（网球动量）的方法
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

from fan_vote_estimator_ssm import StateSpaceFanVoteEstimator
import pandas as pd
import numpy as np
import json


def main():
    """运行阶段2的状态空间模型版本"""
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 状态空间模型版本")
    print("（借鉴2024年MCM C题：状态空间模型 + 卡尔曼滤波）")
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
        
        # 2. 创建状态空间估计器
        print("步骤2: 创建状态空间估计器...")
        print("  核心思想：")
        print("  - 粉丝投票是隐变量（不可直接观测）")
        print("  - 通过评委评分、历史表现等观测变量推断")
        print("  - 使用状态空间模型建模粉丝投票的动态演化")
        print("  - 使用卡尔曼滤波实时估计粉丝投票的概率分布")
        estimator = StateSpaceFanVoteEstimator(df)
        print("✓ 估计器创建成功")
        print()
        
        # 3. 估计所有周次的粉丝投票
        print("步骤3: 估计粉丝投票（使用状态空间模型 + 卡尔曼滤波）...")
        estimates_df = estimator.estimate_all_weeks_ssm()
        
        print()
        print(f"✓ 估计完成")
        print(f"  估计结果数量: {len(estimates_df)} 条")
        print()
        
        # 4. 保存估计结果
        print("步骤4: 保存估计结果...")
        estimates_df.to_csv('fan_vote_estimates_ssm.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 估计结果已保存到: fan_vote_estimates_ssm.csv")
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
        with open('validation_results_ssm.json', 'w', encoding='utf-8') as f:
            json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 验证结果已保存到: validation_results_ssm.json")
        print()
        
        # 6. 不确定性量化（状态空间模型自带不确定性估计）
        print("步骤6: 不确定性分析...")
        if 'uncertainty' in estimates_df.columns:
            print("  状态空间模型提供了每个估计的不确定性（协方差）")
            print(f"  平均不确定性: {estimates_df['uncertainty'].mean():.4f}")
            print(f"  不确定性范围: [{estimates_df['uncertainty'].min():.4f}, {estimates_df['uncertainty'].max():.4f}]")
        else:
            print("  使用蒙特卡洛模拟进行不确定性量化...")
            uncertainty_df = estimator.quantify_uncertainty_monte_carlo(
                estimates_df,
                n_simulations=500
            )
            uncertainty_df.to_csv('fan_vote_uncertainty_ssm.csv', index=False, encoding='utf-8-sig')
            print(f"✓ 不确定性分析结果已保存到: fan_vote_uncertainty_ssm.csv")
        print()
        
        print("=" * 70)
        print("阶段2完成（状态空间模型版本）！")
        print("=" * 70)
        print()
        print("输出文件:")
        print("  - fan_vote_estimates_ssm.csv: 粉丝投票估计结果（含不确定性）")
        print("  - validation_results_ssm.json: 模型验证结果")
        print()
        
        return estimator, estimates_df, validation_results
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
