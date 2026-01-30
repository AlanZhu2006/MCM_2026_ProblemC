"""
阶段5：基于机器学习的投票系统 - 防过拟合版本
使用时间序列交叉验证、正则化、特征重要性分析
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_voting_system_robust import RobustMLVotingSystem
from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor


def convert_to_native(obj):
    """将numpy/pandas类型转换为Python原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
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


def main():
    """主函数"""
    print("=" * 70)
    print("阶段5：基于机器学习的投票系统 - 防过拟合版本")
    print("=" * 70)
    print("\n⚠️  本版本使用时间序列交叉验证和正则化，防止过拟合")
    
    # 选择模型类型
    print("\n可用的模型类型:")
    print("  1. mlp - 多层感知机 (MLP)")
    print("  2. rf - 随机森林 (Random Forest)")
    print("  3. xgb - XGBoost")
    print("  4. lgb - LightGBM")
    print("  5. gbdt - Gradient Boosting")
    print("  6. sgd - 随机梯度下降 (SGD)")
    
    model_choice = input("\n请选择模型类型 (1-6，默认3): ").strip()
    model_map = {
        '1': 'mlp',
        '2': 'rf',
        '3': 'xgb',
        '4': 'lgb',
        '5': 'gbdt',
        '6': 'sgd'
    }
    model_type = model_map.get(model_choice, 'xgb')
    
    print(f"\n使用模型: {model_type.upper()} (防过拟合版本)")
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    try:
        raw_df = load_data()
        print(f"✓ 已加载原始数据: {len(raw_df)} 行")
        
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print(f"✓ 已加载预处理数据: {len(processed_df)} 行")
        
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print(f"✓ 已加载粉丝投票估计: {len(estimates_df)} 行")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: 初始化防过拟合ML投票系统
    print("\n步骤2: 初始化防过拟合ML投票系统...")
    try:
        robust_system = RobustMLVotingSystem(
            estimates_df=estimates_df,
            processed_df=processed_df,
            factor_analysis_path='factor_impact_analysis.json',
            model_type=model_type,
            use_time_split=True
        )
        print(f"✓ 防过拟合ML投票系统初始化完成")
        print(f"  - 模型类型: {model_type.upper()}")
        print(f"  - 使用时间序列交叉验证: 是")
        print(f"  - 使用正则化: 是")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 使用时间序列交叉验证训练模型
    print("\n步骤3: 使用时间序列交叉验证训练模型...")
    try:
        training_results = robust_system.train_with_time_split(n_splits=5)
        
        print(f"\n训练结果:")
        print(f"  - 总样本数: {training_results['n_samples']}")
        print(f"  - 特征数: {training_results['n_features']}")
        print(f"  - 淘汰样本数: {training_results['n_eliminated']}")
        print(f"  - 平均训练准确率: {training_results['train_accuracy']:.4f}")
        print(f"  - 平均交叉验证准确率: {training_results['cv_accuracy']:.4f} (±{training_results['cv_std']:.4f})")
        
        # 分析过拟合风险
        gap = training_results['train_accuracy'] - training_results['cv_accuracy']
        print(f"\n过拟合分析:")
        print(f"  - 训练集-验证集差距: {gap:.4f}")
        if gap > 0.05:
            print(f"  ⚠️  警告: 差距较大，可能存在过拟合")
        elif gap > 0.02:
            print(f"  ⚠️  注意: 差距适中，需要关注")
        else:
            print(f"  ✓ 差距较小，过拟合风险较低")
        
        # 保存训练结果
        training_results_native = convert_to_native(training_results)
        with open(f'ml_robust_training_{model_type}.json', 'w', encoding='utf-8') as f:
            json.dump(training_results_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 已保存训练结果到: ml_robust_training_{model_type}.json")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 应用ML系统到所有周次
    print("\n步骤4: 应用ML系统到所有周次...")
    try:
        ml_system_results = robust_system.apply_to_all_weeks()
        print(f"✓ 已处理 {len(ml_system_results)} 条记录")
        print(f"  - 涉及 {ml_system_results['season'].nunique()} 季")
        print(f"  - 涉及 {ml_system_results.groupby(['season', 'week']).ngroups} 个周次")
        
        output_file = f'ml_robust_results_{model_type}.csv'
        ml_system_results.to_csv(output_file, index=False)
        print(f"✓ 已保存ML系统结果到: {output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 比较ML系统与原始系统
    print("\n步骤5: 比较ML系统与原始系统...")
    try:
        comparison = robust_system.compare_with_original_systems(ml_system_results)
        comparison_native = convert_to_native(comparison)
        
        print("\n比较结果:")
        print(f"  - 总周次数: {comparison['total_weeks']}")
        print(f"  - 原始系统准确率: {comparison['original_system_accuracy']:.2%}")
        print(f"  - ML系统准确率: {comparison['ml_system_accuracy']:.2%}")
        print(f"  - 准确率提升: {comparison['accuracy_improvement']:.2%}")
        print(f"  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})")
        
        # 分析准确率合理性
        print(f"\n准确率合理性分析:")
        cv_acc = training_results['cv_accuracy']
        pred_acc = comparison['ml_system_accuracy']
        diff = abs(cv_acc - pred_acc)
        
        print(f"  - 交叉验证准确率: {cv_acc:.4f}")
        print(f"  - 预测准确率: {pred_acc:.4f}")
        print(f"  - 差异: {diff:.4f}")
        
        if diff < 0.02:
            print(f"  ✓ 差异很小，模型泛化能力良好")
        elif diff < 0.05:
            print(f"  ⚠️  差异适中，需要关注")
        else:
            print(f"  ⚠️  差异较大，可能存在数据泄露或过拟合")
        
        comparison_file = f'ml_robust_comparison_{model_type}.json'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 已保存比较结果到: {comparison_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤6: 生成综合报告
    print("\n步骤6: 生成综合报告...")
    try:
        report = f"""
阶段5：基于机器学习的投票系统 - 防过拟合版本报告
======================================================================

一、系统概述
----------------------------------------------------------------------
ML投票系统模型: {model_type.upper()} (防过拟合版本)

核心改进:
1. 使用时间序列交叉验证（按季次分割）
2. 增加正则化（降低模型复杂度）
3. 特征重要性分析
4. 过拟合风险分析

二、模型训练（时间序列交叉验证）
----------------------------------------------------------------------
总样本数: {training_results['n_samples']}
特征数: {training_results['n_features']}
淘汰样本数: {training_results['n_eliminated']} ({training_results['n_eliminated']/training_results['n_samples']*100:.1f}%)

平均训练准确率: {training_results['train_accuracy']:.2%}
平均交叉验证准确率: {training_results['cv_accuracy']:.2%} (±{training_results['cv_std']:.2%})
训练-验证差距: {training_results['train_accuracy'] - training_results['cv_accuracy']:.2%}

过拟合风险: {'高' if (training_results['train_accuracy'] - training_results['cv_accuracy']) > 0.05 else '中' if (training_results['train_accuracy'] - training_results['cv_accuracy']) > 0.02 else '低'}

三、系统性能
----------------------------------------------------------------------
总周次数: {comparison['total_weeks']}

原始系统:
  - 准确率: {comparison['original_system_accuracy']:.2%}
  - 正确预测数: {comparison['original_correct_count']}

ML系统（防过拟合）:
  - 准确率: {comparison['ml_system_accuracy']:.2%}
  - 正确预测数: {comparison['ml_correct_count']}
  - 准确率提升: {comparison['accuracy_improvement']:.2%}

系统差异:
  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})

四、过拟合分析
----------------------------------------------------------------------
训练集-验证集差距: {training_results['train_accuracy'] - training_results['cv_accuracy']:.4f}
交叉验证-预测准确率差异: {abs(training_results['cv_accuracy'] - comparison['ml_system_accuracy']):.4f}

{'⚠️  警告: 存在过拟合风险' if (training_results['train_accuracy'] - training_results['cv_accuracy']) > 0.05 else '✓ 过拟合风险较低'}

五、特征重要性
----------------------------------------------------------------------
{chr(10).join([f"{feat}: {imp:.4f}" for feat, imp in sorted(training_results.get('feature_importance', {}).items(), key=lambda x: x[1], reverse=True)[:5]]) if training_results.get('feature_importance') else '未计算'}

六、输出文件
----------------------------------------------------------------------
1. ml_robust_results_{model_type}.csv - ML系统的详细结果
2. ml_robust_comparison_{model_type}.json - 与原始系统的比较结果
3. ml_robust_training_{model_type}.json - 模型训练结果（含交叉验证）
4. stage5_ml_robust_report_{model_type}.txt - 本报告

======================================================================
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_file = f'stage5_ml_robust_report_{model_type}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 已保存综合报告到: {report_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("防过拟合版本测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
