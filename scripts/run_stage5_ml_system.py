"""
阶段5：基于机器学习的投票系统 - 运行脚本
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

from ml_voting_system import MLVotingSystem
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
    print("阶段5：基于机器学习的投票系统")
    print("=" * 70)
    
    # 选择模型类型
    print("\n可用的模型类型:")
    print("  1. mlp - 多层感知机 (MLP)")
    print("  2. rf - 随机森林 (Random Forest)")
    print("  3. xgb - XGBoost")
    print("  4. lgb - LightGBM")
    print("  5. gbdt - Gradient Boosting")
    
    model_choice = input("\n请选择模型类型 (1-5，默认1): ").strip()
    model_map = {
        '1': 'mlp',
        '2': 'rf',
        '3': 'xgb',
        '4': 'lgb',
        '5': 'gbdt'
    }
    model_type = model_map.get(model_choice, 'mlp')
    
    print(f"\n使用模型: {model_type.upper()}")
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    try:
        raw_df = load_data()
        print(f"✓ 已加载原始数据: {len(raw_df)} 行")
        
        # 加载预处理后的数据
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print(f"✓ 已加载预处理数据: {len(processed_df)} 行")
        
        # 加载阶段2的估计结果
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print(f"✓ 已加载粉丝投票估计: {len(estimates_df)} 行")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: 初始化ML投票系统
    print("\n步骤2: 初始化ML投票系统...")
    try:
        ml_system = MLVotingSystem(
            estimates_df=estimates_df,
            processed_df=processed_df,
            factor_analysis_path='factor_impact_analysis.json',
            model_type=model_type
        )
        print(f"✓ ML投票系统初始化完成")
        print(f"  - 模型类型: {model_type.upper()}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 训练模型
    print("\n步骤3: 训练模型...")
    try:
        training_results = ml_system.train(test_size=0.2, random_state=42)
        print(f"\n训练结果:")
        print(f"  - 总样本数: {training_results['n_samples']}")
        print(f"  - 特征数: {training_results['n_features']}")
        print(f"  - 淘汰样本数: {training_results['n_eliminated']}")
        print(f"  - 训练集准确率: {training_results['train_accuracy']:.4f}")
        print(f"  - 测试集准确率: {training_results['test_accuracy']:.4f}")
        
        # 保存训练结果
        with open(f'ml_system_training_{model_type}.json', 'w', encoding='utf-8') as f:
            json.dump(convert_to_native(training_results), f, indent=2, ensure_ascii=False)
        print(f"✓ 已保存训练结果到: ml_system_training_{model_type}.json")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 应用ML系统到所有周次
    print("\n步骤4: 应用ML系统到所有周次...")
    try:
        ml_system_results = ml_system.apply_to_all_weeks()
        print(f"✓ 已处理 {len(ml_system_results)} 条记录")
        print(f"  - 涉及 {ml_system_results['season'].nunique()} 季")
        print(f"  - 涉及 {ml_system_results.groupby(['season', 'week']).ngroups} 个周次")
        
        # 保存结果
        output_file = f'ml_voting_system_results_{model_type}.csv'
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
        comparison = ml_system.compare_with_original_systems(ml_system_results)
        comparison_native = convert_to_native(comparison)
        
        print("\n比较结果:")
        print(f"  - 总周次数: {comparison['total_weeks']}")
        print(f"  - 原始系统准确率: {comparison['original_system_accuracy']:.2%}")
        print(f"  - ML系统准确率: {comparison['ml_system_accuracy']:.2%}")
        print(f"  - 准确率提升: {comparison['accuracy_improvement']:.2%}")
        print(f"  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})")
        
        # 保存比较结果
        comparison_file = f'ml_system_comparison_{model_type}.json'
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
阶段5：基于机器学习的投票系统 - 综合报告
======================================================================

一、系统概述
----------------------------------------------------------------------
ML投票系统模型: {model_type.upper()}

核心特点:
1. 使用机器学习模型学习如何组合评委评分和粉丝投票
2. 特征包括：评委评分、粉丝投票、年龄、专业舞者、行业、地区等
3. 自动学习最优的组合方式，无需手动设计规则

二、模型训练
----------------------------------------------------------------------
总样本数: {training_results['n_samples']}
特征数: {training_results['n_features']}
淘汰样本数: {training_results['n_eliminated']} ({training_results['n_eliminated']/training_results['n_samples']*100:.1f}%)

训练集准确率: {training_results['train_accuracy']:.2%}
测试集准确率: {training_results['test_accuracy']:.2%}

三、系统性能
----------------------------------------------------------------------
总周次数: {comparison['total_weeks']}

原始系统:
  - 准确率: {comparison['original_system_accuracy']:.2%}
  - 正确预测数: {comparison['original_correct_count']}

ML系统:
  - 准确率: {comparison['ml_system_accuracy']:.2%}
  - 正确预测数: {comparison['ml_correct_count']}
  - 准确率提升: {comparison['accuracy_improvement']:.2%}

系统差异:
  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})

四、系统优势
----------------------------------------------------------------------
1. 自动学习
   - 无需手动设计规则
   - 模型自动学习最优组合方式

2. 综合考虑
   - 同时考虑多个影响因素
   - 特征工程丰富（标准化、排名、百分比、相对值等）

3. 可扩展性
   - 可以轻松添加新特征
   - 可以尝试不同的模型架构

五、输出文件
----------------------------------------------------------------------
1. ml_voting_system_results_{model_type}.csv - ML系统的详细结果
2. ml_system_comparison_{model_type}.json - 与原始系统的比较结果
3. ml_system_training_{model_type}.json - 模型训练结果
4. stage5_ml_system_report_{model_type}.txt - 本报告

======================================================================
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_file = f'stage5_ml_system_report_{model_type}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 已保存综合报告到: {report_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("阶段5完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
