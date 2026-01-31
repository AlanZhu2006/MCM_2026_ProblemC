"""
阶段5：ML投票系统模型深度分析
分析模型的内部机制：特征重要性、权重、决策规则等
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_voting_system_robust import RobustMLVotingSystem
from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor


def analyze_model_weights(model, feature_names, model_type):
    """分析模型权重（适用于线性模型）"""
    weights_info = {}
    
    if model_type == 'sgd':
        # SGD模型：提取系数
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            weights_info = {
                'type': 'linear_coefficients',
                'weights': dict(zip(feature_names, coef.tolist())),
                'intercept': float(model.intercept_[0]) if hasattr(model, 'intercept_') else None
            }
    elif model_type == 'mlp':
        # MLP模型：提取第一层权重
        if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
            first_layer_weights = model.coefs_[0]  # [n_features, n_hidden]
            # 计算每个特征的平均权重（绝对值）
            avg_weights = np.abs(first_layer_weights).mean(axis=1)
            weights_info = {
                'type': 'mlp_first_layer_weights',
                'weights': dict(zip(feature_names, avg_weights.tolist())),
                'layer_structure': [w.shape for w in model.coefs_]
            }
    
    return weights_info


def analyze_feature_importance(model, feature_names, model_type):
    """分析特征重要性"""
    importance = {}
    
    if hasattr(model, 'feature_importances_'):
        # 树模型：特征重要性
        importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    elif model_type == 'mlp':
        # MLP：使用权重分析
        if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
            first_layer_weights = model.coefs_[0]
            # 计算每个特征的平均权重（绝对值）
            avg_weights = np.abs(first_layer_weights).mean(axis=1)
            total = avg_weights.sum()
            if total > 0:
                importance = dict(zip(feature_names, (avg_weights / total).tolist()))
    elif model_type == 'sgd':
        # SGD：使用系数绝对值
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            abs_coef = np.abs(coef)
            total = abs_coef.sum()
            if total > 0:
                importance = dict(zip(feature_names, (abs_coef / total).tolist()))
    
    return importance


def analyze_decision_rules(model, feature_names, model_type, sample_data=None):
    """分析决策规则（适用于树模型）"""
    rules_info = {}
    
    if model_type in ['rf', 'xgb', 'lgb', 'gbdt']:
        # 树模型：提取重要特征
        if hasattr(model, 'feature_importances_'):
            top_features = sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            rules_info = {
                'type': 'tree_importance',
                'top_features': [{'feature': f, 'importance': float(imp)} for f, imp in top_features]
            }
    
    return rules_info


def generate_detailed_analysis_report(
    ml_system: RobustMLVotingSystem,
    training_results: dict,
    comparison: dict,
    model_type: str
):
    """生成详细的算法分析报告"""
    
    model = ml_system.model
    feature_names = ml_system.feature_columns
    
    # 1. 特征重要性分析
    feature_importance = analyze_feature_importance(model, feature_names, model_type)
    
    # 2. 模型权重分析
    weights_info = analyze_model_weights(model, feature_names, model_type)
    
    # 3. 决策规则分析
    rules_info = analyze_decision_rules(model, feature_names, model_type)
    
    # 4. 特征贡献度分析
    feature_contribution = {}
    if feature_importance:
        # 归一化特征重要性
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_contribution = {
                k: v / total_importance * 100 
                for k, v in feature_importance.items()
            }
    
    # 5. 特征分类统计
    feature_categories = {
        '基础特征（评委+粉丝）': ['judge_score_normalized', 'fan_votes_normalized'],
        '排名特征（评委+粉丝）': ['judge_rank_normalized', 'fan_rank_normalized'],
        '百分比特征（评委+粉丝）': ['judge_percent', 'fan_percent'],
        '相对特征（评委+粉丝）': ['judge_relative', 'fan_relative'],
        '年龄特征': ['age_normalized'],
        '专业舞者特征': ['pro_dancer_encoded'],
        '行业特征': ['industry_encoded'],
        '地区特征': ['region_encoded']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_importance[category] = sum(
            feature_importance.get(f, 0) for f in features
        )
    
    # 计算粉丝投票相关特征的总重要性
    fan_related_features = ['fan_votes_normalized', 'fan_rank_normalized', 'fan_percent', 'fan_relative']
    judge_related_features = ['judge_score_normalized', 'judge_rank_normalized', 'judge_percent', 'judge_relative']
    
    fan_total_importance = sum(feature_importance.get(f, 0) for f in fan_related_features)
    judge_total_importance = sum(feature_importance.get(f, 0) for f in judge_related_features)
    
    # 生成报告
    report = f"""
阶段5：ML投票系统算法深度分析报告
======================================================================

一、模型概述
----------------------------------------------------------------------
模型类型: {model_type.upper()}
总样本数: {training_results['n_samples']}
特征数: {training_results['n_features']}
淘汰样本数: {training_results['n_eliminated']} ({training_results['n_eliminated']/training_results['n_samples']*100:.1f}%)

训练准确率: {training_results['train_accuracy']:.2%}
交叉验证准确率: {training_results['cv_accuracy']:.2%} (±{training_results['cv_std']:.2%})
预测准确率: {comparison['ml_system_accuracy']:.2%}

二、特征重要性分析
----------------------------------------------------------------------

2.1 所有特征重要性（按重要性排序）
"""
    
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features, 1):
            contribution = feature_contribution.get(feat, 0)
            report += f"{i:2d}. {feat:30s}: {imp:.6f} ({contribution:.2f}%)\n"
    
    report += f"""
2.2 特征类别重要性
"""
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    for category, imp in sorted_categories:
        report += f"  {category:20s}: {imp:.6f} ({imp/sum(category_importance.values())*100:.2f}%)\n"
    
    report += f"""
三、模型权重分析
----------------------------------------------------------------------
"""
    
    if weights_info:
        if weights_info['type'] == 'linear_coefficients':
            report += f"模型类型: 线性模型（SGD）\n"
            report += f"截距项: {weights_info.get('intercept', 'N/A')}\n\n"
            report += f"特征权重（系数）:\n"
            sorted_weights = sorted(weights_info['weights'].items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, weight in sorted_weights[:10]:
                report += f"  {feat:30s}: {weight:+.6f}\n"
        elif weights_info['type'] == 'mlp_first_layer_weights':
            report += f"模型类型: 多层感知机（MLP）\n"
            report += f"网络结构: {weights_info.get('layer_structure', 'N/A')}\n\n"
            report += f"第一层平均权重（绝对值）:\n"
            sorted_weights = sorted(weights_info['weights'].items(), key=lambda x: x[1], reverse=True)
            for feat, weight in sorted_weights[:10]:
                report += f"  {feat:30s}: {weight:.6f}\n"
    else:
        report += f"模型类型: {model_type.upper()}（树模型或集成模型）\n"
        report += f"使用特征重要性而非权重\n"
    
    report += f"""
四、决策规则分析
----------------------------------------------------------------------
"""
    
    if rules_info:
        if rules_info['type'] == 'tree_importance':
            report += f"最重要的5个特征（用于决策）:\n"
            for i, feat_info in enumerate(rules_info['top_features'], 1):
                report += f"  {i}. {feat_info['feature']:30s}: {feat_info['importance']:.6f}\n"
    else:
        report += f"模型使用复杂的非线性决策边界\n"
        report += f"主要依赖特征重要性进行决策\n"
    
    report += f"""
五、算法工作原理
----------------------------------------------------------------------

5.1 特征工程
系统使用12个特征，包括：
  - 基础特征：评委评分和粉丝投票（标准化到0-1）
  - 排名特征：评委排名和粉丝排名（归一化）
  - 百分比特征：评委百分比和粉丝百分比
  - 相对特征：相对于组内平均值的比例
  - 影响因素：年龄、专业舞者、行业、地区（编码为数值）

5.2 模型训练
  - 使用时间序列交叉验证（按季次分割）
  - 目标：预测谁会被淘汰（二分类问题）
  - 使用概率预测，选择概率最高的选手作为淘汰预测

5.3 预测过程
  1. 对每个周次的选手提取12个特征
  2. 使用StandardScaler标准化特征
  3. 模型预测每个选手被淘汰的概率
  4. 选择概率最高的选手作为淘汰预测

5.4 高准确率的原因分析
"""
    
    # 分析高准确率的原因
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    report += f"  1. 最重要的特征:\n"
    for feat, imp in top_features:
        report += f"     - {feat}: {imp:.6f}\n"
    
    report += f"\n  2. 特征组合优势:\n"
    report += f"     - 同时使用排名和百分比特征，捕获不同维度的信息\n"
    report += f"     - 相对特征能够消除组间差异\n"
    report += f"     - 影响因素特征提供额外的上下文信息\n"
    
    report += f"\n  3. 模型优势:\n"
    if model_type in ['xgb', 'lgb']:
        report += f"     - 梯度提升模型能够学习复杂的非线性关系\n"
        report += f"     - 特征重要性显示模型主要依赖粉丝投票相关特征\n"
    elif model_type == 'mlp':
        report += f"     - 多层感知机能够学习复杂的特征交互\n"
        report += f"     - 非线性激活函数捕获复杂的决策边界\n"
    elif model_type == 'rf':
        report += f"     - 随机森林通过集成多个决策树提高准确性\n"
        report += f"     - 特征重要性显示模型主要依赖粉丝投票相关特征\n"
    
    report += f"""
六、关键发现
----------------------------------------------------------------------

1. 最重要的特征类别: {sorted_categories[0][0]} ({sorted_categories[0][1]/sum(category_importance.values())*100:.2f}%)
2. 最重要的单个特征: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[0][0]}
3. 粉丝投票相关特征的总重要性: {fan_total_importance:.6f} ({fan_total_importance/sum(feature_importance.values())*100:.2f}%)
4. 评委评分相关特征的总重要性: {judge_total_importance:.6f} ({judge_total_importance/sum(feature_importance.values())*100:.2f}%)

七、模型参数
----------------------------------------------------------------------
"""
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        important_params = {k: v for k, v in params.items() 
                           if k in ['max_depth', 'n_estimators', 'learning_rate', 
                                   'alpha', 'reg_alpha', 'reg_lambda', 'min_samples_split',
                                   'hidden_layer_sizes', 'activation', 'solver']}
        for param, value in important_params.items():
            report += f"  {param:30s}: {value}\n"
    
    report += f"""
======================================================================
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report, {
        'feature_importance': feature_importance,
        'feature_contribution': feature_contribution,
        'category_importance': category_importance,
        'weights_info': weights_info,
        'rules_info': rules_info,
        'fan_total_importance': float(fan_total_importance),
        'judge_total_importance': float(judge_total_importance)
    }


def main():
    """主函数"""
    print("=" * 70)
    print("阶段5：ML投票系统算法深度分析")
    print("=" * 70)
    
    # 选择模型类型
    print("\n可用的模型类型:")
    print("  1. mlp - 多层感知机 (MLP)")
    print("  2. rf - 随机森林 (Random Forest)")
    print("  3. xgb - XGBoost")
    print("  4. lgb - LightGBM")
    print("  5. gbdt - Gradient Boosting")
    print("  6. sgd - 随机梯度下降 (SGD)")
    
    model_choice = input("\n请选择模型类型 (1-6，默认4): ").strip()
    model_map = {
        '1': 'mlp',
        '2': 'rf',
        '3': 'xgb',
        '4': 'lgb',
        '5': 'gbdt',
        '6': 'sgd'
    }
    model_type = model_map.get(model_choice, 'lgb')
    
    print(f"\n分析模型: {model_type.upper()}")
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    try:
        raw_df = load_data()
        print("✓ 原始数据加载成功")
        
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print("✓ 预处理数据加载成功")
        
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print("✓ 粉丝投票估计数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 步骤2: 创建ML系统
    print("\n步骤2: 创建ML投票系统...")
    try:
        ml_system = RobustMLVotingSystem(
            estimates_df=estimates_df,
            processed_df=processed_df,
            factor_analysis_path='factor_impact_analysis.json',
            model_type=model_type,
            use_time_split=True
        )
        print("✓ ML系统创建成功")
    except Exception as e:
        print(f"❌ ML系统创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 训练模型
    print("\n步骤3: 训练模型...")
    try:
        training_results = ml_system.train_with_time_split(n_splits=5)
        print("✓ 模型训练成功")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 应用模型
    print("\n步骤4: 应用模型到所有周次...")
    try:
        ml_system_results = ml_system.apply_to_all_weeks()
        comparison = ml_system.compare_with_original_systems(ml_system_results)
        print("✓ 模型应用成功")
    except Exception as e:
        print(f"❌ 模型应用失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 生成详细分析报告
    print("\n步骤5: 生成详细分析报告...")
    try:
        report, analysis_data = generate_detailed_analysis_report(
            ml_system, training_results, comparison, model_type
        )
        
        # 保存报告
        report_file = f'stage5_model_analysis_{model_type}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 详细分析报告已保存到: {report_file}")
        
        # 保存分析数据（JSON）
        analysis_file = f'stage5_model_analysis_{model_type}.json'
        # 转换numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        analysis_native = convert_to_native(analysis_data)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_native, f, indent=2, ensure_ascii=False)
        print(f"✓ 分析数据已保存到: {analysis_file}")
        
        # 打印报告摘要
        print("\n" + "=" * 70)
        print("算法分析摘要")
        print("=" * 70)
        print(report[:2000])  # 打印前2000个字符
        print("\n... (完整报告请查看文件)")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("算法深度分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
