"""
阶段5：基于机器学习的投票系统 - 自动化测试和集成
自动测试所有模型，比较结果，创建集成模型，选择最佳模型
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_voting_system import MLVotingSystem
from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor

# 集成学习
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import lightgbm as lgb


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


class EnsembleMLVotingSystem(MLVotingSystem):
    """集成ML投票系统"""
    
    def __init__(self, estimates_df, processed_df, factor_analysis_path='factor_impact_analysis.json'):
        # 调用父类初始化，但model_type设为'ensemble'（虽然不会用到）
        super().__init__(estimates_df, processed_df, factor_analysis_path, model_type='mlp')
        self.base_models = []
        self.ensemble_model = None
    
    def train_ensemble(self, test_size=0.2, random_state=42):
        """训练集成模型"""
        print("训练集成模型...")
        
        # 准备训练数据
        X_list = []
        y_list = []
        
        # 合并数据（检查列是否存在）
        info_cols = ['season', 'celebrity_name', 'celebrity_age_during_season',
                     'celebrity_industry', 'celebrity_homecountry/region']
        
        # 检查专业舞者列名
        if 'ballroompartner' in self.processed_df.columns:
            info_cols.append('ballroompartner')
        elif 'ballroom_partner' in self.processed_df.columns:
            info_cols.append('ballroom_partner')
        
        # 只选择存在的列
        available_cols = [col for col in info_cols if col in self.processed_df.columns]
        processed_info = self.processed_df[available_cols].drop_duplicates(
            subset=['season', 'celebrity_name']
        )
        
        merged_df = self.estimates_df.merge(
            processed_info,
            on=['season', 'celebrity_name'],
            how='left'
        )
        
        for (season, week), group in merged_df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            
            X_week, feature_names = self._prepare_features(group)
            self.feature_columns = feature_names
            y_week = (group['eliminated'] == True).astype(int).values
            
            X_list.append(X_week)
            y_list.append(y_week)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 创建基础模型
        n_features = X_train_scaled.shape[1]
        
        models = [
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )),
            ('gbdt', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )),
            ('sgd', SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                learning_rate='adaptive',
                eta0=0.01,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        # 训练所有基础模型
        print("  训练基础模型...")
        trained_models = []
        for name, model in models:
            print(f"    - {name.upper()}...", end=' ')
            model.fit(X_train_scaled, y_train)
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            print(f"训练: {train_acc:.4f}, 测试: {test_acc:.4f}")
            trained_models.append((name, model))
        
        # 创建集成模型（投票分类器）
        print("  创建集成模型...")
        self.ensemble_model = VotingClassifier(
            estimators=trained_models,
            voting='soft',  # 使用概率投票
            n_jobs=-1
        )
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # 评估集成模型
        train_acc = self.ensemble_model.score(X_train_scaled, y_train)
        test_acc = self.ensemble_model.score(X_test_scaled, y_test)
        
        print(f"  集成模型 - 训练: {train_acc:.4f}, 测试: {test_acc:.4f}")
        
        self.model = self.ensemble_model
        self.base_models = trained_models
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_eliminated': y.sum(),
            'base_models': {name: {'train_acc': m.score(X_train_scaled, y_train), 
                                   'test_acc': m.score(X_test_scaled, y_test)} 
                            for name, m in trained_models}
        }


def test_all_models(estimates_df, processed_df):
    """测试所有模型并比较结果"""
    print("=" * 70)
    print("自动测试所有模型")
    print("=" * 70)
    
    model_types = ['mlp', 'rf', 'xgb', 'lgb', 'gbdt', 'sgd']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"测试模型: {model_type.upper()}")
        print(f"{'='*70}")
        
        try:
            # 初始化系统
            ml_system = MLVotingSystem(
                estimates_df=estimates_df,
                processed_df=processed_df,
                factor_analysis_path='factor_impact_analysis.json',
                model_type=model_type
            )
            
            # 训练模型
            training_results = ml_system.train(test_size=0.2, random_state=42)
            
            # 应用到所有周次
            ml_system_results = ml_system.apply_to_all_weeks()
            
            # 比较结果（包含训练集和测试集的分别准确率）
            comparison = ml_system.compare_with_original_systems(ml_system_results)
            
            results[model_type] = {
                'training': training_results,
                'comparison': comparison,
                'system': ml_system
            }
            
            print(f"\n✓ {model_type.upper()} 完成")
            print(f"  训练时测试集准确率: {training_results['test_accuracy']:.4f}")
            if 'test_ml_accuracy' in comparison:
                print(f"  ⚠️  注意: 以下准确率包含训练集（存在数据泄露）")
                print(f"  全部数据预测准确率: {comparison['ml_system_accuracy']:.4f}")
                print(f"  ✓ 测试集预测准确率（真实泛化）: {comparison['test_ml_accuracy']:.4f}")
            else:
                print(f"  预测准确率: {comparison['ml_system_accuracy']:.4f}")
                print(f"  ⚠️  警告: 此准确率包含训练集，存在数据泄露风险")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def test_ensemble(estimates_df, processed_df):
    """测试集成模型"""
    print(f"\n{'='*70}")
    print("测试集成模型")
    print(f"{'='*70}")
    
    try:
        ensemble_system = EnsembleMLVotingSystem(
            estimates_df=estimates_df,
            processed_df=processed_df,
            factor_analysis_path='factor_impact_analysis.json'
        )
        
        training_results = ensemble_system.train_ensemble(test_size=0.2, random_state=42)
        
        ml_system_results = ensemble_system.apply_to_all_weeks()
        comparison = ensemble_system.compare_with_original_systems(ml_system_results)
        
        return {
            'training': training_results,
            'comparison': comparison,
            'system': ensemble_system
        }
    except Exception as e:
        print(f"❌ 集成模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_comparison_report(all_results, ensemble_result):
    """生成比较报告"""
    print("\n" + "=" * 70)
    print("模型比较结果")
    print("=" * 70)
    
    # 创建比较表
    comparison_data = []
    
    for model_type, result in all_results.items():
        comparison_data.append({
            '模型': model_type.upper(),
            '训练集准确率': result['training']['train_accuracy'],
            '测试集准确率': result['training']['test_accuracy'],
            '预测准确率': result['comparison']['ml_system_accuracy'],
            '准确率提升': result['comparison']['accuracy_improvement'],
            '正确预测数': result['comparison']['ml_correct_count']
        })
    
    if ensemble_result:
        comparison_data.append({
            '模型': 'ENSEMBLE',
            '训练集准确率': ensemble_result['training']['train_accuracy'],
            '测试集准确率': ensemble_result['training']['test_accuracy'],
            '预测准确率': ensemble_result['comparison']['ml_system_accuracy'],
            '准确率提升': ensemble_result['comparison']['accuracy_improvement'],
            '正确预测数': ensemble_result['comparison']['ml_correct_count']
        })
    
    # 检查是否有结果
    if not comparison_data:
        print("⚠️  警告: 没有成功的模型测试结果")
        return pd.DataFrame(), None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 按预测准确率排序
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('预测准确率', ascending=False)
        
        print("\n模型性能比较:")
        print(comparison_df.to_string(index=False))
        
        # 找出最佳模型
        best_model = comparison_df.iloc[0]
        print(f"\n最佳模型: {best_model['模型']}")
        print(f"  预测准确率: {best_model['预测准确率']:.2%}")
        print(f"  准确率提升: {best_model['准确率提升']:.2%}")
        
        return comparison_df, best_model
    else:
        return pd.DataFrame(), None


def main():
    """主函数"""
    print("=" * 70)
    print("阶段5：基于机器学习的投票系统 - 自动化测试和集成")
    print("=" * 70)
    
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
    
    # 步骤2: 测试所有模型
    print("\n步骤2: 测试所有模型...")
    all_results = test_all_models(estimates_df, processed_df)
    
    # 步骤3: 测试集成模型
    print("\n步骤3: 测试集成模型...")
    ensemble_result = test_ensemble(estimates_df, processed_df)
    
    # 步骤4: 比较结果
    print("\n步骤4: 比较所有模型...")
    comparison_df, best_model = generate_comparison_report(all_results, ensemble_result)
    
    # 步骤5: 保存结果
    print("\n步骤5: 保存结果...")
    
    # 保存比较表（如果有结果）
    if len(comparison_df) > 0:
        comparison_df.to_csv('ml_models_comparison.csv', index=False, encoding='utf-8-sig')
        print("✓ 已保存模型比较表到: ml_models_comparison.csv")
    else:
        print("⚠️  没有结果可保存")
    
    # 保存详细结果
    all_results_dict = {}
    for model_type, result in all_results.items():
        all_results_dict[model_type] = {
            'training': convert_to_native(result['training']),
            'comparison': convert_to_native(result['comparison'])
        }
    
    if ensemble_result:
        all_results_dict['ensemble'] = {
            'training': convert_to_native(ensemble_result['training']),
            'comparison': convert_to_native(ensemble_result['comparison'])
        }
    
    with open('ml_models_all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)
    print("✓ 已保存所有模型结果到: ml_models_all_results.json")
    
    # 生成综合报告
    report = f"""
阶段5：基于机器学习的投票系统 - 自动化测试报告
======================================================================

一、测试概述
----------------------------------------------------------------------
测试了以下模型:
1. MLP - 多层感知机
2. RF - 随机森林
3. XGB - XGBoost
4. LGB - LightGBM
5. GBDT - Gradient Boosting
6. SGD - 随机梯度下降
7. ENSEMBLE - 集成模型（投票分类器）

二、模型性能比较
----------------------------------------------------------------------
{comparison_df.to_string(index=False)}

三、最佳模型
----------------------------------------------------------------------
模型名称: {best_model['模型']}
预测准确率: {best_model['预测准确率']:.2%}
准确率提升: {best_model['准确率提升']:.2%}
正确预测数: {int(best_model['正确预测数'])} / {all_results[list(all_results.keys())[0]]['comparison']['total_weeks']}

四、详细结果
----------------------------------------------------------------------
详细结果请参考:
- ml_models_comparison.csv - 模型比较表
- ml_models_all_results.json - 所有模型的详细结果

======================================================================
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('stage5_ml_auto_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ 已保存综合报告到: stage5_ml_auto_report.txt")
    
    print("\n" + "=" * 70)
    print("自动化测试完成！")
    if best_model is not None:
        print(f"最佳模型: {best_model['模型']}")
    else:
        print("⚠️  没有成功的模型，请检查错误信息")
    print("=" * 70)


if __name__ == '__main__':
    main()
