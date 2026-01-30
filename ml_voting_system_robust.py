"""
阶段5：基于机器学习的投票系统（防过拟合版本）
添加时间序列交叉验证、正则化、特征重要性分析
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 机器学习模型
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb


class RobustMLVotingSystem:
    """
    防过拟合的ML投票系统
    
    改进：
    1. 使用时间序列交叉验证（按季次分割）
    2. 增加正则化
    3. 降低模型复杂度
    4. 特征重要性分析
    """
    
    def __init__(
        self,
        estimates_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        factor_analysis_path: str = 'factor_impact_analysis.json',
        model_type: str = 'mlp',
        use_time_split: bool = True  # 使用时间序列分割
    ):
        self.estimates_df = estimates_df.copy()
        self.processed_df = processed_df.copy()
        self.model_type = model_type
        self.use_time_split = use_time_split
        
        try:
            with open(factor_analysis_path, 'r', encoding='utf-8') as f:
                self.factor_analysis = json.load(f)
        except:
            self.factor_analysis = {}
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_importance = None
    
    def _prepare_features(self, group: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """准备特征数据"""
        features = []
        feature_names = []
        
        n = len(group)
        
        # 1. 基础特征：评委评分和粉丝投票
        judge_totals = group['judge_total'].values
        fan_votes = group['fan_votes'].values
        
        # 标准化到0-1范围
        judge_normalized = (judge_totals - judge_totals.min()) / (judge_totals.max() - judge_totals.min() + 1e-6)
        fan_normalized = (fan_votes - fan_votes.min()) / (fan_votes.max() - fan_votes.min() + 1e-6)
        
        features.append(judge_normalized)
        features.append(fan_normalized)
        feature_names.extend(['judge_score_normalized', 'fan_votes_normalized'])
        
        # 2. 排名特征
        judge_ranks = pd.Series(judge_totals).rank(ascending=False, method='min').values / n
        fan_ranks = pd.Series(fan_votes).rank(ascending=False, method='min').values / n
        features.append(judge_ranks)
        features.append(fan_ranks)
        feature_names.extend(['judge_rank_normalized', 'fan_rank_normalized'])
        
        # 3. 百分比特征
        judge_percents = (judge_totals / judge_totals.sum())
        fan_percents = (fan_votes / fan_votes.sum())
        features.append(judge_percents)
        features.append(fan_percents)
        feature_names.extend(['judge_percent', 'fan_percent'])
        
        # 4. 年龄特征
        if 'celebrity_age_during_season' in group.columns:
            ages = group['celebrity_age_during_season'].fillna(group['celebrity_age_during_season'].mean()).values
            age_normalized = (ages - ages.min()) / (ages.max() - ages.min() + 1e-6)
            features.append(age_normalized)
            feature_names.append('age_normalized')
        else:
            features.append(np.zeros(n))
            feature_names.append('age_normalized')
        
        # 5. 专业舞者特征
        if 'ballroompartner' in group.columns or 'ballroom_partner' in group.columns:
            pro_dancer_col = 'ballroompartner' if 'ballroompartner' in group.columns else 'ballroom_partner'
            pro_dancers = group[pro_dancer_col].fillna('Unknown').astype(str).values
            
            if 'pro_dancer' not in self.label_encoders:
                self.label_encoders['pro_dancer'] = LabelEncoder()
                all_dancers = self.estimates_df.get('ballroompartner', pd.Series()).fillna('Unknown').astype(str)
                if all_dancers.empty:
                    all_dancers = self.estimates_df.get('ballroom_partner', pd.Series()).fillna('Unknown').astype(str)
                if not all_dancers.empty:
                    self.label_encoders['pro_dancer'].fit(all_dancers.unique())
            
            if 'pro_dancer' in self.label_encoders:
                try:
                    pro_dancer_encoded = self.label_encoders['pro_dancer'].transform(pro_dancers)
                    pro_dancer_normalized = pro_dancer_encoded / (len(self.label_encoders['pro_dancer'].classes_) + 1e-6)
                    features.append(pro_dancer_normalized)
                    feature_names.append('pro_dancer_encoded')
                except:
                    features.append(np.zeros(n))
                    feature_names.append('pro_dancer_encoded')
            else:
                features.append(np.zeros(n))
                feature_names.append('pro_dancer_encoded')
        else:
            features.append(np.zeros(n))
            feature_names.append('pro_dancer_encoded')
        
        # 6. 行业特征
        if 'celebrity_industry' in group.columns:
            industries = group['celebrity_industry'].fillna('Unknown').astype(str).values
            
            if 'industry' not in self.label_encoders:
                all_industries = self.processed_df['celebrity_industry'].fillna('Unknown').astype(str).unique()
                self.label_encoders['industry'] = LabelEncoder()
                self.label_encoders['industry'].fit(all_industries)
            
            try:
                industry_encoded = self.label_encoders['industry'].transform(industries)
                industry_normalized = industry_encoded / (len(self.label_encoders['industry'].classes_) + 1e-6)
                features.append(industry_normalized)
                feature_names.append('industry_encoded')
            except:
                features.append(np.zeros(n))
                feature_names.append('industry_encoded')
        else:
            features.append(np.zeros(n))
            feature_names.append('industry_encoded')
        
        # 7. 地区特征
        if 'celebrity_homecountry/region' in group.columns:
            regions = group['celebrity_homecountry/region'].fillna('Unknown').astype(str).values
            
            if 'region' not in self.label_encoders:
                all_regions = self.processed_df['celebrity_homecountry/region'].fillna('Unknown').astype(str).unique()
                self.label_encoders['region'] = LabelEncoder()
                self.label_encoders['region'].fit(all_regions)
            
            try:
                region_encoded = self.label_encoders['region'].transform(regions)
                region_normalized = region_encoded / (len(self.label_encoders['region'].classes_) + 1e-6)
                features.append(region_normalized)
                feature_names.append('region_encoded')
            except:
                features.append(np.zeros(n))
                feature_names.append('region_encoded')
        else:
            features.append(np.zeros(n))
            feature_names.append('region_encoded')
        
        # 8. 相对特征
        judge_relative = judge_totals / (judge_totals.mean() + 1e-6)
        fan_relative = fan_votes / (fan_votes.mean() + 1e-6)
        features.append(judge_relative)
        features.append(fan_relative)
        feature_names.extend(['judge_relative', 'fan_relative'])
        
        feature_matrix = np.column_stack(features)
        return feature_matrix, feature_names
    
    def _build_model(self, n_features: int, regularized: bool = True):
        """
        构建ML模型（增加正则化）
        
        Parameters:
        -----------
        n_features : int
            特征数量
        regularized : bool
            是否使用正则化版本
        """
        if self.model_type == 'mlp':
            if regularized:
                # 更保守的MLP（减少层数和神经元）
                self.model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),  # 2层，减少神经元
                    activation='relu',
                    solver='adam',
                    alpha=0.01,  # 增加L2正则化
                    batch_size='auto',
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2,  # 增加验证集比例
                    n_iter_no_change=10
                )
            else:
                self.model = MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
        elif self.model_type == 'rf':
            if regularized:
                # 更保守的随机森林
                self.model = RandomForestClassifier(
                    n_estimators=100,  # 减少树数量
                    max_depth=5,  # 大幅降低深度
                    min_samples_split=10,  # 增加最小分割样本
                    min_samples_leaf=5,  # 增加叶子节点最小样本
                    max_features='sqrt',  # 限制特征数
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
        elif self.model_type == 'xgb':
            if regularized:
                # 更保守的XGBoost
                self.model = xgb.XGBClassifier(
                    n_estimators=100,  # 减少树数量
                    max_depth=4,  # 降低深度
                    learning_rate=0.05,  # 降低学习率
                    subsample=0.7,  # 降低采样比例
                    colsample_bytree=0.7,
                    reg_alpha=0.1,  # L1正则化
                    reg_lambda=1.0,  # L2正则化
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
        elif self.model_type == 'lgb':
            if regularized:
                # 更保守的LightGBM
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_samples=20,  # 增加最小子样本
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
        elif self.model_type == 'gbdt':
            if regularized:
                # 更保守的Gradient Boosting
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.7,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
        elif self.model_type == 'sgd':
            if regularized:
                # 更保守的SGD（增加正则化）
                self.model = SGDClassifier(
                    loss='log_loss',
                    penalty='elasticnet',  # Elastic Net（L1+L2）
                    alpha=0.001,  # 增加正则化强度
                    l1_ratio=0.15,
                    max_iter=2000,
                    learning_rate='adaptive',
                    eta0=0.001,  # 降低初始学习率
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=15,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = SGDClassifier(
                    loss='log_loss',
                    penalty='l2',
                    alpha=0.0001,
                    l1_ratio=0.15,
                    max_iter=1000,
                    learning_rate='adaptive',
                    eta0=0.01,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train_with_time_split(self, n_splits: int = 5):
        """
        使用时间序列交叉验证训练模型
        
        Parameters:
        -----------
        n_splits : int
            交叉验证折数
        """
        print(f"训练 {self.model_type.upper()} 模型（时间序列交叉验证）...")
        
        # 准备数据（按季次排序）
        X_list = []
        y_list = []
        season_list = []
        
        # 合并数据
        info_cols = ['season', 'celebrity_name', 'celebrity_age_during_season',
                     'celebrity_industry', 'celebrity_homecountry/region']
        
        if 'ballroompartner' in self.processed_df.columns:
            info_cols.append('ballroompartner')
        elif 'ballroom_partner' in self.processed_df.columns:
            info_cols.append('ballroom_partner')
        
        available_cols = [col for col in info_cols if col in self.processed_df.columns]
        processed_info = self.processed_df[available_cols].drop_duplicates(
            subset=['season', 'celebrity_name']
        )
        
        merged_df = self.estimates_df.merge(
            processed_info,
            on=['season', 'celebrity_name'],
            how='left'
        )
        
        # 按季次和周次排序
        merged_df = merged_df.sort_values(['season', 'week'])
        
        for (season, week), group in merged_df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            
            X_week, feature_names = self._prepare_features(group)
            self.feature_columns = feature_names
            y_week = (group['eliminated'] == True).astype(int).values
            
            X_list.append(X_week)
            y_list.append(y_week)
            season_list.extend([season] * len(X_week))
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        seasons = np.array(season_list)
        
        print(f"  总样本数: {len(X)}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  淘汰样本数: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"  季次范围: {seasons.min()} - {seasons.max()}")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型（使用正则化版本）
        self._build_model(X_scaled.shape[1], regularized=True)
        
        # 时间序列交叉验证
        unique_seasons = sorted(merged_df['season'].unique())
        print(f"\n  时间序列交叉验证（按季次分割）...")
        
        cv_scores = []
        train_scores = []
        
        # 使用前N-1季训练，最后一季测试
        for i in range(1, min(n_splits + 1, len(unique_seasons))):
            train_seasons = unique_seasons[:-i]
            test_season = unique_seasons[-i]
            
            train_mask = np.isin(seasons, train_seasons)
            test_mask = (seasons == test_season)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            X_train = X_scaled[train_mask]
            y_train = y[train_mask]
            X_test = X_scaled[test_mask]
            y_test = y[test_mask]
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 评估
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            train_scores.append(train_acc)
            cv_scores.append(test_acc)
            
            print(f"    Fold {i}: 训练季次 {train_seasons[0]}-{train_seasons[-1]}, "
                  f"测试季次 {test_season}")
            print(f"      训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        # 使用所有数据训练最终模型
        print(f"\n  使用所有数据训练最终模型...")
        self.model.fit(X_scaled, y)
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            print(f"\n  特征重要性:")
            for feat, imp in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {feat}: {imp:.4f}")
        
        avg_train_score = np.mean(train_scores) if train_scores else 0.0
        avg_cv_score = np.mean(cv_scores) if cv_scores else 0.0
        std_cv_score = np.std(cv_scores) if cv_scores else 0.0
        
        print(f"\n  平均训练准确率: {avg_train_score:.4f}")
        print(f"  平均交叉验证准确率: {avg_cv_score:.4f} (±{std_cv_score:.4f})")
        
        return {
            'train_accuracy': avg_train_score,
            'cv_accuracy': avg_cv_score,
            'cv_std': std_cv_score,
            'cv_scores': cv_scores,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_eliminated': y.sum(),
            'feature_importance': self.feature_importance
        }
    
    def predict_eliminated(self, group: pd.DataFrame) -> int:
        """预测该周被淘汰的选手"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self._prepare_features(group)
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            eliminated_idx = np.argmax(proba[:, 1])
        else:
            predictions = self.model.predict(X_scaled)
            eliminated_idx = np.argmax(predictions)
        
        return eliminated_idx
    
    def apply_to_all_weeks(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """对所有周次应用ML投票系统"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        results = []
        
        info_cols = ['season', 'celebrity_name', 'celebrity_age_during_season',
                     'celebrity_industry', 'celebrity_homecountry/region']
        
        if 'ballroompartner' in self.processed_df.columns:
            info_cols.append('ballroompartner')
        elif 'ballroom_partner' in self.processed_df.columns:
            info_cols.append('ballroom_partner')
        
        available_cols = [col for col in info_cols if col in self.processed_df.columns]
        processed_info = self.processed_df[available_cols].drop_duplicates(
            subset=['season', 'celebrity_name']
        )
        
        merged_df = self.estimates_df.merge(
            processed_info,
            on=['season', 'celebrity_name'],
            how='left'
        )
        
        if seasons:
            merged_df = merged_df[merged_df['season'].isin(seasons)]
        
        for (season, week), group in merged_df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            
            judge_totals = group['judge_total'].values
            fan_votes = group['fan_votes'].values
            
            try:
                eliminated_idx = self.predict_eliminated(group)
            except Exception as e:
                continue
            
            for i, (idx, row) in enumerate(group.iterrows()):
                results.append({
                    'season': int(season),
                    'week': int(week),
                    'celebrity_name': row['celebrity_name'],
                    'judge_total': float(judge_totals[i]),
                    'fan_votes': float(fan_votes[i]),
                    'is_eliminated_ml_system': (i == eliminated_idx),
                    'age': float(row.get('celebrity_age_during_season', np.nan)) if pd.notna(row.get('celebrity_age_during_season')) else np.nan,
                    'pro_dancer': str(row.get('ballroompartner', '') or row.get('ballroom_partner', '')),
                    'industry': str(row.get('celebrity_industry', '')),
                    'region': str(row.get('celebrity_homecountry/region', ''))
                })
        
        return pd.DataFrame(results)
    
    def compare_with_original_systems(self, ml_system_results: pd.DataFrame) -> Dict:
        """比较ML系统与原始系统"""
        comparison_df = ml_system_results.merge(
            self.estimates_df[['season', 'week', 'celebrity_name', 'eliminated']],
            on=['season', 'week', 'celebrity_name'],
            how='left'
        )
        
        def get_original_eliminated_rank(group):
            group = group.copy()
            group['judge_rank'] = group['judge_total'].rank(ascending=False, method='min')
            group['fan_rank'] = group['fan_votes'].rank(ascending=False, method='min')
            group['combined_rank_original'] = group['judge_rank'] + group['fan_rank']
            return group.loc[group['combined_rank_original'].idxmax(), 'celebrity_name']
        
        def get_original_eliminated_percent(group):
            group = group.copy()
            group['judge_percent'] = group['judge_total'] / group['judge_total'].sum() * 100
            group['fan_percent'] = group['fan_votes'] / group['fan_votes'].sum() * 100
            group['combined_percent_original'] = group['judge_percent'] + group['fan_percent']
            return group.loc[group['combined_percent_original'].idxmin(), 'celebrity_name']
        
        original_predictions = []
        for (season, week), group in comparison_df.groupby(['season', 'week']):
            if season in [1, 2] or season >= 28:
                original_eliminated = get_original_eliminated_rank(group)
            else:
                original_eliminated = get_original_eliminated_percent(group)
            
            original_predictions.append({
                'season': season,
                'week': week,
                'original_eliminated': original_eliminated
            })
        
        original_pred_df = pd.DataFrame(original_predictions)
        original_eliminated_dict = {}
        for _, row in original_pred_df.iterrows():
            key = (int(row['season']), int(row['week']))
            original_eliminated_dict[key] = row['original_eliminated']
        
        ml_eliminated = comparison_df[comparison_df['is_eliminated_ml_system'] == True]
        ml_eliminated_dict = {}
        for _, row in ml_eliminated.iterrows():
            key = (int(row['season']), int(row['week']))
            ml_eliminated_dict[key] = row['celebrity_name']
        
        actual_eliminated = comparison_df[comparison_df['eliminated'] == True]
        actual_eliminated_dict = {}
        for _, row in actual_eliminated.iterrows():
            key = (int(row['season']), int(row['week']))
            actual_eliminated_dict[key] = row['celebrity_name']
        
        all_weeks = set(comparison_df[['season', 'week']].drop_duplicates().apply(
            lambda x: (int(x['season']), int(x['week'])), axis=1
        ))
        
        original_correct = 0
        ml_correct = 0
        different_predictions = 0
        total_weeks = len(all_weeks)
        
        for week_key in all_weeks:
            original_pred = original_eliminated_dict.get(week_key)
            ml_pred = ml_eliminated_dict.get(week_key)
            actual = actual_eliminated_dict.get(week_key)
            
            if actual:
                if original_pred == actual:
                    original_correct += 1
                if ml_pred == actual:
                    ml_correct += 1
                if original_pred != ml_pred:
                    different_predictions += 1
        
        return {
            'total_weeks': total_weeks,
            'original_system_accuracy': original_correct / total_weeks if total_weeks > 0 else 0.0,
            'ml_system_accuracy': ml_correct / total_weeks if total_weeks > 0 else 0.0,
            'accuracy_improvement': (ml_correct - original_correct) / total_weeks if total_weeks > 0 else 0.0,
            'different_predictions': different_predictions,
            'different_predictions_rate': different_predictions / total_weeks if total_weeks > 0 else 0.0,
            'original_correct_count': original_correct,
            'ml_correct_count': ml_correct
        }
