"""
阶段5：基于机器学习的投票系统
使用MLP、随机森林等模型动态学习如何组合评委评分和粉丝投票
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb


class MLVotingSystem:
    """
    基于机器学习的投票系统
    
    核心思想：
    1. 使用ML模型学习如何组合评委评分和粉丝投票
    2. 特征包括：评委评分、粉丝投票、年龄、专业舞者、行业、地区等
    3. 目标：预测谁会被淘汰
    4. 支持多种模型：MLP、随机森林、XGBoost、LightGBM等
    """
    
    def __init__(
        self,
        estimates_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        factor_analysis_path: str = 'factor_impact_analysis.json',
        model_type: str = 'mlp'  # 'mlp', 'rf', 'xgb', 'lgb', 'gbdt'
    ):
        """
        初始化ML投票系统
        
        Parameters:
        -----------
        estimates_df : pd.DataFrame
            阶段2估计的粉丝投票数据
        processed_df : pd.DataFrame
            预处理后的数据
        factor_analysis_path : str
            阶段4的影响因素分析结果文件路径
        model_type : str
            模型类型：'mlp', 'rf', 'xgb', 'lgb', 'gbdt'
        """
        self.estimates_df = estimates_df.copy()
        self.processed_df = processed_df.copy()
        self.model_type = model_type
        
        # 加载阶段4的分析结果（用于特征编码）
        try:
            with open(factor_analysis_path, 'r', encoding='utf-8') as f:
                self.factor_analysis = json.load(f)
        except:
            self.factor_analysis = {}
        
        # 模型和编码器
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # 特征列名
        self.feature_columns = []
        
    def _prepare_features(self, group: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        准备特征数据
        
        Parameters:
        -----------
        group : pd.DataFrame
            当前周次的选手数据
        
        Returns:
        --------
        Tuple[np.ndarray, List[str]]: (特征矩阵, 特征列名)
        """
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
            # 标准化年龄
            age_normalized = (ages - ages.min()) / (ages.max() - ages.min() + 1e-6)
            features.append(age_normalized)
            feature_names.append('age_normalized')
        else:
            features.append(np.zeros(n))
            feature_names.append('age_normalized')
        
        # 5. 专业舞者特征（编码为数值）
        if 'ballroompartner' in group.columns or 'ballroom_partner' in group.columns:
            pro_dancer_col = 'ballroompartner' if 'ballroompartner' in group.columns else 'ballroom_partner'
            pro_dancers = group[pro_dancer_col].fillna('Unknown').astype(str).values
            
            # 使用LabelEncoder编码
            if 'pro_dancer' not in self.label_encoders:
                self.label_encoders['pro_dancer'] = LabelEncoder()
                # 先fit所有可能的专业舞者
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
        
        # 6. 行业特征（编码为数值）
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
        
        # 7. 地区特征（编码为数值）
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
        
        # 8. 相对特征（相对于组内平均）
        judge_relative = judge_totals / (judge_totals.mean() + 1e-6)
        fan_relative = fan_votes / (fan_votes.mean() + 1e-6)
        features.append(judge_relative)
        features.append(fan_relative)
        feature_names.extend(['judge_relative', 'fan_relative'])
        
        # 组合所有特征
        feature_matrix = np.column_stack(features)
        
        return feature_matrix, feature_names
    
    def _build_model(self, n_features: int):
        """
        构建ML模型
        
        Parameters:
        -----------
        n_features : int
            特征数量
        """
        if self.model_type == 'mlp':
            # 多层感知机（MLP）
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # 3层：128 -> 64 -> 32
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2正则化
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.model_type == 'rf':
            # 随机森林
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgb':
            # XGBoost
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
            # LightGBM
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
            # Gradient Boosting
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'sgd':
            # 随机梯度下降（SGD）
            self.model = SGDClassifier(
                loss='log_loss',  # 逻辑回归损失（用于概率预测）
                penalty='l2',  # L2正则化
                alpha=0.0001,  # 正则化强度
                l1_ratio=0.15,  # Elastic Net混合比例
                max_iter=1000,
                learning_rate='adaptive',
                eta0=0.01,  # 初始学习率
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, test_size: float = 0.2, random_state: int = 42):
        """
        训练模型
        
        Parameters:
        -----------
        test_size : float
            测试集比例
        random_state : int
            随机种子
        """
        print(f"训练 {self.model_type.upper()} 模型...")
        
        # 准备训练数据
        X_list = []
        y_list = []
        week_keys = []
        
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
        
        # 按周次分组
        for (season, week), group in merged_df.groupby(['season', 'week']):
            # 跳过太小的组
            if len(group) < 2:
                continue
            
            # 准备特征
            X_week, feature_names = self._prepare_features(group)
            self.feature_columns = feature_names  # 保存特征列名
            
            # 准备标签（1表示被淘汰，0表示未淘汰）
            y_week = (group['eliminated'] == True).astype(int).values
            
            X_list.append(X_week)
            y_list.append(y_week)
            week_keys.append((season, week))
        
        # 合并所有周次的数据
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"  总样本数: {len(X)}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  淘汰样本数: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        
        # 划分训练集和测试集
        # 注意：使用样本索引分割，以便后续可以追踪哪些周次在测试集中
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        # 保存测试集的周次信息（用于后续只在测试集上评估）
        self.test_week_keys = [week_keys[i] for i in test_indices]
        self.train_week_keys = [week_keys[i] for i in train_indices]
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 构建模型
        self._build_model(X_train_scaled.shape[1])
        
        # 训练模型
        print("  训练中...")
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"  训练集准确率: {train_acc:.4f}")
        print(f"  测试集准确率: {test_acc:.4f}")
        print(f"  ⚠️  注意: 后续预测准确率应只在测试集上计算")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_samples': len(X),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X.shape[1],
            'n_eliminated': y.sum()
        }
    
    def predict_eliminated(self, group: pd.DataFrame) -> int:
        """
        预测该周被淘汰的选手
        
        Parameters:
        -----------
        group : pd.DataFrame
            当前周次的选手数据
        
        Returns:
        --------
        int: 被淘汰选手的索引
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # 准备特征
        X, _ = self._prepare_features(group)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            # 返回被淘汰概率最高的索引
            eliminated_idx = np.argmax(proba[:, 1])  # 第二列是淘汰概率
        else:
            # 如果没有predict_proba，使用predict
            predictions = self.model.predict(X_scaled)
            eliminated_idx = np.argmax(predictions)
        
        return eliminated_idx
    
    def apply_to_all_weeks(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        对所有周次应用ML投票系统
        
        Parameters:
        -----------
        seasons : Optional[List[int]]
            要处理的季次列表，None表示所有季次
        
        Returns:
        --------
        pd.DataFrame: 包含新系统结果的数据框
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        results = []
        
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
        
        if seasons:
            merged_df = merged_df[merged_df['season'].isin(seasons)]
        
        # 按季和周分组
        for (season, week), group in merged_df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            
            # 获取评委总分和粉丝投票
            judge_totals = group['judge_total'].values
            fan_votes = group['fan_votes'].values
            
            # 使用ML模型预测淘汰者
            try:
                eliminated_idx = self.predict_eliminated(group)
            except Exception as e:
                print(f"警告: Season {season}, Week {week} 预测失败: {e}")
                continue
            
            # 记录结果
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
    
    def compare_with_original_systems(
        self,
        ml_system_results: pd.DataFrame,
        test_only: bool = False
    ) -> Dict:
        """
        比较ML系统与原始系统的差异
        
        Parameters:
        -----------
        ml_system_results : pd.DataFrame
            ML系统的结果
        test_only : bool
            如果为True，只在测试集上计算准确率（避免数据泄露）
        
        Returns:
        --------
        Dict: 比较分析结果
        """
        # 合并原始估计数据
        comparison_df = ml_system_results.merge(
            self.estimates_df[['season', 'week', 'celebrity_name', 'eliminated']],
            on=['season', 'week', 'celebrity_name'],
            how='left'
        )
        
        # 如果test_only=True，只保留测试集的周次
        if test_only and hasattr(self, 'test_week_keys'):
            test_week_set = set(self.test_week_keys)
            comparison_df = comparison_df[
                comparison_df.apply(lambda row: (int(row['season']), int(row['week'])) in test_week_set, axis=1)
            ]
            print(f"  ⚠️  只在测试集上评估（{len(test_week_set)}个周次）")
        
        # 计算原始系统的淘汰预测（排名法）
        def get_original_eliminated_rank(group):
            group = group.copy()
            group['judge_rank'] = group['judge_total'].rank(ascending=False, method='min')
            group['fan_rank'] = group['fan_votes'].rank(ascending=False, method='min')
            group['combined_rank_original'] = group['judge_rank'] + group['fan_rank']
            return group.loc[group['combined_rank_original'].idxmax(), 'celebrity_name']
        
        # 计算原始系统的淘汰预测（百分比法）
        def get_original_eliminated_percent(group):
            group = group.copy()
            group['judge_percent'] = group['judge_total'] / group['judge_total'].sum() * 100
            group['fan_percent'] = group['fan_votes'] / group['fan_votes'].sum() * 100
            group['combined_percent_original'] = group['judge_percent'] + group['fan_percent']
            return group.loc[group['combined_percent_original'].idxmin(), 'celebrity_name']
        
        # 对每个周次计算原始系统的淘汰预测
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
        
        # 原始系统的淘汰预测字典
        original_eliminated_dict = {}
        for _, row in original_pred_df.iterrows():
            key = (int(row['season']), int(row['week']))
            original_eliminated_dict[key] = row['original_eliminated']
        
        # ML系统的淘汰预测
        ml_eliminated = comparison_df[comparison_df['is_eliminated_ml_system'] == True]
        ml_eliminated_dict = {}
        for _, row in ml_eliminated.iterrows():
            key = (int(row['season']), int(row['week']))
            ml_eliminated_dict[key] = row['celebrity_name']
        
        # 实际淘汰
        actual_eliminated = comparison_df[comparison_df['eliminated'] == True]
        actual_eliminated_dict = {}
        for _, row in actual_eliminated.iterrows():
            key = (int(row['season']), int(row['week']))
            actual_eliminated_dict[key] = row['celebrity_name']
        
        # 统计比较
        all_weeks = set(comparison_df[['season', 'week']].drop_duplicates().apply(
            lambda x: (int(x['season']), int(x['week'])), axis=1
        ))
        
        original_correct = 0
        ml_correct = 0
        different_predictions = 0
        total_weeks = len(all_weeks)
        
        # 计算训练集和测试集的分别准确率（如果可用）
        train_original_correct = 0
        train_ml_correct = 0
        test_original_correct = 0
        test_ml_correct = 0
        n_train_weeks = 0
        n_test_weeks = 0
        
        if hasattr(self, 'train_week_keys') and hasattr(self, 'test_week_keys'):
            train_week_set = set(self.train_week_keys)
            test_week_set = set(self.test_week_keys)
        else:
            train_week_set = set()
            test_week_set = set()
        
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
                
                # 分别统计训练集和测试集
                if week_key in train_week_set:
                    n_train_weeks += 1
                    if original_pred == actual:
                        train_original_correct += 1
                    if ml_pred == actual:
                        train_ml_correct += 1
                elif week_key in test_week_set:
                    n_test_weeks += 1
                    if original_pred == actual:
                        test_original_correct += 1
                    if ml_pred == actual:
                        test_ml_correct += 1
        
        result = {
            'total_weeks': total_weeks,
            'original_system_accuracy': original_correct / total_weeks if total_weeks > 0 else 0.0,
            'ml_system_accuracy': ml_correct / total_weeks if total_weeks > 0 else 0.0,
            'accuracy_improvement': (ml_correct - original_correct) / total_weeks if total_weeks > 0 else 0.0,
            'different_predictions': different_predictions,
            'different_predictions_rate': different_predictions / total_weeks if total_weeks > 0 else 0.0,
            'original_correct_count': original_correct,
            'ml_correct_count': ml_correct
        }
        
        # 添加训练集和测试集的分别准确率（避免数据泄露）
        if n_train_weeks > 0 and n_test_weeks > 0:
            result['train_weeks'] = n_train_weeks
            result['test_weeks'] = n_test_weeks
            result['train_original_accuracy'] = train_original_correct / n_train_weeks if n_train_weeks > 0 else 0.0
            result['train_ml_accuracy'] = train_ml_correct / n_train_weeks if n_train_weeks > 0 else 0.0
            result['test_original_accuracy'] = test_original_correct / n_test_weeks if n_test_weeks > 0 else 0.0
            result['test_ml_accuracy'] = test_ml_correct / n_test_weeks if n_test_weeks > 0 else 0.0
            result['test_ml_accuracy_improvement'] = (test_ml_correct - test_original_correct) / n_test_weeks if n_test_weeks > 0 else 0.0
            result['note'] = 'test_ml_accuracy是真正的泛化能力（只在测试集上）'
        
        return result
