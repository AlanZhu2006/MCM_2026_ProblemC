"""
阶段2：粉丝投票估计模型 - 机器学习版本（集成学习）
使用多个机器学习模型进行集成预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入基础估计器
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fan_vote_estimator import FanVoteEstimator


class MLFanVoteEstimator(FanVoteEstimator):
    """基于机器学习的粉丝投票估计器（继承自基础估计器）"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化机器学习估计器
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据
        """
        super().__init__(df)
        self.models = {}  # 存储训练好的模型
        self.scalers = {}  # 存储特征缩放器
        self.label_encoders = {}  # 存储标签编码器
        self.feature_importance = {}  # 特征重要性
        
    def prepare_ml_features(self, season: int, week: int) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        准备机器学习特征
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[str]]: 特征数据框和被淘汰选手名称
        """
        # 使用父类方法提取基础特征
        features_df = self.extract_features(season, week)
        
        if len(features_df) == 0:
            return pd.DataFrame(), None
        
        # 获取被淘汰选手
        eliminated_name = self.get_eliminated_contestant(season, week)
        
        # 准备机器学习特征
        ml_features = features_df.copy()
        
        # 1. 评委评分特征（已存在）
        # judge_total, judge_rank, judge_percent
        
        # 2. 历史表现特征（已存在，需要补充）
        if 'avg_historical_score' not in ml_features.columns:
            ml_features['avg_historical_score'] = ml_features.get('judge_total', 0)
        if 'avg_historical_rank' not in ml_features.columns:
            ml_features['avg_historical_rank'] = ml_features.get('judge_rank', 0)
        
        # 3. 选手特征编码
        if 'celebrity_industry' in ml_features.columns:
            if 'industry_encoder' not in self.label_encoders:
                self.label_encoders['industry_encoder'] = LabelEncoder()
                ml_features['industry_encoded'] = self.label_encoders['industry_encoder'].fit_transform(
                    ml_features['celebrity_industry'].astype(str)
                )
            else:
                # 处理新类别
                known_classes = set(self.label_encoders['industry_encoder'].classes_)
                current_classes = set(ml_features['celebrity_industry'].astype(str).unique())
                new_classes = current_classes - known_classes
                if new_classes:
                    # 添加新类别
                    all_classes = list(known_classes) + list(new_classes)
                    self.label_encoders['industry_encoder'].classes_ = np.array(all_classes)
                ml_features['industry_encoded'] = self.label_encoders['industry_encoder'].transform(
                    ml_features['celebrity_industry'].astype(str)
                )
        
        # 4. 年龄特征
        if 'celebrity_age_during_season' in ml_features.columns:
            ml_features['age'] = ml_features['celebrity_age_during_season'].fillna(
                ml_features['celebrity_age_during_season'].median()
            )
            # 年龄分组
            ml_features['age_group'] = pd.cut(
                ml_features['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        else:
            ml_features['age'] = 0
            ml_features['age_group'] = 0
        
        # 5. 专业舞者特征（已存在，需要补充）
        if 'partner_avg_placement' not in ml_features.columns:
            ml_features['partner_avg_placement'] = ml_features.get('placement', 10)
        
        # 6. 周次特征
        ml_features['week_number'] = week
        ml_features['contestants_remaining'] = len(ml_features)
        ml_features['week_ratio'] = week / ml_features['contestants_remaining']  # 周次比例
        
        # 7. 排名趋势（如果有历史数据）
        if week > 1:
            prev_week = week - 1
            prev_total_col = f'week{prev_week}_total_score'
            if prev_total_col in ml_features.columns:
                ml_features['score_change'] = ml_features['judge_total'] - ml_features[prev_total_col].fillna(0)
                prev_rank_col = f'week{prev_week}_judge_rank'
                if prev_rank_col in ml_features.columns:
                    ml_features['rank_change'] = ml_features['judge_rank'] - ml_features[prev_rank_col].fillna(0)
            else:
                ml_features['score_change'] = 0
                ml_features['rank_change'] = 0
        else:
            ml_features['score_change'] = 0
            ml_features['rank_change'] = 0
        
        # 8. 是否被淘汰（标签）
        ml_features['is_eliminated'] = (ml_features['celebrity_name'] == eliminated_name).astype(int)
        
        # 9. 填充缺失值（更严格的处理）
        numeric_cols = ml_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if ml_features[col].isna().any():
                median_val = ml_features[col].median()
                if pd.isna(median_val):
                    # 如果中位数也是NaN，用0填充
                    ml_features[col] = ml_features[col].fillna(0.0)
                else:
                    ml_features[col] = ml_features[col].fillna(median_val)
        
        # 最终检查：确保所有数值列都没有NaN
        for col in numeric_cols:
            if ml_features[col].isna().any():
                ml_features[col] = ml_features[col].fillna(0.0)
        
        return ml_features, eliminated_name
    
    def train_models(self, training_data: List[Tuple[int, int, pd.DataFrame, Optional[str]]]) -> Dict:
        """
        训练多个机器学习模型
        
        改进：使用基础方法生成"伪标签"作为训练目标
        
        Parameters:
        -----------
        training_data : List[Tuple[int, int, pd.DataFrame, Optional[str]]]
            训练数据列表，每个元素是(season, week, features_df, eliminated_name)
        
        Returns:
        --------
        Dict: 训练结果和模型性能
        """
        print("\n开始训练机器学习模型...")
        print("  策略：使用基础方法生成训练标签，然后学习改进...")
        
        # 准备训练数据
        X_list = []
        y_rank_list = []  # 粉丝排名（归一化）
        y_percent_list = []  # 粉丝百分比
        voting_methods = []  # 记录投票方法
        
        for season, week, features_df, eliminated_name in training_data:
            if eliminated_name is None or len(features_df) == 0:
                continue
            
            # 准备特征
            ml_features, _ = self.prepare_ml_features(season, week)
            if len(ml_features) == 0:
                continue
            
            # 选择特征列（排除非特征列）
            exclude_cols = ['celebrity_name', 'ballroompartner', 'celebrity_industry', 
                          'celebrity_homestate', 'celebrity_homecountry/region',
                          'results', 'placement', 'is_eliminated']
            
            # 获取所有数值列
            numeric_cols = [col for col in ml_features.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(ml_features[col])]
            
            # 确保特征列一致
            if not hasattr(self, '_feature_cols'):
                self._feature_cols = numeric_cols
            else:
                # 使用之前确定的特征列，缺失的用0填充
                for col in self._feature_cols:
                    if col not in ml_features.columns:
                        ml_features[col] = 0
            
            X = ml_features[self._feature_cols].values
            
            # 处理NaN值（在训练数据准备阶段）
            if np.isnan(X).any():
                # 使用中位数填充每列的NaN
                for i in range(X.shape[1]):
                    col_data = X[:, i]
                    if np.isnan(col_data).any():
                        median_val = np.nanmedian(col_data)
                        if np.isnan(median_val) or np.isinf(median_val):
                            # 如果中位数也是NaN或Inf，用0填充
                            median_val = 0.0
                        X[:, i] = np.where(np.isnan(col_data), median_val, col_data)
            
            # 最终检查：确保没有NaN或Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 使用基础方法生成"伪标签"
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                base_estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
                if base_estimate:
                    # 归一化排名到[0, 1]
                    n_contestants = len(features_df)
                    fan_ranks = base_estimate['fan_ranks']
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                    y_percent = None
                else:
                    continue
            else:
                base_estimate = super().estimate_fan_votes_percent_method(season, week, features_df)
                if base_estimate:
                    # 百分比已经归一化
                    y_percent = base_estimate['fan_percents'] / 100.0
                    n_contestants = len(features_df)
                    # 从百分比反推排名（近似）
                    fan_ranks = (1 - y_percent).argsort().argsort() + 1
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                else:
                    continue
            
            X_list.append(X)
            y_rank_list.append(y_rank)
            y_percent_list.append(y_percent)
            voting_methods.append(voting_method)
        
        if len(X_list) == 0:
            print("⚠️  没有足够的训练数据")
            return {}
        
        # 合并所有数据（确保所有数组的列数一致）
        if len(X_list) == 0:
            print("⚠️  没有训练数据")
            return {}
        
        # 检查所有数组的列数是否一致（行数可以不同，因为不同周次选手数量不同）
        first_n_features = X_list[0].shape[1]
        inconsistent_count = 0
        
        for i, X_arr in enumerate(X_list):
            if X_arr.shape[1] != first_n_features:
                inconsistent_count += 1
                # 如果列数不一致，填充或截断
                if X_arr.shape[1] < first_n_features:
                    # 列数少，用0填充
                    padding = np.zeros((X_arr.shape[0], first_n_features - X_arr.shape[1]))
                    X_arr = np.hstack([X_arr, padding])
                    X_list[i] = X_arr
                elif X_arr.shape[1] > first_n_features:
                    # 列数多，截断
                    X_arr = X_arr[:, :first_n_features]
                    X_list[i] = X_arr
        
        if inconsistent_count > 0:
            print(f"⚠️  发现 {inconsistent_count} 个数组列数不一致，已自动调整")
        
        X_all = np.vstack(X_list)
        y_rank_all = np.hstack(y_rank_list)
        
        print(f"训练数据形状: {X_all.shape}")
        
        # 检查并处理NaN值（关键步骤！）
        nan_count_X = np.isnan(X_all).sum()
        nan_count_y = np.isnan(y_rank_all).sum()
        
        if nan_count_X > 0 or nan_count_y > 0:
            print(f"⚠️  发现 NaN值: X中有{nan_count_X}个, y中有{nan_count_y}个")
            
            # 处理X中的NaN
            if nan_count_X > 0:
                # 使用中位数填充
                imputer = SimpleImputer(strategy='median')
                X_all = imputer.fit_transform(X_all)
                self.imputer = imputer
                print(f"✓ X中的NaN已用中位数填充")
            else:
                self.imputer = None
            
            # 处理y中的NaN
            if nan_count_y > 0:
                y_rank_all = np.nan_to_num(y_rank_all, nan=0.5)
                print(f"✓ y中的NaN已填充为0.5")
        else:
            self.imputer = None
            print("✓ 未发现NaN值")
        
        # 处理Inf值
        if np.isinf(X_all).any():
            print("⚠️  发现Inf值，正在处理...")
            X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isinf(y_rank_all).any():
            print("⚠️  目标变量中发现Inf值，正在处理...")
            y_rank_all = np.nan_to_num(y_rank_all, nan=0.5, posinf=0.5, neginf=0.5)
        
        print(f"目标变量范围: [{y_rank_all.min():.3f}, {y_rank_all.max():.3f}]")
        
        # 特征缩放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        self.scalers['main'] = scaler
        
        # 最终安全检查：确保没有NaN或Inf
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("⚠️  缩放后仍有NaN/Inf，使用0填充...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(y_rank_all).any() or np.isinf(y_rank_all).any():
            y_rank_all = np.nan_to_num(y_rank_all, nan=0.5, posinf=0.5, neginf=0.5)
        
        # 训练多个模型（预测归一化的粉丝排名）
        models_to_train = {
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                min_samples_split=5,
                random_state=42
            ),
            'ridge': Ridge(alpha=0.5, random_state=42),
            'lasso': Lasso(alpha=0.05, random_state=42, max_iter=2000),
            'elastic_net': ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=2000)
        }
        
        model_scores = {}
        
        for name, model in models_to_train.items():
            print(f"  训练 {name}...", end=' ')
            
            # 训练模型
            model.fit(X_scaled, y_rank_all)
            self.models[name] = model
            
            # 评估模型（使用交叉验证）
            scores = cross_val_score(model, X_scaled, y_rank_all, cv=5, scoring='neg_mean_squared_error')
            model_scores[name] = -scores.mean()
            
            print(f"完成 (MSE: {model_scores[name]:.4f})")
        
        # 创建集成模型（投票回归器）- 只使用最好的3个模型
        print("  创建集成模型...", end=' ')
        # 选择MSE最低的3个模型
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1])[:3]
        ensemble_models = [(name, self.models[name]) for name, _ in sorted_models]
        
        self.models['ensemble'] = VotingRegressor(ensemble_models)
        self.models['ensemble'].fit(X_scaled, y_rank_all)
        
        ensemble_scores = cross_val_score(
            self.models['ensemble'], X_scaled, y_rank_all, 
            cv=5, scoring='neg_mean_squared_error'
        )
        model_scores['ensemble'] = -ensemble_scores.mean()
        print(f"完成 (MSE: {model_scores['ensemble']:.4f})")
        
        # 保存投票方法信息（用于预测时选择）
        self.voting_methods = voting_methods
        
        # 特征重要性（使用随机森林）
        if 'random_forest' in self.models and hasattr(self, '_feature_cols'):
            self.feature_importance = dict(zip(
                self._feature_cols,
                self.models['random_forest'].feature_importances_
            ))
        
        print("✓ 模型训练完成")
        return model_scores
    
    def predict_fan_votes_ml(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame,
        eliminated_name: Optional[str]
    ) -> Dict:
        """
        使用机器学习模型预测粉丝投票
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        features_df : pd.DataFrame
            特征数据框
        eliminated_name : Optional[str]
            被淘汰选手名称
        
        Returns:
        --------
        Dict: 包含估计的粉丝投票和相关信息
        """
        if len(features_df) == 0 or eliminated_name is None:
            return None
        
        # 准备特征
        ml_features, _ = self.prepare_ml_features(season, week)
        if len(ml_features) == 0:
            return None
        
        # 使用训练时确定的特征列
        if not hasattr(self, '_feature_cols'):
            # 如果没有训练过，使用基础方法
            return super().estimate_fan_votes_rank_method(season, week, features_df) if self.determine_voting_method(season) == 'rank' else super().estimate_fan_votes_percent_method(season, week, features_df)
        
        # 确保所有特征列都存在
        for col in self._feature_cols:
            if col not in ml_features.columns:
                ml_features[col] = 0
        
        X = ml_features[self._feature_cols].values
        
        # 处理NaN值（使用训练时的imputer或直接填充）
        if np.isnan(X).any():
            if hasattr(self, 'imputer') and self.imputer is not None:
                X = self.imputer.transform(X)
            else:
                # 如果没有imputer，使用中位数填充每列
                for i in range(X.shape[1]):
                    col_data = X[:, i]
                    if np.isnan(col_data).any():
                        median_val = np.nanmedian(col_data)
                        if np.isnan(median_val):
                            median_val = 0.0  # 如果整列都是NaN，用0填充
                        X[:, i] = np.where(np.isnan(col_data), median_val, col_data)
        
        # 最终检查：确保没有NaN
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        # 特征缩放
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # 使用集成模型预测归一化的粉丝排名 [0, 1]
        if 'ensemble' in self.models:
            predicted_rank_norm = self.models['ensemble'].predict(X_scaled)
            # 确保在[0, 1]范围内
            predicted_rank_norm = np.clip(predicted_rank_norm, 0, 1)
        else:
            # 如果没有训练好的模型，使用基础方法
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super().estimate_fan_votes_rank_method(season, week, features_df)
            else:
                return super().estimate_fan_votes_percent_method(season, week, features_df)
        
        # 将预测的归一化排名转换为实际排名
        n_contestants = len(features_df)
        voting_method = self.determine_voting_method(season)
        
        # 从归一化排名转换为实际排名
        # predicted_rank_norm 越大，排名越高（越不受欢迎）
        fan_ranks = (predicted_rank_norm * (n_contestants - 1) + 1).astype(int)
        fan_ranks = np.clip(fan_ranks, 1, n_contestants)
        
        # 找到被淘汰选手的索引
        eliminated_mask = ml_features['celebrity_name'] == eliminated_name
        eliminated_idx = np.where(eliminated_mask)[0]
        
        if len(eliminated_idx) == 0:
            # 如果找不到被淘汰选手，尝试在features_df中查找
            eliminated_mask_df = features_df['celebrity_name'] == eliminated_name
            if eliminated_mask_df.any():
                eliminated_idx = np.where(eliminated_mask_df)[0]
            else:
                # 如果还是找不到，使用基础方法
                if voting_method == 'rank':
                    return super().estimate_fan_votes_rank_method(season, week, features_df)
                else:
                    return super().estimate_fan_votes_percent_method(season, week, features_df)
        
        eliminated_idx = eliminated_idx[0]
        
        # 关键改进：使用优化算法微调，确保约束条件满足
        # 先使用ML预测作为初始值，然后用优化算法调整
        if voting_method == 'rank':
            # 排名法：综合排名 = 评委排名 + 粉丝排名
            judge_ranks = features_df['judge_rank'].values
            
            # 优化目标：最小化与ML预测的差异，同时满足约束
            def objective(fan_ranks_opt):
                # 惩罚与ML预测的差异
                ml_penalty = np.sum((fan_ranks_opt - fan_ranks) ** 2)
                # 惩罚与评委排名的差异（粉丝投票应该与评委评分相关）
                judge_penalty = np.sum((fan_ranks_opt - judge_ranks) ** 2) * 0.3
                return ml_penalty + judge_penalty
            
            def constraint_eliminated(fan_ranks_opt):
                # 约束：被淘汰选手的综合排名应该最高
                combined_ranks = judge_ranks + fan_ranks_opt
                eliminated_combined = combined_ranks[eliminated_idx]
                return eliminated_combined - np.max(combined_ranks)  # 应该 >= 0
            
            bounds = [(1, n_contestants)] * n_contestants
            x0 = fan_ranks.copy().astype(float)
            
            # 确保初始值满足约束（如果不满足，调整）
            combined_ranks_init = judge_ranks + x0
            if combined_ranks_init[eliminated_idx] < np.max(combined_ranks_init):
                # 调整被淘汰选手的粉丝排名，使其综合排名最高
                max_combined = np.max(combined_ranks_init)
                needed_rank = max_combined - judge_ranks[eliminated_idx] + 1
                x0[eliminated_idx] = min(needed_rank, n_contestants)
            
            constraints = [{'type': 'ineq', 'fun': constraint_eliminated}]
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                fan_ranks_final = result.x.astype(int)
                fan_ranks_final = np.clip(fan_ranks_final, 1, n_contestants)
            else:
                # 如果优化失败，使用调整后的初始值
                fan_ranks_final = x0.astype(int)
            
            # 转换为投票数
            fan_votes = (n_contestants + 1 - fan_ranks_final) * 1000
            combined_ranks = judge_ranks + fan_ranks_final
            
            return {
                'fan_ranks': fan_ranks_final,
                'fan_votes': fan_votes,
                'combined_ranks': combined_ranks,
                'eliminated_idx': eliminated_idx,
                'method': 'ml_ensemble_rank_optimized'
            }
        else:
            # 百分比法：综合百分比 = 评委百分比 + 粉丝百分比
            judge_percents = features_df['judge_percent'].values
            
            # 从排名转换为百分比（近似）
            # 排名低的（受欢迎）百分比高
            fan_percents_ml = (n_contestants + 1 - fan_ranks) / np.sum(n_contestants + 1 - fan_ranks) * 100
            
            # 优化目标
            def objective(fan_percents_opt):
                # 惩罚与ML预测的差异
                ml_penalty = np.sum((fan_percents_opt - fan_percents_ml) ** 2)
                # 惩罚与评委百分比的差异
                judge_penalty = np.sum((fan_percents_opt - judge_percents) ** 2) * 0.2
                return ml_penalty + judge_penalty
            
            def constraint_sum(fan_percents_opt):
                return np.sum(fan_percents_opt) - 100.0
            
            def constraint_eliminated(fan_percents_opt):
                # 约束：被淘汰选手的综合百分比应该最低
                combined_percents = judge_percents + fan_percents_opt
                eliminated_combined = combined_percents[eliminated_idx]
                return eliminated_combined - np.min(combined_percents)  # 应该 <= 0
            
            bounds = [(0, 100)] * n_contestants
            x0 = fan_percents_ml.copy()
            x0 = x0 / np.sum(x0) * 100  # 归一化
            
            # 确保初始值满足约束
            combined_percents_init = judge_percents + x0
            if combined_percents_init[eliminated_idx] > np.min(combined_percents_init):
                # 调整被淘汰选手的百分比
                min_combined = np.min(combined_percents_init)
                needed_percent = max(0, min_combined - judge_percents[eliminated_idx] - 1)
                x0[eliminated_idx] = max(needed_percent, 1)
                x0 = x0 / np.sum(x0) * 100
            
            constraints = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_eliminated}
            ]
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                fan_percents_final = result.x
                fan_percents_final = np.maximum(fan_percents_final, 0)
                fan_percents_final = fan_percents_final / np.sum(fan_percents_final) * 100
            else:
                fan_percents_final = x0
            
            # 转换为投票数
            total_votes = 10_000_000
            fan_votes = fan_percents_final / 100 * total_votes
            combined_percents = judge_percents + fan_percents_final
            
            return {
                'fan_percents': fan_percents_final,
                'fan_votes': fan_votes,
                'combined_percents': combined_percents,
                'eliminated_idx': eliminated_idx,
                'method': 'ml_ensemble_percent_optimized'
            }
    
    def estimate_all_weeks_ml(
        self,
        seasons: Optional[List[int]] = None,
        train_on_all: bool = True
    ) -> pd.DataFrame:
        """
        使用机器学习方法估计所有周次的粉丝投票
        
        Parameters:
        -----------
        seasons : Optional[List[int]]
            要处理的季数列表
        train_on_all : bool
            是否在所有数据上训练（True）或使用交叉验证（False）
        
        Returns:
        --------
        pd.DataFrame: 包含估计结果的 DataFrame
        """
        if seasons is None:
            seasons = sorted(self.df['season'].unique())
        
        # 准备训练数据
        print("\n准备训练数据...")
        training_data = []
        
        for season in seasons:
            season_data = self.df[self.df['season'] == season]
            total_cols = [col for col in season_data.columns if '_total_score' in col]
            weeks = []
            for col in total_cols:
                week_num = self._extract_week_number(col)
                if week_num:
                    weeks.append(week_num)
            weeks = sorted(set(weeks))
            
            for week in weeks:
                features_df = self.extract_features(season, week)
                if len(features_df) == 0:
                    continue
                
                eliminated_name = self.get_eliminated_contestant(season, week)
                if eliminated_name is None:
                    continue
                
                # 只保存必要的数据，避免内存问题
                training_data.append((season, week, features_df.copy(), eliminated_name))
        
        print(f"收集到 {len(training_data)} 个训练样本")
        
        # 训练模型
        if train_on_all:
            # 在所有数据上训练
            model_scores = self.train_models(training_data)
        else:
            # 使用交叉验证：用前N-1季训练，预测第N季
            print("使用交叉验证模式...")
            # 简化：使用前80%的数据训练，预测后20%
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            test_data = training_data[split_idx:]
            
            model_scores = self.train_models(train_data)
            print(f"训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")
        
        # 估计所有周次
        print("\n开始估计粉丝投票...")
        results = []
        
        for season in seasons:
            print(f"处理第 {season} 季...")
            season_data = self.df[self.df['season'] == season]
            voting_method = self.determine_voting_method(season)
            
            total_cols = [col for col in season_data.columns if '_total_score' in col]
            weeks = []
            for col in total_cols:
                week_num = self._extract_week_number(col)
                if week_num:
                    weeks.append(week_num)
            weeks = sorted(set(weeks))
            
            for week in weeks:
                print(f"  周 {week}...", end=' ')
                
                features_df = self.extract_features(season, week)
                if len(features_df) == 0:
                    print("跳过（无数据）")
                    continue
                
                eliminated_name = self.get_eliminated_contestant(season, week)
                
                # 使用ML方法估计
                estimate = self.predict_fan_votes_ml(season, week, features_df, eliminated_name)
                
                if estimate is None:
                    # 如果ML方法失败，回退到基础方法
                    if voting_method == 'rank':
                        estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
                    else:
                        estimate = super().estimate_fan_votes_percent_method(season, week, features_df)
                
                # 再次检查estimate是否为None（基础方法也可能失败）
                if estimate is None:
                    print(f"跳过（无法估计）", end=' ')
                    continue
                
                # 确保estimate中的数组长度与features_df一致
                n_contestants = len(features_df)
                if 'fan_votes' not in estimate:
                    print(f"跳过（缺少fan_votes）", end=' ')
                    continue
                
                if len(estimate['fan_votes']) != n_contestants:
                    print(f"跳过（数组长度不匹配: {len(estimate['fan_votes'])} vs {n_contestants}）", end=' ')
                    continue
                
                # 保存结果
                for i, (idx, row) in enumerate(features_df.iterrows()):
                    # 确保索引在范围内
                    if i >= len(estimate['fan_votes']):
                        # 如果索引超出范围，使用最后一个值
                        fan_votes_val = estimate['fan_votes'][-1] if len(estimate['fan_votes']) > 0 else 0
                        eliminated_flag = False
                    else:
                        fan_votes_val = estimate['fan_votes'][i]
                        eliminated_flag = (i == estimate.get('eliminated_idx', -1))
                    
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': row['celebrity_name'],
                        'fan_votes': fan_votes_val,
                        'judge_total': row['judge_total'],
                        'voting_method': voting_method,
                        'eliminated': eliminated_flag,
                        'estimation_method': estimate.get('method', 'fallback')
                    })
                
                print("完成")
        
        return pd.DataFrame(results)


def main():
    """主函数：运行机器学习版本的粉丝投票估计"""
    from loader import load_data
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 机器学习版本（集成学习）")
    print("=" * 70)
    
    # 加载数据
    try:
        df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print("✓ 加载处理后的数据成功")
    except FileNotFoundError:
        print("⚠️  未找到处理后的数据，使用原始数据...")
        from preprocess_dwts import DWTSDataPreprocessor
        from loader import load_data
        
        raw_df = load_data()
        preprocessor = DWTSDataPreprocessor(raw_df)
        preprocessor.calculate_weekly_scores_and_ranks()
        df = preprocessor.get_processed_data()
    
    # 创建ML估计器
    estimator = MLFanVoteEstimator(df)
    
    # 估计所有周次（使用ML方法）
    print("\n开始估计粉丝投票（使用机器学习集成方法）...")
    estimates_df = estimator.estimate_all_weeks_ml(train_on_all=True)
    
    # 保存估计结果
    estimates_df.to_csv('fan_vote_estimates_ml.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 估计结果已保存到: fan_vote_estimates_ml.csv")
    
    # 验证模型
    validation_results = estimator.validate_estimates(estimates_df)
    
    # 保存验证结果
    import json
    with open('validation_results_ml.json', 'w', encoding='utf-8') as f:
        # 转换numpy类型
        def convert_to_native(obj):
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
        json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 验证结果已保存到: validation_results_ml.json")
    
    # 显示特征重要性
    if estimator.feature_importance:
        print("\n特征重要性（Top 10）:")
        sorted_features = sorted(
            estimator.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
    
    print("\n" + "=" * 70)
    print("阶段2完成（机器学习版本）！")
    print("=" * 70)
    
    return estimator, estimates_df, validation_results


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
