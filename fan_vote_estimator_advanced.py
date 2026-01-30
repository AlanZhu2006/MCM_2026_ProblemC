"""
阶段2：粉丝投票估计模型 - 高级版本
使用更先进的模型和特征工程方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 尝试导入高级模型
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost未安装，将使用基础模型")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM未安装，将使用基础模型")

# 基础模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fan_vote_estimator import FanVoteEstimator


class AdvancedFanVoteEstimator(FanVoteEstimator):
    """高级粉丝投票估计器 - 使用更先进的模型和特征工程"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化高级估计器
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据
        """
        super().__init__(df)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.best_model_name = None
        
    def create_advanced_features(self, season: int, week: int) -> pd.DataFrame:
        """
        创建高级特征（增强版特征工程）
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        pd.DataFrame: 包含高级特征的 DataFrame
        """
        # 使用基础特征提取
        features_df = self.extract_features(season, week)
        
        if len(features_df) == 0:
            return pd.DataFrame()
        
        # 获取被淘汰选手
        eliminated_name = self.get_eliminated_contestant(season, week)
        
        ml_features = features_df.copy()
        
        # === 基础特征（已有） ===
        # judge_total, judge_rank, judge_percent
        # avg_historical_score, avg_historical_rank
        
        # === 高级特征1：时间序列特征 ===
        if week > 1:
            # 计算趋势（斜率）
            prev_weeks = []
            for prev_week in range(max(1, week-3), week):  # 最近3周
                prev_total_col = f'week{prev_week}_total_score'
                if prev_total_col in ml_features.columns:
                    prev_weeks.append(prev_total_col)
            
            if prev_weeks:
                # 评分趋势（线性回归斜率）
                scores_history = ml_features[prev_weeks].values
                if len(prev_weeks) >= 2:
                    # 计算斜率
                    x = np.arange(len(prev_weeks))
                    slopes = []
                    for row in scores_history:
                        if not np.isnan(row).all():
                            valid_mask = ~np.isnan(row)
                            if valid_mask.sum() >= 2:
                                slope = np.polyfit(x[valid_mask], row[valid_mask], 1)[0]
                                slopes.append(slope)
                            else:
                                slopes.append(0)
                        else:
                            slopes.append(0)
                    ml_features['score_trend'] = slopes
                else:
                    ml_features['score_trend'] = 0
                
                # 评分波动性（标准差）
                ml_features['score_volatility'] = ml_features[prev_weeks].std(axis=1).fillna(0)
            else:
                ml_features['score_trend'] = 0
                ml_features['score_volatility'] = 0
        else:
            ml_features['score_trend'] = 0
            ml_features['score_volatility'] = 0
        
        # === 高级特征2：相对排名特征 ===
        n_contestants = len(ml_features)
        if n_contestants > 1:
            ml_features['judge_rank_normalized'] = (ml_features['judge_rank'] - 1) / (n_contestants - 1)
            ml_features['rank_percentile'] = ml_features['judge_rank'] / n_contestants
        else:
            ml_features['judge_rank_normalized'] = 0.5
            ml_features['rank_percentile'] = 0.5
        
        # === 高级特征3：与平均值的差异 ===
        mean_score = ml_features['judge_total'].mean()
        ml_features['score_vs_mean'] = ml_features['judge_total'] - mean_score
        ml_features['score_vs_mean_pct'] = (ml_features['judge_total'] - mean_score) / mean_score if mean_score > 0 else 0
        
        # === 高级特征4：选手特征增强 ===
        if 'celebrity_age_during_season' in ml_features.columns:
            ml_features['age'] = ml_features['celebrity_age_during_season'].fillna(
                ml_features['celebrity_age_during_season'].median()
            )
            # 年龄分组（更细粒度）
            ml_features['age_group_fine'] = pd.cut(
                ml_features['age'],
                bins=[0, 20, 25, 30, 35, 40, 50, 100],
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(float).fillna(3)
        else:
            ml_features['age'] = 0
            ml_features['age_group_fine'] = 0
        
        # 行业编码（如果存在）
        if 'celebrity_industry' in ml_features.columns:
            if 'industry_encoder' not in self.label_encoders:
                self.label_encoders['industry_encoder'] = LabelEncoder()
                ml_features['industry_encoded'] = self.label_encoders['industry_encoder'].fit_transform(
                    ml_features['celebrity_industry'].astype(str)
                )
            else:
                try:
                    ml_features['industry_encoded'] = self.label_encoders['industry_encoder'].transform(
                        ml_features['celebrity_industry'].astype(str)
                    )
                except ValueError:
                    # 新类别，使用最常见的编码
                    ml_features['industry_encoded'] = 0
        
        # === 高级特征5：专业舞者特征增强 ===
        if 'ballroompartner' in ml_features.columns:
            # 计算专业舞者的详细统计
            partner_stats = self._calculate_advanced_partner_stats(season, week)
            ml_features = ml_features.merge(
                partner_stats,
                on='ballroompartner',
                how='left'
            )
            # 填充缺失值
            if 'partner_win_rate' in ml_features.columns:
                ml_features['partner_win_rate'] = ml_features['partner_win_rate'].fillna(0.5)
            if 'partner_avg_placement' in ml_features.columns:
                ml_features['partner_avg_placement'] = ml_features['partner_avg_placement'].fillna(10)
        
        # === 高级特征6：周次和竞争环境特征 ===
        ml_features['week_number'] = week
        ml_features['contestants_remaining'] = n_contestants
        ml_features['week_ratio'] = week / max(n_contestants, 1)
        ml_features['competition_intensity'] = n_contestants / 15.0  # 归一化到[0,1]
        
        # === 高级特征7：交互特征 ===
        if 'age' in ml_features.columns and 'judge_total' in ml_features.columns:
            ml_features['age_score_interaction'] = ml_features['age'] * ml_features['judge_total'] / 100
        
        if 'judge_rank' in ml_features.columns and 'contestants_remaining' in ml_features.columns:
            ml_features['rank_competition_interaction'] = ml_features['judge_rank'] / ml_features['contestants_remaining']
        
        # === 高级特征8：是否被淘汰（标签） ===
        ml_features['is_eliminated'] = (ml_features['celebrity_name'] == eliminated_name).astype(int)
        
        # === 填充所有缺失值 ===
        numeric_cols = ml_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if ml_features[col].isna().any():
                median_val = ml_features[col].median()
                if pd.isna(median_val):
                    ml_features[col] = ml_features[col].fillna(0.0)
                else:
                    ml_features[col] = ml_features[col].fillna(median_val)
        
        return ml_features
    
    def _calculate_advanced_partner_stats(self, season: int, week: int) -> pd.DataFrame:
        """
        计算专业舞者的高级统计数据
        
        Returns:
        --------
        pd.DataFrame: 包含专业舞者详细统计的 DataFrame
        """
        # 计算每个专业舞者的历史表现
        partner_stats = self.df.groupby('ballroompartner').agg({
            'placement': ['mean', 'min', 'count'],  # 平均排名、最佳排名、参与次数
        }).reset_index()
        
        partner_stats.columns = ['ballroompartner', 'partner_avg_placement', 'partner_best_placement', 'partner_experience']
        
        # 计算胜率（排名<=3的比例）
        partner_wins = self.df[self.df['placement'] <= 3].groupby('ballroompartner').size()
        partner_total = self.df.groupby('ballroompartner').size()
        partner_stats['partner_win_rate'] = partner_stats['ballroompartner'].map(
            lambda x: partner_wins.get(x, 0) / partner_total.get(x, 1)
        ).fillna(0.5)
        
        return partner_stats
    
    def train_advanced_models(self, training_data: List[Tuple[int, int, pd.DataFrame, Optional[str]]]) -> Dict:
        """
        训练高级模型（包括XGBoost、LightGBM等）
        
        Parameters:
        -----------
        training_data : List[Tuple[int, int, pd.DataFrame, Optional[str]]]
            训练数据列表
        
        Returns:
        --------
        Dict: 训练结果和模型性能
        """
        print("\n开始训练高级模型...")
        print("  策略：使用基础方法生成训练标签，然后使用高级模型学习...")
        
        # 准备训练数据
        X_list = []
        y_rank_list = []
        voting_methods = []
        
        for season, week, features_df, eliminated_name in training_data:
            if eliminated_name is None or len(features_df) == 0:
                continue
            
            # 准备高级特征
            ml_features = self.create_advanced_features(season, week)
            if len(ml_features) == 0:
                continue
            
            # 选择特征列
            exclude_cols = ['celebrity_name', 'ballroompartner', 'celebrity_industry', 
                          'celebrity_homestate', 'celebrity_homecountry/region',
                          'results', 'placement', 'is_eliminated']
            
            numeric_cols = [col for col in ml_features.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(ml_features[col])]
            
            if not hasattr(self, '_feature_cols'):
                self._feature_cols = numeric_cols
            else:
                for col in self._feature_cols:
                    if col not in ml_features.columns:
                        ml_features[col] = 0
            
            X = ml_features[self._feature_cols].values
            
            # 处理NaN
            if np.isnan(X).any():
                for i in range(X.shape[1]):
                    col_data = X[:, i]
                    if np.isnan(col_data).any():
                        median_val = np.nanmedian(col_data)
                        if np.isnan(median_val) or np.isinf(median_val):
                            median_val = 0.0
                        X[:, i] = np.where(np.isnan(col_data), median_val, col_data)
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 使用基础方法生成标签
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                base_estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
                if base_estimate:
                    n_contestants = len(features_df)
                    fan_ranks = base_estimate['fan_ranks']
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                else:
                    continue
            else:
                base_estimate = super().estimate_fan_votes_percent_method(season, week, features_df)
                if base_estimate:
                    y_percent = base_estimate['fan_percents'] / 100.0
                    n_contestants = len(features_df)
                    fan_ranks = (1 - y_percent).argsort().argsort() + 1
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                else:
                    continue
            
            X_list.append(X)
            y_rank_list.append(y_rank)
            voting_methods.append(voting_method)
        
        if len(X_list) == 0:
            print("⚠️  没有足够的训练数据")
            return {}
        
        # 合并数据（只检查列数）
        first_n_features = X_list[0].shape[1]
        for i, X_arr in enumerate(X_list):
            if X_arr.shape[1] != first_n_features:
                if X_arr.shape[1] < first_n_features:
                    padding = np.zeros((X_arr.shape[0], first_n_features - X_arr.shape[1]))
                    X_arr = np.hstack([X_arr, padding])
                    X_list[i] = X_arr
                else:
                    X_arr = X_arr[:, :first_n_features]
                    X_list[i] = X_arr
        
        X_all = np.vstack(X_list)
        y_rank_all = np.hstack(y_rank_list)
        
        print(f"训练数据形状: {X_all.shape}")
        
        # 处理NaN和Inf
        nan_count = np.isnan(X_all).sum() + np.isnan(y_rank_all).sum()
        if nan_count > 0:
            imputer = SimpleImputer(strategy='median')
            X_all = imputer.fit_transform(X_all)
            self.imputer = imputer
            y_rank_all = np.nan_to_num(y_rank_all, nan=0.5)
        else:
            self.imputer = None
        
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        y_rank_all = np.nan_to_num(y_rank_all, nan=0.5, posinf=0.5, neginf=0.5)
        
        print(f"目标变量范围: [{y_rank_all.min():.3f}, {y_rank_all.max():.3f}]")
        
        # 特征缩放（使用RobustScaler，对异常值更鲁棒）
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_all)
        self.scalers['main'] = scaler
        
        # 训练多个模型
        models_to_train = {}
        
        # 基础模型
        models_to_train['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models_to_train['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            min_samples_split=5,
            random_state=42
        )
        
        # 高级模型（如果可用）
        if HAS_XGBOOST:
            models_to_train['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        if HAS_LIGHTGBM:
            models_to_train['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # 线性模型
        models_to_train['ridge'] = Ridge(alpha=0.5, random_state=42)
        models_to_train['elastic_net'] = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=2000)
        
        model_scores = {}
        
        # 使用时间序列交叉验证（因为数据有时间顺序）
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models_to_train.items():
            print(f"  训练 {name}...", end=' ')
            
            try:
                model.fit(X_scaled, y_rank_all)
                self.models[name] = model
                
                # 评估模型
                scores = cross_val_score(model, X_scaled, y_rank_all, cv=tscv, scoring='neg_mean_squared_error')
                model_scores[name] = -scores.mean()
                
                print(f"完成 (MSE: {model_scores[name]:.4f})")
            except Exception as e:
                print(f"失败: {str(e)}")
                continue
        
        # 选择最佳模型进行集成
        if len(model_scores) > 0:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])[:3]
            ensemble_models = [(name, self.models[name]) for name, _ in sorted_models if name in self.models]
            
            if len(ensemble_models) >= 2:
                print("  创建集成模型...", end=' ')
                self.models['ensemble'] = VotingRegressor(ensemble_models)
                self.models['ensemble'].fit(X_scaled, y_rank_all)
                
                ensemble_scores = cross_val_score(
                    self.models['ensemble'], X_scaled, y_rank_all,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                model_scores['ensemble'] = -ensemble_scores.mean()
                print(f"完成 (MSE: {model_scores['ensemble']:.4f})")
                
                # 选择最佳模型
                self.best_model_name = min(model_scores.items(), key=lambda x: x[1])[0]
                print(f"  最佳模型: {self.best_model_name} (MSE: {model_scores[self.best_model_name]:.4f})")
        
        # 特征重要性
        if 'random_forest' in self.models and hasattr(self, '_feature_cols'):
            self.feature_importance = dict(zip(
                self._feature_cols,
                self.models['random_forest'].feature_importances_
            ))
        
        print("✓ 模型训练完成")
        return model_scores
    
    def predict_fan_votes_advanced(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame,
        eliminated_name: Optional[str]
    ) -> Dict:
        """
        使用高级模型预测粉丝投票
        
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
        
        # 准备高级特征
        ml_features = self.create_advanced_features(season, week)
        if len(ml_features) == 0:
            return None
        
        # 使用训练时确定的特征列
        if not hasattr(self, '_feature_cols'):
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super().estimate_fan_votes_rank_method(season, week, features_df)
            else:
                return super().estimate_fan_votes_percent_method(season, week, features_df)
        
        # 确保所有特征列都存在
        for col in self._feature_cols:
            if col not in ml_features.columns:
                ml_features[col] = 0
        
        X = ml_features[self._feature_cols].values
        
        # 处理NaN
        if np.isnan(X).any():
            if hasattr(self, 'imputer') and self.imputer is not None:
                X = self.imputer.transform(X)
            else:
                X = np.nan_to_num(X, nan=0.0)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 特征缩放
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # 使用最佳模型或集成模型预测
        if self.best_model_name and self.best_model_name in self.models:
            predicted_rank_norm = self.models[self.best_model_name].predict(X_scaled)
        elif 'ensemble' in self.models:
            predicted_rank_norm = self.models['ensemble'].predict(X_scaled)
        else:
            # 回退到基础方法
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super().estimate_fan_votes_rank_method(season, week, features_df)
            else:
                return super().estimate_fan_votes_percent_method(season, week, features_df)
        
        predicted_rank_norm = np.clip(predicted_rank_norm, 0, 1)
        
        # 转换为粉丝投票（使用与ML版本相同的优化逻辑）
        n_contestants = len(features_df)
        voting_method = self.determine_voting_method(season)
        
        fan_ranks = (predicted_rank_norm * (n_contestants - 1) + 1).astype(int)
        fan_ranks = np.clip(fan_ranks, 1, n_contestants)
        
        # 找到被淘汰选手
        eliminated_mask = ml_features['celebrity_name'] == eliminated_name
        eliminated_idx = np.where(eliminated_mask)[0]
        if len(eliminated_idx) == 0:
            eliminated_mask_df = features_df['celebrity_name'] == eliminated_name
            if eliminated_mask_df.any():
                eliminated_idx = np.where(eliminated_mask_df)[0]
            else:
                voting_method = self.determine_voting_method(season)
                if voting_method == 'rank':
                    return super().estimate_fan_votes_rank_method(season, week, features_df)
                else:
                    return super().estimate_fan_votes_percent_method(season, week, features_df)
        
        eliminated_idx = eliminated_idx[0]
        
        # 使用优化算法微调（与ML版本相同）
        if voting_method == 'rank':
            judge_ranks = features_df['judge_rank'].values
            
            def objective(fan_ranks_opt):
                ml_penalty = np.sum((fan_ranks_opt - fan_ranks) ** 2)
                judge_penalty = np.sum((fan_ranks_opt - judge_ranks) ** 2) * 0.3
                return ml_penalty + judge_penalty
            
            def constraint_eliminated(fan_ranks_opt):
                combined_ranks = judge_ranks + fan_ranks_opt
                eliminated_combined = combined_ranks[eliminated_idx]
                return eliminated_combined - np.max(combined_ranks)
            
            bounds = [(1, n_contestants)] * n_contestants
            x0 = fan_ranks.copy().astype(float)
            
            combined_ranks_init = judge_ranks + x0
            if combined_ranks_init[eliminated_idx] < np.max(combined_ranks_init):
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
                fan_ranks_final = x0.astype(int)
            
            fan_votes = (n_contestants + 1 - fan_ranks_final) * 1000
            combined_ranks = judge_ranks + fan_ranks_final
            
            return {
                'fan_ranks': fan_ranks_final,
                'fan_votes': fan_votes,
                'combined_ranks': combined_ranks,
                'eliminated_idx': eliminated_idx,
                'method': 'advanced_ensemble_rank'
            }
        else:
            judge_percents = features_df['judge_percent'].values
            fan_percents_ml = (n_contestants + 1 - fan_ranks) / np.sum(n_contestants + 1 - fan_ranks) * 100
            
            def objective(fan_percents_opt):
                ml_penalty = np.sum((fan_percents_opt - fan_percents_ml) ** 2)
                judge_penalty = np.sum((fan_percents_opt - judge_percents) ** 2) * 0.2
                return ml_penalty + judge_penalty
            
            def constraint_sum(fan_percents_opt):
                return np.sum(fan_percents_opt) - 100.0
            
            def constraint_eliminated(fan_percents_opt):
                combined_percents = judge_percents + fan_percents_opt
                eliminated_combined = combined_percents[eliminated_idx]
                return eliminated_combined - np.min(combined_percents)
            
            bounds = [(0, 100)] * n_contestants
            x0 = fan_percents_ml.copy()
            x0 = x0 / np.sum(x0) * 100
            
            combined_percents_init = judge_percents + x0
            if combined_percents_init[eliminated_idx] > np.min(combined_percents_init):
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
            
            total_votes = 10_000_000
            fan_votes = fan_percents_final / 100 * total_votes
            combined_percents = judge_percents + fan_percents_final
            
            return {
                'fan_percents': fan_percents_final,
                'fan_votes': fan_votes,
                'combined_percents': combined_percents,
                'eliminated_idx': eliminated_idx,
                'method': 'advanced_ensemble_percent'
            }
    
    def estimate_all_weeks_advanced(
        self,
        seasons: Optional[List[int]] = None,
        train_on_all: bool = True
    ) -> pd.DataFrame:
        """
        使用高级方法估计所有周次的粉丝投票
        
        Parameters:
        -----------
        seasons : Optional[List[int]]
            要处理的季数列表
        train_on_all : bool
            是否在所有数据上训练
        
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
                
                training_data.append((season, week, features_df.copy(), eliminated_name))
        
        print(f"收集到 {len(training_data)} 个训练样本")
        
        # 训练模型
        if train_on_all:
            model_scores = self.train_advanced_models(training_data)
        else:
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            test_data = training_data[split_idx:]
            model_scores = self.train_advanced_models(train_data)
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
                
                # 使用高级方法估计
                estimate = self.predict_fan_votes_advanced(season, week, features_df, eliminated_name)
                
                if estimate is None:
                    # 回退到基础方法
                    if voting_method == 'rank':
                        estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
                    else:
                        estimate = super().estimate_fan_votes_percent_method(season, week, features_df)
                
                if estimate is None:
                    print(f"跳过（无法估计）", end=' ')
                    continue
                
                if 'fan_votes' not in estimate or len(estimate['fan_votes']) != len(features_df):
                    print(f"跳过（数组长度不匹配）", end=' ')
                    continue
                
                # 保存结果
                for i, (idx, row) in enumerate(features_df.iterrows()):
                    if i >= len(estimate['fan_votes']):
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
    """主函数：运行高级版本的粉丝投票估计"""
    from loader import load_data
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 高级版本（XGBoost/LightGBM + 高级特征）")
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
    
    # 创建高级估计器
    estimator = AdvancedFanVoteEstimator(df)
    
    # 估计所有周次
    print("\n开始估计粉丝投票（使用高级方法）...")
    estimates_df = estimator.estimate_all_weeks_advanced(train_on_all=True)
    
    # 保存估计结果
    estimates_df.to_csv('fan_vote_estimates_advanced.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 估计结果已保存到: fan_vote_estimates_advanced.csv")
    
    # 验证模型
    validation_results = estimator.validate_estimates(estimates_df)
    
    # 保存验证结果
    import json
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
    
    with open('validation_results_advanced.json', 'w', encoding='utf-8') as f:
        validation_results_native = convert_to_native(validation_results)
        json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 验证结果已保存到: validation_results_advanced.json")
    
    # 显示特征重要性
    if estimator.feature_importance:
        print("\n特征重要性（Top 15）:")
        sorted_features = sorted(
            estimator.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {feature:35s}: {importance:.4f}")
    
    print("\n" + "=" * 70)
    print("阶段2完成（高级版本）！")
    print("=" * 70)
    
    return estimator, estimates_df, validation_results


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
