"""
阶段2：粉丝投票估计模型 - 增强版本
整合2024年MCM C题的方法 + 智能NaN处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import norm
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

# 尝试导入状态空间模型相关
try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False

# 基础模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 尝试导入ARIMA模型（借鉴2301192的方法）
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    print("⚠️  statsmodels未安装，ARIMA功能将不可用")
    print("   安装命令: pip install statsmodels")

# 尝试导入TabNet（表格数据专用模型）
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("⚠️  pytorch-tabnet未安装，TabNet功能将不可用")
    print("   安装命令: pip install pytorch-tabnet")

# 尝试导入CatBoost（另一个强大的表格数据模型）
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost未安装，将使用其他模型")
    print("   安装命令: pip install catboost")

from fan_vote_estimator_advanced import AdvancedFanVoteEstimator


class EnhancedFanVoteEstimator(AdvancedFanVoteEstimator):
    """
    增强版粉丝投票估计器
    
    改进点：
    1. 智能NaN处理（缺失值指示器、时间序列插值、KNN插值）
    2. 集成状态空间模型思想（卡尔曼滤波处理缺失值）
    3. 使用XGBoost/LightGBM的内置缺失值处理
    4. MCMC风格的不确定性量化
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化增强估计器
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据
        """
        super().__init__(df)
        self.missing_indicators = {}  # 存储缺失值模式
        self.kalman_filters = {}  # 用于处理缺失值的卡尔曼滤波器
        self.imputation_strategies = {}  # 存储插值策略
        self.kmeans_models = {}  # 存储K-means聚类模型（用于NaN处理）
        self.arima_models = {}  # 存储ARIMA模型（用于时间序列预测）
        
    def create_missing_value_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建缺失值相关特征（借鉴2024年C题的信息熵思想）
        
        思想：缺失值本身可能包含信息
        - 缺失值比例（信息缺失度）
        - 缺失值模式（哪些特征缺失）
        - 缺失值位置（时间序列中的位置）
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            原始特征数据框
        
        Returns:
        --------
        pd.DataFrame: 添加了缺失值特征的 DataFrame
        """
        enhanced_features = features_df.copy()
        
        # 1. 缺失值指示器（Missing Indicator）
        # 为每个可能有缺失值的特征创建指示器
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if features_df[col].isna().any():
                # 创建缺失值指示器
                indicator_col = f'{col}_is_missing'
                enhanced_features[indicator_col] = features_df[col].isna().astype(int)
        
        # 2. 缺失值比例（信息缺失度，类似信息熵）
        # 计算每个样本的缺失值比例
        missing_ratio = features_df[numeric_cols].isna().sum(axis=1) / len(numeric_cols)
        enhanced_features['missing_ratio'] = missing_ratio
        enhanced_features['missing_entropy'] = -missing_ratio * np.log(missing_ratio + 1e-10)  # 信息熵
        
        # 3. 缺失值模式（聚类特征）
        # 将缺失值模式编码为特征
        missing_pattern = features_df[numeric_cols].isna().astype(int)
        if missing_pattern.sum().sum() > 0:
            # 使用PCA或简单的哈希编码
            missing_hash = missing_pattern.sum(axis=1) % 10  # 简单的哈希编码
            enhanced_features['missing_pattern_hash'] = missing_hash
        
        return enhanced_features
    
    def smart_imputation(self, features_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
        """
        智能插值（借鉴2024年C题的时间序列方法）
        
        策略：
        1. 时间序列插值：对于有时间顺序的特征，使用时间序列插值
        2. KNN插值：对于其他特征，使用KNN插值（考虑相似样本）
        3. 卡尔曼滤波插值：对于关键特征，使用卡尔曼滤波预测
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据框
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        pd.DataFrame: 插值后的特征数据框
        """
        imputed_features = features_df.copy()
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # 策略1：时间序列插值（对于历史评分特征）
        for col in numeric_cols:
            if 'week' in col.lower() and 'score' in col.lower():
                # 这是历史评分特征，使用时间序列插值
                if features_df[col].isna().any():
                    # 获取该选手的历史数据
                    for idx in features_df.index:
                        if pd.isna(features_df.loc[idx, col]):
                            # 尝试从其他周次插值
                            prev_weeks = []
                            next_weeks = []
                            
                            # 提取周次数字
                            import re
                            match = re.search(r'week[_\s]*(\d+)', col.lower())
                            if match:
                                current_week_num = int(match.group(1))
                                
                                # 查找前后周次的数据
                                for w in range(1, current_week_num):
                                    prev_col = col.replace(f'week{current_week_num}', f'week{w}')
                                    if prev_col in features_df.columns:
                                        prev_val = features_df.loc[idx, prev_col]
                                        if not pd.isna(prev_val):
                                            prev_weeks.append((w, prev_val))
                                
                                for w in range(current_week_num + 1, 20):  # 假设最多20周
                                    next_col = col.replace(f'week{current_week_num}', f'week{w}')
                                    if next_col in features_df.columns:
                                        next_val = features_df.loc[idx, next_col]
                                        if not pd.isna(next_val):
                                            next_weeks.append((w, next_val))
                                
                                # 使用线性插值
                                if len(prev_weeks) >= 1 or len(next_weeks) >= 1:
                                    all_points = sorted(prev_weeks + next_weeks)
                                    if len(all_points) >= 2:
                                        weeks = [p[0] for p in all_points]
                                        values = [p[1] for p in all_points]
                                        try:
                                            interp_func = interp1d(weeks, values, kind='linear', 
                                                                 fill_value='extrapolate')
                                            imputed_value = float(interp_func(current_week_num))
                                            imputed_features.loc[idx, col] = imputed_value
                                        except:
                                            # 如果插值失败，使用中位数
                                            imputed_features.loc[idx, col] = features_df[col].median()
        
        # 策略2：K-means聚类插值（借鉴2301192的方法）
        # 使用K-means对样本进行聚类，然后用同一簇内样本的中位数填充
        remaining_missing_cols = [col for col in numeric_cols 
                                 if imputed_features[col].isna().any()]
        
        if remaining_missing_cols and len(imputed_features) > 2:
            try:
                # 选择用于聚类的特征（没有缺失值或缺失值较少的特征）
                cluster_features = [col for col in numeric_cols 
                                  if col not in remaining_missing_cols and
                                  imputed_features[col].isna().sum() < len(imputed_features) * 0.3]
                
                if len(cluster_features) >= 2:
                    # 准备聚类数据（只使用非缺失值）
                    cluster_data = imputed_features[cluster_features].fillna(
                        imputed_features[cluster_features].median()
                    ).values
                    
                    # 标准化数据用于聚类
                    from sklearn.preprocessing import StandardScaler
                    scaler_cluster = StandardScaler()
                    cluster_data_scaled = scaler_cluster.fit_transform(cluster_data)
                    
                    # K-means聚类（借鉴2301192：将"难度"量化为简单、中等、困难）
                    n_clusters = min(5, max(2, len(imputed_features) // 3))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(cluster_data_scaled)
                    imputed_features['_cluster'] = clusters
                    
                    # 对每个缺失值，使用同一簇内样本的中位数填充
                    for col in remaining_missing_cols:
                        missing_mask = imputed_features[col].isna()
                        if missing_mask.any():
                            for cluster_id in range(n_clusters):
                                cluster_mask = (clusters == cluster_id) & missing_mask
                                if cluster_mask.any():
                                    # 获取同一簇内非缺失值的中位数
                                    cluster_values = imputed_features.loc[
                                        (clusters == cluster_id) & ~missing_mask, col
                                    ]
                                    if len(cluster_values) > 0:
                                        fill_value = cluster_values.median()
                                    else:
                                        # 如果簇内没有非缺失值，使用全局中位数
                                        fill_value = imputed_features[col].median()
                                    
                                    if pd.isna(fill_value):
                                        fill_value = 0.0
                                    
                                    imputed_features.loc[cluster_mask, col] = fill_value
                    
                    # 移除临时聚类列
                    if '_cluster' in imputed_features.columns:
                        imputed_features = imputed_features.drop('_cluster', axis=1)
            except Exception as e:
                # 如果K-means失败，尝试KNN插值
                try:
                    knn_features = [col for col in numeric_cols 
                                  if not imputed_features[col].isna().any() and 
                                  col not in remaining_missing_cols]
                    
                    if len(knn_features) >= 2:
                        knn_imputer = KNNImputer(n_neighbors=min(5, len(imputed_features) - 1))
                        knn_data = imputed_features[knn_features + remaining_missing_cols].values
                        knn_imputed = knn_imputer.fit_transform(knn_data)
                        
                        for i, col in enumerate(remaining_missing_cols):
                            col_idx = len(knn_features) + i
                            if col_idx < knn_imputed.shape[1]:
                                imputed_features[col] = knn_imputed[:, col_idx]
                except:
                    # 如果都失败，使用中位数
                    for col in remaining_missing_cols:
                        imputed_features[col] = imputed_features[col].fillna(
                            imputed_features[col].median()
                        )
        
        # 策略3：最终填充（对于仍有缺失值的特征）
        for col in numeric_cols:
            if imputed_features[col].isna().any():
                median_val = imputed_features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                imputed_features[col] = imputed_features[col].fillna(median_val)
        
        return imputed_features
    
    def create_enhanced_features(self, season: int, week: int) -> pd.DataFrame:
        """
        创建增强特征（整合状态空间模型思想）
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        pd.DataFrame: 包含增强特征的 DataFrame
        """
        # 使用父类方法创建基础高级特征
        features_df = self.create_advanced_features(season, week)
        
        if len(features_df) == 0:
            return pd.DataFrame()
        
        # 1. 添加缺失值特征（在插值之前）
        features_df = self.create_missing_value_features(features_df)
        
        # 2. 智能插值
        features_df = self.smart_imputation(features_df, season, week)
        
        # 3. 添加状态空间模型特征（借鉴2024年C题）
        # 3.1 隐变量估计（粉丝投票的潜在状态）
        if week > 1:
            # 使用历史数据估计"动量"（类似2024年C题的动量概念）
            prev_features = self.extract_features(season, week - 1)
            if len(prev_features) > 0:
                # 计算"动量"（评分变化率）
                for idx in features_df.index:
                    contestant_name = features_df.loc[idx, 'celebrity_name']
                    prev_row = prev_features[prev_features['celebrity_name'] == contestant_name]
                    
                    if len(prev_row) > 0:
                        prev_score = prev_row['judge_total'].values[0]
                        curr_score = features_df.loc[idx, 'judge_total']
                        momentum = (curr_score - prev_score) / (prev_score + 1e-10)  # 相对变化率
                        features_df.loc[idx, 'momentum'] = momentum
                    else:
                        features_df.loc[idx, 'momentum'] = 0.0
            else:
                features_df['momentum'] = 0.0
        else:
            features_df['momentum'] = 0.0
        
        # 3.2 状态转移特征（类似马尔可夫模型）
        if week > 1:
            prev_features = self.extract_features(season, week - 1)
            if len(prev_features) > 0:
                for idx in features_df.index:
                    contestant_name = features_df.loc[idx, 'celebrity_name']
                    prev_row = prev_features[prev_features['celebrity_name'] == contestant_name]
                    
                    if len(prev_row) > 0:
                        prev_rank = prev_row['judge_rank'].values[0]
                        curr_rank = features_df.loc[idx, 'judge_rank']
                        rank_change = curr_rank - prev_rank  # 排名变化（正数表示下降）
                        features_df.loc[idx, 'rank_change'] = rank_change
                        features_df.loc[idx, 'rank_improved'] = 1 if rank_change < 0 else 0
                    else:
                        features_df.loc[idx, 'rank_change'] = 0
                        features_df.loc[idx, 'rank_improved'] = 0
            else:
                features_df['rank_change'] = 0
                features_df['rank_improved'] = 0
        else:
            features_df['rank_change'] = 0
            features_df['rank_improved'] = 0
        
        # 4. ARIMA时间序列特征（借鉴Team 2301192的方法）
        # 使用ARIMA模型捕捉周期性波动
        if week > 2 and HAS_ARIMA:
            try:
                # 对每个选手，使用历史评分构建ARIMA模型
                for idx in features_df.index:
                    contestant_name = features_df.loc[idx, 'celebrity_name']
                    # 获取历史评分序列
                    historical_scores = []
                    for prev_week in range(1, week):
                        prev_col = f'week{prev_week}_total_score'
                        if prev_col in self.df.columns:
                            prev_data = self.df[
                                (self.df['season'] == season) & 
                                (self.df['celebrity_name'] == contestant_name)
                            ]
                            if len(prev_data) > 0 and prev_col in prev_data.columns:
                                score = prev_data[prev_col].values[0]
                                if not pd.isna(score):
                                    historical_scores.append(score)
                    
                    if len(historical_scores) >= 3:
                        try:
                            # 使用ARIMA(1,1,1)模型（可以调整参数）
                            arima_model = ARIMA(historical_scores, order=(1, 1, 1))
                            arima_fitted = arima_model.fit()
                            # 预测下一期的值
                            forecast = arima_fitted.forecast(steps=1)
                            features_df.loc[idx, 'arima_forecast'] = forecast[0]
                            # 预测的标准误差
                            features_df.loc[idx, 'arima_std'] = arima_fitted.resid.std()
                        except:
                            features_df.loc[idx, 'arima_forecast'] = np.mean(historical_scores)
                            features_df.loc[idx, 'arima_std'] = np.std(historical_scores)
                    else:
                        features_df.loc[idx, 'arima_forecast'] = features_df.loc[idx, 'judge_total']
                        features_df.loc[idx, 'arima_std'] = 0.0
            except Exception as e:
                # 如果ARIMA失败，使用简单均值
                features_df['arima_forecast'] = features_df['judge_total']
                features_df['arima_std'] = 0.0
        else:
            features_df['arima_forecast'] = features_df['judge_total']
            features_df['arima_std'] = 0.0
        
        # 5. 信息熵特征（借鉴Team 2301192的方法）
        # 计算特征的信息熵（不确定性）
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['is_eliminated', 'missing_ratio', 'missing_entropy', 'arima_forecast', 'arima_std']:
                col_data = features_df[col].values
                if len(np.unique(col_data)) > 1:
                    # 计算信息熵
                    hist, _ = np.histogram(col_data, bins=min(10, len(np.unique(col_data))))
                    hist = hist[hist > 0]
                    prob = hist / hist.sum()
                    entropy = -np.sum(prob * np.log(prob + 1e-10))
                    features_df[f'{col}_entropy'] = entropy
                else:
                    features_df[f'{col}_entropy'] = 0.0
        
        # 5. 填充所有剩余的NaN
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                features_df[col] = features_df[col].fillna(median_val)
        
        return features_df
    
    def train_enhanced_models(self, training_data: List[Tuple[int, int, pd.DataFrame, Optional[str]]]) -> Dict:
        """
        训练增强模型（利用XGBoost/LightGBM的内置缺失值处理）
        
        Parameters:
        -----------
        training_data : List[Tuple[int, int, pd.DataFrame, Optional[str]]]
            训练数据列表
        
        Returns:
        --------
        Dict: 训练结果和模型性能
        """
        print("\n开始训练增强模型...")
        print("  策略：")
        print("  - 使用缺失值指示器作为特征")
        print("  - 利用XGBoost/LightGBM的内置缺失值处理")
        print("  - 整合状态空间模型思想")
        
        # 准备训练数据
        X_list = []
        y_rank_list = []
        voting_methods = []
        missing_masks_list = []  # 存储缺失值掩码
        
        for season, week, features_df, eliminated_name in training_data:
            if eliminated_name is None or len(features_df) == 0:
                continue
            
            # 准备增强特征
            ml_features = self.create_enhanced_features(season, week)
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
            
            # 记录缺失值掩码（用于XGBoost/LightGBM）
            missing_mask = np.isnan(X)
            missing_masks_list.append(missing_mask)
            
            # 对于不支持缺失值的模型，使用智能插值
            # 但对于XGBoost/LightGBM，保留NaN（它们可以处理）
            X_for_other = X.copy()
            if np.isnan(X_for_other).any():
                for i in range(X_for_other.shape[1]):
                    col_data = X_for_other[:, i]
                    if np.isnan(col_data).any():
                        median_val = np.nanmedian(col_data)
                        if np.isnan(median_val) or np.isinf(median_val):
                            median_val = 0.0
                        X_for_other[:, i] = np.where(np.isnan(col_data), median_val, col_data)
            
            X_for_other = np.nan_to_num(X_for_other, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 使用基础方法生成标签
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                base_estimate = super(AdvancedFanVoteEstimator, self).estimate_fan_votes_rank_method(
                    season, week, features_df
                )
                if base_estimate:
                    n_contestants = len(features_df)
                    fan_ranks = base_estimate['fan_ranks']
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                else:
                    continue
            else:
                base_estimate = super(AdvancedFanVoteEstimator, self).estimate_fan_votes_percent_method(
                    season, week, features_df
                )
                if base_estimate:
                    y_percent = base_estimate['fan_percents'] / 100.0
                    n_contestants = len(features_df)
                    fan_ranks = (1 - y_percent).argsort().argsort() + 1
                    y_rank = (fan_ranks - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
                else:
                    continue
            
            X_list.append((X, X_for_other))  # 存储两个版本：带NaN和插值后的
            y_rank_list.append(y_rank)
            voting_methods.append(voting_method)
        
        if len(X_list) == 0:
            print("⚠️  没有足够的训练数据")
            return {}
        
        # 合并数据
        first_n_features = X_list[0][0].shape[1]
        X_all_list = []
        X_all_other_list = []
        
        for (X, X_other) in X_list:
            if X.shape[1] != first_n_features:
                if X.shape[1] < first_n_features:
                    padding = np.zeros((X.shape[0], first_n_features - X.shape[1]))
                    X = np.hstack([X, padding])
                    X_other = np.hstack([X_other, padding])
                else:
                    X = X[:, :first_n_features]
                    X_other = X_other[:, :first_n_features]
            X_all_list.append(X)
            X_all_other_list.append(X_other)
        
        X_all = np.vstack(X_all_list)  # 带NaN的版本（用于XGBoost/LightGBM）
        X_all_other = np.vstack(X_all_other_list)  # 插值后的版本（用于其他模型）
        y_rank_all = np.hstack(y_rank_list)
        missing_mask_all = np.vstack(missing_masks_list)
        
        print(f"训练数据形状: {X_all.shape}")
        print(f"缺失值比例: {missing_mask_all.sum() / missing_mask_all.size * 100:.2f}%")
        
        # 处理y中的NaN
        y_rank_all = np.nan_to_num(y_rank_all, nan=0.5, posinf=0.5, neginf=0.5)
        
        # 特征缩放（只对插值后的版本）
        scaler = RobustScaler()
        X_scaled_other = scaler.fit_transform(X_all_other)
        self.scalers['main'] = scaler
        
        # 训练多个模型
        models_to_train = {}
        
        # 基础模型（使用插值后的数据）
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
        
        # 高级模型（如果可用，使用带NaN的数据）
        if HAS_XGBOOST:
            # XGBoost可以处理缺失值（用NaN表示）
            models_to_train['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',  # 使用hist方法可以更好地处理缺失值
                enable_categorical=False
            )
        
        if HAS_LIGHTGBM:
            # LightGBM可以处理缺失值（用NaN表示）
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
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models_to_train.items():
            print(f"  训练 {name}...", end=' ')
            
            try:
                # 选择合适的数据版本
                if name in ['xgboost', 'lightgbm', 'catboost']:
                    # 这些模型可以处理缺失值（用NaN表示）
                    X_train = X_all.copy()
                elif name == 'tabnet':
                    # TabNet需要先处理NaN
                    X_train = X_scaled_other.copy()
                else:
                    # 使用插值后的数据
                    X_train = X_scaled_other.copy()
                
                model.fit(X_train, y_rank_all)
                self.models[name] = model
                
                # 评估模型
                scores = cross_val_score(model, X_train, y_rank_all, cv=tscv, scoring='neg_mean_squared_error')
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
                
                # 集成模型使用插值后的数据
                self.models['ensemble'].fit(X_scaled_other, y_rank_all)
                
                ensemble_scores = cross_val_score(
                    self.models['ensemble'], X_scaled_other, y_rank_all,
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
    
    def predict_fan_votes_enhanced(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame,
        eliminated_name: Optional[str]
    ) -> Dict:
        """
        使用增强模型预测粉丝投票
        
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
        
        # 准备增强特征
        ml_features = self.create_enhanced_features(season, week)
        if len(ml_features) == 0:
            return None
        
        # 使用训练时确定的特征列
        if not hasattr(self, '_feature_cols'):
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_rank_method(
                    season, week, features_df
                )
            else:
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_percent_method(
                    season, week, features_df
                )
        
        # 确保所有特征列都存在
        for col in self._feature_cols:
            if col not in ml_features.columns:
                ml_features[col] = 0
        
        X = ml_features[self._feature_cols].values
        
        # 对于XGBoost/LightGBM，保留NaN
        # 对于其他模型，使用插值
        X_for_other = X.copy()
        if np.isnan(X_for_other).any():
            if hasattr(self, 'imputer') and self.imputer is not None:
                X_for_other = self.imputer.transform(X_for_other)
            else:
                for i in range(X_for_other.shape[1]):
                    col_data = X_for_other[:, i]
                    if np.isnan(col_data).any():
                        median_val = np.nanmedian(col_data)
                        if np.isnan(median_val):
                            median_val = 0.0
                        X_for_other[:, i] = np.where(np.isnan(col_data), median_val, col_data)
        
        X_for_other = np.nan_to_num(X_for_other, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 特征缩放（只对插值后的版本）
        if 'main' in self.scalers:
            X_scaled_other = self.scalers['main'].transform(X_for_other)
        else:
            X_scaled_other = X_for_other
        
        # 使用最佳模型或集成模型预测
        if self.best_model_name and self.best_model_name in self.models:
            model_name = self.best_model_name
        elif 'ensemble' in self.models:
            model_name = 'ensemble'
        else:
            # 回退到基础方法
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_rank_method(
                    season, week, features_df
                )
            else:
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_percent_method(
                    season, week, features_df
                )
        
        # 选择合适的数据版本
        if model_name in ['xgboost', 'lightgbm', 'catboost']:
            X_pred = X.copy()  # 使用带NaN的数据
        elif model_name == 'tabnet':
            X_pred = X_scaled_other.copy()  # TabNet需要先处理NaN
        else:
            X_pred = X_scaled_other.copy()  # 使用插值后的数据
        
        predicted_rank_norm = self.models[model_name].predict(X_pred)
        predicted_rank_norm = np.clip(predicted_rank_norm, 0, 1)
        
        # 后续处理与父类相同
        n_contestants = len(features_df)
        voting_method = self.determine_voting_method(season)
        
        fan_ranks = (predicted_rank_norm * (n_contestants - 1) + 1).astype(int)
        fan_ranks = np.clip(fan_ranks, 1, n_contestants)
        
        eliminated_idx = np.where(ml_features['celebrity_name'] == eliminated_name)[0]
        if len(eliminated_idx) == 0:
            voting_method = self.determine_voting_method(season)
            if voting_method == 'rank':
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_rank_method(
                    season, week, features_df
                )
            else:
                return super(AdvancedFanVoteEstimator, self).estimate_fan_votes_percent_method(
                    season, week, features_df
                )
        
        eliminated_idx = eliminated_idx[0]
        
        # 使用优化算法微调（与父类相同）
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
                'method': f'enhanced_{model_name}_rank'
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
            x0 = x0 / np.sum(x0) * 100.0
            
            combined_percents_init = judge_percents + x0
            if combined_percents_init[eliminated_idx] > np.min(combined_percents_init):
                min_combined = np.min(combined_percents_init)
                needed_percent = max(0, min_combined - judge_percents[eliminated_idx] - 1)
                x0[eliminated_idx] = max(needed_percent, 1)
                x0 = x0 / np.sum(x0) * 100.0
            
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
                'method': f'enhanced_{model_name}_percent'
            }
    
    def estimate_all_weeks_enhanced(
        self,
        seasons: Optional[List[int]] = None,
        train_on_all: bool = True
    ) -> pd.DataFrame:
        """
        使用增强方法估计所有周次的粉丝投票
        
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
            model_scores = self.train_enhanced_models(training_data)
        else:
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            test_data = training_data[split_idx:]
            model_scores = self.train_enhanced_models(train_data)
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
                
                # 使用增强方法估计
                estimate = self.predict_fan_votes_enhanced(season, week, features_df, eliminated_name)
                
                if estimate is None:
                    # 回退到基础方法
                    if voting_method == 'rank':
                        estimate = super(AdvancedFanVoteEstimator, self).estimate_fan_votes_rank_method(
                            season, week, features_df
                        )
                    else:
                        estimate = super(AdvancedFanVoteEstimator, self).estimate_fan_votes_percent_method(
                            season, week, features_df
                        )
                
                if estimate is None:
                    print("跳过（无法估计）")
                    continue
                
                if 'fan_votes' not in estimate or len(estimate['fan_votes']) != len(features_df):
                    print("跳过（数组长度不匹配）")
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
    """主函数：运行增强版本的粉丝投票估计"""
    from loader import load_data
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 增强版本")
    print("（智能NaN处理 + 状态空间模型思想 + XGBoost/LightGBM）")
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
    
    # 创建增强估计器
    estimator = EnhancedFanVoteEstimator(df)
    
    # 估计所有周次
    print("\n开始估计粉丝投票（使用增强方法）...")
    estimates_df = estimator.estimate_all_weeks_enhanced(train_on_all=True)
    
    # 保存估计结果
    estimates_df.to_csv('fan_vote_estimates_enhanced.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 估计结果已保存到: fan_vote_estimates_enhanced.csv")
    
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
    
    with open('validation_results_enhanced.json', 'w', encoding='utf-8') as f:
        validation_results_native = convert_to_native(validation_results)
        json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 验证结果已保存到: validation_results_enhanced.json")
    
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
    print("阶段2完成（增强版本）！")
    print("=" * 70)
    
    return estimator, estimates_df, validation_results


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
