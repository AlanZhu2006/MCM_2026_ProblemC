"""
阶段2：粉丝投票估计模型 - 状态空间模型版本
借鉴2024年MCM C题（网球动量）的方法
使用状态空间模型（State Space Model）和卡尔曼滤波（Kalman Filter）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# 尝试导入卡尔曼滤波库
try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("⚠️  filterpy未安装，将使用简化版卡尔曼滤波实现")
    print("   安装命令: pip install filterpy")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels未安装，将使用简化版ARIMA实现")

from fan_vote_estimator import FanVoteEstimator


class SimpleKalmanFilter:
    """简化版卡尔曼滤波实现（如果filterpy不可用）"""
    
    def __init__(self, dim_x=1, dim_z=1):
        """
        初始化卡尔曼滤波
        
        Parameters:
        -----------
        dim_x : int
            状态维度
        dim_z : int
            观测维度
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # 状态向量
        self.x = np.zeros((dim_x, 1))  # 状态估计
        
        # 协方差矩阵
        self.P = np.eye(dim_x)  # 状态协方差
        
        # 状态转移矩阵
        self.F = np.eye(dim_x)  # 状态转移
        
        # 观测矩阵
        self.H = np.eye(dim_z, dim_x)  # 观测模型
        
        # 过程噪声协方差
        self.Q = np.eye(dim_x) * 0.1
        
        # 观测噪声协方差
        self.R = np.eye(dim_z) * 1.0
        
    def predict(self, u=None, B=None, F=None, Q=None):
        """
        预测步骤
        
        Parameters:
        -----------
        u : np.ndarray, optional
            控制输入
        B : np.ndarray, optional
            控制输入矩阵
        F : np.ndarray, optional
            状态转移矩阵
        Q : np.ndarray, optional
            过程噪声协方差
        """
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        
        # 预测状态
        self.x = F @ self.x
        if u is not None and B is not None:
            self.x = self.x + B @ u
        
        # 预测协方差
        self.P = F @ self.P @ F.T + Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, z, R=None, H=None):
        """
        更新步骤
        
        Parameters:
        -----------
        z : np.ndarray
            观测值
        R : np.ndarray, optional
            观测噪声协方差
        H : np.ndarray, optional
            观测矩阵
        """
        if R is None:
            R = self.R
        if H is None:
            H = self.H
        
        z = np.array(z).reshape(-1, 1)
        
        # 计算残差
        y = z - H @ self.x
        
        # 残差协方差
        S = H @ self.P @ H.T + R
        
        # 卡尔曼增益
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新协方差
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P
        
        return self.x.copy(), self.P.copy()


class StateSpaceFanVoteEstimator(FanVoteEstimator):
    """
    基于状态空间模型的粉丝投票估计器
    
    核心思想（借鉴2024年C题）：
    1. 粉丝投票是隐变量（不可直接观测）
    2. 通过评委评分、历史表现等观测变量推断
    3. 使用状态空间模型建模粉丝投票的动态演化
    4. 使用卡尔曼滤波实时估计粉丝投票的概率分布
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化状态空间估计器
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据
        """
        super().__init__(df)
        self.kalman_filters = {}  # 存储每个选手的卡尔曼滤波器
        self.state_history = {}  # 存储状态历史
        self.observation_models = {}  # 存储观测模型参数
        
    def build_state_space_model(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        构建状态空间模型
        
        状态方程（State Equation）：
        fan_vote[t] = A * fan_vote[t-1] + B * judge_score[t] + w[t]
        
        观测方程（Observation Equation）：
        y[t] = C * fan_vote[t] + D * judge_score[t] + v[t]
        
        其中：
        - fan_vote[t]: 隐状态（粉丝投票）
        - judge_score[t]: 已知输入（评委评分）
        - y[t]: 观测值（综合排名/百分比）
        - w[t]: 过程噪声
        - v[t]: 观测噪声
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        features_df : pd.DataFrame
            特征数据框
        
        Returns:
        --------
        Dict: 状态空间模型参数
        """
        n_contestants = len(features_df)
        voting_method = self.determine_voting_method(season)
        
        # 获取历史数据（如果有）
        prev_week = week - 1
        prev_features = None
        if prev_week >= 1:
            prev_features = self.extract_features(season, prev_week)
        
        # 状态转移矩阵 A（自回归系数）
        # 假设粉丝投票有惯性（与上周相关）
        A = 0.7  # 自回归系数（0.7表示70%的惯性）
        
        # 输入矩阵 B（评委评分对粉丝投票的影响）
        # 评委评分越高，粉丝投票可能越高（但相关性较弱）
        B = 0.2  # 评委评分的影响系数
        
        # 观测矩阵 C（粉丝投票对综合得分的影响）
        if voting_method == 'rank':
            C = 1.0  # 粉丝排名直接影响综合排名
        else:
            C = 1.0  # 粉丝百分比直接影响综合百分比
        
        # 观测矩阵 D（评委评分对综合得分的影响）
        D = 1.0  # 评委评分直接影响综合得分
        
        # 过程噪声协方差 Q（粉丝投票的不确定性）
        Q = 0.1  # 较小的过程噪声（粉丝投票相对稳定）
        
        # 观测噪声协方差 R（综合得分的不确定性）
        R = 0.5  # 较大的观测噪声（综合得分受多种因素影响）
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'Q': Q,
            'R': R,
            'voting_method': voting_method,
            'prev_features': prev_features
        }
    
    def estimate_fan_votes_kalman(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        使用卡尔曼滤波估计粉丝投票
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        features_df : pd.DataFrame
            特征数据框
        
        Returns:
        --------
        Dict: 包含估计的粉丝投票和相关信息
        """
        n_contestants = len(features_df)
        eliminated_name = self.get_eliminated_contestant(season, week)
        if eliminated_name is None:
            return None
        
        # 构建状态空间模型
        ssm_params = self.build_state_space_model(season, week, features_df)
        voting_method = ssm_params['voting_method']
        
        # 获取观测值（综合排名或百分比）
        if voting_method == 'rank':
            judge_ranks = features_df['judge_rank'].values
            # 观测值：综合排名（初始值设为评委排名）
            observed_ranks = judge_ranks.copy()
        else:
            judge_percents = features_df['judge_percent'].values
            # 观测值：综合百分比（初始值设为评委百分比）
            observed_percents = judge_percents.copy()
        
        # 初始化或更新每个选手的卡尔曼滤波器
        fan_votes_estimates = np.zeros(n_contestants)
        fan_votes_uncertainty = np.zeros(n_contestants)
        
        for i, (idx, row) in enumerate(features_df.iterrows()):
            contestant_name = row['celebrity_name']
            key = f"{season}_{week}_{contestant_name}"
            
            # 初始化卡尔曼滤波器
            if key not in self.kalman_filters:
                if HAS_FILTERPY:
                    kf = KalmanFilter(dim_x=1, dim_z=1)
                    kf.F = np.array([[ssm_params['A']]])  # 状态转移
                    kf.H = np.array([[ssm_params['C']]])  # 观测矩阵
                    kf.Q = np.array([[ssm_params['Q']]])  # 过程噪声
                    kf.R = np.array([[ssm_params['R']]])  # 观测噪声
                    kf.x = np.array([[0.5]])  # 初始状态（归一化的粉丝投票）
                    kf.P = np.array([[1.0]])  # 初始协方差
                else:
                    kf = SimpleKalmanFilter(dim_x=1, dim_z=1)
                    kf.F = np.array([[ssm_params['A']]])
                    kf.H = np.array([[ssm_params['C']]])
                    kf.Q = np.array([[ssm_params['Q']]])
                    kf.R = np.array([[ssm_params['R']]])
                    kf.x = np.array([[0.5]]).reshape(-1, 1)
                    kf.P = np.array([[1.0]])
                
                self.kalman_filters[key] = kf
            else:
                kf = self.kalman_filters[key]
            
            # 获取输入（评委评分）
            if voting_method == 'rank':
                judge_input = judge_ranks[i]
                # 归一化到[0,1]
                judge_input_norm = (judge_input - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
            else:
                judge_input = judge_percents[i]
                # 归一化到[0,1]
                judge_input_norm = judge_input / 100.0
            
            # 预测步骤（考虑输入：评委评分）
            if HAS_FILTERPY:
                # 如果有控制输入，需要扩展状态空间
                kf.predict()
            else:
                kf.predict()
            
            # 获取观测值（综合排名/百分比）
            if voting_method == 'rank':
                observed_value = observed_ranks[i]
                observed_value_norm = (observed_value - 1) / (n_contestants - 1) if n_contestants > 1 else 0.5
            else:
                observed_value = observed_percents[i]
                observed_value_norm = observed_value / 100.0
            
            # 更新步骤（使用观测值）
            if HAS_FILTERPY:
                kf.update(np.array([[observed_value_norm]]))
                fan_vote_norm = kf.x[0, 0]
                fan_vote_uncertainty_val = kf.P[0, 0]
            else:
                kf.update(np.array([[observed_value_norm]]))
                fan_vote_norm = kf.x[0, 0]
                fan_vote_uncertainty_val = kf.P[0, 0]
            
            # 反归一化
            if voting_method == 'rank':
                fan_rank = fan_vote_norm * (n_contestants - 1) + 1
                fan_rank = np.clip(fan_rank, 1, n_contestants)
                fan_votes_estimates[i] = (n_contestants + 1 - fan_rank) * 1000
            else:
                fan_percent = fan_vote_norm * 100.0
                fan_percent = np.clip(fan_percent, 0, 100)
                total_votes = 10_000_000
                fan_votes_estimates[i] = fan_percent / 100.0 * total_votes
            
            fan_votes_uncertainty[i] = fan_vote_uncertainty_val
        
        # 使用优化算法微调（确保约束满足）
        eliminated_mask = features_df['celebrity_name'] == eliminated_name
        eliminated_idx = np.where(eliminated_mask)[0][0]
        
        if voting_method == 'rank':
            # 微调粉丝排名，确保被淘汰选手的综合排名最高
            judge_ranks = features_df['judge_rank'].values
            fan_ranks_initial = (n_contestants + 1 - fan_votes_estimates / 1000).astype(int)
            fan_ranks_initial = np.clip(fan_ranks_initial, 1, n_contestants)
            
            def objective(fan_ranks_opt):
                # 最小化与卡尔曼滤波估计的差异
                kalman_penalty = np.sum((fan_ranks_opt - fan_ranks_initial) ** 2)
                # 最小化与评委排名的差异
                judge_penalty = np.sum((fan_ranks_opt - judge_ranks) ** 2) * 0.3
                return kalman_penalty + judge_penalty
            
            def constraint_eliminated(fan_ranks_opt):
                combined_ranks = judge_ranks + fan_ranks_opt
                eliminated_combined = combined_ranks[eliminated_idx]
                return eliminated_combined - np.max(combined_ranks)
            
            bounds = [(1, n_contestants)] * n_contestants
            x0 = fan_ranks_initial.copy().astype(float)
            
            # 确保初始值满足约束
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
                fan_votes_final = (n_contestants + 1 - fan_ranks_final) * 1000
            else:
                fan_ranks_final = fan_ranks_initial
                fan_votes_final = fan_votes_estimates
            
            combined_ranks = judge_ranks + fan_ranks_final
            
            return {
                'fan_ranks': fan_ranks_final,
                'fan_votes': fan_votes_final,
                'combined_ranks': combined_ranks,
                'eliminated_idx': eliminated_idx,
                'method': 'kalman_filter_rank',
                'uncertainty': fan_votes_uncertainty
            }
        else:
            # 微调粉丝百分比，确保被淘汰选手的综合百分比最低
            judge_percents = features_df['judge_percent'].values
            fan_percents_initial = fan_votes_estimates / 10_000_000 * 100.0
            fan_percents_initial = np.clip(fan_percents_initial, 0, 100)
            fan_percents_initial = fan_percents_initial / np.sum(fan_percents_initial) * 100.0
            
            def objective(fan_percents_opt):
                kalman_penalty = np.sum((fan_percents_opt - fan_percents_initial) ** 2)
                judge_penalty = np.sum((fan_percents_opt - judge_percents) ** 2) * 0.2
                return kalman_penalty + judge_penalty
            
            def constraint_sum(fan_percents_opt):
                return np.sum(fan_percents_opt) - 100.0
            
            def constraint_eliminated(fan_percents_opt):
                combined_percents = judge_percents + fan_percents_opt
                eliminated_combined = combined_percents[eliminated_idx]
                return eliminated_combined - np.min(combined_percents)
            
            bounds = [(0, 100)] * n_contestants
            x0 = fan_percents_initial.copy()
            
            # 确保初始值满足约束
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
                fan_percents_final = fan_percents_initial
            
            total_votes = 10_000_000
            fan_votes_final = fan_percents_final / 100 * total_votes
            combined_percents = judge_percents + fan_percents_final
            
            return {
                'fan_percents': fan_percents_final,
                'fan_votes': fan_votes_final,
                'combined_percents': combined_percents,
                'eliminated_idx': eliminated_idx,
                'method': 'kalman_filter_percent',
                'uncertainty': fan_votes_uncertainty
            }
    
    def estimate_all_weeks_ssm(
        self,
        seasons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        使用状态空间模型估计所有周次的粉丝投票
        
        Parameters:
        -----------
        seasons : Optional[List[int]]
            要处理的季数列表
        
        Returns:
        --------
        pd.DataFrame: 包含估计结果的 DataFrame
        """
        if seasons is None:
            seasons = sorted(self.df['season'].unique())
        
        print("\n开始估计粉丝投票（使用状态空间模型 + 卡尔曼滤波）...")
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
                
                # 使用卡尔曼滤波估计
                estimate = self.estimate_fan_votes_kalman(season, week, features_df)
                
                if estimate is None:
                    # 回退到基础方法
                    if voting_method == 'rank':
                        estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
                    else:
                        estimate = super().estimate_fan_votes_percent_method(season, week, features_df)
                
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
                        uncertainty_val = 0.0
                    else:
                        fan_votes_val = estimate['fan_votes'][i]
                        eliminated_flag = (i == estimate.get('eliminated_idx', -1))
                        uncertainty_val = estimate.get('uncertainty', [0.0] * len(estimate['fan_votes']))[i] if 'uncertainty' in estimate else 0.0
                    
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': row['celebrity_name'],
                        'fan_votes': fan_votes_val,
                        'uncertainty': uncertainty_val,
                        'judge_total': row['judge_total'],
                        'voting_method': voting_method,
                        'eliminated': eliminated_flag,
                        'estimation_method': estimate.get('method', 'fallback')
                    })
                
                print("完成")
        
        return pd.DataFrame(results)


def main():
    """主函数：运行状态空间模型版本的粉丝投票估计"""
    from loader import load_data
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型 - 状态空间模型版本")
    print("（借鉴2024年MCM C题：状态空间模型 + 卡尔曼滤波）")
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
    
    # 创建状态空间估计器
    estimator = StateSpaceFanVoteEstimator(df)
    
    # 估计所有周次
    print("\n开始估计粉丝投票（使用状态空间模型）...")
    estimates_df = estimator.estimate_all_weeks_ssm()
    
    # 保存估计结果
    estimates_df.to_csv('fan_vote_estimates_ssm.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 估计结果已保存到: fan_vote_estimates_ssm.csv")
    
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
    
    with open('validation_results_ssm.json', 'w', encoding='utf-8') as f:
        validation_results_native = convert_to_native(validation_results)
        json.dump(validation_results_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 验证结果已保存到: validation_results_ssm.json")
    
    print("\n" + "=" * 70)
    print("阶段2完成（状态空间模型版本）！")
    print("=" * 70)
    
    return estimator, estimates_df, validation_results


if __name__ == "__main__":
    estimator, estimates_df, validation_results = main()
