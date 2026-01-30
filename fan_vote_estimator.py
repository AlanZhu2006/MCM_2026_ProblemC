"""
阶段2：粉丝投票估计模型
从已知的淘汰结果反推粉丝投票（逆问题求解）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class FanVoteEstimator:
    """粉丝投票估计器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化估计器
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据（应包含每周的总分和排名）
        """
        self.df = df.copy()
        self.estimates = {}  # 存储估计结果
        self.uncertainty = {}  # 存储不确定性信息
        
    def extract_features(self, season: int, week: int) -> pd.DataFrame:
        """
        提取特征：评委评分、历史表现、选手特征、专业舞者等
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        pd.DataFrame: 包含所有特征的 DataFrame
        """
        # 筛选该季该周的数据
        season_data = self.df[self.df['season'] == season].copy()
        
        # 获取该周仍在比赛的选手（总分不为0且不为N/A）
        total_col = f'week{week}_total_score'
        if total_col not in season_data.columns:
            return pd.DataFrame()
        
        week_data = season_data[
            season_data[total_col].notna() & (season_data[total_col] > 0)
        ].copy()
        
        if len(week_data) == 0:
            return pd.DataFrame()
        
        # 特征1: 评委评分相关
        week_data['judge_total'] = week_data[total_col]
        week_data['judge_rank'] = week_data[f'week{week}_judge_rank']
        week_data['judge_percent'] = week_data[f'week{week}_judge_percent']
        
        # 特征2: 历史表现（前几周的平均评分和排名）
        history_features = []
        for prev_week in range(1, week):
            prev_total = f'week{prev_week}_total_score'
            prev_rank = f'week{prev_week}_judge_rank'
            if prev_total in week_data.columns:
                week_data[f'avg_score_prev{prev_week}'] = week_data[prev_total]
                week_data[f'avg_rank_prev{prev_week}'] = week_data[prev_rank]
                history_features.extend([f'avg_score_prev{prev_week}', f'avg_rank_prev{prev_week}'])
        
        # 计算历史平均评分和排名
        if history_features:
            score_cols = [col for col in history_features if 'score' in col]
            rank_cols = [col for col in history_features if 'rank' in col]
            if score_cols:
                week_data['avg_historical_score'] = week_data[score_cols].mean(axis=1)
            if rank_cols:
                week_data['avg_historical_rank'] = week_data[rank_cols].mean(axis=1)
        
        # 特征3: 选手特征
        if 'celebrity_age_during_season' in week_data.columns:
            week_data['age'] = week_data['celebrity_age_during_season']
        if 'celebrity_industry' in week_data.columns:
            # 将行业转换为数值（可以进一步优化）
            week_data['industry_encoded'] = pd.Categorical(week_data['celebrity_industry']).codes
        
        # 特征4: 专业舞者特征（可以统计专业舞者的历史表现）
        if 'ballroompartner' in week_data.columns:
            # 计算每个专业舞者的平均历史表现
            partner_stats = self._calculate_partner_stats(season, week)
            week_data = week_data.merge(
                partner_stats, 
                on='ballroompartner', 
                how='left'
            )
        
        # 特征5: 周次特征（后期可能有更多关注）
        week_data['week_number'] = week
        week_data['contestants_remaining'] = len(week_data)
        
        return week_data
    
    def _calculate_partner_stats(self, season: int, week: int) -> pd.DataFrame:
        """
        计算专业舞者的历史统计数据
        
        Returns:
        --------
        pd.DataFrame: 包含专业舞者统计信息的 DataFrame
        """
        # 简化版本：可以扩展为更复杂的统计
        partner_stats = self.df.groupby('ballroompartner').agg({
            'placement': 'mean',  # 平均排名（越小越好）
        }).reset_index()
        partner_stats.columns = ['ballroompartner', 'partner_avg_placement']
        return partner_stats
    
    def determine_voting_method(self, season: int) -> str:
        """
        确定该季使用的投票组合方法
        
        Returns:
        --------
        str: 'rank' 或 'percent'
        """
        # 根据问题说明：
        # 第1-2季：排名法
        # 第3-27季：百分比法
        # 第28-34季：排名法
        if season <= 2:
            return 'rank'
        elif 3 <= season <= 27:
            return 'percent'
        else:  # season >= 28
            return 'rank'
    
    def get_eliminated_contestant(self, season: int, week: int) -> Optional[str]:
        """
        获取该周被淘汰的选手
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        str or None: 被淘汰选手的姓名，如果无法确定则返回None
        """
        season_data = self.df[self.df['season'] == season].copy()
        
        # 检查下一周的数据，找出谁消失了
        next_week = week + 1
        next_total_col = f'week{next_week}_total_score'
        
        if next_total_col in season_data.columns:
            # 找出本周有分数但下周没有分数（或为0）的选手
            week_total_col = f'week{week}_total_score'
            if week_total_col in season_data.columns:
                this_week = season_data[
                    season_data[week_total_col].notna() & (season_data[week_total_col] > 0)
                ]['celebrity_name'].tolist()
                
                next_week_active = season_data[
                    season_data[next_total_col].notna() & (season_data[next_total_col] > 0)
                ]['celebrity_name'].tolist()
                
                eliminated = [name for name in this_week if name not in next_week_active]
                if eliminated:
                    return eliminated[0]  # 返回第一个被淘汰的
        
        # 如果无法从数据推断，尝试从results列解析
        if 'results' in season_data.columns:
            results = season_data['results'].astype(str)
            eliminated_pattern = f'Eliminated Week {week}'
            eliminated = season_data[results.str.contains(eliminated_pattern, case=False, na=False)]
            if len(eliminated) > 0:
                return eliminated.iloc[0]['celebrity_name']
        
        return None
    
    def estimate_fan_votes_rank_method(
        self, 
        season: int, 
        week: int,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        使用排名法估计粉丝投票
        
        约束：被淘汰选手的综合排名（评委排名 + 粉丝排名）应该最高（最差）
        
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
        judge_ranks = features_df['judge_rank'].values
        
        # 获取被淘汰的选手
        eliminated_name = self.get_eliminated_contestant(season, week)
        if eliminated_name is None:
            # 如果无法确定，假设排名最低的评委评分选手被淘汰
            eliminated_idx = np.argmax(judge_ranks)  # 排名最高（最差）
        else:
            eliminated_mask = features_df['celebrity_name'] == eliminated_name
            if eliminated_mask.any():
                eliminated_idx = np.where(eliminated_mask)[0][0]
            else:
                eliminated_idx = np.argmax(judge_ranks)
        
        # 目标函数：最小化粉丝投票的方差，同时满足约束
        def objective(fan_ranks):
            """
            目标函数：最小化粉丝投票与评委评分的差异
            """
            # 惩罚项：粉丝排名应该与某些特征相关
            penalty = 0
            
            # 如果评委评分高，粉丝排名应该相对较低（受欢迎）
            for i, judge_rank in enumerate(judge_ranks):
                # 评委排名低（好）的选手，粉丝排名也应该相对低（受欢迎）
                expected_fan_rank = judge_rank * 0.7 + n_contestants * 0.3  # 混合模型
                penalty += (fan_ranks[i] - expected_fan_rank) ** 2
            
            return penalty
        
        # 约束：被淘汰选手的综合排名应该最高
        def constraint_eliminated(fan_ranks):
            """
            约束：被淘汰选手的（评委排名 + 粉丝排名）应该最大
            """
            combined_ranks = judge_ranks + fan_ranks
            eliminated_combined = combined_ranks[eliminated_idx]
            # 被淘汰选手的综合排名应该大于等于所有其他选手
            return eliminated_combined - np.max(combined_ranks)
        
        # 边界：粉丝排名在 [1, n_contestants] 之间
        bounds = [(1, n_contestants)] * n_contestants
        
        # 初始猜测：粉丝排名与评委排名相似但有一定随机性
        x0 = judge_ranks.copy() + np.random.normal(0, 0.5, n_contestants)
        x0 = np.clip(x0, 1, n_contestants)
        
        # 优化
        constraints = [{'type': 'ineq', 'fun': constraint_eliminated}]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            fan_ranks = result.x
        else:
            # 如果优化失败，使用启发式方法
            fan_ranks = self._heuristic_fan_ranks(judge_ranks, eliminated_idx, n_contestants)
        
        # 将排名转换为投票数（相对值）
        # 排名越低，投票数越多
        fan_votes = (n_contestants + 1 - fan_ranks) * 1000  # 缩放因子
        
        return {
            'fan_ranks': fan_ranks,
            'fan_votes': fan_votes,
            'combined_ranks': judge_ranks + fan_ranks,
            'eliminated_idx': eliminated_idx,
            'optimization_success': result.success if 'result' in locals() else False
        }
    
    def estimate_fan_votes_percent_method(
        self,
        season: int,
        week: int,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        使用百分比法估计粉丝投票
        
        约束：被淘汰选手的综合百分比（评委百分比 + 粉丝百分比）应该最低
        
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
        judge_percents = features_df['judge_percent'].values
        
        # 获取被淘汰的选手
        eliminated_name = self.get_eliminated_contestant(season, week)
        if eliminated_name is None:
            eliminated_idx = np.argmin(judge_percents)
        else:
            eliminated_mask = features_df['celebrity_name'] == eliminated_name
            if eliminated_mask.any():
                eliminated_idx = np.where(eliminated_mask)[0][0]
            else:
                eliminated_idx = np.argmin(judge_percents)
        
        # 目标函数：最小化粉丝投票与评委评分的差异
        def objective(fan_percents):
            """
            目标函数：粉丝百分比应该与评委百分比有一定相关性
            """
            penalty = 0
            for i, judge_percent in enumerate(judge_percents):
                # 评委百分比高的，粉丝百分比也应该相对高
                expected_fan_percent = judge_percent * 0.6 + (100/n_contestants) * 0.4
                penalty += (fan_percents[i] - expected_fan_percent) ** 2
            return penalty
        
        # 约束1：百分比总和为100
        def constraint_sum(fan_percents):
            return np.sum(fan_percents) - 100.0
        
        # 约束2：被淘汰选手的综合百分比应该最低
        def constraint_eliminated(fan_percents):
            combined_percents = judge_percents + fan_percents
            eliminated_combined = combined_percents[eliminated_idx]
            return np.min(combined_percents) - eliminated_combined
        
        # 边界：每个选手的粉丝百分比在 [0, 100] 之间
        bounds = [(0, 100)] * n_contestants
        
        # 初始猜测
        x0 = judge_percents.copy() * 0.8 + np.random.normal(0, 2, n_contestants)
        x0 = np.clip(x0, 1, 99)
        x0 = x0 / np.sum(x0) * 100  # 归一化到100
        
        # 优化
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
            options={'maxiter': 1000}
        )
        
        if result.success:
            fan_percents = result.x
        else:
            # 启发式方法
            fan_percents = self._heuristic_fan_percents(judge_percents, eliminated_idx, n_contestants)
        
        # 将百分比转换为投票数（假设总投票数为1000万）
        total_votes = 10_000_000
        fan_votes = fan_percents / 100 * total_votes
        
        return {
            'fan_percents': fan_percents,
            'fan_votes': fan_votes,
            'combined_percents': judge_percents + fan_percents,
            'eliminated_idx': eliminated_idx,
            'optimization_success': result.success if 'result' in locals() else False
        }
    
    def _heuristic_fan_ranks(
        self, 
        judge_ranks: np.ndarray, 
        eliminated_idx: int, 
        n_contestants: int
    ) -> np.ndarray:
        """
        启发式方法：当优化失败时使用
        """
        # 简单启发式：粉丝排名与评委排名相关，但被淘汰选手的粉丝排名应该较高（不受欢迎）
        fan_ranks = judge_ranks.copy()
        
        # 被淘汰选手的粉丝排名设为最高（最不受欢迎）
        fan_ranks[eliminated_idx] = n_contestants
        
        # 其他选手的粉丝排名基于评委排名，但加入一些随机性
        for i in range(n_contestants):
            if i != eliminated_idx:
                # 评委排名好的，粉丝排名也相对好
                fan_ranks[i] = judge_ranks[i] * 0.8 + np.random.uniform(0.5, 2)
        
        fan_ranks = np.clip(fan_ranks, 1, n_contestants)
        return fan_ranks
    
    def _heuristic_fan_percents(
        self,
        judge_percents: np.ndarray,
        eliminated_idx: int,
        n_contestants: int
    ) -> np.ndarray:
        """
        启发式方法：当优化失败时使用
        """
        fan_percents = judge_percents.copy() * 0.7
        
        # 被淘汰选手的粉丝百分比设为最低
        fan_percents[eliminated_idx] = max(5, 100 / n_contestants * 0.5)
        
        # 归一化到100
        fan_percents = fan_percents / np.sum(fan_percents) * 100
        return fan_percents
    
    def estimate_all_weeks(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        估计所有周次的粉丝投票
        
        Parameters:
        -----------
        seasons : List[int], optional
            要处理的季数列表，如果为None则处理所有季
        
        Returns:
        --------
        pd.DataFrame: 包含估计结果的 DataFrame
        """
        if seasons is None:
            seasons = sorted(self.df['season'].unique())
        
        results = []
        
        for season in seasons:
            print(f"\n处理第 {season} 季...")
            season_data = self.df[self.df['season'] == season]
            
            # 确定投票方法
            voting_method = self.determine_voting_method(season)
            
            # 找出该季的所有周次
            total_cols = [col for col in season_data.columns if '_total_score' in col]
            weeks = []
            for col in total_cols:
                week_num = self._extract_week_number(col)
                if week_num:
                    weeks.append(week_num)
            weeks = sorted(set(weeks))
            
            for week in weeks:
                print(f"  周 {week}...", end=' ')
                
                # 提取特征
                features_df = self.extract_features(season, week)
                if len(features_df) == 0:
                    print("跳过（无数据）")
                    continue
                
                # 根据投票方法估计
                if voting_method == 'rank':
                    estimate = self.estimate_fan_votes_rank_method(season, week, features_df)
                else:
                    estimate = self.estimate_fan_votes_percent_method(season, week, features_df)
                
                # 保存结果
                for i, (idx, row) in enumerate(features_df.iterrows()):
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': row['celebrity_name'],
                        'fan_votes': estimate['fan_votes'][i],
                        'judge_total': row['judge_total'],
                        'voting_method': voting_method,
                        'eliminated': (i == estimate['eliminated_idx'])
                    })
                
                print("完成")
        
        return pd.DataFrame(results)
    
    def _extract_week_number(self, column_name: str) -> int:
        """从列名中提取周次数字"""
        import re
        match = re.search(r'week[_\s]*(\d+)', column_name.lower())
        if match:
            return int(match.group(1))
        return None
    
    def quantify_uncertainty_monte_carlo(
        self,
        estimates_df: pd.DataFrame,
        n_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        使用蒙特卡洛方法量化不确定性
        
        Parameters:
        -----------
        estimates_df : pd.DataFrame
            估计结果
        n_simulations : int
            模拟次数
        
        Returns:
        --------
        pd.DataFrame: 包含不确定性信息的 DataFrame
        """
        print(f"\n进行蒙特卡洛不确定性分析（{n_simulations}次模拟）...")
        
        uncertainty_results = []
        
        # 按季和周分组
        grouped = estimates_df.groupby(['season', 'week'])
        
        for (season, week), group in grouped:
            # 对每个周次进行多次模拟
            fan_votes_samples = []
            
            for sim in range(n_simulations):
                # 添加随机噪声到估计值
                noise = np.random.normal(0, 0.1, len(group))  # 10%的标准差
                simulated_votes = group['fan_votes'].values * (1 + noise)
                simulated_votes = np.maximum(simulated_votes, 0)  # 确保非负
                fan_votes_samples.append(simulated_votes)
            
            fan_votes_samples = np.array(fan_votes_samples)
            
            # 计算统计量
            for i, (idx, row) in enumerate(group.iterrows()):
                votes_dist = fan_votes_samples[:, i]
                
                uncertainty_results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': row['celebrity_name'],
                    'fan_votes_mean': np.mean(votes_dist),
                    'fan_votes_std': np.std(votes_dist),
                    'fan_votes_ci_lower': np.percentile(votes_dist, 2.5),  # 95%置信区间下界
                    'fan_votes_ci_upper': np.percentile(votes_dist, 97.5),  # 95%置信区间上界
                    'fan_votes_median': np.median(votes_dist),
                })
        
        return pd.DataFrame(uncertainty_results)
    
    def validate_estimates(
        self,
        estimates_df: pd.DataFrame
    ) -> Dict:
        """
        验证估计值是否与已知淘汰结果一致
        
        Parameters:
        -----------
        estimates_df : pd.DataFrame
            估计结果
        
        Returns:
        --------
        Dict: 验证结果
        """
        print("\n验证估计值...")
        
        validation_results = {
            'total_weeks': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'details': []
        }
        
        # 按季和周分组
        grouped = estimates_df.groupby(['season', 'week'])
        
        for (season, week), group in grouped:
            validation_results['total_weeks'] += 1
            
            # 确定投票方法
            voting_method = self.determine_voting_method(season)
            
            # 获取实际被淘汰的选手
            eliminated_name = self.get_eliminated_contestant(season, week)
            
            if eliminated_name is None:
                continue
            
            # 计算综合得分
            if voting_method == 'rank':
                # 排名法：综合排名 = 评委排名 + 粉丝排名
                group = group.copy()
                group['judge_rank'] = group['judge_total'].rank(ascending=False)
                group['fan_rank'] = group['fan_votes'].rank(ascending=False)
                group['combined_rank'] = group['judge_rank'] + group['fan_rank']
                
                # 综合排名最高的应该被淘汰
                predicted_eliminated = group.loc[group['combined_rank'].idxmax(), 'celebrity_name']
            else:
                # 百分比法：综合百分比 = 评委百分比 + 粉丝百分比
                group = group.copy()
                group['judge_percent'] = group['judge_total'] / group['judge_total'].sum() * 100
                group['fan_percent'] = group['fan_votes'] / group['fan_votes'].sum() * 100
                group['combined_percent'] = group['judge_percent'] + group['fan_percent']
                
                # 综合百分比最低的应该被淘汰
                predicted_eliminated = group.loc[group['combined_percent'].idxmin(), 'celebrity_name']
            
            # 检查预测是否正确
            is_correct = (predicted_eliminated == eliminated_name)
            
            if is_correct:
                validation_results['correct_predictions'] += 1
            else:
                validation_results['incorrect_predictions'] += 1
            
            validation_results['details'].append({
                'season': int(season),
                'week': int(week),
                'actual_eliminated': str(eliminated_name),
                'predicted_eliminated': str(predicted_eliminated),
                'correct': bool(is_correct)
            })
        
        # 计算准确率
        if validation_results['total_weeks'] > 0:
            validation_results['accuracy'] = float(
                validation_results['correct_predictions'] / 
                validation_results['total_weeks']
            )
        else:
            validation_results['accuracy'] = 0.0
        
        # 确保所有数值都是Python原生类型
        validation_results['total_weeks'] = int(validation_results['total_weeks'])
        validation_results['correct_predictions'] = int(validation_results['correct_predictions'])
        validation_results['incorrect_predictions'] = int(validation_results['incorrect_predictions'])
        
        print(f"总周数: {validation_results['total_weeks']}")
        print(f"正确预测: {validation_results['correct_predictions']}")
        print(f"错误预测: {validation_results['incorrect_predictions']}")
        print(f"准确率: {validation_results['accuracy']:.2%}")
        
        return validation_results


def main():
    """主函数：运行粉丝投票估计"""
    from loader import load_data
    
    print("=" * 70)
    print("阶段2：粉丝投票估计模型")
    print("=" * 70)
    
    # 加载处理后的数据
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
    
    # 创建估计器
    estimator = FanVoteEstimator(df)
    
    # 估计所有周次的粉丝投票
    print("\n开始估计粉丝投票...")
    estimates_df = estimator.estimate_all_weeks()
    
    # 保存估计结果
    estimates_df.to_csv('fan_vote_estimates.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 估计结果已保存到: fan_vote_estimates.csv")
    
    # 验证模型
    validation_results = estimator.validate_estimates(estimates_df)
    
    # 不确定性量化
    uncertainty_df = estimator.quantify_uncertainty_monte_carlo(estimates_df, n_simulations=500)
    uncertainty_df.to_csv('fan_vote_uncertainty.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 不确定性分析结果已保存到: fan_vote_uncertainty.csv")
    
    # 保存验证结果
    import json
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 验证结果已保存到: validation_results.json")
    
    print("\n" + "=" * 70)
    print("阶段2完成！")
    print("=" * 70)
    
    return estimator, estimates_df, uncertainty_df, validation_results


if __name__ == "__main__":
    estimator, estimates_df, uncertainty_df, validation_results = main()
