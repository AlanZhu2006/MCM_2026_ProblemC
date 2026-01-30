"""
阶段5：新投票系统设计
设计一个更公平的投票组合系统，基于阶段4的影响因素分析结果
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FairnessAdjustedVotingSystem:
    """
    公平性调整的投票系统（Fairness-Adjusted Voting System）
    
    核心思想：
    1. 基于影响因素分析，对评分进行标准化调整，减少不公平因素
    2. 使用动态权重平衡评委和粉丝的影响
    3. 考虑专业舞者、年龄、行业、地区等因素的平衡
    """
    
    def __init__(
        self,
        estimates_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        factor_analysis_path: str = 'factor_impact_analysis.json'
    ):
        """
        初始化新投票系统
        
        Parameters:
        -----------
        estimates_df : pd.DataFrame
            阶段2估计的粉丝投票数据
        processed_df : pd.DataFrame
            预处理后的数据
        factor_analysis_path : str
            阶段4的影响因素分析结果文件路径
        """
        self.estimates_df = estimates_df.copy()
        self.processed_df = processed_df.copy()
        
        # 加载阶段4的分析结果
        with open(factor_analysis_path, 'r', encoding='utf-8') as f:
            self.factor_analysis = json.load(f)
        
        # 构建影响因素查找表
        self._build_factor_lookup_tables()
        
        # 系统参数（可调整）
        # 注意：经过测试，过度调整会降低准确率，因此采用保守策略
        self.age_adjustment_enabled = True  # 是否启用年龄调整
        self.pro_dancer_adjustment_enabled = True  # 是否启用专业舞者调整
        self.industry_adjustment_enabled = False  # 是否启用行业调整（影响较小，默认禁用）
        self.region_adjustment_enabled = False  # 是否启用地区调整（影响较小，默认禁用）
        self.dynamic_weight_enabled = True  # 是否启用动态权重
        
        # 权重参数
        self.base_judge_weight = 0.5  # 基础评委权重
        self.base_fan_weight = 0.5  # 基础粉丝权重
        self.adjustment_strength = 0.1  # 调整强度（0-1，从0.3降低到0.1以减少过度调整）
    
    def _build_factor_lookup_tables(self):
        """构建影响因素查找表"""
        # 专业舞者影响表
        self.pro_dancer_impact = {}
        for dancer_stat in self.factor_analysis['pro_dancer_impact']['pro_dancer_stats']:
            dancer_name = dancer_stat['pro_dancer']
            # 计算相对于平均水平的提升/降低
            avg_judge = dancer_stat['avg_judge_score']
            avg_fan = dancer_stat['avg_fan_votes']
            avg_placement = dancer_stat['avg_placement']
            
            # 标准化：排名越低（越好），影响因子越大
            # 使用倒数关系：排名1 -> 因子1.0，排名10 -> 因子0.1
            impact_factor = 1.0 / (avg_placement + 0.5)  # 避免除零
            
            self.pro_dancer_impact[dancer_name] = {
                'judge_impact': avg_judge,
                'fan_impact': avg_fan,
                'placement_impact': avg_placement,
                'adjustment_factor': impact_factor
            }
        
        # 年龄影响参数（从阶段4分析得到）
        age_analysis = self.factor_analysis['celebrity_features_impact']['age']
        self.age_judge_correlation = age_analysis['correlation_with_judge_score']['correlation']
        self.age_fan_correlation = age_analysis['correlation_with_fan_votes']['correlation']
        self.age_range = age_analysis['age_range']
        
        # 行业影响表
        self.industry_impact = {}
        # 从industry_statistics中提取数据
        industry_stats = self.factor_analysis['celebrity_features_impact']['industry']
        if 'industry_performance' in industry_stats:
            for industry_stat in industry_stats['industry_performance']:
                industry = industry_stat['industry']
                avg_placement = industry_stat['avg_placement']
                impact_factor = 1.0 / (avg_placement + 0.5)
                self.industry_impact[industry] = {
                    'avg_placement': avg_placement,
                    'adjustment_factor': impact_factor
                }
        else:
            # 如果没有industry_performance，从industry_statistics计算
            placement_means = industry_stats.get('industry_statistics', {}).get('placement_mean', {})
            for industry, avg_placement in placement_means.items():
                if avg_placement is not None:
                    impact_factor = 1.0 / (avg_placement + 0.5)
                    self.industry_impact[industry] = {
                        'avg_placement': float(avg_placement),
                        'adjustment_factor': impact_factor
                    }
        
        # 地区影响表
        self.region_impact = {}
        region_stats = self.factor_analysis['celebrity_features_impact']['region']
        if 'region_performance' in region_stats:
            for region_stat in region_stats['region_performance']:
                region = region_stat['region']
                avg_placement = region_stat['avg_placement']
                impact_factor = 1.0 / (avg_placement + 0.5)
                self.region_impact[region] = {
                    'avg_placement': avg_placement,
                    'adjustment_factor': impact_factor
                }
        else:
            # 如果没有region_performance，从region_statistics计算
            placement_means = region_stats.get('region_statistics', {}).get('placement_mean', {})
            for region, avg_placement in placement_means.items():
                if avg_placement is not None:
                    impact_factor = 1.0 / (avg_placement + 0.5)
                    self.region_impact[region] = {
                        'avg_placement': float(avg_placement),
                        'adjustment_factor': impact_factor
                    }
    
    def _calculate_age_adjustment(self, age: float) -> Dict[str, float]:
        """
        计算年龄调整因子
        
        Parameters:
        -----------
        age : float
            选手年龄
        
        Returns:
        --------
        Dict: 包含评委和粉丝的调整因子
        """
        if not self.age_adjustment_enabled:
            return {'judge': 1.0, 'fan': 1.0}
        
        # 基于相关性，年龄越大，评分越低
        # 调整：使不同年龄段的选手在同等水平下获得相似评分
        age_mean = self.age_range['mean']
        age_std = (self.age_range['max'] - self.age_range['min']) / 4  # 近似标准差
        
        if age_std > 0:
            # 标准化年龄
            z_score = (age - age_mean) / age_std
            
            # 根据相关性调整
            # 年龄越大，需要向上调整（补偿）
            judge_adjustment = 1.0 - self.age_judge_correlation * z_score * 0.1
            fan_adjustment = 1.0 - self.age_fan_correlation * z_score * 0.1
            
            # 限制调整范围
            judge_adjustment = np.clip(judge_adjustment, 0.8, 1.2)
            fan_adjustment = np.clip(fan_adjustment, 0.8, 1.2)
        else:
            judge_adjustment = 1.0
            fan_adjustment = 1.0
        
        return {
            'judge': judge_adjustment,
            'fan': fan_adjustment
        }
    
    def _calculate_pro_dancer_adjustment(self, pro_dancer: str) -> Dict[str, float]:
        """
        计算专业舞者调整因子
        
        Parameters:
        -----------
        pro_dancer : str
            专业舞者姓名
        
        Returns:
        --------
        Dict: 包含评委和粉丝的调整因子
        """
        if not self.pro_dancer_adjustment_enabled:
            return {'judge': 1.0, 'fan': 1.0}
        
        if pro_dancer in self.pro_dancer_impact:
            impact = self.pro_dancer_impact[pro_dancer]
            # 改进逻辑：反向调整专业舞者影响
            # 排名越好的专业舞者，向下调整（减少其优势）
            # 排名越差的专业舞者，向上调整（补偿其劣势）
            avg_placement = impact['placement_impact']
            
            # 计算调整因子：排名越好（数值越小），调整因子越小（向下调整）
            # 排名越差（数值越大），调整因子越大（向上调整）
            # 使用线性映射：排名1 -> 0.9，排名10 -> 1.1
            placement_normalized = (avg_placement - 1.0) / 9.0  # 归一化到0-1
            adjustment = 0.9 + 0.2 * placement_normalized  # 映射到0.9-1.1
            
            return {
                'judge': adjustment,
                'fan': adjustment
            }
        else:
            # 未知专业舞者，使用中性调整
            return {'judge': 1.0, 'fan': 1.0}
    
    def _calculate_industry_adjustment(self, industry: str) -> float:
        """
        计算行业调整因子
        
        Parameters:
        -----------
        industry : str
            选手行业
        
        Returns:
        --------
        float: 调整因子
        """
        if not self.industry_adjustment_enabled:
            return 1.0
        
        if industry in self.industry_impact:
            impact = self.industry_impact[industry]
            adjustment = impact['adjustment_factor']
            normalized_adjustment = 0.5 + 0.5 * (adjustment / 2.0)
            return normalized_adjustment
        else:
            return 1.0
    
    def _calculate_region_adjustment(self, region: str) -> float:
        """
        计算地区调整因子
        
        Parameters:
        -----------
        region : str
            选手地区
        
        Returns:
        --------
        float: 调整因子
        """
        if not self.region_adjustment_enabled:
            return 1.0
        
        if region in self.region_impact:
            impact = self.region_impact[region]
            adjustment = impact['adjustment_factor']
            normalized_adjustment = 0.5 + 0.5 * (adjustment / 2.0)
            return normalized_adjustment
        else:
            return 1.0
    
    def _calculate_dynamic_weights(
        self,
        judge_scores: np.ndarray,
        fan_votes: np.ndarray,
        ages: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        计算动态权重
        
        根据阶段4的发现：年龄对粉丝投票的影响（-0.26）略大于对评委评分的影响（-0.24）
        这意味着粉丝更关注年龄，因此可以动态调整权重
        
        Parameters:
        -----------
        judge_scores : np.ndarray
            评委评分数组
        fan_votes : np.ndarray
            粉丝投票数组
        ages : Optional[np.ndarray]
            年龄数组（可选）
        
        Returns:
        --------
        Tuple[float, float]: (评委权重, 粉丝权重)
        """
        if not self.dynamic_weight_enabled:
            return self.base_judge_weight, self.base_fan_weight
        
        # 计算评委和粉丝的变异系数（CV），反映分散程度
        judge_cv = np.std(judge_scores) / (np.mean(judge_scores) + 1e-6)
        fan_cv = np.std(fan_votes) / (np.mean(fan_votes) + 1e-6)
        
        # 如果粉丝投票更分散，说明粉丝意见分歧更大，可以适当降低粉丝权重
        # 如果评委评分更分散，说明评委意见分歧更大，可以适当降低评委权重
        total_cv = judge_cv + fan_cv
        
        if total_cv > 0:
            # 根据变异系数调整权重
            judge_weight = self.base_judge_weight * (1 + (fan_cv - judge_cv) * 0.2)
            fan_weight = self.base_fan_weight * (1 + (judge_cv - fan_cv) * 0.2)
            
            # 归一化
            total_weight = judge_weight + fan_weight
            judge_weight = judge_weight / total_weight
            fan_weight = fan_weight / total_weight
            
            # 限制权重变化范围，避免过度调整（±10%）
            judge_weight = np.clip(judge_weight, 0.45, 0.55)
            fan_weight = np.clip(fan_weight, 0.45, 0.55)
            
            # 重新归一化
            total_weight = judge_weight + fan_weight
            judge_weight = judge_weight / total_weight
            fan_weight = fan_weight / total_weight
        else:
            judge_weight = self.base_judge_weight
            fan_weight = self.base_fan_weight
        
        return judge_weight, fan_weight
    
    def calculate_fairness_adjusted_scores(
        self,
        group: pd.DataFrame,
        judge_totals: np.ndarray,
        fan_votes: np.ndarray
    ) -> Dict:
        """
        计算公平性调整后的综合得分
        
        Parameters:
        -----------
        group : pd.DataFrame
            当前周次的选手数据
        judge_totals : np.ndarray
            评委总分数组
        fan_votes : np.ndarray
            粉丝投票数组
        
        Returns:
        --------
        Dict: 包含调整后的评分和综合得分
        """
        n = len(judge_totals)
        if n == 0:
            return {}
        
        # 初始化调整后的评分
        adjusted_judge_scores = judge_totals.copy()
        adjusted_fan_votes = fan_votes.copy()
        
        # 对每个选手应用调整
        for i, (idx, row) in enumerate(group.iterrows()):
            # 年龄调整
            age = row.get('celebrity_age_during_season', np.nan)
            if pd.notna(age):
                age_adj = self._calculate_age_adjustment(float(age))
                adjusted_judge_scores[i] *= (1.0 + (age_adj['judge'] - 1.0) * self.adjustment_strength)
                adjusted_fan_votes[i] *= (1.0 + (age_adj['fan'] - 1.0) * self.adjustment_strength)
            
            # 专业舞者调整
            pro_dancer = row.get('ballroompartner', '') or row.get('ballroom_partner', '')
            if pro_dancer:
                dancer_adj = self._calculate_pro_dancer_adjustment(str(pro_dancer))
                adjusted_judge_scores[i] *= (1.0 + (dancer_adj['judge'] - 1.0) * self.adjustment_strength)
                adjusted_fan_votes[i] *= (1.0 + (dancer_adj['fan'] - 1.0) * self.adjustment_strength)
            
            # 行业调整
            industry = row.get('celebrity_industry', '')
            if industry:
                industry_adj = self._calculate_industry_adjustment(str(industry))
                combined_adj = (industry_adj - 1.0) * self.adjustment_strength * 0.5  # 行业影响较小
                adjusted_judge_scores[i] *= (1.0 + combined_adj)
                adjusted_fan_votes[i] *= (1.0 + combined_adj)
            
            # 地区调整
            region = row.get('celebrity_homecountry/region', '')
            if region:
                region_adj = self._calculate_region_adjustment(str(region))
                combined_adj = (region_adj - 1.0) * self.adjustment_strength * 0.5  # 地区影响较小
                adjusted_judge_scores[i] *= (1.0 + combined_adj)
                adjusted_fan_votes[i] *= (1.0 + combined_adj)
        
        # 计算动态权重
        ages = group.get('celebrity_age_during_season', pd.Series()).values
        judge_weight, fan_weight = self._calculate_dynamic_weights(
            adjusted_judge_scores, adjusted_fan_votes, ages
        )
        
        # 标准化调整后的评分（用于排名法）
        judge_ranks = pd.Series(adjusted_judge_scores).rank(
            ascending=False, method='min'
        ).astype(int).values
        fan_ranks = pd.Series(adjusted_fan_votes).rank(
            ascending=False, method='min'
        ).astype(int).values
        
        # 加权综合排名
        combined_ranks = judge_weight * judge_ranks + fan_weight * fan_ranks
        
        # 标准化调整后的评分（用于百分比法）
        judge_percents = (adjusted_judge_scores / adjusted_judge_scores.sum()) * 100
        fan_percents = (adjusted_fan_votes / adjusted_fan_votes.sum()) * 100
        
        # 加权综合百分比
        combined_percents = judge_weight * judge_percents + fan_weight * fan_percents
        
        return {
            'adjusted_judge_scores': adjusted_judge_scores,
            'adjusted_fan_votes': adjusted_fan_votes,
            'judge_weight': judge_weight,
            'fan_weight': fan_weight,
            'judge_ranks': judge_ranks,
            'fan_ranks': fan_ranks,
            'combined_ranks': combined_ranks,
            'judge_percents': judge_percents,  # 已经是numpy数组，不需要.values
            'fan_percents': fan_percents,  # 已经是numpy数组，不需要.values
            'combined_percents': combined_percents,  # 已经是numpy数组，不需要.values
            'eliminated_idx_rank': np.argmax(combined_ranks),
            'eliminated_idx_percent': np.argmin(combined_percents)
        }
    
    def apply_to_all_weeks(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        对所有周次应用新投票系统
        
        Parameters:
        -----------
        seasons : Optional[List[int]]
            要处理的季次列表，None表示所有季次
        
        Returns:
        --------
        pd.DataFrame: 包含新系统结果的数据框
        """
        results = []
        
        # 合并数据（processed_df没有week列，只按season和celebrity_name合并）
        # 从processed_df中获取选手的基本信息（年龄、专业舞者、行业、地区等）
        info_cols = ['season', 'celebrity_name', 'celebrity_age_during_season',
                     'celebrity_industry', 'celebrity_homecountry/region']
        
        # 检查专业舞者列名（可能是ballroompartner或ballroom_partner）
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
            # 获取评委总分和粉丝投票
            judge_totals = group['judge_total'].values
            fan_votes = group['fan_votes'].values
            
            # 计算新系统的综合得分
            new_system_result = self.calculate_fairness_adjusted_scores(
                group, judge_totals, fan_votes
            )
            
            if not new_system_result:
                continue
            
            # 确定使用的投票方法（根据季次）
            if season in [1, 2] or season >= 28:
                voting_method = 'rank'
                eliminated_idx = new_system_result['eliminated_idx_rank']
            else:
                voting_method = 'percent'
                eliminated_idx = new_system_result['eliminated_idx_percent']
            
            # 记录结果
            for i, (idx, row) in enumerate(group.iterrows()):
                results.append({
                    'season': int(season),
                    'week': int(week),
                    'celebrity_name': row['celebrity_name'],
                    'judge_total': float(judge_totals[i]),
                    'fan_votes': float(fan_votes[i]),
                    'adjusted_judge_score': float(new_system_result['adjusted_judge_scores'][i]),
                    'adjusted_fan_votes': float(new_system_result['adjusted_fan_votes'][i]),
                    'judge_weight': float(new_system_result['judge_weight']),
                    'fan_weight': float(new_system_result['fan_weight']),
                    'combined_rank': float(new_system_result['combined_ranks'][i]),
                    'combined_percent': float(new_system_result['combined_percents'][i]),
                    'voting_method': voting_method,
                    'is_eliminated_new_system': (i == eliminated_idx),
                    'age': float(row.get('celebrity_age_during_season', np.nan)) if pd.notna(row.get('celebrity_age_during_season')) else np.nan,
                    'pro_dancer': str(row.get('ballroompartner', '') or row.get('ballroom_partner', '')),
                    'industry': str(row.get('celebrity_industry', '')),
                    'region': str(row.get('celebrity_homecountry/region', ''))
                })
        
        return pd.DataFrame(results)
    
    def compare_with_original_systems(
        self,
        new_system_results: pd.DataFrame
    ) -> Dict:
        """
        比较新系统与原始系统的差异
        
        Parameters:
        -----------
        new_system_results : pd.DataFrame
            新系统的结果
        
        Returns:
        --------
        Dict: 比较分析结果
        """
        # 合并原始估计数据
        comparison_df = new_system_results.merge(
            self.estimates_df[['season', 'week', 'celebrity_name', 'eliminated']],
            on=['season', 'week', 'celebrity_name'],
            how='left'
        )
        
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
        comparison_df = comparison_df.merge(
            original_pred_df,
            on=['season', 'week'],
            how='left'
        )
        
        # 原始系统的淘汰预测字典
        original_eliminated_dict = {}
        for _, row in original_pred_df.iterrows():
            key = (int(row['season']), int(row['week']))
            original_eliminated_dict[key] = row['original_eliminated']
        
        # 新系统的淘汰预测
        new_eliminated = comparison_df[comparison_df['is_eliminated_new_system'] == True]
        new_eliminated_dict = {}
        for _, row in new_eliminated.iterrows():
            key = (int(row['season']), int(row['week']))
            new_eliminated_dict[key] = row['celebrity_name']
        
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
        new_correct = 0
        different_predictions = 0
        total_weeks = len(all_weeks)
        
        for week_key in all_weeks:
            original_pred = original_eliminated_dict.get(week_key)
            new_pred = new_eliminated_dict.get(week_key)
            actual = actual_eliminated_dict.get(week_key)
            
            if actual:
                if original_pred == actual:
                    original_correct += 1
                if new_pred == actual:
                    new_correct += 1
                if original_pred != new_pred:
                    different_predictions += 1
        
        return {
            'total_weeks': total_weeks,
            'original_system_accuracy': original_correct / total_weeks if total_weeks > 0 else 0.0,
            'new_system_accuracy': new_correct / total_weeks if total_weeks > 0 else 0.0,
            'accuracy_improvement': (new_correct - original_correct) / total_weeks if total_weeks > 0 else 0.0,
            'different_predictions': different_predictions,
            'different_predictions_rate': different_predictions / total_weeks if total_weeks > 0 else 0.0,
            'original_correct_count': original_correct,
            'new_correct_count': new_correct
        }
