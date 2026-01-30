"""
阶段4：影响因素分析
分析专业舞者和选手特征对评委评分和粉丝投票的影响
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FactorImpactAnalyzer:
    """影响因素分析器：分析专业舞者和选手特征的影响"""
    
    def __init__(
        self,
        processed_df: pd.DataFrame,
        estimates_df: pd.DataFrame
    ):
        """
        初始化分析器
        
        Parameters:
        -----------
        processed_df : pd.DataFrame
            预处理后的数据（包含选手特征和评委评分）
        estimates_df : pd.DataFrame
            阶段2估计的粉丝投票数据
        """
        self.processed_df = processed_df.copy()
        self.estimates_df = estimates_df.copy()
        
        # 合并数据
        self.merged_df = self._merge_data()
        
    def _merge_data(self) -> pd.DataFrame:
        """合并预处理数据和粉丝投票估计数据"""
        # 检查列名（可能是ballroom_partner或ballroompartner）
        partner_col = None
        for col in ['ballroom_partner', 'ballroompartner']:
            if col in self.processed_df.columns:
                partner_col = col
                break
        
        if partner_col is None:
            warnings.warn("未找到专业舞者列，将使用空值")
            partner_col = 'ballroom_partner'
            self.processed_df[partner_col] = None
        
        # 准备合并列
        merge_cols = ['celebrity_name', 'season', 
                     'celebrity_industry', 'celebrity_homestate',
                     'celebrity_homecountry/region', 'celebrity_age_during_season',
                     'placement']
        merge_cols.append(partner_col)
        
        # 只选择存在的列
        available_cols = [col for col in merge_cols if col in self.processed_df.columns]
        
        # 合并数据
        merged = self.estimates_df.merge(
            self.processed_df[available_cols],
            on=['celebrity_name', 'season'],
            how='left'
        )
        
        # 统一列名
        if 'ballroompartner' in merged.columns:
            merged['ballroom_partner'] = merged['ballroompartner']
        
        return merged
    
    def analyze_pro_dancer_impact(self) -> Dict:
        """
        分析专业舞者的影响
        
        分析内容：
        1. 不同专业舞者的平均评委评分
        2. 不同专业舞者的平均粉丝投票
        3. 专业舞者对最终排名的影响
        4. 专业舞者对评委评分和粉丝投票的影响差异
        """
        print("\n分析专业舞者的影响...")
        
        # 检查列名
        dancer_col = 'ballroom_partner' if 'ballroom_partner' in self.merged_df.columns else 'ballroompartner'
        if dancer_col not in self.merged_df.columns:
            warnings.warn("未找到专业舞者列，跳过专业舞者分析")
            return {}
        
        # 获取所有专业舞者
        pro_dancers = self.merged_df[dancer_col].dropna().unique()
        
        results = []
        
        for dancer in pro_dancers:
            dancer_data = self.merged_df[self.merged_df[dancer_col] == dancer]
            
            if len(dancer_data) == 0:
                continue
            
            # 计算平均评委评分
            avg_judge_score = dancer_data['judge_total'].mean()
            
            # 计算平均粉丝投票
            avg_fan_votes = dancer_data['fan_votes'].mean()
            
            # 计算平均最终排名（1为最好）
            avg_placement = dancer_data['placement'].mean()
            
            # 统计该专业舞者合作的选手数量
            n_celebrities = dancer_data['celebrity_name'].nunique()
            
            # 统计该专业舞者参与的周次数量
            n_weeks = len(dancer_data)
            
            # 统计该专业舞者获得的最佳排名
            best_placement = dancer_data['placement'].min()
            
            # 统计该专业舞者进入决赛的次数
            finalists = dancer_data[dancer_data['placement'] <= 3]
            n_finalists = len(finalists)
            
            results.append({
                'pro_dancer': dancer,
                'n_celebrities': n_celebrities,
                'n_weeks': n_weeks,
                'avg_judge_score': avg_judge_score,
                'avg_fan_votes': avg_fan_votes,
                'avg_placement': avg_placement,
                'best_placement': best_placement,
                'n_finalists': n_finalists,
                'finalist_rate': n_finalists / n_weeks if n_weeks > 0 else 0
            })
        
        results_df = pd.DataFrame(results)
        
        # 排序：按平均排名（越小越好）
        results_df = results_df.sort_values('avg_placement')
        
        # 计算相关性
        judge_correlation = self._calculate_dancer_correlation('judge_total')
        fan_correlation = self._calculate_dancer_correlation('fan_votes')
        
        analysis = {
            'pro_dancer_stats': results_df.to_dict('records'),
            'correlations': {
                'judge_score_correlation': judge_correlation,
                'fan_votes_correlation': fan_correlation
            },
            'summary': {
                'total_pro_dancers': len(results_df),
                'top_5_dancers_by_placement': results_df.head(5)[['pro_dancer', 'avg_placement', 'avg_judge_score', 'avg_fan_votes']].to_dict('records'),
                'bottom_5_dancers_by_placement': results_df.tail(5)[['pro_dancer', 'avg_placement', 'avg_judge_score', 'avg_fan_votes']].to_dict('records')
            }
        }
        
        return analysis
    
    def _calculate_dancer_correlation(self, target_col: str) -> float:
        """计算专业舞者对目标变量的相关性"""
        # 检查列名
        dancer_col = 'ballroom_partner' if 'ballroom_partner' in self.merged_df.columns else 'ballroompartner'
        if dancer_col not in self.merged_df.columns:
            return 0.0
        
        # 使用专业舞者作为分类变量，计算与目标变量的相关性
        dancer_encoded = pd.Categorical(self.merged_df[dancer_col]).codes
        target = self.merged_df[target_col].values
        
        # 移除NaN
        mask = ~(np.isnan(dancer_encoded) | np.isnan(target))
        if mask.sum() < 10:
            return 0.0
        
        dancer_clean = dancer_encoded[mask]
        target_clean = target[mask]
        
        # 计算相关系数
        if len(np.unique(dancer_clean)) < 2:
            return 0.0
        
        try:
            correlation, p_value = stats.pearsonr(dancer_clean, target_clean)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def analyze_celebrity_features_impact(self) -> Dict:
        """
        分析选手特征的影响
        
        分析内容：
        1. 年龄对评委评分和粉丝投票的影响
        2. 行业对评委评分和粉丝投票的影响
        3. 地区对评委评分和粉丝投票的影响
        4. 这些特征对两种投票的不同影响
        """
        print("\n分析选手特征的影响...")
        
        analysis = {}
        
        # 1. 年龄影响分析
        age_analysis = self._analyze_age_impact()
        analysis['age'] = age_analysis
        
        # 2. 行业影响分析
        industry_analysis = self._analyze_industry_impact()
        analysis['industry'] = industry_analysis
        
        # 3. 地区影响分析
        region_analysis = self._analyze_region_impact()
        analysis['region'] = region_analysis
        
        return analysis
    
    def _analyze_age_impact(self) -> Dict:
        """分析年龄的影响"""
        data = self.merged_df[
            self.merged_df['celebrity_age_during_season'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        ages = data['celebrity_age_during_season'].values
        judge_scores = data['judge_total'].values
        fan_votes = data['fan_votes'].values
        
        # 计算相关系数
        judge_corr, judge_p = stats.pearsonr(ages, judge_scores) if len(ages) > 1 else (0, 1)
        fan_corr, fan_p = stats.pearsonr(ages, fan_votes) if len(ages) > 1 else (0, 1)
        
        # 按年龄分组分析
        age_groups = pd.cut(ages, bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
        data['age_group'] = age_groups
        
        group_stats = data.groupby('age_group').agg({
            'judge_total': ['mean', 'std', 'count'],
            'fan_votes': ['mean', 'std'],
            'placement': 'mean'
        }).round(2)
        
        # 将MultiIndex转换为字符串键（JSON不支持tuple键）
        group_stats_dict = {}
        for col in group_stats.columns:
            if isinstance(col, tuple):
                key = f"{col[0]}_{col[1]}"
            else:
                key = str(col)
            group_stats_dict[key] = group_stats[col].to_dict()
        
        return {
            'correlation_with_judge_score': {
                'correlation': float(judge_corr) if not np.isnan(judge_corr) else 0.0,
                'p_value': float(judge_p) if not np.isnan(judge_p) else 1.0,
                'significant': bool(judge_p < 0.05 if not np.isnan(judge_p) else False)
            },
            'correlation_with_fan_votes': {
                'correlation': float(fan_corr) if not np.isnan(fan_corr) else 0.0,
                'p_value': float(fan_p) if not np.isnan(fan_p) else 1.0,
                'significant': bool(fan_p < 0.05 if not np.isnan(fan_p) else False)
            },
            'group_statistics': group_stats_dict,
            'age_range': {
                'min': float(ages.min()),
                'max': float(ages.max()),
                'mean': float(ages.mean()),
                'median': float(np.median(ages))
            }
        }
    
    def _analyze_industry_impact(self) -> Dict:
        """分析行业的影响"""
        data = self.merged_df[
            self.merged_df['celebrity_industry'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        # 按行业分组统计
        industry_stats = data.groupby('celebrity_industry').agg({
            'judge_total': ['mean', 'std', 'count'],
            'fan_votes': ['mean', 'std'],
            'placement': 'mean'
        }).round(2)
        
        # 将MultiIndex转换为字符串键（JSON不支持tuple键）
        industry_stats_dict = {}
        for col in industry_stats.columns:
            if isinstance(col, tuple):
                key = f"{col[0]}_{col[1]}"
            else:
                key = str(col)
            industry_stats_dict[key] = industry_stats[col].to_dict()
        
        # 计算每个行业的平均表现
        industry_performance = []
        for industry in industry_stats.index:
            industry_data = data[data['celebrity_industry'] == industry]
            industry_performance.append({
                'industry': industry,
                'avg_judge_score': float(industry_data['judge_total'].mean()),
                'avg_fan_votes': float(industry_data['fan_votes'].mean()),
                'avg_placement': float(industry_data['placement'].mean()),
                'n_celebrities': int(industry_data['celebrity_name'].nunique()),
                'n_weeks': int(len(industry_data))
            })
        
        # 排序：按平均排名
        industry_performance = sorted(industry_performance, key=lambda x: x['avg_placement'])
        
        return {
            'industry_statistics': industry_stats_dict,
            'industry_performance': industry_performance,
            'top_industries_by_judge_score': sorted(industry_performance, key=lambda x: x['avg_judge_score'], reverse=True)[:5],
            'top_industries_by_fan_votes': sorted(industry_performance, key=lambda x: x['avg_fan_votes'], reverse=True)[:5],
            'top_industries_by_placement': industry_performance[:5]
        }
    
    def _analyze_region_impact(self) -> Dict:
        """分析地区的影响"""
        data = self.merged_df[
            self.merged_df['celebrity_homecountry/region'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        # 按国家/地区分组统计
        region_stats = data.groupby('celebrity_homecountry/region').agg({
            'judge_total': ['mean', 'std', 'count'],
            'fan_votes': ['mean', 'std'],
            'placement': 'mean'
        }).round(2)
        
        # 将MultiIndex转换为字符串键（JSON不支持tuple键）
        region_stats_dict = {}
        for col in region_stats.columns:
            if isinstance(col, tuple):
                key = f"{col[0]}_{col[1]}"
            else:
                key = str(col)
            region_stats_dict[key] = region_stats[col].to_dict()
        
        # 计算每个地区的平均表现
        region_performance = []
        for region in region_stats.index:
            region_data = data[data['celebrity_homecountry/region'] == region]
            region_performance.append({
                'region': region,
                'avg_judge_score': float(region_data['judge_total'].mean()),
                'avg_fan_votes': float(region_data['fan_votes'].mean()),
                'avg_placement': float(region_data['placement'].mean()),
                'n_celebrities': int(region_data['celebrity_name'].nunique()),
                'n_weeks': int(len(region_data))
            })
        
        # 排序：按平均排名
        region_performance = sorted(region_performance, key=lambda x: x['avg_placement'])
        
        return {
            'region_statistics': region_stats_dict,
            'region_performance': region_performance,
            'top_regions_by_judge_score': sorted(region_performance, key=lambda x: x['avg_judge_score'], reverse=True)[:5],
            'top_regions_by_fan_votes': sorted(region_performance, key=lambda x: x['avg_fan_votes'], reverse=True)[:5],
            'top_regions_by_placement': region_performance[:5]
        }
    
    def compare_judge_vs_fan_impacts(self) -> Dict:
        """
        比较影响因素对评委评分和粉丝投票的不同影响
        
        分析内容：
        1. 专业舞者对评委评分和粉丝投票的影响差异
        2. 年龄对两种投票的影响差异
        3. 行业对两种投票的影响差异
        4. 地区对两种投票的影响差异
        """
        print("\n比较影响因素对评委评分和粉丝投票的不同影响...")
        
        comparison = {}
        
        # 1. 专业舞者影响比较
        pro_dancer_comparison = self._compare_dancer_impact_judge_vs_fan()
        comparison['pro_dancer'] = pro_dancer_comparison
        
        # 2. 年龄影响比较
        age_comparison = self._compare_age_impact_judge_vs_fan()
        comparison['age'] = age_comparison
        
        # 3. 行业影响比较
        industry_comparison = self._compare_industry_impact_judge_vs_fan()
        comparison['industry'] = industry_comparison
        
        # 4. 地区影响比较
        region_comparison = self._compare_region_impact_judge_vs_fan()
        comparison['region'] = region_comparison
        
        return comparison
    
    def _compare_dancer_impact_judge_vs_fan(self) -> Dict:
        """比较专业舞者对评委评分和粉丝投票的影响差异"""
        # 检查列名
        dancer_col = 'ballroom_partner' if 'ballroom_partner' in self.merged_df.columns else 'ballroompartner'
        if dancer_col not in self.merged_df.columns:
            return {}
        
        data = self.merged_df[
            self.merged_df[dancer_col].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        # 计算每个专业舞者对评委评分和粉丝投票的影响
        dancer_impact = []
        
        for dancer in data[dancer_col].unique():
            dancer_data = data[data[dancer_col] == dancer]
            
            if len(dancer_data) < 5:  # 至少需要5个数据点
                continue
            
            avg_judge = dancer_data['judge_total'].mean()
            avg_fan = dancer_data['fan_votes'].mean()
            
            # 与总体平均比较
            overall_judge = data['judge_total'].mean()
            overall_fan = data['fan_votes'].mean()
            
            judge_impact = avg_judge - overall_judge
            fan_impact = avg_fan - overall_fan
            
            dancer_impact.append({
                'pro_dancer': dancer,
                'judge_impact': float(judge_impact),
                'fan_impact': float(fan_impact),
                'impact_difference': float(judge_impact - fan_impact),
                'n_weeks': len(dancer_data)
            })
        
        dancer_impact_df = pd.DataFrame(dancer_impact)
        
        # 计算影响差异的统计量
        impact_diff = dancer_impact_df['impact_difference'].values
        impact_diff_clean = impact_diff[~np.isnan(impact_diff)]
        
        return {
            'dancer_impacts': dancer_impact_df.to_dict('records'),
            'summary': {
                'mean_impact_difference': float(np.mean(impact_diff_clean)) if len(impact_diff_clean) > 0 else 0.0,
                'std_impact_difference': float(np.std(impact_diff_clean)) if len(impact_diff_clean) > 0 else 0.0,
                'dancers_with_positive_judge_impact': int((dancer_impact_df['judge_impact'] > 0).sum()),
                'dancers_with_positive_fan_impact': int((dancer_impact_df['fan_impact'] > 0).sum())
            }
        }
    
    def _compare_age_impact_judge_vs_fan(self) -> Dict:
        """比较年龄对评委评分和粉丝投票的影响差异"""
        data = self.merged_df[
            self.merged_df['celebrity_age_during_season'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        ages = data['celebrity_age_during_season'].values
        judge_scores = data['judge_total'].values
        fan_votes = data['fan_votes'].values
        
        # 计算相关系数
        judge_corr, _ = stats.pearsonr(ages, judge_scores) if len(ages) > 1 else (0, 1)
        fan_corr, _ = stats.pearsonr(ages, fan_votes) if len(ages) > 1 else (0, 1)
        
        # 线性回归分析
        judge_slope, judge_intercept = np.polyfit(ages, judge_scores, 1) if len(ages) > 1 else (0, 0)
        fan_slope, fan_intercept = np.polyfit(ages, fan_votes, 1) if len(ages) > 1 else (0, 0)
        
        return {
            'correlation_comparison': {
                'judge_score_correlation': float(judge_corr) if not np.isnan(judge_corr) else 0.0,
                'fan_votes_correlation': float(fan_corr) if not np.isnan(fan_corr) else 0.0,
                'correlation_difference': float(judge_corr - fan_corr) if not (np.isnan(judge_corr) or np.isnan(fan_corr)) else 0.0
            },
            'regression_comparison': {
                'judge_score_slope': float(judge_slope),
                'fan_votes_slope': float(fan_slope),
                'slope_difference': float(judge_slope - fan_slope)
            },
            'interpretation': {
                'age_affects_judge_more': bool(abs(judge_corr) > abs(fan_corr) if not (np.isnan(judge_corr) or np.isnan(fan_corr)) else False),
                'age_affects_fan_more': bool(abs(fan_corr) > abs(judge_corr) if not (np.isnan(judge_corr) or np.isnan(fan_corr)) else False)
            }
        }
    
    def _compare_industry_impact_judge_vs_fan(self) -> Dict:
        """比较行业对评委评分和粉丝投票的影响差异"""
        data = self.merged_df[
            self.merged_df['celebrity_industry'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        # 计算每个行业对两种投票的影响
        industry_impact = []
        
        overall_judge = data['judge_total'].mean()
        overall_fan = data['fan_votes'].mean()
        
        for industry in data['celebrity_industry'].unique():
            industry_data = data[data['celebrity_industry'] == industry]
            
            if len(industry_data) < 5:  # 至少需要5个数据点
                continue
            
            avg_judge = industry_data['judge_total'].mean()
            avg_fan = industry_data['fan_votes'].mean()
            
            judge_impact = avg_judge - overall_judge
            fan_impact = avg_fan - overall_fan
            
            industry_impact.append({
                'industry': industry,
                'judge_impact': float(judge_impact),
                'fan_impact': float(fan_impact),
                'impact_difference': float(judge_impact - fan_impact),
                'n_weeks': len(industry_data)
            })
        
        industry_impact_df = pd.DataFrame(industry_impact)
        
        return {
            'industry_impacts': industry_impact_df.to_dict('records'),
            'summary': {
                'mean_impact_difference': float(industry_impact_df['impact_difference'].mean()) if len(industry_impact_df) > 0 else 0.0,
                'industries_with_positive_judge_impact': int((industry_impact_df['judge_impact'] > 0).sum()),
                'industries_with_positive_fan_impact': int((industry_impact_df['fan_impact'] > 0).sum())
            }
        }
    
    def _compare_region_impact_judge_vs_fan(self) -> Dict:
        """比较地区对评委评分和粉丝投票的影响差异"""
        data = self.merged_df[
            self.merged_df['celebrity_homecountry/region'].notna()
        ].copy()
        
        if len(data) == 0:
            return {}
        
        # 计算每个地区对两种投票的影响
        region_impact = []
        
        overall_judge = data['judge_total'].mean()
        overall_fan = data['fan_votes'].mean()
        
        for region in data['celebrity_homecountry/region'].unique():
            region_data = data[data['celebrity_homecountry/region'] == region]
            
            if len(region_data) < 5:  # 至少需要5个数据点
                continue
            
            avg_judge = region_data['judge_total'].mean()
            avg_fan = region_data['fan_votes'].mean()
            
            judge_impact = avg_judge - overall_judge
            fan_impact = avg_fan - overall_fan
            
            region_impact.append({
                'region': region,
                'judge_impact': float(judge_impact),
                'fan_impact': float(fan_impact),
                'impact_difference': float(judge_impact - fan_impact),
                'n_weeks': len(region_data)
            })
        
        region_impact_df = pd.DataFrame(region_impact)
        
        return {
            'region_impacts': region_impact_df.to_dict('records'),
            'summary': {
                'mean_impact_difference': float(region_impact_df['impact_difference'].mean()) if len(region_impact_df) > 0 else 0.0,
                'regions_with_positive_judge_impact': int((region_impact_df['judge_impact'] > 0).sum()),
                'regions_with_positive_fan_impact': int((region_impact_df['fan_impact'] > 0).sum())
            }
        }
    
    def generate_comprehensive_analysis(self) -> Dict:
        """
        生成综合分析报告
        
        Returns:
        --------
        Dict: 包含所有分析结果的字典
        """
        print("=" * 70)
        print("阶段4：影响因素分析")
        print("=" * 70)
        
        # 1. 专业舞者影响分析
        pro_dancer_analysis = self.analyze_pro_dancer_impact()
        
        # 2. 选手特征影响分析
        celebrity_features_analysis = self.analyze_celebrity_features_impact()
        
        # 3. 比较不同影响
        comparison_analysis = self.compare_judge_vs_fan_impacts()
        
        comprehensive_analysis = {
            'pro_dancer_impact': pro_dancer_analysis,
            'celebrity_features_impact': celebrity_features_analysis,
            'judge_vs_fan_comparison': comparison_analysis
        }
        
        return comprehensive_analysis


def main():
    """主函数：运行影响因素分析"""
    print("=" * 70)
    print("阶段4：影响因素分析")
    print("=" * 70)
    
    # 加载数据
    print("\n步骤1: 加载数据...")
    try:
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print(f"✓ 加载预处理数据成功 ({len(processed_df)} 条记录)")
    except FileNotFoundError:
        print("❌ 错误: 未找到 2026_MCM_Problem_C_Data_processed.csv")
        print("   请先运行阶段1进行数据预处理")
        return None
    
    try:
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print(f"✓ 加载粉丝投票估计数据成功 ({len(estimates_df)} 条记录)")
    except FileNotFoundError:
        print("❌ 错误: 未找到 fan_vote_estimates.csv")
        print("   请先运行阶段2生成粉丝投票估计数据")
        return None
    
    # 创建分析器
    print("\n步骤2: 创建影响因素分析器...")
    analyzer = FactorImpactAnalyzer(processed_df, estimates_df)
    print("✓ 分析器创建成功")
    
    # 执行综合分析
    print("\n步骤3: 执行综合分析...")
    analysis = analyzer.generate_comprehensive_analysis()
    
    # 保存分析结果
    print("\n步骤4: 保存分析结果...")
    import json
    
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
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
    
    analysis_native = convert_to_native(analysis)
    
    output_path = 'factor_impact_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 分析结果已保存到: {output_path}")
    
    # 保存专业舞者统计
    if 'pro_dancer_impact' in analysis and 'pro_dancer_stats' in analysis['pro_dancer_impact']:
        pro_dancer_df = pd.DataFrame(analysis['pro_dancer_impact']['pro_dancer_stats'])
        pro_dancer_path = 'pro_dancer_impact.csv'
        pro_dancer_df.to_csv(pro_dancer_path, index=False, encoding='utf-8-sig')
        print(f"✓ 专业舞者影响统计已保存到: {pro_dancer_path}")
    
    # 生成文本报告
    print("\n步骤5: 生成文本报告...")
    report_path = 'stage4_factor_impact_report.txt'
    generate_text_report(analysis, report_path)
    print(f"✓ 文本报告已保存到: {report_path}")
    
    print("\n" + "=" * 70)
    print("阶段4完成！所有任务已成功执行。")
    print("=" * 70)
    
    return analyzer, analysis


def generate_text_report(analysis: Dict, output_path: str):
    """生成文本格式的报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("阶段4：影响因素分析 - 摘要报告\n")
        f.write("=" * 70 + "\n\n")
        
        # 1. 专业舞者影响
        f.write("1. 专业舞者影响分析\n")
        f.write("-" * 70 + "\n")
        if 'pro_dancer_impact' in analysis:
            pro_dancer = analysis['pro_dancer_impact']
            if 'summary' in pro_dancer:
                summary = pro_dancer['summary']
                f.write(f"   总专业舞者数: {summary.get('total_pro_dancers', 0)}\n\n")
                
                f.write("   表现最好的5位专业舞者（按平均排名）:\n")
                for i, dancer in enumerate(summary.get('top_5_dancers_by_placement', [])[:5], 1):
                    f.write(f"   {i}. {dancer.get('pro_dancer', 'N/A')}: "
                           f"平均排名 {dancer.get('avg_placement', 0):.2f}, "
                           f"平均评委评分 {dancer.get('avg_judge_score', 0):.2f}, "
                           f"平均粉丝投票 {dancer.get('avg_fan_votes', 0):.2f}\n")
                f.write("\n")
        
        # 2. 年龄影响
        f.write("2. 年龄影响分析\n")
        f.write("-" * 70 + "\n")
        if 'celebrity_features_impact' in analysis and 'age' in analysis['celebrity_features_impact']:
            age = analysis['celebrity_features_impact']['age']
            if 'correlation_with_judge_score' in age:
                judge_corr = age['correlation_with_judge_score']
                f.write(f"   年龄与评委评分的相关性: {judge_corr.get('correlation', 0):.4f} "
                       f"(p={judge_corr.get('p_value', 1):.4f}, "
                       f"显著: {'是' if judge_corr.get('significant', False) else '否'})\n")
            if 'correlation_with_fan_votes' in age:
                fan_corr = age['correlation_with_fan_votes']
                f.write(f"   年龄与粉丝投票的相关性: {fan_corr.get('correlation', 0):.4f} "
                       f"(p={fan_corr.get('p_value', 1):.4f}, "
                       f"显著: {'是' if fan_corr.get('significant', False) else '否'})\n")
            f.write("\n")
        
        # 3. 行业影响
        f.write("3. 行业影响分析\n")
        f.write("-" * 70 + "\n")
        if 'celebrity_features_impact' in analysis and 'industry' in analysis['celebrity_features_impact']:
            industry = analysis['celebrity_features_impact']['industry']
            if 'top_industries_by_placement' in industry:
                f.write("   表现最好的5个行业（按平均排名）:\n")
                for i, ind in enumerate(industry['top_industries_by_placement'][:5], 1):
                    f.write(f"   {i}. {ind.get('industry', 'N/A')}: "
                           f"平均排名 {ind.get('avg_placement', 0):.2f}, "
                           f"平均评委评分 {ind.get('avg_judge_score', 0):.2f}, "
                           f"平均粉丝投票 {ind.get('avg_fan_votes', 0):.2f}\n")
                f.write("\n")
        
        # 4. 地区影响
        f.write("4. 地区影响分析\n")
        f.write("-" * 70 + "\n")
        if 'celebrity_features_impact' in analysis and 'region' in analysis['celebrity_features_impact']:
            region = analysis['celebrity_features_impact']['region']
            if 'top_regions_by_placement' in region:
                f.write("   表现最好的5个地区（按平均排名）:\n")
                for i, reg in enumerate(region['top_regions_by_placement'][:5], 1):
                    f.write(f"   {i}. {reg.get('region', 'N/A')}: "
                           f"平均排名 {reg.get('avg_placement', 0):.2f}, "
                           f"平均评委评分 {reg.get('avg_judge_score', 0):.2f}, "
                           f"平均粉丝投票 {reg.get('avg_fan_votes', 0):.2f}\n")
                f.write("\n")
        
        # 5. 评委评分 vs 粉丝投票的影响差异
        f.write("5. 影响因素对评委评分和粉丝投票的不同影响\n")
        f.write("-" * 70 + "\n")
        if 'judge_vs_fan_comparison' in analysis:
            comparison = analysis['judge_vs_fan_comparison']
            
            # 年龄影响差异
            if 'age' in comparison:
                age_comp = comparison['age']
                if 'correlation_comparison' in age_comp:
                    corr_comp = age_comp['correlation_comparison']
                    f.write("   年龄影响差异:\n")
                    f.write(f"     评委评分相关性: {corr_comp.get('judge_score_correlation', 0):.4f}\n")
                    f.write(f"     粉丝投票相关性: {corr_comp.get('fan_votes_correlation', 0):.4f}\n")
                    f.write(f"     差异: {corr_comp.get('correlation_difference', 0):.4f}\n")
                    if 'interpretation' in age_comp:
                        interp = age_comp['interpretation']
                        if interp.get('age_affects_judge_more', False):
                            f.write("     结论: 年龄对评委评分的影响更大\n")
                        elif interp.get('age_affects_fan_more', False):
                            f.write("     结论: 年龄对粉丝投票的影响更大\n")
                    f.write("\n")
            
            # 专业舞者影响差异
            if 'pro_dancer' in comparison:
                dancer_comp = comparison['pro_dancer']
                if 'summary' in dancer_comp:
                    summary = dancer_comp['summary']
                    f.write("   专业舞者影响差异:\n")
                    f.write(f"     平均影响差异: {summary.get('mean_impact_difference', 0):.2f}\n")
                    f.write(f"     对评委评分有正面影响的专业舞者数: {summary.get('dancers_with_positive_judge_impact', 0)}\n")
                    f.write(f"     对粉丝投票有正面影响的专业舞者数: {summary.get('dancers_with_positive_fan_impact', 0)}\n")
                    f.write("\n")


if __name__ == "__main__":
    analyzer, analysis = main()
