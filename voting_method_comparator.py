"""
阶段3：投票方法比较分析
比较排名法和百分比法在所有季次中的表现差异
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class VotingMethodComparator:
    """投票方法比较器：比较排名法和百分比法"""
    
    def __init__(self, estimates_df: pd.DataFrame, processed_df: pd.DataFrame):
        """
        初始化比较器
        
        Parameters:
        -----------
        estimates_df : pd.DataFrame
            阶段2估计的粉丝投票数据（fan_vote_estimates.csv）
        processed_df : pd.DataFrame
            预处理后的数据（2026_MCM_Problem_C_Data_processed.csv）
        """
        self.estimates_df = estimates_df.copy()
        self.processed_df = processed_df.copy()
        self.comparison_results = []
        
    def calculate_rank_method(
        self,
        judge_totals: np.ndarray,
        fan_votes: np.ndarray
    ) -> Dict:
        """
        计算排名法的综合排名
        
        Parameters:
        -----------
        judge_totals : np.ndarray
            评委总分数组
        fan_votes : np.ndarray
            粉丝投票数组
        
        Returns:
        --------
        Dict: 包含排名信息的字典
        """
        n = len(judge_totals)
        
        if n == 0:
            return {}
        
        # 计算排名（1=最好，n=最差）
        # 评委排名：总分越高排名越好（排名越小）
        judge_ranks = pd.Series(judge_totals).rank(
            ascending=False, 
            method='min'
        ).astype(int).values
        
        # 粉丝排名：投票越多排名越好（排名越小）
        fan_ranks = pd.Series(fan_votes).rank(
            ascending=False, 
            method='min'
        ).astype(int).values
        
        # 综合排名 = 评委排名 + 粉丝排名
        combined_ranks = judge_ranks + fan_ranks
        
        # 找出应该被淘汰的（综合排名最高，即最差）
        eliminated_idx = np.argmax(combined_ranks)
        
        return {
            'judge_ranks': judge_ranks,
            'fan_ranks': fan_ranks,
            'combined_ranks': combined_ranks,
            'eliminated_idx': eliminated_idx,
            'eliminated_combined_rank': combined_ranks[eliminated_idx]
        }
    
    def calculate_percent_method(
        self,
        judge_totals: np.ndarray,
        fan_votes: np.ndarray
    ) -> Dict:
        """
        计算百分比法的综合百分比
        
        Parameters:
        -----------
        judge_totals : np.ndarray
            评委总分数组
        fan_votes : np.ndarray
            粉丝投票数组
        
        Returns:
        --------
        Dict: 包含百分比信息的字典
        """
        n = len(judge_totals)
        
        if n == 0:
            return {}
        
        # 计算百分比
        judge_sum = np.sum(judge_totals)
        fan_sum = np.sum(fan_votes)
        
        if judge_sum == 0 or fan_sum == 0:
            return {}
        
        judge_percents = (judge_totals / judge_sum) * 100
        fan_percents = (fan_votes / fan_sum) * 100
        
        # 综合百分比 = 评委百分比 + 粉丝百分比
        combined_percents = judge_percents + fan_percents
        
        # 找出应该被淘汰的（综合百分比最低）
        eliminated_idx = np.argmin(combined_percents)
        
        return {
            'judge_percents': judge_percents,
            'fan_percents': fan_percents,
            'combined_percents': combined_percents,
            'eliminated_idx': eliminated_idx,
            'eliminated_combined_percent': combined_percents[eliminated_idx]
        }
    
    def compare_methods_for_week(
        self,
        season: int,
        week: int
    ) -> Optional[Dict]:
        """
        对某一周比较两种方法
        
        Parameters:
        -----------
        season : int
            季数
        week : int
            周次
        
        Returns:
        --------
        Dict or None: 比较结果，如果数据不足则返回None
        """
        # 获取该周的数据
        week_estimates = self.estimates_df[
            (self.estimates_df['season'] == season) & 
            (self.estimates_df['week'] == week)
        ].copy()
        
        if len(week_estimates) == 0:
            return None
        
        # 获取评委总分和粉丝投票
        judge_totals = week_estimates['judge_total'].values
        fan_votes = week_estimates['fan_votes'].values
        celebrity_names = week_estimates['celebrity_name'].values
        
        # 计算排名法
        rank_result = self.calculate_rank_method(judge_totals, fan_votes)
        if not rank_result:
            return None
        
        # 计算百分比法
        percent_result = self.calculate_percent_method(judge_totals, fan_votes)
        if not percent_result:
            return None
        
        # 获取实际被淘汰的选手
        actual_eliminated = week_estimates[week_estimates['eliminated'] == True]
        actual_eliminated_name = None
        if len(actual_eliminated) > 0:
            actual_eliminated_name = actual_eliminated.iloc[0]['celebrity_name']
        
        # 找出两种方法预测的被淘汰者
        rank_eliminated_idx = rank_result['eliminated_idx']
        percent_eliminated_idx = percent_result['eliminated_idx']
        
        rank_eliminated_name = celebrity_names[rank_eliminated_idx]
        percent_eliminated_name = celebrity_names[percent_eliminated_idx]
        
        # 检查两种方法是否一致
        methods_agree = (rank_eliminated_idx == percent_eliminated_idx)
        
        # 检查每种方法是否与实际一致
        rank_correct = (rank_eliminated_name == actual_eliminated_name) if actual_eliminated_name else None
        percent_correct = (percent_eliminated_name == actual_eliminated_name) if actual_eliminated_name else None
        
        # 构建结果
        result = {
            'season': season,
            'week': week,
            'n_contestants': len(week_estimates),
            'actual_eliminated': actual_eliminated_name,
            # 排名法结果
            'rank_method': {
                'eliminated': rank_eliminated_name,
                'eliminated_idx': int(rank_eliminated_idx),
                'correct': rank_correct
            },
            # 百分比法结果
            'percent_method': {
                'eliminated': percent_eliminated_name,
                'eliminated_idx': int(percent_eliminated_idx),
                'correct': percent_correct
            },
            # 比较结果
            'methods_agree': methods_agree,
            # 详细数据（用于后续分析）
            'details': {
                'celebrity_names': celebrity_names.tolist(),
                'judge_totals': judge_totals.tolist(),
                'fan_votes': fan_votes.tolist(),
                'judge_ranks': rank_result['judge_ranks'].tolist(),
                'fan_ranks': rank_result['fan_ranks'].tolist(),
                'combined_ranks': rank_result['combined_ranks'].tolist(),
                'judge_percents': percent_result['judge_percents'].tolist(),
                'fan_percents': percent_result['fan_percents'].tolist(),
                'combined_percents': percent_result['combined_percents'].tolist()
            }
        }
        
        return result
    
    def compare_all_weeks(
        self,
        seasons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        比较所有周次的两种方法
        
        Parameters:
        -----------
        seasons : List[int], optional
            要处理的季数列表，如果为None则处理所有季
        
        Returns:
        --------
        pd.DataFrame: 包含所有比较结果的DataFrame
        """
        if seasons is None:
            seasons = sorted(self.estimates_df['season'].unique())
        
        results = []
        
        for season in seasons:
            print(f"\n处理第 {season} 季...")
            season_estimates = self.estimates_df[self.estimates_df['season'] == season]
            weeks = sorted(season_estimates['week'].unique())
            
            for week in weeks:
                print(f"  周 {week}...", end=' ')
                result = self.compare_methods_for_week(season, week)
                
                if result is None:
                    print("跳过（数据不足）")
                    continue
                
                # 展开结果到扁平结构
                flat_result = {
                    'season': result['season'],
                    'week': result['week'],
                    'n_contestants': result['n_contestants'],
                    'actual_eliminated': result['actual_eliminated'],
                    'rank_method_eliminated': result['rank_method']['eliminated'],
                    'rank_method_correct': result['rank_method']['correct'],
                    'percent_method_eliminated': result['percent_method']['eliminated'],
                    'percent_method_correct': result['percent_method']['correct'],
                    'methods_agree': result['methods_agree']
                }
                
                results.append(flat_result)
                print("完成")
        
        self.comparison_results = results
        return pd.DataFrame(results)
    
    def analyze_differences(self, comparison_df: pd.DataFrame) -> Dict:
        """
        分析两种方法的差异
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            比较结果DataFrame
        
        Returns:
        --------
        Dict: 差异分析结果
        """
        total_weeks = len(comparison_df)
        
        # 统计差异
        methods_agree_count = comparison_df['methods_agree'].sum()
        methods_disagree_count = total_weeks - methods_agree_count
        
        # 统计准确率
        rank_correct = comparison_df['rank_method_correct'].sum()
        percent_correct = comparison_df['percent_method_correct'].sum()
        
        rank_total = comparison_df['rank_method_correct'].notna().sum()
        percent_total = comparison_df['percent_method_correct'].notna().sum()
        
        rank_accuracy = rank_correct / rank_total if rank_total > 0 else 0
        percent_accuracy = percent_correct / percent_total if percent_total > 0 else 0
        
        # 按季统计差异
        season_stats = []
        for season in sorted(comparison_df['season'].unique()):
            season_data = comparison_df[comparison_df['season'] == season]
            season_total = len(season_data)
            season_disagree = (~season_data['methods_agree']).sum()
            
            season_stats.append({
                'season': season,
                'total_weeks': season_total,
                'disagree_count': season_disagree,
                'disagree_rate': season_disagree / season_total if season_total > 0 else 0
            })
        
        # 按选手数量统计差异
        n_contestants_stats = []
        for n in sorted(comparison_df['n_contestants'].unique()):
            n_data = comparison_df[comparison_df['n_contestants'] == n]
            n_total = len(n_data)
            n_disagree = (~n_data['methods_agree']).sum()
            
            n_contestants_stats.append({
                'n_contestants': n,
                'total_weeks': n_total,
                'disagree_count': n_disagree,
                'disagree_rate': n_disagree / n_total if n_total > 0 else 0
            })
        
        analysis = {
            'total_weeks': int(total_weeks),
            'methods_agree_count': int(methods_agree_count),
            'methods_disagree_count': int(methods_disagree_count),
            'disagree_rate': float(methods_disagree_count / total_weeks if total_weeks > 0 else 0),
            'rank_method': {
                'correct_predictions': int(rank_correct),
                'total_predictions': int(rank_total),
                'accuracy': float(rank_accuracy)
            },
            'percent_method': {
                'correct_predictions': int(percent_correct),
                'total_predictions': int(percent_total),
                'accuracy': float(percent_accuracy)
            },
            'season_stats': [
                {
                    'season': int(stat['season']),
                    'total_weeks': int(stat['total_weeks']),
                    'disagree_count': int(stat['disagree_count']),
                    'disagree_rate': float(stat['disagree_rate'])
                }
                for stat in season_stats
            ],
            'n_contestants_stats': [
                {
                    'n_contestants': int(stat['n_contestants']),
                    'total_weeks': int(stat['total_weeks']),
                    'disagree_count': int(stat['disagree_count']),
                    'disagree_rate': float(stat['disagree_rate'])
                }
                for stat in n_contestants_stats
            ]
        }
        
        return analysis
    
    def analyze_controversial_cases(
        self,
        comparison_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        分析争议案例
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            比较结果DataFrame
        
        Returns:
        --------
        pd.DataFrame: 争议案例分析结果
        """
        # 定义争议案例（根据README）
        controversial_cases = [
            {'season': 2, 'celebrity': 'Jerry Rice', 'description': '尽管评委评分最低（5周），但仍进入决赛获得亚军'},
            {'season': 4, 'celebrity': 'Billy Ray Cyrus', 'description': '尽管6周评委评分垫底，仍获得第5名'},
            {'season': 11, 'celebrity': 'Bristol Palin', 'description': '尽管12次评委评分最低，仍获得第3名'},
            {'season': 27, 'celebrity': 'Bobby Bones', 'description': '尽管评委评分持续偏低，仍获得冠军'}
        ]
        
        results = []
        
        for case in controversial_cases:
            season = case['season']
            celebrity = case['celebrity']
            
            # 获取该选手在该季的所有周次数据
            case_data = comparison_df[
                (comparison_df['season'] == season) &
                (
                    (comparison_df['actual_eliminated'] == celebrity) |
                    (comparison_df['rank_method_eliminated'] == celebrity) |
                    (comparison_df['percent_method_eliminated'] == celebrity)
                )
            ].copy()
            
            if len(case_data) == 0:
                # 尝试从estimates_df中查找
                case_estimates = self.estimates_df[
                    (self.estimates_df['season'] == season) &
                    (self.estimates_df['celebrity_name'] == celebrity)
                ]
                
                if len(case_estimates) > 0:
                    weeks = sorted(case_estimates['week'].unique())
                    for week in weeks:
                        week_data = case_estimates[case_estimates['week'] == week]
                        if len(week_data) > 0:
                            row = week_data.iloc[0]
                            results.append({
                                'season': season,
                                'celebrity': celebrity,
                                'week': week,
                                'description': case['description'],
                                'judge_total': row['judge_total'],
                                'fan_votes': row['fan_votes'],
                                'judge_rank': None,  # 需要从processed_df获取
                                'fan_rank': None,
                                'rank_method_would_eliminate': None,
                                'percent_method_would_eliminate': None,
                                'actual_result': 'Survived' if not row['eliminated'] else 'Eliminated'
                            })
                continue
            
            # 分析该案例
            for _, row in case_data.iterrows():
                # 获取详细信息
                week = row['week']
                week_estimates = self.estimates_df[
                    (self.estimates_df['season'] == season) &
                    (self.estimates_df['week'] == week) &
                    (self.estimates_df['celebrity_name'] == celebrity)
                ]
                
                if len(week_estimates) == 0:
                    continue
                
                celeb_data = week_estimates.iloc[0]
                
                # 获取该周所有选手数据以计算排名
                all_week_data = self.estimates_df[
                    (self.estimates_df['season'] == season) &
                    (self.estimates_df['week'] == week)
                ]
                
                judge_rank = all_week_data['judge_total'].rank(ascending=False, method='min')
                fan_rank = all_week_data['fan_votes'].rank(ascending=False, method='min')
                
                celeb_judge_rank = judge_rank[all_week_data['celebrity_name'] == celebrity].values[0]
                celeb_fan_rank = fan_rank[all_week_data['celebrity_name'] == celebrity].values[0]
                
                results.append({
                    'season': season,
                    'celebrity': celebrity,
                    'week': week,
                    'description': case['description'],
                    'judge_total': celeb_data['judge_total'],
                    'fan_votes': celeb_data['fan_votes'],
                    'judge_rank': int(celeb_judge_rank),
                    'fan_rank': int(celeb_fan_rank),
                    'rank_method_would_eliminate': (row['rank_method_eliminated'] == celebrity),
                    'percent_method_would_eliminate': (row['percent_method_eliminated'] == celebrity),
                    'actual_result': 'Survived' if not celeb_data['eliminated'] else 'Eliminated',
                    'methods_agree': row['methods_agree']
                })
        
        return pd.DataFrame(results)


def main():
    """主函数：运行投票方法比较"""
    print("=" * 70)
    print("阶段3：投票方法比较分析")
    print("=" * 70)
    
    # 加载数据
    print("\n步骤1: 加载数据...")
    try:
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print(f"✓ 加载粉丝投票估计数据成功 ({len(estimates_df)} 条记录)")
    except FileNotFoundError:
        print("❌ 错误: 未找到 fan_vote_estimates.csv")
        print("   请先运行阶段2生成粉丝投票估计数据")
        return None
    
    try:
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print(f"✓ 加载预处理数据成功 ({len(processed_df)} 条记录)")
    except FileNotFoundError:
        print("❌ 错误: 未找到 2026_MCM_Problem_C_Data_processed.csv")
        print("   请先运行阶段1进行数据预处理")
        return None
    
    # 创建比较器
    print("\n步骤2: 创建投票方法比较器...")
    comparator = VotingMethodComparator(estimates_df, processed_df)
    print("✓ 比较器创建成功")
    
    # 执行比较
    print("\n步骤3: 比较所有周次的两种方法...")
    print("  这可能需要一些时间，请耐心等待...")
    comparison_df = comparator.compare_all_weeks()
    
    # 保存比较结果
    print(f"\n步骤4: 保存比较结果...")
    output_path = 'voting_method_comparison.csv'
    comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 比较结果已保存到: {output_path}")
    
    # 分析差异
    print("\n步骤5: 分析差异...")
    analysis = comparator.analyze_differences(comparison_df)
    
    print(f"\n差异分析结果:")
    print(f"  总周数: {analysis['total_weeks']}")
    print(f"  两种方法一致: {analysis['methods_agree_count']} 周 ({analysis['methods_agree_count']/analysis['total_weeks']*100:.1f}%)")
    print(f"  两种方法不一致: {analysis['methods_disagree_count']} 周 ({analysis['methods_disagree_count']/analysis['total_weeks']*100:.1f}%)")
    print(f"  排名法准确率: {analysis['rank_method']['accuracy']:.2%}")
    print(f"  百分比法准确率: {analysis['percent_method']['accuracy']:.2%}")
    
    # 保存差异分析
    import json
    analysis_path = 'method_differences_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✓ 差异分析已保存到: {analysis_path}")
    
    # 分析争议案例
    print("\n步骤6: 分析争议案例...")
    controversial_df = comparator.analyze_controversial_cases(comparison_df)
    
    if len(controversial_df) > 0:
        controversial_path = 'controversial_cases_analysis.csv'
        controversial_df.to_csv(controversial_path, index=False, encoding='utf-8-sig')
        print(f"✓ 争议案例分析已保存到: {controversial_path}")
        print(f"  找到 {len(controversial_df)} 条争议案例记录")
    else:
        print("⚠️  未找到争议案例数据")
    
    # 生成报告
    print("\n步骤7: 生成摘要报告...")
    report_path = 'stage3_comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("阶段3：投票方法比较分析 - 摘要报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. 总体统计\n")
        f.write(f"   总周数: {analysis['total_weeks']}\n")
        f.write(f"   两种方法一致: {analysis['methods_agree_count']} 周\n")
        f.write(f"   两种方法不一致: {analysis['methods_disagree_count']} 周\n")
        f.write(f"   不一致率: {analysis['disagree_rate']:.2%}\n\n")
        
        f.write("2. 方法准确率\n")
        f.write(f"   排名法准确率: {analysis['rank_method']['accuracy']:.2%}\n")
        f.write(f"   百分比法准确率: {analysis['percent_method']['accuracy']:.2%}\n\n")
        
        f.write("3. 按季统计差异\n")
        for stat in analysis['season_stats']:
            f.write(f"   第{stat['season']}季: {stat['disagree_count']}/{stat['total_weeks']} 周不一致 "
                   f"({stat['disagree_rate']:.2%})\n")
        
        f.write("\n4. 按选手数量统计差异\n")
        for stat in analysis['n_contestants_stats']:
            f.write(f"   {stat['n_contestants']}位选手: {stat['disagree_count']}/{stat['total_weeks']} 周不一致 "
                   f"({stat['disagree_rate']:.2%})\n")
    
    print(f"✓ 摘要报告已保存到: {report_path}")
    
    print("\n" + "=" * 70)
    print("阶段3完成！所有任务已成功执行。")
    print("=" * 70)
    print("\n生成的文件:")
    print(f"  - {output_path}")
    print(f"  - {analysis_path}")
    if len(controversial_df) > 0:
        print(f"  - {controversial_path}")
    print(f"  - {report_path}")
    
    return comparator, comparison_df, analysis, controversial_df


if __name__ == "__main__":
    comparator, comparison_df, analysis, controversial_df = main()
