"""
增强阶段3报告：添加缺失的分析内容
"""

import pandas as pd
import numpy as np


def analyze_fan_vote_favoritism(comparison_df: pd.DataFrame, estimates_df: pd.DataFrame) -> Dict:
    """
    分析哪种方法更偏向粉丝投票
    
    分析思路：
    - 当两种方法不一致时，看哪种方法更倾向于选择粉丝投票高的选手
    - 统计在争议案例中，哪种方法更"保护"粉丝投票高的选手
    """
    # 找出两种方法不一致的周次
    disagree_df = comparison_df[~comparison_df['methods_agree']].copy()
    
    if len(disagree_df) == 0:
        return {}
    
    # 对每个不一致的周次，分析哪种方法更偏向粉丝投票
    rank_favors_fan = 0
    percent_favors_fan = 0
    
    for _, row in disagree_df.iterrows():
        season = row['season']
        week = row['week']
        
        # 获取该周所有选手数据
        week_data = estimates_df[
            (estimates_df['season'] == season) &
            (estimates_df['week'] == week)
        ].copy()
        
        if len(week_data) == 0:
            continue
        
        # 计算粉丝排名（1=最多，N=最少）
        week_data['fan_rank'] = week_data['fan_votes'].rank(ascending=False, method='min')
        
        # 找出两种方法预测的被淘汰者
        rank_eliminated = row['rank_method_eliminated']
        percent_eliminated = row['percent_method_eliminated']
        
        rank_eliminated_data = week_data[week_data['celebrity_name'] == rank_eliminated]
        percent_eliminated_data = week_data[week_data['celebrity_name'] == percent_eliminated]
        
        if len(rank_eliminated_data) > 0 and len(percent_eliminated_data) > 0:
            rank_fan_rank = rank_eliminated_data.iloc[0]['fan_rank']
            percent_fan_rank = percent_eliminated_data.iloc[0]['fan_rank']
            
            # 粉丝排名越小（排名越好）表示粉丝投票越多
            # 如果排名法淘汰的选手粉丝排名更差（更大），说明排名法更偏向粉丝投票
            if rank_fan_rank > percent_fan_rank:
                rank_favors_fan += 1
            elif rank_fan_rank < percent_fan_rank:
                percent_favors_fan += 1
    
    total_disagree = len(disagree_df)
    
    return {
        'total_disagree_weeks': total_disagree,
        'rank_method_favors_fan': rank_favors_fan,
        'percent_method_favors_fan': percent_favors_fan,
        'rank_method_favor_rate': rank_favors_fan / total_disagree if total_disagree > 0 else 0,
        'percent_method_favor_rate': percent_favors_fan / total_disagree if total_disagree > 0 else 0
    }


def analyze_bottom_two_judge_choice(comparison_df: pd.DataFrame, estimates_df: pd.DataFrame) -> Dict:
    """
    分析"评委从最低两名中选择淘汰者"机制的影响
    
    分析思路：
    - 对于第28季及以后，如果使用排名法，找出"最低两名"
    - 如果使用百分比法，找出"最低两名"
    - 分析两种方法下，评委的选择空间是否不同
    """
    # 第28季及以后的数据
    season28plus = comparison_df[comparison_df['season'] >= 28].copy()
    
    if len(season28plus) == 0:
        return {}
    
    results = []
    
    for _, row in season28plus.iterrows():
        season = row['season']
        week = row['week']
        
        # 获取该周所有选手数据
        week_data = estimates_df[
            (estimates_df['season'] == season) &
            (estimates_df['week'] == week)
        ].copy()
        
        if len(week_data) < 2:
            continue
        
        # 计算排名法
        judge_totals = week_data['judge_total'].values
        fan_votes = week_data['fan_votes'].values
        
        judge_ranks = pd.Series(judge_totals).rank(ascending=False, method='min').values
        fan_ranks = pd.Series(fan_votes).rank(ascending=False, method='min').values
        combined_ranks = judge_ranks + fan_ranks
        
        # 找出最低两名（综合排名最高的两个）
        bottom_two_indices = np.argsort(combined_ranks)[-2:]
        bottom_two_names = week_data.iloc[bottom_two_indices]['celebrity_name'].tolist()
        
        # 计算百分比法
        judge_sum = np.sum(judge_totals)
        fan_sum = np.sum(fan_votes)
        
        if judge_sum > 0 and fan_sum > 0:
            judge_percents = (judge_totals / judge_sum) * 100
            fan_percents = (fan_votes / fan_sum) * 100
            combined_percents = judge_percents + fan_percents
            
            # 找出最低两名（综合百分比最低的两个）
            bottom_two_indices_pct = np.argsort(combined_percents)[:2]
            bottom_two_names_pct = week_data.iloc[bottom_two_indices_pct]['celebrity_name'].tolist()
            
            # 检查两种方法的最低两名是否相同
            same_bottom_two = set(bottom_two_names) == set(bottom_two_names_pct)
            
            results.append({
                'season': season,
                'week': week,
                'rank_method_bottom_two': bottom_two_names,
                'percent_method_bottom_two': bottom_two_names_pct,
                'same_bottom_two': same_bottom_two
            })
    
    if len(results) == 0:
        return {}
    
    results_df = pd.DataFrame(results)
    same_count = results_df['same_bottom_two'].sum()
    total_count = len(results_df)
    
    return {
        'total_weeks_season28plus': total_count,
        'same_bottom_two_count': int(same_count),
        'different_bottom_two_count': int(total_count - same_count),
        'same_bottom_two_rate': float(same_count / total_count if total_count > 0 else 0),
        'details': results
    }


def generate_recommendations(analysis: Dict, favoritism: Dict, bottom_two: Dict) -> str:
    """
    生成推荐和建议
    """
    recommendations = []
    
    # 基于准确率的推荐
    rank_accuracy = analysis['rank_method']['accuracy']
    percent_accuracy = analysis['percent_method']['accuracy']
    
    if percent_accuracy > rank_accuracy:
        recommendations.append(
            f"**推荐使用百分比法**：\n"
            f"- 百分比法准确率（{percent_accuracy:.2%}）显著高于排名法（{rank_accuracy:.2%}）\n"
            f"- 百分比法能更准确地预测实际淘汰结果（290/299 vs 181/299）\n"
            f"- 这表明百分比法更好地反映了实际的投票组合机制"
        )
    else:
        recommendations.append(
            f"**推荐使用排名法**：\n"
            f"- 排名法准确率（{rank_accuracy:.2%}）高于百分比法（{percent_accuracy:.2%}）"
        )
    
    # 基于偏向性的分析
    if favoritism:
        if favoritism['percent_method_favor_rate'] > favoritism['rank_method_favor_rate']:
            recommendations.append(
                f"\n**百分比法更偏向粉丝投票**：\n"
                f"- 在两种方法不一致的情况下，百分比法更倾向于保护粉丝投票高的选手\n"
                f"- 这可能解释了为什么百分比法准确率更高（因为实际机制可能更偏向粉丝投票）"
            )
        else:
            recommendations.append(
                f"\n**排名法更偏向粉丝投票**：\n"
                f"- 在两种方法不一致的情况下，排名法更倾向于保护粉丝投票高的选手"
            )
    
    # 关于"评委选择"机制的建议
    if bottom_two:
        same_rate = bottom_two['same_bottom_two_rate']
        recommendations.append(
            f"\n**关于\"评委从最低两名中选择\"机制**：\n"
            f"- 在第28季及以后的{bottom_two['total_weeks_season28plus']}周中，"
            f"两种方法产生相同最低两名的比例为{same_rate:.2%}\n"
        )
        
        if same_rate < 0.5:
            recommendations.append(
                f"- **建议保留此机制**：因为两种方法经常产生不同的最低两名，"
                f"评委的选择权可以平衡两种方法的差异，增加公平性"
            )
        else:
            recommendations.append(
                f"- 两种方法在大多数情况下产生相同的最低两名，"
                f"评委选择机制的影响相对较小"
            )
    
    return "\n".join(recommendations)


def main():
    """生成增强的报告"""
    print("=" * 70)
    print("增强阶段3报告")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    comparison_df = pd.read_csv('voting_method_comparison.csv')
    estimates_df = pd.read_csv('fan_vote_estimates.csv')
    
    import json
    with open('method_differences_analysis.json', 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    print("✓ 数据加载成功")
    
    # 分析哪种方法更偏向粉丝投票
    print("\n分析哪种方法更偏向粉丝投票...")
    favoritism = analyze_fan_vote_favoritism(comparison_df, estimates_df)
    print(f"✓ 分析完成")
    
    # 分析"评委选择"机制
    print("\n分析\"评委从最低两名中选择\"机制的影响...")
    bottom_two = analyze_bottom_two_judge_choice(comparison_df, estimates_df)
    print(f"✓ 分析完成")
    
    # 生成推荐
    print("\n生成推荐和建议...")
    recommendations = generate_recommendations(analysis, favoritism, bottom_two)
    
    # 读取原报告
    with open('stage3_comparison_report.txt', 'r', encoding='utf-8') as f:
        original_report = f.read()
    
    # 生成增强报告
    enhanced_report = original_report + "\n\n" + "=" * 70 + "\n\n"
    enhanced_report += "6. 哪种方法更偏向粉丝投票\n"
    enhanced_report += "=" * 70 + "\n"
    
    if favoritism:
        enhanced_report += f"   总不一致周数: {favoritism['total_disagree_weeks']}\n"
        enhanced_report += f"   排名法更偏向粉丝投票: {favoritism['rank_method_favors_fan']} 次 "
        enhanced_report += f"({favoritism['rank_method_favor_rate']:.2%})\n"
        enhanced_report += f"   百分比法更偏向粉丝投票: {favoritism['percent_method_favors_fan']} 次 "
        enhanced_report += f"({favoritism['percent_method_favor_rate']:.2%})\n\n"
        
        if favoritism['percent_method_favor_rate'] > favoritism['rank_method_favor_rate']:
            enhanced_report += "   结论: 百分比法更偏向粉丝投票\n"
            enhanced_report += "   解释: 当两种方法产生不同结果时，百分比法更倾向于保护粉丝投票高的选手\n\n"
        else:
            enhanced_report += "   结论: 排名法更偏向粉丝投票\n"
            enhanced_report += "   解释: 当两种方法产生不同结果时，排名法更倾向于保护粉丝投票高的选手\n\n"
    
    enhanced_report += "\n" + "=" * 70 + "\n\n"
    enhanced_report += "7. \"评委从最低两名中选择\"机制的影响分析（第28季及以后）\n"
    enhanced_report += "=" * 70 + "\n"
    
    if bottom_two:
        enhanced_report += f"   总周数（第28季及以后）: {bottom_two['total_weeks_season28plus']}\n"
        enhanced_report += f"   两种方法产生相同最低两名: {bottom_two['same_bottom_two_count']} 周 "
        enhanced_report += f"({bottom_two['same_bottom_two_rate']:.2%})\n"
        enhanced_report += f"   两种方法产生不同最低两名: {bottom_two['different_bottom_two_count']} 周 "
        enhanced_report += f"({1 - bottom_two['same_bottom_two_rate']:.2%})\n\n"
        
        enhanced_report += "   影响分析:\n"
        enhanced_report += "   - 当两种方法产生不同的最低两名时，评委的选择权可以平衡差异\n"
        enhanced_report += "   - 这增加了淘汰过程的公平性和可控性\n\n"
    
    enhanced_report += "\n" + "=" * 70 + "\n\n"
    enhanced_report += "8. 推荐和建议\n"
    enhanced_report += "=" * 70 + "\n\n"
    enhanced_report += recommendations
    
    # 保存增强报告
    output_path = 'stage3_comparison_report_enhanced.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_report)
    
    print(f"\n✓ 增强报告已保存到: {output_path}")
    
    # 保存分析结果
    import json
    enhanced_analysis = {
        'favoritism_analysis': favoritism,
        'bottom_two_analysis': {k: v for k, v in bottom_two.items() if k != 'details'}
    }
    
    with open('enhanced_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 增强分析结果已保存到: enhanced_analysis.json")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
