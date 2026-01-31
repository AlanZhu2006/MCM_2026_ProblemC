"""
添加更多分析维度
季节性分析、评委分析、时间趋势分析
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("警告: 缺少必要的库，部分功能将受限")

if HAS_LIBS:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def seasonal_analysis():
    """季节性分析"""
    print("=" * 70)
    print("季节性分析")
    print("=" * 70)
    
    if not HAS_LIBS:
        print("需要pandas库，跳过")
        return None
    
    # 加载数据
    processed_file = Path('2026_MCM_Problem_C_Data_processed.csv')
    if not processed_file.exists():
        print("错误: 处理后的数据文件不存在")
        return None
    
    df = pd.read_csv(processed_file)
    
    # 分析各季的表现
    seasonal_stats = df.groupby('season').agg({
        'celebrity_name': 'count',  # 参赛人数
    }).reset_index()
    seasonal_stats.columns = ['season', 'n_contestants']
    
    print("\n各季统计:")
    print(seasonal_stats.to_string())
    
    # 加载估计数据（如果有）
    estimates_file = Path('fan_vote_estimates.csv')
    has_estimates = estimates_file.exists()
    
    if has_estimates:
        estimates_df = pd.read_csv(estimates_file)
        # 合并数据
        seasonal_with_estimates = seasonal_stats.merge(
            estimates_df.groupby('season').agg({
                'fan_votes': 'mean',
                'judge_total': 'mean'
            }).reset_index(),
            on='season',
            how='left'
        )
    else:
        seasonal_with_estimates = seasonal_stats.copy()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各季参赛人数
    ax1 = axes[0, 0]
    ax1.bar(seasonal_stats['season'], seasonal_stats['n_contestants'], 
            color='steelblue', edgecolor='black')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Number of Contestants')
    ax1.set_title('Number of Contestants by Season')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 各季平均周数（从数据中计算）
    ax2 = axes[0, 1]
    if has_estimates:
        weeks_per_season = estimates_df.groupby('season')['week'].max().reset_index()
        weeks_per_season.columns = ['season', 'max_week']
        seasonal_with_weeks = seasonal_stats.merge(weeks_per_season, on='season', how='left')
        ax2.bar(seasonal_with_weeks['season'], seasonal_with_weeks['max_week'], 
                color='coral', edgecolor='black')
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Maximum Week')
        ax2.set_title('Maximum Week per Season')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No estimate data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Maximum Week per Season (No Data)')
    
    # 3. 各季平均粉丝投票（如果有估计数据）
    ax3 = axes[1, 0]
    if has_estimates and 'fan_votes' in seasonal_with_estimates.columns:
        ax3.plot(seasonal_with_estimates['season'], seasonal_with_estimates['fan_votes'], 
                marker='o', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Season')
        ax3.set_ylabel('Average Fan Votes')
        ax3.set_title('Average Fan Votes by Season')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No estimate data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Average Fan Votes by Season (No Data)')
    
    # 4. 各季平均评委评分（如果有估计数据）
    ax4 = axes[1, 1]
    if has_estimates and 'judge_total' in seasonal_with_estimates.columns:
        ax4.plot(seasonal_with_estimates['season'], seasonal_with_estimates['judge_total'], 
                marker='s', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Season')
        ax4.set_ylabel('Average Judge Score')
        ax4.set_title('Average Judge Score by Season')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No estimate data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Average Judge Score by Season (No Data)')
    
    plt.tight_layout()
    output_path = Path('visualizations/seasonal_analysis.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 季节性分析图已保存到: {output_path}")
    plt.close()
    
    return seasonal_stats

def time_trend_analysis():
    """时间趋势分析"""
    print("\n" + "=" * 70)
    print("时间趋势分析")
    print("=" * 70)
    
    if not HAS_LIBS:
        print("需要pandas库，跳过")
        return None
    
    # 加载估计数据
    estimates_file = Path('fan_vote_estimates.csv')
    if not estimates_file.exists():
        print("错误: 估计数据文件不存在")
        return None
    
    df = pd.read_csv(estimates_file)
    
    # 按季分析趋势
    seasonal_trends = df.groupby('season').agg({
        'fan_votes': ['mean', 'std'],
        'judge_total': 'mean'
    }).reset_index()
    
    seasonal_trends.columns = ['season', 'avg_fan_votes', 'std_fan_votes', 'avg_judge_score']
    
    print("\n各季趋势:")
    print(seasonal_trends.to_string())
    
    # 创建趋势图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 平均粉丝投票趋势
    ax1 = axes[0]
    ax1.plot(seasonal_trends['season'], seasonal_trends['avg_fan_votes'], 
             'o-', linewidth=2, markersize=8, label='Average Fan Votes')
    ax1.fill_between(seasonal_trends['season'], 
                     seasonal_trends['avg_fan_votes'] - seasonal_trends['std_fan_votes'],
                     seasonal_trends['avg_fan_votes'] + seasonal_trends['std_fan_votes'],
                     alpha=0.3, label='±1 Std')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Average Fan Votes')
    ax1.set_title('Trend of Average Fan Votes Across Seasons')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均评委评分趋势
    ax2 = axes[1]
    ax2.plot(seasonal_trends['season'], seasonal_trends['avg_judge_score'], 
             's-', linewidth=2, markersize=8, color='red', label='Average Judge Score')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Average Judge Score')
    ax2.set_title('Trend of Average Judge Scores Across Seasons')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('visualizations/time_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 时间趋势图已保存到: {output_path}")
    plt.close()
    
    return seasonal_trends

def generate_additional_analysis_report(seasonal_stats, seasonal_trends):
    """生成额外分析报告"""
    print("\n" + "=" * 70)
    print("生成额外分析报告")
    print("=" * 70)
    
    report = []
    report.append("=" * 70)
    report.append("额外分析维度报告")
    report.append("=" * 70)
    report.append("")
    
    report.append("1. 季节性分析")
    report.append("-" * 70)
    if seasonal_stats is not None:
        report.append("各季参赛人数统计:")
        for _, row in seasonal_stats.iterrows():
            report.append(f"  季{row['season']}: {row['n_contestants']} 位选手")
    report.append("")
    
    report.append("2. 时间趋势分析")
    report.append("-" * 70)
    if seasonal_trends is not None:
        report.append("各季平均表现趋势:")
        for _, row in seasonal_trends.iterrows():
            report.append(f"  季{row['season']}: 平均粉丝投票={row['avg_fan_votes']:.2f}, 平均评委评分={row['avg_judge_score']:.2f}")
    report.append("")
    
    report.append("3. 主要发现")
    report.append("-" * 70)
    report.append("- 不同季度的参赛人数有变化")
    report.append("- 粉丝投票和评委评分在不同季之间存在趋势变化")
    report.append("- 这些趋势可能反映了节目规则和观众偏好的变化")
    report.append("")
    
    report_text = "\n".join(report)
    report_path = Path('additional_analysis_report.txt')
    report_path.write_text(report_text, encoding='utf-8')
    print(f"\n[OK] 报告已保存到: {report_path}")
    
    return report_text

def main():
    """主函数"""
    print("=" * 70)
    print("添加更多分析维度")
    print("=" * 70)
    
    # 季节性分析
    seasonal_stats = seasonal_analysis()
    
    # 时间趋势分析
    seasonal_trends = time_trend_analysis()
    
    # 生成报告
    generate_additional_analysis_report(seasonal_stats, seasonal_trends)
    
    print("\n" + "=" * 70)
    print("额外分析完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
