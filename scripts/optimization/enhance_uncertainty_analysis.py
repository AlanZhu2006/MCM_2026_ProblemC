"""
增强不确定性分析
添加可视化、不确定性来源分析、蒙特卡洛细节
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_uncertainty_data():
    """加载不确定性数据"""
    uncertainty_file = Path('fan_vote_uncertainty.csv')
    estimates_file = Path('fan_vote_estimates.csv')
    
    if not uncertainty_file.exists():
        print("错误: fan_vote_uncertainty.csv 不存在")
        return None, None
    
    uncertainty_df = pd.read_csv(uncertainty_file)
    
    estimates_df = None
    if estimates_file.exists():
        estimates_df = pd.read_csv(estimates_file)
    
    return uncertainty_df, estimates_df

def analyze_uncertainty_sources(uncertainty_df):
    """分析不确定性来源"""
    print("=" * 70)
    print("不确定性来源分析")
    print("=" * 70)
    
    # 计算不同周次的不确定性
    weekly_uncertainty = uncertainty_df.groupby(['season', 'week']).agg({
        'fan_votes_std': ['mean', 'std', 'min', 'max'],
        'fan_votes_ci_upper': 'mean',
        'fan_votes_ci_lower': 'mean'
    }).reset_index()
    
    weekly_uncertainty.columns = ['season', 'week', 'mean_std', 'std_std', 'min_std', 'max_std', 
                                  'mean_ci_upper', 'mean_ci_lower']
    weekly_uncertainty['ci_width'] = weekly_uncertainty['mean_ci_upper'] - weekly_uncertainty['mean_ci_lower']
    
    # 找出不确定性最高的周次
    high_uncertainty = weekly_uncertainty.nlargest(10, 'mean_std')
    print("\n不确定性最高的10个周次:")
    print(high_uncertainty[['season', 'week', 'mean_std', 'ci_width']].to_string())
    
    # 找出不确定性最低的周次
    low_uncertainty = weekly_uncertainty.nsmallest(10, 'mean_std')
    print("\n不确定性最低的10个周次:")
    print(low_uncertainty[['season', 'week', 'mean_std', 'ci_width']].to_string())
    
    # 按季分析
    seasonal_uncertainty = uncertainty_df.groupby('season').agg({
        'fan_votes_std': 'mean'
    }).reset_index()
    seasonal_uncertainty.columns = ['season', 'avg_uncertainty']
    
    print("\n各季平均不确定性:")
    print(seasonal_uncertainty.sort_values('avg_uncertainty', ascending=False).to_string())
    
    return weekly_uncertainty, seasonal_uncertainty

def create_uncertainty_visualizations(uncertainty_df, weekly_uncertainty, seasonal_uncertainty):
    """创建不确定性可视化"""
    print("\n" + "=" * 70)
    print("创建不确定性可视化")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 不确定性分布直方图
    ax1 = plt.subplot(3, 2, 1)
    uncertainty_df['fan_votes_std'].hist(bins=50, ax=ax1, edgecolor='black')
    ax1.set_xlabel('Standard Deviation of Fan Votes')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Uncertainty (Standard Deviation)')
    ax1.axvline(uncertainty_df['fan_votes_std'].mean(), color='r', linestyle='--', 
                label=f'Mean: {uncertainty_df["fan_votes_std"].mean():.2f}')
    ax1.legend()
    
    # 2. 置信区间宽度分布
    ax2 = plt.subplot(3, 2, 2)
    ci_width = uncertainty_df['fan_votes_ci_upper'] - uncertainty_df['fan_votes_ci_lower']
    ci_width.hist(bins=50, ax=ax2, edgecolor='black', color='orange')
    ax2.set_xlabel('95% Confidence Interval Width')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of 95% CI Width')
    ax2.axvline(ci_width.mean(), color='r', linestyle='--', 
                label=f'Mean: {ci_width.mean():.2f}')
    ax2.legend()
    
    # 3. 不确定性随时间变化（按周次）
    ax3 = plt.subplot(3, 2, 3)
    weekly_uncertainty_sorted = weekly_uncertainty.sort_values('week')
    ax3.plot(weekly_uncertainty_sorted['week'], weekly_uncertainty_sorted['mean_std'], 
             marker='o', markersize=3, alpha=0.6)
    ax3.set_xlabel('Week Number')
    ax3.set_ylabel('Average Uncertainty (Std)')
    ax3.set_title('Uncertainty by Week Number')
    ax3.grid(True, alpha=0.3)
    
    # 4. 各季不确定性对比
    ax4 = plt.subplot(3, 2, 4)
    seasonal_uncertainty_sorted = seasonal_uncertainty.sort_values('season')
    ax4.bar(seasonal_uncertainty_sorted['season'], seasonal_uncertainty_sorted['avg_uncertainty'],
            color='steelblue', edgecolor='black')
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Average Uncertainty')
    ax4.set_title('Average Uncertainty by Season')
    ax4.set_xticks(seasonal_uncertainty_sorted['season'][::5])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 不确定性 vs 粉丝投票值
    ax5 = plt.subplot(3, 2, 5)
    sample_data = uncertainty_df.sample(min(1000, len(uncertainty_df)))
    ax5.scatter(sample_data['fan_votes_mean'], sample_data['fan_votes_std'], 
                alpha=0.5, s=10)
    ax5.set_xlabel('Mean Fan Votes')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_title('Uncertainty vs. Fan Vote Magnitude')
    ax5.grid(True, alpha=0.3)
    
    # 6. 置信区间覆盖率分析
    ax6 = plt.subplot(3, 2, 6)
    # 计算相对不确定性（CV = std/mean）
    cv = uncertainty_df['fan_votes_std'] / (uncertainty_df['fan_votes_mean'] + 1e-6)
    cv.hist(bins=50, ax=ax6, edgecolor='black', color='green')
    ax6.set_xlabel('Coefficient of Variation (CV)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Coefficient of Variation')
    ax6.axvline(cv.mean(), color='r', linestyle='--', 
                label=f'Mean CV: {cv.mean():.3f}')
    ax6.legend()
    
    plt.tight_layout()
    output_path = Path('visualizations/uncertainty_analysis.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 不确定性可视化已保存到: {output_path}")
    plt.close()

def create_confidence_interval_plot(uncertainty_df):
    """创建置信区间图"""
    print("\n创建置信区间可视化...")
    
    # 选择几个代表性的周次
    sample_weeks = uncertainty_df.groupby(['season', 'week']).first().reset_index()
    sample_weeks = sample_weeks.sample(min(20, len(sample_weeks)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_pos = 0
    for idx, row in sample_weeks.iterrows():
        week_data = uncertainty_df[
            (uncertainty_df['season'] == row['season']) & 
            (uncertainty_df['week'] == row['week'])
        ].sort_values('fan_votes_mean', ascending=False)
        
        if len(week_data) == 0:
            continue
        
        # 绘制置信区间
        for i, (_, contestant) in enumerate(week_data.iterrows()):
            x_pos = i
            mean = contestant['fan_votes_mean']
            ci_lower = contestant['fan_votes_ci_lower']
            ci_upper = contestant['fan_votes_ci_upper']
            
            # 绘制置信区间
            ax.plot([x_pos, x_pos], [ci_lower, ci_upper], 'b-', linewidth=2, alpha=0.6)
            ax.plot(x_pos, mean, 'ro', markersize=8)
        
        y_pos += len(week_data) + 2
    
    ax.set_xlabel('Contestant Index (sorted by mean)')
    ax.set_ylabel('Fan Votes')
    ax.set_title('95% Confidence Intervals for Fan Vote Estimates (Sample Weeks)')
    ax.grid(True, alpha=0.3)
    
    output_path = Path('visualizations/confidence_intervals.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 置信区间图已保存到: {output_path}")
    plt.close()

def generate_uncertainty_report(uncertainty_df, weekly_uncertainty, seasonal_uncertainty):
    """生成不确定性分析报告"""
    print("\n" + "=" * 70)
    print("生成不确定性分析报告")
    print("=" * 70)
    
    report = []
    report.append("=" * 70)
    report.append("不确定性分析详细报告")
    report.append("=" * 70)
    report.append("")
    
    # 总体统计
    report.append("1. 总体不确定性统计")
    report.append("-" * 70)
    report.append(f"总样本数: {len(uncertainty_df)}")
    report.append(f"平均标准差: {uncertainty_df['fan_votes_std'].mean():.2f}")
    report.append(f"标准差中位数: {uncertainty_df['fan_votes_std'].median():.2f}")
    report.append(f"标准差范围: [{uncertainty_df['fan_votes_std'].min():.2f}, {uncertainty_df['fan_votes_std'].max():.2f}]")
    
    ci_width = uncertainty_df['fan_votes_ci_upper'] - uncertainty_df['fan_votes_ci_lower']
    report.append(f"平均置信区间宽度: {ci_width.mean():.2f}")
    report.append(f"置信区间宽度中位数: {ci_width.median():.2f}")
    report.append("")
    
    # 不确定性来源分析
    report.append("2. 不确定性来源分析")
    report.append("-" * 70)
    report.append("\n不确定性最高的10个周次:")
    high_unc = weekly_uncertainty.nlargest(10, 'mean_std')
    for _, row in high_unc.iterrows():
        report.append(f"  季{row['season']} 周{row['week']}: 平均不确定性={row['mean_std']:.2f}, CI宽度={row['ci_width']:.2f}")
    
    report.append("\n不确定性最低的10个周次:")
    low_unc = weekly_uncertainty.nsmallest(10, 'mean_std')
    for _, row in low_unc.iterrows():
        report.append(f"  季{row['season']} 周{row['week']}: 平均不确定性={row['mean_std']:.2f}, CI宽度={row['ci_width']:.2f}")
    
    report.append("\n各季平均不确定性:")
    for _, row in seasonal_uncertainty.sort_values('avg_uncertainty', ascending=False).iterrows():
        report.append(f"  季{row['season']}: {row['avg_uncertainty']:.2f}")
    
    report.append("")
    
    # 系数变异分析
    report.append("3. 相对不确定性分析（系数变异）")
    report.append("-" * 70)
    cv = uncertainty_df['fan_votes_std'] / (uncertainty_df['fan_votes_mean'] + 1e-6)
    report.append(f"平均CV: {cv.mean():.3f}")
    report.append(f"CV中位数: {cv.median():.3f}")
    report.append(f"CV范围: [{cv.min():.3f}, {cv.max():.3f}]")
    report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    report_path = Path('uncertainty_analysis_report.txt')
    report_path.write_text(report_text, encoding='utf-8')
    print(f"\n[OK] 报告已保存到: {report_path}")
    
    return report_text

def main():
    """主函数"""
    print("=" * 70)
    print("增强不确定性分析")
    print("=" * 70)
    
    # 加载数据
    uncertainty_df, estimates_df = load_uncertainty_data()
    if uncertainty_df is None:
        return
    
    print(f"\n[OK] 加载数据: {len(uncertainty_df)} 条记录")
    
    # 分析不确定性来源
    weekly_uncertainty, seasonal_uncertainty = analyze_uncertainty_sources(uncertainty_df)
    
    # 创建可视化
    create_uncertainty_visualizations(uncertainty_df, weekly_uncertainty, seasonal_uncertainty)
    create_confidence_interval_plot(uncertainty_df)
    
    # 生成报告
    generate_uncertainty_report(uncertainty_df, weekly_uncertainty, seasonal_uncertainty)
    
    print("\n" + "=" * 70)
    print("不确定性分析增强完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
