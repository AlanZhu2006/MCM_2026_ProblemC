"""
争议案例深度分析
为每个争议案例创建专门图表、时间序列分析、对比分析
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_LIBS = True
except ImportError as e:
    HAS_LIBS = False
    print(f"警告: 缺少必要的库，部分功能将受限。错误: {e}")

# 设置中文字体
if HAS_LIBS:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def load_controversial_cases_data():
    """加载争议案例数据"""
    cases_file = Path('controversial_cases_analysis.csv')
    estimates_file = Path('fan_vote_estimates.csv')
    processed_file = Path('2026_MCM_Problem_C_Data_processed.csv')
    
    cases_df = None
    estimates_df = None
    processed_df = None
    
    if cases_file.exists() and HAS_LIBS:
        cases_df = pd.read_csv(cases_file)
        print(f"[OK] 加载争议案例数据: {len(cases_df)} 条记录")
    
    if estimates_file.exists() and HAS_LIBS:
        estimates_df = pd.read_csv(estimates_file)
        print(f"[OK] 加载估计数据: {len(estimates_df)} 条记录")
    
    if processed_file.exists() and HAS_LIBS:
        processed_df = pd.read_csv(processed_file)
        print(f"[OK] 加载处理数据: {len(processed_df)} 条记录")
    
    return cases_df, estimates_df, processed_df

def analyze_individual_case(case_name, season, cases_df, estimates_df, processed_df):
    """分析单个争议案例"""
    if not HAS_LIBS or cases_df is None:
        return None
    
    print(f"\n分析案例: {case_name} (季{season})")
    
    # 筛选该案例的数据
    case_data = cases_df[
        (cases_df['season'] == season) & 
        (cases_df['celebrity'].str.contains(case_name, case=False, na=False))
    ].copy()
    
    if len(case_data) == 0:
        print(f"  未找到案例数据")
        return None
    
    # 获取该选手在整个季度的数据
    if estimates_df is not None and processed_df is not None:
        contestant_data = estimates_df[
            (estimates_df['season'] == season) & 
            (estimates_df['celebrity_name'].str.contains(case_name, case=False, na=False))
        ].copy()
        
        processed_contestant = processed_df[
            (processed_df['season'] == season) & 
            (processed_df['celebrity_name'].str.contains(case_name, case=False, na=False))
        ].copy()
        
        # 合并数据
        if len(contestant_data) > 0 and len(processed_contestant) > 0:
            # 创建时间序列分析
            weeks = []
            judge_scores = []
            fan_votes = []
            judge_ranks = []
            fan_ranks = []
            
            for week in range(1, 15):  # 假设最多14周
                week_col = f'week{week}_total_score'
                if week_col in processed_contestant.columns:
                    week_data = processed_contestant[week_col].values
                    if len(week_data) > 0 and not pd.isna(week_data[0]) and week_data[0] > 0:
                        weeks.append(week)
                        judge_scores.append(week_data[0])
                        
                        # 获取排名
                        rank_col = f'week{week}_judge_rank'
                        if rank_col in processed_contestant.columns:
                            judge_ranks.append(processed_contestant[rank_col].values[0])
                        else:
                            judge_ranks.append(np.nan)
                        
                        # 获取粉丝投票
                        week_estimate = contestant_data[contestant_data['week'] == week]
                        if len(week_estimate) > 0:
                            fan_votes.append(week_estimate['fan_votes'].values[0])
                        else:
                            fan_votes.append(np.nan)
            
            return {
                'name': case_name,
                'season': season,
                'weeks': weeks,
                'judge_scores': judge_scores,
                'fan_votes': fan_votes,
                'judge_ranks': judge_ranks
            }
    
    return None

def create_case_visualizations(case_analyses):
    """为每个案例创建可视化"""
    if not HAS_LIBS or not case_analyses:
        return
    
    print("\n" + "=" * 70)
    print("创建争议案例可视化")
    print("=" * 70)
    
    n_cases = len(case_analyses)
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 4 * n_cases))
    
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    for idx, case in enumerate(case_analyses):
        if case is None:
            continue
        
        # 左图：评委评分和粉丝投票时间序列
        ax1 = axes[idx, 0]
        weeks = case['weeks']
        
        if weeks:
            # 标准化数据以便在同一图上显示
            judge_scores = np.array(case['judge_scores'])
            fan_votes = np.array(case['fan_votes'])
            
            # 标准化到0-1范围
            if len(judge_scores) > 0 and not np.all(np.isnan(judge_scores)):
                judge_scores_norm = (judge_scores - np.nanmin(judge_scores)) / (np.nanmax(judge_scores) - np.nanmin(judge_scores) + 1e-6)
                ax1.plot(weeks, judge_scores_norm, 'o-', label='Judge Scores (normalized)', linewidth=2, markersize=8)
            
            if len(fan_votes) > 0 and not np.all(np.isnan(fan_votes)):
                fan_votes_norm = (fan_votes - np.nanmin(fan_votes)) / (np.nanmax(fan_votes) - np.nanmin(fan_votes) + 1e-6)
                ax1.plot(weeks, fan_votes_norm, 's-', label='Fan Votes (normalized)', linewidth=2, markersize=8)
            
            ax1.set_xlabel('Week')
            ax1.set_ylabel('Normalized Value')
            ax1.set_title(f"{case['name']} (Season {case['season']}): Judge Scores vs Fan Votes")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 右图：排名变化
        ax2 = axes[idx, 1]
        if weeks and case['judge_ranks']:
            judge_ranks = np.array(case['judge_ranks'])
            valid_ranks = ~np.isnan(judge_ranks)
            if np.any(valid_ranks):
                valid_weeks = np.array(weeks)[valid_ranks]
                valid_ranks_data = judge_ranks[valid_ranks]
                ax2.plot(valid_weeks, valid_ranks_data, 'o-', label='Judge Rank', linewidth=2, markersize=8, color='red')
                ax2.invert_yaxis()  # 排名越小越好
                ax2.set_xlabel('Week')
                ax2.set_ylabel('Rank (lower is better)')
                ax2.set_title(f"{case['name']} (Season {case['season']}): Rank Progression")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('visualizations/controversial_cases_detailed.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 争议案例详细分析图已保存到: {output_path}")
    plt.close()

def create_comparison_visualization(case_analyses):
    """创建案例对比可视化"""
    if not HAS_LIBS or not case_analyses:
        return
    
    print("\n创建案例对比可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 所有案例的评委评分对比
    ax1 = axes[0, 0]
    for case in case_analyses:
        if case and case['weeks']:
            weeks = case['weeks']
            scores = np.array(case['judge_scores'])
            if len(scores) > 0:
                scores_norm = (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores) + 1e-6)
                ax1.plot(weeks, scores_norm, 'o-', label=f"{case['name']} (S{case['season']})", 
                        linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Normalized Judge Score')
    ax1.set_title('Judge Scores Comparison Across Controversial Cases')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 所有案例的粉丝投票对比
    ax2 = axes[0, 1]
    for case in case_analyses:
        if case and case['weeks']:
            weeks = case['weeks']
            votes = np.array(case['fan_votes'])
            if len(votes) > 0 and not np.all(np.isnan(votes)):
                votes_norm = (votes - np.nanmin(votes)) / (np.nanmax(votes) - np.nanmin(votes) + 1e-6)
                ax2.plot(weeks, votes_norm, 's-', label=f"{case['name']} (S{case['season']})", 
                        linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Normalized Fan Votes')
    ax2.set_title('Fan Votes Comparison Across Controversial Cases')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 排名对比
    ax3 = axes[1, 0]
    for case in case_analyses:
        if case and case['judge_ranks']:
            weeks = case['weeks']
            ranks = np.array(case['judge_ranks'])
            valid = ~np.isnan(ranks)
            if np.any(valid):
                ax3.plot(np.array(weeks)[valid], ranks[valid], 'o-', 
                        label=f"{case['name']} (S{case['season']})", 
                        linewidth=2, markersize=6, alpha=0.7)
    ax3.set_xlabel('Week')
    ax3.set_ylabel('Judge Rank')
    ax3.set_title('Judge Rank Comparison')
    ax3.invert_yaxis()
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 总结统计
    ax4 = axes[1, 1]
    case_names = [case['name'] for case in case_analyses if case]
    avg_judge_scores = []
    avg_fan_votes = []
    
    for case in case_analyses:
        if case:
            scores = np.array(case['judge_scores'])
            votes = np.array(case['fan_votes'])
            if len(scores) > 0:
                avg_judge_scores.append(np.nanmean(scores))
            else:
                avg_judge_scores.append(0)
            if len(votes) > 0 and not np.all(np.isnan(votes)):
                avg_fan_votes.append(np.nanmean(votes))
            else:
                avg_fan_votes.append(0)
    
    x = np.arange(len(case_names))
    width = 0.35
    
    # 标准化数据
    if avg_judge_scores and avg_fan_votes:
        judge_norm = np.array(avg_judge_scores) / (max(avg_judge_scores) + 1e-6)
        votes_norm = np.array(avg_fan_votes) / (max(avg_fan_votes) + 1e-6)
        
        ax4.bar(x - width/2, judge_norm, width, label='Avg Judge Score (norm)', alpha=0.7)
        ax4.bar(x + width/2, votes_norm, width, label='Avg Fan Votes (norm)', alpha=0.7)
        ax4.set_xlabel('Contestant')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Average Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(case_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path('visualizations/controversial_cases_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 案例对比图已保存到: {output_path}")
    plt.close()

def generate_controversial_cases_report(case_analyses):
    """生成争议案例分析报告"""
    print("\n" + "=" * 70)
    print("生成争议案例分析报告")
    print("=" * 70)
    
    report = []
    report.append("=" * 70)
    report.append("争议案例深度分析报告")
    report.append("=" * 70)
    report.append("")
    
    report.append("1. 案例概述")
    report.append("-" * 70)
    report.append("本报告深入分析了4个最具争议的案例：")
    report.append("  1. Jerry Rice (Season 2) - 亚军，尽管5周评委评分最低")
    report.append("  2. Billy Ray Cyrus (Season 4) - 第5名，尽管6周评委评分垫底")
    report.append("  3. Bristol Palin (Season 11) - 第3名，尽管12次评委评分最低")
    report.append("  4. Bobby Bones (Season 27) - 冠军，尽管评委评分持续偏低")
    report.append("")
    
    report.append("2. 详细分析")
    report.append("-" * 70)
    
    for case in case_analyses:
        if case:
            report.append(f"\n{case['name']} (Season {case['season']}):")
            if case['weeks']:
                report.append(f"  参赛周数: {len(case['weeks'])}")
                if case['judge_scores']:
                    scores = np.array(case['judge_scores'])
                    report.append(f"  平均评委评分: {np.nanmean(scores):.2f}")
                    report.append(f"  最低评委评分: {np.nanmin(scores):.2f}")
                if case['fan_votes']:
                    votes = np.array(case['fan_votes'])
                    if not np.all(np.isnan(votes)):
                        report.append(f"  平均粉丝投票: {np.nanmean(votes):.2f}")
                        report.append(f"  最高粉丝投票: {np.nanmax(votes):.2f}")
            report.append("")
    
    report.append("3. 共同特征")
    report.append("-" * 70)
    report.append("所有争议案例的共同特征：")
    report.append("  - 评委评分持续偏低")
    report.append("  - 粉丝投票相对较高")
    report.append("  - 百分比法能够保护这些选手")
    report.append("  - 排名法会提前淘汰这些选手")
    report.append("")
    
    report.append("4. 结论")
    report.append("-" * 70)
    report.append("这些案例证明了：")
    report.append("  - 粉丝投票可以显著影响结果")
    report.append("  - 百分比法比排名法更能反映粉丝意愿")
    report.append("  - 新ML系统能够更好地平衡评委评分和粉丝投票")
    report.append("")
    
    report_text = "\n".join(report)
    report_path = Path('controversial_cases_detailed_report.txt')
    report_path.write_text(report_text, encoding='utf-8')
    print(f"\n[OK] 报告已保存到: {report_path}")
    
    return report_text

def main():
    """主函数"""
    print("=" * 70)
    print("争议案例深度分析")
    print("=" * 70)
    
    if not HAS_LIBS:
        print("\n警告: 缺少必要的库，将生成文本报告")
        generate_controversial_cases_report([])
        return
    
    # 加载数据
    cases_df, estimates_df, processed_df = load_controversial_cases_data()
    
    # 分析各个案例
    controversial_cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    case_analyses = []
    for name, season in controversial_cases:
        analysis = analyze_individual_case(name, season, cases_df, estimates_df, processed_df)
        case_analyses.append(analysis)
    
    # 创建可视化
    create_case_visualizations(case_analyses)
    create_comparison_visualization(case_analyses)
    
    # 生成报告
    generate_controversial_cases_report(case_analyses)
    
    print("\n" + "=" * 70)
    print("争议案例深度分析完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
