"""
为2026 MCM Problem C生成数据可视化图表
用于最终报告的图表生成
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor

# 创建输出目录
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)


def load_all_data():
    """加载所有需要的数据"""
    print("加载数据...")
    
    # 原始数据
    raw_df = load_data()
    
    # 预处理数据
    processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
    
    # Stage 2: 粉丝投票估计
    estimates_df = pd.read_csv('fan_vote_estimates.csv')
    
    # Stage 3: 投票方法比较
    comparison_df = pd.read_csv('voting_method_comparison.csv')
    controversial_df = pd.read_csv('controversial_cases_analysis.csv')
    
    # Stage 4: 影响因素分析
    with open('factor_impact_analysis.json', 'r', encoding='utf-8') as f:
        factor_analysis = json.load(f)
    
    # Stage 5: ML模型分析
    with open('stage5_model_analysis_lgb.json', 'r', encoding='utf-8') as f:
        ml_analysis = json.load(f)
    
    print("✓ 数据加载完成")
    return {
        'raw_df': raw_df,
        'processed_df': processed_df,
        'estimates_df': estimates_df,
        'comparison_df': comparison_df,
        'controversial_df': controversial_df,
        'factor_analysis': factor_analysis,
        'ml_analysis': ml_analysis
    }


def plot_stage2_fan_vote_estimation(data):
    """Stage 2: 粉丝投票估计可视化"""
    print("\n生成Stage 2可视化...")
    
    estimates_df = data['estimates_df']
    
    # 1. 准确率趋势图（按季次）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 每季的周数分布
    season_weeks = estimates_df.groupby('season')['week'].nunique().reset_index()
    season_weeks.columns = ['season', 'weeks']
    
    axes[0, 0].bar(season_weeks['season'], season_weeks['weeks'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('季次', fontsize=12)
    axes[0, 0].set_ylabel('周数', fontsize=12)
    axes[0, 0].set_title('每季的周数分布', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 1.2 粉丝投票分布（箱线图）
    fan_votes = estimates_df['fan_votes'].dropna()
    if len(fan_votes) > 0:
        axes[0, 1].boxplot(fan_votes.values, vert=True)
        axes[0, 1].set_ylabel('粉丝投票数', fontsize=12)
        axes[0, 1].set_title('粉丝投票分布（箱线图）', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('粉丝投票分布（箱线图）', fontsize=14, fontweight='bold')
    
    # 1.3 评委评分 vs 粉丝投票散点图（抽样）
    if 'judge_total' in estimates_df.columns and 'fan_votes' in estimates_df.columns:
        valid_data = estimates_df[['judge_total', 'fan_votes']].dropna()
        if len(valid_data) > 0:
            sample_size = min(1000, len(valid_data))
            sample_df = valid_data.sample(sample_size, random_state=42)
            axes[1, 0].scatter(sample_df['judge_total'], sample_df['fan_votes'], 
                              alpha=0.5, s=20, color='coral')
            axes[1, 0].set_xlabel('评委总分', fontsize=12)
            axes[1, 0].set_ylabel('粉丝投票数', fontsize=12)
            axes[1, 0].set_title('评委评分 vs 粉丝投票（散点图）', fontsize=14, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '无有效数据', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('评委评分 vs 粉丝投票（散点图）', fontsize=14, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, '缺少必要列', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('评委评分 vs 粉丝投票（散点图）', fontsize=14, fontweight='bold')
    
    # 1.4 不确定性分析（如果有）
    uncertainty_path = project_root / 'fan_vote_uncertainty.csv'
    if uncertainty_path.exists():
        try:
            uncertainty_df = pd.read_csv(uncertainty_path)
            # 检查可能的列名：fan_votes_std 或 std
            std_col = None
            if 'fan_votes_std' in uncertainty_df.columns:
                std_col = 'fan_votes_std'
            elif 'std' in uncertainty_df.columns:
                std_col = 'std'
            
            if std_col:
                std_data = uncertainty_df[std_col].dropna()
                if len(std_data) > 0:
                    axes[1, 1].hist(std_data, bins=50, color='green', alpha=0.7)
                    axes[1, 1].set_xlabel('标准差', fontsize=12)
                    axes[1, 1].set_ylabel('频数', fontsize=12)
                    axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
                    axes[1, 1].grid(axis='y', alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, '无不确定性数据', ha='center', va='center', fontsize=14)
                    axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
            else:
                # 如果没有std列，尝试使用验证结果中的准确率信息
                validation_path = project_root / 'validation_results.json'
                if validation_path.exists():
                    try:
                        with open(validation_path, 'r', encoding='utf-8') as f:
                            validation = json.load(f)
                            accuracy = validation.get('accuracy', 0) * 100
                            axes[1, 1].text(0.5, 0.5, f'预测准确率: {accuracy:.2f}%', 
                                           ha='center', va='center', fontsize=14, fontweight='bold')
                            axes[1, 1].set_title('Stage 2 模型性能', fontsize=14, fontweight='bold')
                    except:
                        axes[1, 1].text(0.5, 0.5, '缺少std列', ha='center', va='center', fontsize=14)
                        axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
                else:
                    axes[1, 1].text(0.5, 0.5, '缺少std列', ha='center', va='center', fontsize=14)
                    axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'读取失败: {str(e)[:30]}', ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
    else:
        # 使用验证结果中的准确率信息
        validation_path = project_root / 'validation_results.json'
        if validation_path.exists():
            try:
                with open(validation_path, 'r', encoding='utf-8') as f:
                    validation = json.load(f)
                    accuracy = validation.get('accuracy', 0) * 100
                    axes[1, 1].text(0.5, 0.5, f'预测准确率: {accuracy:.2f}%', 
                                   ha='center', va='center', fontsize=14, fontweight='bold')
                    axes[1, 1].set_title('Stage 2 模型性能', fontsize=14, fontweight='bold')
            except:
                axes[1, 1].text(0.5, 0.5, '无不确定性数据', ha='center', va='center', fontsize=14)
                axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, '无不确定性数据', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_fan_vote_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stage 2可视化已保存")


def plot_stage3_voting_comparison(data):
    """Stage 3: 投票方法比较可视化"""
    print("\n生成Stage 3可视化...")
    
    comparison_df = data['comparison_df']
    controversial_df = data['controversial_df']
    
    if comparison_df is None or len(comparison_df) == 0:
        print("  ⚠️  警告: 投票方法比较数据为空，跳过Stage 3可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2.1 两种方法的准确率对比
    if 'rank_method_correct' in comparison_df.columns and 'percent_method_correct' in comparison_df.columns:
        rank_accuracy = comparison_df['rank_method_correct'].mean() * 100
        percent_accuracy = comparison_df['percent_method_correct'].mean() * 100
        
        methods = ['排名法', '百分比法']
        accuracies = [rank_accuracy, percent_accuracy]
        colors = ['steelblue', 'coral']
        
        bars = axes[0, 0].bar(methods, accuracies, color=colors, alpha=0.7, width=0.6)
        axes[0, 0].set_ylabel('准确率 (%)', fontsize=12)
        axes[0, 0].set_title('两种投票方法的准确率对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 100])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.2f}%', ha='center', va='bottom', fontsize=11)
    else:
        axes[0, 0].text(0.5, 0.5, '缺少准确率数据', ha='center', va='center', fontsize=14)
        axes[0, 0].set_title('两种投票方法的准确率对比', fontsize=14, fontweight='bold')
    
    # 2.2 按季次统计差异
    if 'methods_agree' in comparison_df.columns:
        season_diffs = []
        for season in sorted(comparison_df['season'].unique()):
            season_data = comparison_df[comparison_df['season'] == season]
            # methods_agree=False 表示两种方法不一致
            disagree_count = (~season_data['methods_agree']).sum()
            total = len(season_data)
            season_diffs.append({
                'season': season,
                'diff_rate': disagree_count / total * 100 if total > 0 else 0
            })
        
        if season_diffs:
            diff_df = pd.DataFrame(season_diffs)
            axes[0, 1].plot(diff_df['season'], diff_df['diff_rate'], marker='o', 
                           linewidth=2, markersize=6, color='purple')
            axes[0, 1].set_xlabel('季次', fontsize=12)
            axes[0, 1].set_ylabel('不一致率 (%)', fontsize=12)
            axes[0, 1].set_title('两种方法的不一致率（按季次）', fontsize=14, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, '缺少一致性数据', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('两种方法的不一致率（按季次）', fontsize=14, fontweight='bold')
    
    # 2.3 争议案例分析
    if controversial_df is not None and len(controversial_df) > 0:
        # 检查可能的列名
        if 'celebrity' in controversial_df.columns:
            celebrity_col = 'celebrity'
        elif 'celebrity_name' in controversial_df.columns:
            celebrity_col = 'celebrity_name'
        else:
            celebrity_col = None
        
        if celebrity_col:
            controversial_cases = controversial_df.groupby(celebrity_col).size().sort_values(ascending=False)
            top_cases = controversial_cases.head(10)  # 只显示前10个
            axes[1, 0].barh(range(len(top_cases)), top_cases.values, 
                           color='red', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_cases)))
            axes[1, 0].set_yticklabels(top_cases.index, fontsize=9)
            axes[1, 0].set_xlabel('争议周次数', fontsize=12)
            axes[1, 0].set_title('争议案例统计（Top 10）', fontsize=14, fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '缺少争议案例数据', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('争议案例统计', fontsize=14, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, '无争议案例数据', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('争议案例统计', fontsize=14, fontweight='bold')
    
    # 2.4 方法差异按选手数量统计
    if 'n_contestants' in comparison_df.columns and 'methods_agree' in comparison_df.columns:
        contestant_diffs = comparison_df.groupby('n_contestants').apply(
            lambda x: (~x['methods_agree']).mean() * 100
        )
        if len(contestant_diffs) > 0:
            axes[1, 1].bar(contestant_diffs.index, contestant_diffs.values, 
                          color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('选手数量', fontsize=12)
            axes[1, 1].set_ylabel('不一致率 (%)', fontsize=12)
            axes[1, 1].set_title('方法差异 vs 选手数量', fontsize=14, fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无差异数据', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('方法差异 vs 选手数量', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, '缺少必要列', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('方法差异 vs 选手数量', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stage3_voting_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stage 3可视化已保存")


def plot_stage4_factor_impact(data):
    """Stage 4: 影响因素分析可视化"""
    print("\n生成Stage 4可视化...")
    
    factor_analysis = data['factor_analysis']
    processed_df = data['processed_df']
    estimates_df = data.get('estimates_df', None)
    
    if processed_df is None:
        print("  ⚠️  警告: 预处理数据为空，跳过Stage 4可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 3.1 年龄影响
    if 'celebrity_age_during_season' in processed_df.columns:
        # 尝试从estimates_df获取judge_total，如果没有则计算平均值
        if estimates_df is not None and 'judge_total' in estimates_df.columns:
            # 合并数据
            age_judge_data = estimates_df[['celebrity_name', 'season', 'judge_total']].merge(
                processed_df[['celebrity_name', 'season', 'celebrity_age_during_season']].drop_duplicates(),
                on=['celebrity_name', 'season'],
                how='inner'
            )
            age_data = age_judge_data[['celebrity_age_during_season', 'judge_total']].dropna()
        else:
            # 从processed_df计算所有周的平均评委总分
            week_score_cols = [col for col in processed_df.columns if col.endswith('_total_score')]
            if len(week_score_cols) > 0:
                # 计算每行的平均评委总分（排除0值，因为0表示已淘汰）
                processed_df_copy = processed_df.copy()
                score_matrix = processed_df_copy[week_score_cols].replace(0, np.nan)
                processed_df_copy['avg_judge_total'] = score_matrix.mean(axis=1)
                age_data = processed_df_copy[['celebrity_age_during_season', 'avg_judge_total']].dropna()
                age_data = age_data.rename(columns={'avg_judge_total': 'judge_total'})
            else:
                age_data = pd.DataFrame()
        
        if len(age_data) > 0:
            axes[0, 0].scatter(age_data['celebrity_age_during_season'], age_data['judge_total'],
                             alpha=0.5, s=20, color='steelblue')
            # 添加趋势线
            try:
                z = np.polyfit(age_data['celebrity_age_during_season'], age_data['judge_total'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(age_data['celebrity_age_during_season'], 
                              p(age_data['celebrity_age_during_season']), 
                              "r--", alpha=0.8, linewidth=2, label=f'趋势线 (斜率={z[0]:.2f})')
                axes[0, 0].legend()
            except:
                pass  # 如果拟合失败，只显示散点图
            axes[0, 0].set_xlabel('年龄', fontsize=12)
            axes[0, 0].set_ylabel('评委总分', fontsize=12)
            axes[0, 0].set_title('年龄对评委评分的影响', fontsize=14, fontweight='bold')
            axes[0, 0].grid(alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '无有效数据', ha='center', va='center', fontsize=14)
            axes[0, 0].set_title('年龄对评委评分的影响', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, '缺少年龄数据', ha='center', va='center', fontsize=14)
        axes[0, 0].set_title('年龄对评委评分的影响', fontsize=14, fontweight='bold')
    
    # 3.2 专业舞者影响（Top 10）
    if 'pro_dancer_impact' in factor_analysis:
        pro_dancer_stats = factor_analysis['pro_dancer_impact'].get('pro_dancer_stats', [])
        if len(pro_dancer_stats) > 0:
            # 按平均排名排序，取前10
            top_dancers = sorted(pro_dancer_stats, key=lambda x: x.get('avg_placement', 999))[:10]
            dancer_names = [d['pro_dancer'] for d in top_dancers]
            avg_placements = [d.get('avg_placement', 0) for d in top_dancers]
            
            axes[0, 1].barh(dancer_names, avg_placements, color='coral', alpha=0.7)
            axes[0, 1].set_xlabel('平均排名（越小越好）', fontsize=12)
            axes[0, 1].set_title('表现最好的10位专业舞者', fontsize=14, fontweight='bold')
            axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3.3 行业影响（Top 10）
    if 'celebrity_features_impact' in factor_analysis:
        industry_stats = factor_analysis['celebrity_features_impact'].get('industry', {}).get('industry_performance', [])
        if len(industry_stats) > 0:
            # 按平均排名排序，取前10
            top_industries = sorted(industry_stats, key=lambda x: x.get('avg_placement', 999))[:10]
            industry_names = [d['industry'] for d in top_industries]
            avg_placements = [d.get('avg_placement', 0) for d in top_industries]
            
            axes[1, 0].barh(industry_names, avg_placements, color='green', alpha=0.7)
            axes[1, 0].set_xlabel('平均排名（越小越好）', fontsize=12)
            axes[1, 0].set_title('表现最好的10个行业', fontsize=14, fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 3.4 地区影响（Top 10）
    if 'celebrity_features_impact' in factor_analysis:
        region_stats = factor_analysis['celebrity_features_impact'].get('region', {}).get('region_performance', [])
        if len(region_stats) > 0:
            # 按平均排名排序，取前10
            top_regions = sorted(region_stats, key=lambda x: x.get('avg_placement', 999))[:10]
            region_names = [d['region'] for d in top_regions]
            avg_placements = [d.get('avg_placement', 0) for d in top_regions]
            
            axes[1, 1].barh(region_names, avg_placements, color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('平均排名（越小越好）', fontsize=12)
            axes[1, 1].set_title('表现最好的10个地区', fontsize=14, fontweight='bold')
            axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stage4_factor_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stage 4可视化已保存")


def plot_stage5_ml_system(data):
    """Stage 5: ML系统可视化"""
    print("\n生成Stage 5可视化...")
    
    ml_analysis = data['ml_analysis']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 4.1 特征重要性（Top 10）
    if 'feature_importance' in ml_analysis:
        feature_imp = ml_analysis['feature_importance']
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        axes[0, 0].barh(features, importances, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('重要性', fontsize=12)
        axes[0, 0].set_title('特征重要性（Top 10）', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 4.2 特征类别重要性
    if 'category_importance' in ml_analysis:
        category_imp = ml_analysis['category_importance']
        categories = list(category_imp.keys())
        importances = list(category_imp.values())
        
        # 创建饼图
        axes[0, 1].pie(importances, labels=categories, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('特征类别重要性分布', fontsize=14, fontweight='bold')
    
    # 4.3 粉丝 vs 评委重要性对比
    if 'fan_total_importance' in ml_analysis and 'judge_total_importance' in ml_analysis:
        fan_imp = ml_analysis['fan_total_importance']
        judge_imp = ml_analysis['judge_total_importance']
        total = fan_imp + judge_imp
        
        labels = ['粉丝投票相关', '评委评分相关']
        sizes = [fan_imp / total * 100, judge_imp / total * 100]
        colors = ['coral', 'steelblue']
        
        axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 0].set_title('粉丝投票 vs 评委评分的重要性对比', fontsize=14, fontweight='bold')
    
    # 4.4 模型性能对比（如果有多个模型）
    # 这里可以添加不同模型的准确率对比
    models = ['原始系统', 'ML系统']
    accuracies = [88.96, 97.99]  # 从报告中获取
    
    bars = axes[1, 1].bar(models, accuracies, color=['gray', 'green'], alpha=0.7, width=0.6)
    axes[1, 1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1, 1].set_title('原始系统 vs ML系统准确率对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([80, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stage5_ml_system.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stage 5可视化已保存")


def plot_overall_summary(data):
    """生成总体摘要图表"""
    print("\n生成总体摘要图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 5.1 各阶段完成情况
    stages = ['Stage 1\n数据预处理', 'Stage 2\n粉丝投票估计', 'Stage 3\n方法比较', 
              'Stage 4\n影响因素', 'Stage 5\n新系统设计']
    status = ['完成', '完成', '完成', '完成', '完成']
    colors_status = ['green'] * 5
    
    axes[0, 0].barh(stages, [1]*5, color=colors_status, alpha=0.7)
    axes[0, 0].set_xlabel('完成状态', fontsize=12)
    axes[0, 0].set_title('各阶段完成情况', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim([0, 1.2])
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 5.2 关键指标汇总
    metrics = ['粉丝投票估计\n准确率', '投票方法比较\n不一致率', 'ML系统\n准确率', '准确率\n提升']
    values = [90.0, 37.46, 97.99, 9.03]  # 从报告中获取
    units = ['%', '%', '%', '%']
    
    y_pos = np.arange(len(metrics))
    bars = axes[0, 1].barh(y_pos, values, color=['steelblue', 'coral', 'green', 'purple'], alpha=0.7)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(metrics)
    axes[0, 1].set_xlabel('数值', fontsize=12)
    axes[0, 1].set_title('关键指标汇总', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val, unit) in enumerate(zip(bars, values, units)):
        width = bar.get_width()
        axes[0, 1].text(width, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}{unit}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 5.3 数据覆盖范围
    estimates_df = data['estimates_df']
    total_seasons = estimates_df['season'].nunique()
    total_weeks = estimates_df.groupby('season')['week'].nunique().sum()
    total_contestants = estimates_df['celebrity_name'].nunique()
    
    categories = ['季数', '总周数', '选手数']
    counts = [total_seasons, total_weeks, total_contestants]
    
    axes[1, 0].bar(categories, counts, color=['steelblue', 'coral', 'green'], alpha=0.7)
    axes[1, 0].set_ylabel('数量', fontsize=12)
    axes[1, 0].set_title('数据覆盖范围', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (cat, count) in enumerate(zip(categories, counts)):
        axes[1, 0].text(i, count, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 5.4 系统性能提升
    systems = ['原始系统\n(排名/百分比)', 'ML系统\n(智能组合)']
    accuracies = [88.96, 97.99]
    improvement = accuracies[1] - accuracies[0]
    
    x = np.arange(len(systems))
    bars = axes[1, 1].bar(x, accuracies, color=['gray', 'green'], alpha=0.7, width=0.6)
    axes[1, 1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1, 1].set_title(f'系统性能提升 (+{improvement:.2f}%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(systems)
    axes[1, 1].set_ylim([80, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 添加数值标签和箭头
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加提升箭头
    axes[1, 1].annotate('', xy=(1, accuracies[1]), xytext=(0, accuracies[0]),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    axes[1, 1].text(0.5, (accuracies[0] + accuracies[1])/2, f'+{improvement:.2f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 总体摘要图表已保存")


def main():
    """主函数"""
    print("=" * 70)
    print("生成2026 MCM Problem C 数据可视化图表")
    print("=" * 70)
    
    try:
        # 加载数据
        data = load_all_data()
        
        # 生成各阶段可视化
        plot_stage2_fan_vote_estimation(data)
        plot_stage3_voting_comparison(data)
        plot_stage4_factor_impact(data)
        plot_stage5_ml_system(data)
        plot_overall_summary(data)
        
        print("\n" + "=" * 70)
        print("所有可视化图表已生成完成！")
        print(f"输出目录: {output_dir}")
        print("=" * 70)
        
        # 列出生成的文件
        print("\n生成的图表文件:")
        for file in sorted(output_dir.glob('*.png')):
            print(f"  - {file.name}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
