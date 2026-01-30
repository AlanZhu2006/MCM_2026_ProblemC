"""
DWTS数据预处理模块
完成阶段1：数据探索与预处理的所有任务
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DWTSDataPreprocessor:
    """DWTS数据预处理器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化预处理器
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始DWTS数据
        """
        self.df = df.copy()
        self.processed_df = None
        self.data_summary = {}
        
    def check_data_integrity(self) -> Dict:
        """
        任务1: 加载并检查数据完整性
        
        Returns:
        --------
        Dict: 数据完整性报告
        """
        print("=" * 60)
        print("任务1: 检查数据完整性")
        print("=" * 60)
        
        summary = {
            'shape': self.df.shape,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.df.duplicated().sum(),
            'columns': list(self.df.columns),
        }
        
        # 检查关键列是否存在
        required_columns = [
            'celebrity_name', 'season', 'results', 'placement',
            'ballroompartner', 'celebrity_industry'
        ]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"⚠️  警告: 缺少关键列: {missing_columns}")
        else:
            print("✓ 所有关键列都存在")
        
        # 检查数据范围
        if 'season' in self.df.columns:
            summary['season_range'] = (self.df['season'].min(), self.df['season'].max())
            summary['unique_seasons'] = self.df['season'].nunique()
            print(f"✓ 季数范围: {summary['season_range'][0]} - {summary['season_range'][1]}")
            print(f"✓ 总季数: {summary['unique_seasons']}")
        
        if 'placement' in self.df.columns:
            summary['placement_range'] = (self.df['placement'].min(), self.df['placement'].max())
            print(f"✓ 排名范围: {summary['placement_range'][0]} - {summary['placement_range'][1]}")
        
        print(f"\n数据形状: {summary['shape']}")
        print(f"内存使用: {summary['memory_usage_mb']:.2f} MB")
        print(f"重复行数: {summary['duplicate_rows']}")
        
        self.data_summary['integrity'] = summary
        return summary
    
    def handle_missing_values(self) -> Dict:
        """
        任务2: 处理缺失值（N/A）
        
        Returns:
        --------
        Dict: 缺失值处理报告
        """
        print("\n" + "=" * 60)
        print("任务2: 处理缺失值（N/A）")
        print("=" * 60)
        
        # 识别所有评分列（weekX_judgeY_score格式）
        score_columns = [col for col in self.df.columns if 'judge' in col.lower() and 'score' in col.lower()]
        
        missing_report = {
            'total_missing': self.df.isna().sum().sum(),
            'missing_by_column': {},
            'score_columns_count': len(score_columns),
            'missing_in_scores': 0,
        }
        
        # 统计每列的缺失值
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_report['missing_by_column'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(self.df) * 100)
                }
        
        # 处理评分列的N/A值
        # 根据问题说明，N/A出现在：
        # 1. 第4位评委评分（如果该周只有3位评委）
        # 2. 未进行的周次（如第1季只有6周，第7-11周为N/A）
        
        print(f"总缺失值数: {missing_report['total_missing']}")
        print(f"评分列数量: {missing_report['score_columns_count']}")
        
        # 对于评分列的N/A，我们保留它们（因为这是数据的一部分，表示该周未进行或该评委不存在）
        # 但在计算总分时需要特殊处理
        
        # 对于其他关键列的缺失值，记录但不删除（可能需要进一步分析）
        key_columns = ['celebrity_name', 'season', 'results', 'placement']
        for col in key_columns:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                if missing > 0:
                    print(f"⚠️  {col} 有 {missing} 个缺失值")
        
        self.data_summary['missing_values'] = missing_report
        return missing_report
    
    def handle_eliminated_contestants(self) -> Dict:
        """
        任务3: 处理被淘汰选手的0分
        
        根据问题说明：被淘汰选手后续周次评分为0
        例如：第1季第2周淘汰的选手，第3-6周评分为0
        
        Returns:
        --------
        Dict: 淘汰选手处理报告
        """
        print("\n" + "=" * 60)
        print("任务3: 处理被淘汰选手的0分")
        print("=" * 60)
        
        # 识别评分列
        score_columns = [col for col in self.df.columns if 'judge' in col.lower() and 'score' in col.lower()]
        
        eliminated_report = {
            'total_zero_scores': 0,
            'contestants_with_zeros': [],
            'zero_score_patterns': {},
        }
        
        # 分析每个选手的0分模式
        for idx, row in self.df.iterrows():
            zero_scores = 0
            zero_weeks = []
            
            for col in score_columns:
                if pd.notna(row[col]) and row[col] == 0:
                    zero_scores += 1
                    # 从列名提取周次信息
                    if 'week' in col.lower():
                        week_num = self._extract_week_number(col)
                        if week_num:
                            zero_weeks.append(week_num)
            
            if zero_scores > 0:
                eliminated_report['total_zero_scores'] += zero_scores
                contestant_info = {
                    'name': row.get('celebrity_name', 'Unknown'),
                    'season': row.get('season', 'Unknown'),
                    'zero_count': zero_scores,
                    'zero_weeks': sorted(set(zero_weeks)),
                    'result': row.get('results', 'Unknown')
                }
                eliminated_report['contestants_with_zeros'].append(contestant_info)
        
        print(f"总0分数量: {eliminated_report['total_zero_scores']}")
        print(f"有0分的选手数: {len(eliminated_report['contestants_with_zeros'])}")
        
        # 显示前几个例子
        if eliminated_report['contestants_with_zeros']:
            print("\n前5个有0分的选手示例:")
            for i, contestant in enumerate(eliminated_report['contestants_with_zeros'][:5]):
                print(f"  {i+1}. {contestant['name']} (第{contestant['season']}季) - "
                      f"0分数量: {contestant['zero_count']}, 结果: {contestant['result']}")
        
        # 标记0分（这些是淘汰后的分数，不是真实评分）
        # 在后续计算中，我们需要识别这些0分并正确处理
        
        self.data_summary['eliminated_contestants'] = eliminated_report
        return eliminated_report
    
    def calculate_weekly_scores_and_ranks(self) -> pd.DataFrame:
        """
        任务4: 计算每周的评委总分和排名
        
        Returns:
        --------
        pd.DataFrame: 添加了每周总分和排名的数据框
        """
        print("\n" + "=" * 60)
        print("任务4: 计算每周的评委总分和排名")
        print("=" * 60)
        
        df = self.df.copy()
        
        # 识别所有评分列并按周分组
        score_columns = [col for col in df.columns if 'judge' in col.lower() and 'score' in col.lower()]
        
        # 提取周次信息
        week_numbers = set()
        for col in score_columns:
            week_num = self._extract_week_number(col)
            if week_num:
                week_numbers.add(week_num)
        
        week_numbers = sorted(week_numbers)
        print(f"识别到周次: {week_numbers[:10]}..." if len(week_numbers) > 10 else f"识别到周次: {week_numbers}")
        
        # 为每周计算总分和排名
        for week in week_numbers:
            # 找到该周的所有评分列
            week_cols = [col for col in score_columns if f'week{week}' in col.lower() or f'week_{week}' in col.lower()]
            
            if not week_cols:
                continue
            
            # 计算该周的总分（忽略N/A，忽略0分如果该选手已被淘汰）
            def calc_week_total(row):
                scores = []
                for col in week_cols:
                    val = row[col]
                    # 只计算非N/A且非0的分数（0分表示已淘汰）
                    if pd.notna(val) and val != 0:
                        scores.append(float(val))
                return sum(scores) if scores else np.nan
            
            total_col = f'week{week}_total_score'
            df[total_col] = df.apply(calc_week_total, axis=1)
            
            # 计算该周的排名（按季分组）
            rank_col = f'week{week}_judge_rank'
            df[rank_col] = df.groupby('season')[total_col].rank(method='min', ascending=False, na_option='bottom')
            
            # 计算该周的百分比（用于百分比法）
            percent_col = f'week{week}_judge_percent'
            season_totals = df.groupby('season')[total_col].transform('sum')
            df[percent_col] = df[total_col] / season_totals * 100
        
        # 统计计算了多少周
        total_cols = [col for col in df.columns if '_total_score' in col]
        print(f"✓ 已计算 {len(total_cols)} 周的总分和排名")
        
        self.processed_df = df
        self.data_summary['weekly_scores'] = {
            'weeks_calculated': len(total_cols),
            'week_numbers': week_numbers
        }
        
        return df
    
    def identify_season_info(self) -> Dict:
        """
        任务5: 识别每季的周数和选手数量
        
        Returns:
        --------
        Dict: 每季的详细信息
        """
        print("\n" + "=" * 60)
        print("任务5: 识别每季的周数和选手数量")
        print("=" * 60)
        
        if self.processed_df is None:
            df = self.df.copy()
        else:
            df = self.processed_df.copy()
        
        season_info = {}
        
        # 按季分组分析
        for season in sorted(df['season'].unique()):
            season_data = df[df['season'] == season]
            
            # 统计选手数量
            unique_contestants = season_data['celebrity_name'].nunique()
            
            # 统计周数（通过评分列推断）
            score_columns = [col for col in season_data.columns if 'judge' in col.lower() and 'score' in col.lower()]
            week_numbers = set()
            for col in score_columns:
                week_num = self._extract_week_number(col)
                if week_num:
                    # 检查该周是否有非N/A的评分
                    if season_data[col].notna().any():
                        week_numbers.add(week_num)
            
            # 获取该季的排名范围
            placements = season_data['placement'].dropna().unique()
            
            season_info[int(season)] = {
                'contestant_count': int(unique_contestants),
                'week_count': len(week_numbers),
                'weeks': sorted(week_numbers),
                'placement_range': (int(placements.min()), int(placements.max())) if len(placements) > 0 else (None, None),
                'results': season_data['results'].unique().tolist()[:5]  # 前5个结果示例
            }
        
        # 打印摘要
        print(f"总季数: {len(season_info)}")
        print("\n各季信息摘要:")
        print(f"{'季数':<8} {'选手数':<10} {'周数':<8} {'排名范围':<15}")
        print("-" * 50)
        
        for season in sorted(season_info.keys()):
            info = season_info[season]
            placement_str = f"{info['placement_range'][0]}-{info['placement_range'][1]}" if info['placement_range'][0] else "N/A"
            print(f"{season:<8} {info['contestant_count']:<10} {info['week_count']:<8} {placement_str:<15}")
        
        # 统计信息
        avg_contestants = np.mean([info['contestant_count'] for info in season_info.values()])
        avg_weeks = np.mean([info['week_count'] for info in season_info.values()])
        
        print(f"\n平均选手数: {avg_contestants:.1f}")
        print(f"平均周数: {avg_weeks:.1f}")
        
        self.data_summary['season_info'] = season_info
        return season_info
    
    def _extract_week_number(self, column_name: str) -> int:
        """
        从列名中提取周次数字
        
        Parameters:
        -----------
        column_name : str
            列名，如 'week1_judge1_score' 或 'week_2_judge_3_score'
        
        Returns:
        --------
        int: 周次数字，如果无法提取则返回None
        """
        import re
        # 匹配 weekX 或 week_X 格式
        match = re.search(r'week[_\s]*(\d+)', column_name.lower())
        if match:
            return int(match.group(1))
        return None
    
    def generate_summary_report(self) -> Dict:
        """
        生成完整的预处理摘要报告
        
        Returns:
        --------
        Dict: 完整的摘要报告
        """
        print("\n" + "=" * 60)
        print("预处理摘要报告")
        print("=" * 60)
        
        report = {
            'data_integrity': self.data_summary.get('integrity', {}),
            'missing_values': self.data_summary.get('missing_values', {}),
            'eliminated_contestants': self.data_summary.get('eliminated_contestants', {}),
            'weekly_scores': self.data_summary.get('weekly_scores', {}),
            'season_info': self.data_summary.get('season_info', {}),
        }
        
        print("\n✓ 所有预处理任务已完成！")
        print(f"✓ 处理后的数据形状: {self.processed_df.shape if self.processed_df is not None else self.df.shape}")
        
        return report
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        获取处理后的数据
        
        Returns:
        --------
        pd.DataFrame: 处理后的数据框
        """
        if self.processed_df is not None:
            return self.processed_df
        return self.df


def main():
    """主函数：运行所有预处理任务"""
    from loader import load_data
    
    print("开始DWTS数据预处理...")
    print("=" * 60)
    
    # 加载数据
    print("正在加载数据...")
    df = load_data()
    print(f"✓ 数据加载成功: {df.shape}")
    
    # 创建预处理器
    preprocessor = DWTSDataPreprocessor(df)
    
    # 执行所有任务
    preprocessor.check_data_integrity()
    preprocessor.handle_missing_values()
    preprocessor.handle_eliminated_contestants()
    preprocessor.calculate_weekly_scores_and_ranks()
    preprocessor.identify_season_info()
    
    # 生成报告
    report = preprocessor.generate_summary_report()
    
    # 保存处理后的数据
    processed_df = preprocessor.get_processed_data()
    output_path = '2026_MCM_Problem_C_Data_processed.csv'
    processed_df.to_csv(output_path, index=False)
    print(f"\n✓ 处理后的数据已保存到: {output_path}")
    
    return preprocessor, processed_df, report


if __name__ == "__main__":
    preprocessor, processed_df, report = main()
