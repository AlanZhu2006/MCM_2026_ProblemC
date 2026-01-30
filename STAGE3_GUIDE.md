# 阶段3：投票方法比较分析 - 使用指南

## 概述

阶段3的目标是比较两种投票组合方法（排名法和百分比法）在所有季次中的表现差异，分析哪种方法更偏向粉丝投票，并针对争议案例进行深入分析。

## 前置条件

✅ **阶段2必须已完成**，因为阶段3需要以下数据：

1. **`fan_vote_estimates.csv`** - 阶段2估计的粉丝投票数据
   - 包含所有季、所有周的粉丝投票估计值
   - 列：season, week, celebrity_name, fan_votes, judge_total, voting_method, eliminated

2. **`2026_MCM_Problem_C_Data_processed.csv`** - 预处理后的数据
   - 包含评委评分、排名等信息

3. **`validation_results.json`** - 模型验证结果（可选，用于了解准确率）

## 核心任务

### 1. 实现排名法计算
- **输入**：评委评分、估计的粉丝投票
- **计算**：
  - 评委排名 = 根据评委总分排名（1=最好，N=最差）
  - 粉丝排名 = 根据粉丝投票排名（1=最多，N=最少）
  - 综合排名 = 评委排名 + 粉丝排名
- **输出**：每周每个选手的综合排名

### 2. 实现百分比法计算
- **输入**：评委评分、估计的粉丝投票
- **计算**：
  - 评委百分比 = 选手评委总分 / 所有选手评委总分之和 × 100
  - 粉丝百分比 = 选手粉丝投票 / 所有选手粉丝投票之和 × 100
  - 综合百分比 = 评委百分比 + 粉丝百分比
- **输出**：每周每个选手的综合百分比

### 3. 对所有季应用两种方法
- 对**所有34季**的**每一周**都应用两种方法
- 比较两种方法产生的淘汰结果差异
- 统计差异发生的频率和模式

### 4. 分析差异和模式
- **差异统计**：
  - 有多少周次两种方法产生不同的淘汰结果
  - 差异主要集中在哪些季次
  - 差异是否与选手数量、周次等因素相关
- **模式分析**：
  - 哪种方法更偏向粉丝投票
  - 哪种方法更偏向评委评分
  - 两种方法在不同情况下的表现差异

### 5. 分析争议案例
根据README，争议案例包括：
- **第2季**：Jerry Rice - 尽管评委评分最低（5周），但仍进入决赛获得亚军
- **第4季**：Billy Ray Cyrus - 尽管6周评委评分垫底，仍获得第5名
- **第11季**：Bristol Palin - 尽管12次评委评分最低，仍获得第3名
- **第27季**：Bobby Bones - 尽管评委评分持续偏低，仍获得冠军

**分析内容**：
- 使用排名法，这些选手是否会被淘汰
- 使用百分比法，这些选手是否会被淘汰
- 两种方法对这些争议案例的处理是否相同
- 分析为什么粉丝投票能够"拯救"这些选手

## 实现思路

### 数据结构设计

```python
# 比较结果数据结构
comparison_results = {
    'season': int,
    'week': int,
    'celebrity_name': str,
    'judge_total': float,
    'fan_votes': float,
    # 排名法结果
    'rank_method': {
        'judge_rank': int,
        'fan_rank': int,
        'combined_rank': int,
        'would_be_eliminated': bool
    },
    # 百分比法结果
    'percent_method': {
        'judge_percent': float,
        'fan_percent': float,
        'combined_percent': float,
        'would_be_eliminated': bool
    },
    # 差异标记
    'methods_agree': bool  # 两种方法是否产生相同的淘汰结果
}
```

### 核心算法

#### 排名法计算
```python
def calculate_rank_method(judge_totals, fan_votes):
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
    dict: 包含排名信息的字典
    """
    n = len(judge_totals)
    
    # 计算排名（1=最好，n=最差）
    judge_ranks = pd.Series(judge_totals).rank(ascending=False, method='min').astype(int)
    fan_ranks = pd.Series(fan_votes).rank(ascending=False, method='min').astype(int)
    
    # 综合排名
    combined_ranks = judge_ranks + fan_ranks
    
    # 找出应该被淘汰的（综合排名最高）
    eliminated_idx = combined_ranks.idxmax()
    
    return {
        'judge_ranks': judge_ranks.values,
        'fan_ranks': fan_ranks.values,
        'combined_ranks': combined_ranks.values,
        'eliminated_idx': eliminated_idx
    }
```

#### 百分比法计算
```python
def calculate_percent_method(judge_totals, fan_votes):
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
    dict: 包含百分比信息的字典
    """
    # 计算百分比
    judge_percents = (judge_totals / judge_totals.sum()) * 100
    fan_percents = (fan_votes / fan_votes.sum()) * 100
    
    # 综合百分比
    combined_percents = judge_percents + fan_percents
    
    # 找出应该被淘汰的（综合百分比最低）
    eliminated_idx = combined_percents.idxmin()
    
    return {
        'judge_percents': judge_percents.values,
        'fan_percents': fan_percents.values,
        'combined_percents': combined_percents.values,
        'eliminated_idx': eliminated_idx
    }
```

### 工作流程

```
1. 加载数据
   ├── 加载 fan_vote_estimates.csv（阶段2的输出）
   └── 加载 2026_MCM_Problem_C_Data_processed.csv

2. 对每一周进行计算
   ├── 获取该周所有选手的评委总分
   ├── 获取该周所有选手的粉丝投票（估计值）
   ├── 应用排名法计算
   ├── 应用百分比法计算
   └── 比较两种方法的结果

3. 统计分析
   ├── 统计差异发生的频率
   ├── 分析差异的模式
   └── 识别争议案例

4. 生成报告
   ├── 差异统计表
   ├── 争议案例分析
   └── 可视化图表
```

## 预期输出

### 1. 比较结果CSV文件
- `voting_method_comparison.csv` - 包含所有周次的两种方法比较结果

### 2. 差异统计报告
- `method_differences_report.txt` - 差异统计和分析

### 3. 争议案例分析
- `controversial_cases_analysis.txt` - 针对争议案例的详细分析

### 4. 可视化图表（可选）
- 差异频率图
- 争议案例对比图
- 方法偏向性分析图

## 关键分析问题

### 1. 哪种方法更偏向粉丝投票？

**分析指标**：
- 当评委评分和粉丝投票不一致时，哪种方法更倾向于选择粉丝投票高的选手
- 统计在争议案例中，哪种方法更"保护"粉丝投票高的选手

**判断标准**：
- 如果排名法更经常选择粉丝投票高的选手，说明排名法更偏向粉丝投票
- 如果百分比法更经常选择粉丝投票高的选手，说明百分比法更偏向粉丝投票

### 2. 两种方法的公平性比较

**公平性定义**：
- 评委评分和粉丝投票的权重是否平衡
- 是否给粉丝投票过大的影响力
- 是否给评委评分过大的影响力

**分析方法**：
- 计算两种方法下，评委评分和粉丝投票的"影响力"
- 分析在极端情况下（评委评分很低但粉丝投票很高）两种方法的处理

### 3. 第28季新机制的影响

**新机制**：评委从综合得分最低的两对选手中选择淘汰者

**分析内容**：
- 如果使用排名法，哪些选手会进入"最低两名"
- 如果使用百分比法，哪些选手会进入"最低两名"
- 两种方法下，评委的选择空间是否不同

## 实现说明

### 已实现的代码

✅ **核心文件已创建**：

1. **`voting_method_comparator.py`** - 投票方法比较器类
   - `VotingMethodComparator` 类：主比较器
   - `calculate_rank_method()` - 排名法计算
   - `calculate_percent_method()` - 百分比法计算
   - `compare_methods_for_week()` - 单周比较
   - `compare_all_weeks()` - 批量比较
   - `analyze_differences()` - 差异分析
   - `analyze_controversial_cases()` - 争议案例分析

2. **`scripts/run_stage3_comparison.py`** - 运行脚本
   - 自动加载数据
   - 执行比较分析
   - 生成报告

### 使用方法

#### 方法1：使用Python脚本（推荐）

```bash
python scripts/run_stage3_comparison.py
```

这将自动执行所有阶段3的任务。

#### 方法2：在Python代码中使用

```python
import pandas as pd
from voting_method_comparator import VotingMethodComparator

# 加载数据
estimates_df = pd.read_csv('fan_vote_estimates.csv')
processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')

# 创建比较器
comparator = VotingMethodComparator(estimates_df, processed_df)

# 比较所有周次
comparison_df = comparator.compare_all_weeks()

# 分析差异
analysis = comparator.analyze_differences(comparison_df)

# 分析争议案例
controversial_df = comparator.analyze_controversial_cases(comparison_df)
```

## 注意事项

1. **数据一致性**
   - 确保使用阶段2估计的粉丝投票数据
   - 确保评委评分数据来自预处理后的数据

2. **排名计算**
   - 注意处理并列情况（相同分数/投票）
   - 使用 `method='min'` 确保排名一致

3. **百分比计算**
   - 确保总和为100%
   - 注意处理零值情况

4. **争议案例识别**
   - 需要手动识别或通过规则识别争议案例
   - 确保分析这些案例时使用正确的数据

5. **第28季特殊处理**
   - 第28季及以后有"评委选择"机制
   - 需要分析这个机制对两种方法的影响

## 与阶段2的关系

✅ **阶段2是阶段3的前置条件**：

- 阶段2提供了**估计的粉丝投票数据**，这是阶段3进行方法比较的基础
- 阶段2的准确率（约90%）保证了阶段3比较结果的可靠性
- 阶段2已经实现了两种方法的估计逻辑，阶段3需要重新计算以进行比较

**关键区别**：
- **阶段2**：使用一种方法（根据季次）估计粉丝投票
- **阶段3**：使用两种方法（排名法+百分比法）重新计算，比较差异

## 下一步

完成阶段3后，可以：
- 进入阶段4：影响因素分析（专业舞者、选手特征等）
- 基于阶段3的发现，设计新的投票系统（阶段5）

---

**提示**：阶段2成功后，确实可以开始阶段3！阶段2已经提供了所有必需的数据。
