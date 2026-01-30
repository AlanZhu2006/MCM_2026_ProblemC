# 阶段4：影响因素分析 - 使用指南

## 概述

阶段4的目标是分析各种因素对选手表现的影响，特别是区分这些因素对评委评分和粉丝投票的不同影响。

## 前置条件

✅ **阶段1和阶段2必须已完成**，因为阶段4需要以下数据：

1. **`2026_MCM_Problem_C_Data_processed.csv`** - 预处理后的数据
   - 包含选手特征：年龄、行业、地区、专业舞者等

2. **`fan_vote_estimates.csv`** - 阶段2估计的粉丝投票数据
   - 包含所有周次的粉丝投票估计值

## 核心任务

### 1. 专业舞者影响分析

**分析内容**：
- 不同专业舞者的平均评委评分
- 不同专业舞者的平均粉丝投票
- 专业舞者对最终排名的影响
- 专业舞者进入决赛的频率

**分析方法**：
- 按专业舞者分组统计
- 计算平均表现指标
- 识别表现最好和最差的专业舞者

### 2. 选手特征影响分析

#### 2.1 年龄影响
- **相关性分析**：年龄与评委评分/粉丝投票的相关系数
- **分组分析**：按年龄组（<30, 30-40, 40-50, 50-60, 60+）统计平均表现
- **回归分析**：年龄对两种投票的线性影响

#### 2.2 行业影响
- **分组统计**：按行业统计平均评委评分、粉丝投票、最终排名
- **排名分析**：识别表现最好和最差的行业
- **样本量分析**：每个行业的选手数量和周次数量

#### 2.3 地区影响
- **分组统计**：按国家/地区统计平均表现
- **排名分析**：识别表现最好和最差的地区
- **跨地区比较**：不同地区选手的表现差异

### 3. 区分对评委评分和粉丝投票的不同影响

**核心问题**：这些因素对评委评分和粉丝投票的影响是否相同？

**分析方法**：
- **专业舞者**：比较每个专业舞者对两种投票的影响差异
- **年龄**：比较年龄对两种投票的相关性差异
- **行业**：比较每个行业对两种投票的影响差异
- **地区**：比较每个地区对两种投票的影响差异

**关键指标**：
- 影响差异（impact_difference）= 对评委评分的影响 - 对粉丝投票的影响
- 相关性差异（correlation_difference）= 与评委评分的相关性 - 与粉丝投票的相关性

## 快速开始

### 方法1：使用Python脚本（推荐）

```bash
python scripts/run_stage4_factor_analysis.py
```

这将自动执行所有阶段4的任务。

### 方法2：在Python代码中使用

```python
import pandas as pd
from factor_impact_analyzer import FactorImpactAnalyzer

# 加载数据
processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
estimates_df = pd.read_csv('fan_vote_estimates.csv')

# 创建分析器
analyzer = FactorImpactAnalyzer(processed_df, estimates_df)

# 执行综合分析
analysis = analyzer.generate_comprehensive_analysis()

# 或者分别执行各项分析
pro_dancer_analysis = analyzer.analyze_pro_dancer_impact()
celebrity_features_analysis = analyzer.analyze_celebrity_features_impact()
comparison_analysis = analyzer.compare_judge_vs_fan_impacts()
```

## 输出文件

运行脚本后会生成：

1. **`factor_impact_analysis.json`**
   - 包含所有分析结果的JSON文件
   - 包括专业舞者、年龄、行业、地区的详细统计

2. **`pro_dancer_impact.csv`**
   - 专业舞者影响统计表
   - 列包括：pro_dancer, n_celebrities, n_weeks, avg_judge_score, avg_fan_votes, avg_placement等

3. **`industry_impact.csv`**
   - 行业影响统计表
   - 列包括：industry, avg_judge_score, avg_fan_votes, avg_placement等

4. **`region_impact.csv`**
   - 地区影响统计表
   - 列包括：region, avg_judge_score, avg_fan_votes, avg_placement等

5. **`stage4_factor_impact_report.txt`**
   - 文本格式的摘要报告

## 分析方法详解

### 专业舞者影响分析

**统计指标**：
- `avg_judge_score`: 该专业舞者合作选手的平均评委评分
- `avg_fan_votes`: 该专业舞者合作选手的平均粉丝投票
- `avg_placement`: 该专业舞者合作选手的平均最终排名（1为最好）
- `best_placement`: 该专业舞者获得的最佳排名
- `n_finalists`: 该专业舞者进入决赛的次数
- `finalist_rate`: 进入决赛的比例

**排序方式**：按平均排名（avg_placement）排序，排名越小越好

### 年龄影响分析

**分析方法**：
1. **相关性分析**：使用Pearson相关系数
   - 计算年龄与评委评分的相关性
   - 计算年龄与粉丝投票的相关性
   - 进行显著性检验（p值）

2. **分组分析**：按年龄组统计
   - <30岁
   - 30-40岁
   - 40-50岁
   - 50-60岁
   - 60岁以上

3. **回归分析**：线性回归
   - 计算斜率（slope）和截距（intercept）
   - 比较两种投票的斜率差异

### 行业影响分析

**分析方法**：
1. **分组统计**：按行业分组计算平均表现
2. **排名分析**：识别表现最好和最差的行业
3. **样本量检查**：确保每个行业有足够的样本量（至少5个数据点）

**关键发现**：
- 哪些行业的选手在评委评分方面表现更好？
- 哪些行业的选手在粉丝投票方面表现更好？
- 是否存在行业差异？

### 地区影响分析

**分析方法**：
1. **分组统计**：按国家/地区分组计算平均表现
2. **排名分析**：识别表现最好和最差的地区
3. **跨地区比较**：比较不同地区选手的表现

**注意事项**：
- 美国选手可能按州进一步细分
- 非美国选手按国家/地区分组

## 影响差异分析

### 专业舞者影响差异

**计算方法**：
```
judge_impact = 该专业舞者的平均评委评分 - 总体平均评委评分
fan_impact = 该专业舞者的平均粉丝投票 - 总体平均粉丝投票
impact_difference = judge_impact - fan_impact
```

**解释**：
- `impact_difference > 0`: 该专业舞者对评委评分的影响大于对粉丝投票的影响
- `impact_difference < 0`: 该专业舞者对粉丝投票的影响大于对评委评分的影响

### 年龄影响差异

**计算方法**：
```
judge_correlation = 年龄与评委评分的相关系数
fan_correlation = 年龄与粉丝投票的相关系数
correlation_difference = judge_correlation - fan_correlation
```

**解释**：
- `correlation_difference > 0`: 年龄对评委评分的影响更大
- `correlation_difference < 0`: 年龄对粉丝投票的影响更大

### 行业/地区影响差异

**计算方法**：与专业舞者相同，计算每个行业/地区的影响差异

## 关键发现示例

### 专业舞者影响
- 表现最好的专业舞者通常有更高的平均评委评分和粉丝投票
- 某些专业舞者可能更擅长获得评委高分，而另一些可能更擅长获得粉丝投票

### 年龄影响
- 年龄可能与评委评分呈负相关（年轻选手可能表现更好）
- 年龄对粉丝投票的影响可能不同（可能更关注人气而非技术）

### 行业影响
- 某些行业（如运动员）可能在评委评分方面表现更好
- 某些行业（如演员）可能在粉丝投票方面表现更好

### 影响差异
- 专业舞者对两种投票的影响可能不同
- 年龄、行业、地区对两种投票的影响也可能不同

## 注意事项

1. **样本量**：确保每个分组有足够的样本量（至少5个数据点）才能得出可靠结论

2. **因果关系**：相关性不等于因果关系，需要谨慎解释

3. **混杂因素**：可能存在其他未考虑的因素影响结果

4. **数据质量**：确保数据完整性和准确性

5. **统计显著性**：关注p值，只有显著的结果才值得讨论

## 下一步

完成阶段4后，可以：
- 查看详细的分析结果和报告
- 进入阶段5：新投票系统提案
- 基于分析结果提出改进建议

---

**提示**：阶段4的分析结果可以用于阶段5设计新投票系统时考虑不同因素的影响。
