# 阶段5：新投票系统设计 - 使用指南

## 概述

阶段5的目标是基于阶段4的影响因素分析结果，设计一个更"公平"或更"精彩"的投票组合系统，并提供支持证据。

**核心特点**：我们设计了**两种互补的新投票系统**，它们都综合考量了所有影响因素，并集成了排名法和百分比法的优势。

## 前置条件

✅ **阶段2、阶段3、阶段4必须已完成**，因为阶段5需要以下数据：

1. **`fan_vote_estimates.csv`** - 阶段2估计的粉丝投票数据
2. **`2026_MCM_Problem_C_Data_processed.csv`** - 预处理后的数据
3. **`factor_impact_analysis.json`** - 阶段4的影响因素分析结果

## 核心任务

### 1. 设计新的投票组合方法

我们设计了**两种新投票系统**，它们都综合考量了所有影响因素，并集成了排名法和百分比法：

#### 系统1：公平性调整的投票系统（Fairness-Adjusted Voting System）

基于阶段4的发现，我们设计了一个**公平性调整的投票系统**，它：

1. **综合考量所有影响因素**：
   - ✅ 年龄影响（相关系数：-0.24评委，-0.26粉丝）
   - ✅ 专业舞者影响（60位专业舞者的历史表现）
   - ✅ 行业影响（各行业的平均表现）
   - ✅ 地区影响（各国家/地区的平均表现）

2. **集成排名法和百分比法**：
   - ✅ 根据季次自动选择方法（第1-2季、第28-34季用排名法，第3-27季用百分比法）
   - ✅ 同时计算两种方法的综合得分
   - ✅ 使用调整后的评分进行排名/百分比计算

3. **动态权重调整**：
   - ✅ 根据评委和粉丝的分散程度动态调整权重
   - ✅ 当粉丝投票更分散时，降低粉丝权重
   - ✅ 当评委评分更分散时，降低评委权重

#### 系统2：基于机器学习的投票系统（ML-Based Voting System）

使用机器学习模型动态学习如何组合评委评分和粉丝投票：

1. **综合考量所有影响因素**：
   - ✅ 评委评分和粉丝投票（标准化、排名、百分比、相对值）
   - ✅ 年龄特征（标准化）
   - ✅ 专业舞者特征（编码为数值）
   - ✅ 行业特征（编码为数值）
   - ✅ 地区特征（编码为数值）

2. **支持多种模型**：
   - ✅ MLP（多层感知机）
   - ✅ 随机森林（Random Forest）
   - ✅ XGBoost
   - ✅ LightGBM
   - ✅ Gradient Boosting
   - ✅ SGD（随机梯度下降）

3. **集成学习**：
   - ✅ 投票分类器（Voting Classifier）集成多个模型
   - ✅ 软投票（Soft Voting）使用概率平均

核心改进包括：

#### 1.1 系统1：影响因素标准化（Fairness-Adjusted Voting System）

**年龄标准化**：
- 根据年龄调整评分，使不同年龄段的选手在同等水平下获得相似评分
- 年龄与评委评分的相关系数：-0.24
- 年龄与粉丝投票的相关系数：-0.26
- 调整公式：
  - $A_{judge,i} = 1 - \rho_{age,judge} \cdot z_{age,i} \cdot \alpha$
  - $A_{fan,i} = 1 - \rho_{age,fan} \cdot z_{age,i} \cdot \alpha$
  - 其中 $z_{age,i} = \frac{age_i - \bar{age}}{\sigma_{age}}$（标准化年龄）
  - $\alpha = 0.1$（调整强度，经过测试优化）

**专业舞者平衡**：
- 根据专业舞者的历史表现，对评分进行调整
- 排名越好的专业舞者，调整因子越大
- 调整公式：$D_j = 0.5 + 0.5 \cdot \frac{1}{placement_j + 0.5} \cdot \frac{1}{2}$
- 其中 $placement_j$ 是该专业舞者的平均排名

**行业/地区平衡**：
- 对行业和地区因素进行适度调整（默认禁用，因为影响较小）
- 表现越好的行业/地区，调整因子越大

**综合调整后的评分**：
- 调整后评委评分：$S_{judge,adjusted,i} = S_{judge,i} \cdot A_{judge,i} \cdot D_j \cdot I_i \cdot R_i$
- 调整后粉丝投票：$V_{fan,adjusted,i} = V_{fan,i} \cdot A_{fan,i} \cdot D_j \cdot I_i \cdot R_i$
- 其中 $I_i$ 和 $R_i$ 分别是行业和地区调整因子

#### 1.2 系统1：排名法和百分比法的集成

**自动选择方法**：
- 根据季次自动选择投票方法：
  - 第1-2季、第28-34季：使用**排名法**
  - 第3-27季：使用**百分比法**

**同时计算两种方法**：
- 排名法综合得分：
  $$R_{combined,i} = w_{judge} \cdot R_{judge,i} + w_{fan} \cdot R_{fan,i}$$
  - 其中 $R_{judge,i}$ 和 $R_{fan,i}$ 是基于调整后评分的排名
  - 综合排名**最高**者被淘汰

- 百分比法综合得分：
  $$P_{combined,i} = w_{judge} \cdot P_{judge,i} + w_{fan} \cdot P_{fan,i}$$
  - 其中 $P_{judge,i}$ 和 $P_{fan,i}$ 是基于调整后评分的百分比
  - 综合百分比**最低**者被淘汰

**实现细节**：
- 在 `calculate_fairness_adjusted_scores()` 方法中，同时计算：
  - `combined_ranks`：用于排名法
  - `combined_percents`：用于百分比法
- 根据季次选择对应的淘汰预测

#### 1.3 系统2：机器学习特征工程（ML-Based Voting System）

**特征提取**（`_prepare_features()` 方法）：

1. **基础特征**：
   - 评委评分标准化：$(S_{judge} - S_{min}) / (S_{max} - S_{min})$
   - 粉丝投票标准化：$(V_{fan} - V_{min}) / (V_{max} - V_{min})$

2. **排名特征**：
   - 评委排名标准化：$R_{judge} / n$（n为选手数）
   - 粉丝排名标准化：$R_{fan} / n$

3. **百分比特征**：
   - 评委百分比：$P_{judge} = S_{judge} / \sum S_{judge} \times 100$
   - 粉丝百分比：$P_{fan} = V_{fan} / \sum V_{fan} \times 100$

4. **相对特征**：
   - 评委相对值：$S_{judge} / \bar{S}_{judge}$
   - 粉丝相对值：$V_{fan} / \bar{V}_{fan}$

5. **影响因素特征**：
   - 年龄标准化：$(age - age_{min}) / (age_{max} - age_{min})$
   - 专业舞者编码：使用LabelEncoder编码为数值
   - 行业编码：使用LabelEncoder编码为数值
   - 地区编码：使用LabelEncoder编码为数值

**模型训练**：
- 目标：预测谁会被淘汰（二分类问题）
- 特征：上述所有特征（共12个特征）
- 模型：支持MLP、RF、XGB、LGB、GBDT、SGD等

**预测方法**：
- 使用 `predict_proba()` 获取每个选手被淘汰的概率
- 选择概率最高的选手作为淘汰预测

#### 1.4 系统1：动态权重调整

根据评委和粉丝的偏好差异，动态调整权重比例：

**变异系数计算**：
- 评委评分变异系数：$CV_{judge} = \frac{\sigma_{judge}}{\mu_{judge}}$
- 粉丝投票变异系数：$CV_{fan} = \frac{\sigma_{fan}}{\mu_{fan}}$

**权重调整公式**：
- $w_{judge} = w_{base,judge} \cdot (1 + (CV_{fan} - CV_{judge}) \cdot 0.2)$
- $w_{fan} = 1 - w_{judge}$

**调整逻辑**：
- 当粉丝投票更分散时（$CV_{fan} > CV_{judge}$），降低粉丝权重
- 当评委评分更分散时（$CV_{judge} > CV_{fan}$），降低评委权重
- 权重限制在 [0.3, 0.7] 范围内，避免极端情况

### 2. 理论分析其优势

#### 2.1 公平性提升

1. **减少年龄偏见**：通过年龄标准化，使不同年龄段的选手在同等水平下获得相似评分
2. **平衡专业舞者影响**：减少因专业舞者能力差异导致的不公平
3. **减少地区/行业偏见**：适度调整地区和行业因素的影响

#### 2.2 准确性提升

1. **动态权重**：根据实际情况动态调整评委和粉丝的权重，提高预测准确性
2. **综合考虑**：同时考虑多个影响因素，提供更全面的评估

#### 2.3 可解释性

1. **透明调整**：所有调整因子都可以量化解释
2. **可追溯性**：可以追踪每个调整因子的影响

### 3. 使用历史数据验证

新系统会：
1. 对所有历史周次应用新系统
2. 计算新系统的淘汰预测准确率
3. 与原始系统（排名法/百分比法）进行比较
4. 分析系统差异和改进效果

## 快速开始

### 方法1：使用Python脚本（推荐）

**ML投票系统（自动测试所有模型）**
```bash
python scripts/run_stage5_ml_auto.py
# 自动测试所有模型并生成比较报告
```

**ML投票系统（防过拟合版本）**
```bash
python scripts/run_stage5_ml_robust.py
# 使用时间序列交叉验证，减少过拟合
```

**模型深度分析工具**
```bash
python scripts/analyze_stage5_model.py
# 分析模型的内部机制：特征重要性、权重、决策规则等
```

### 方法2：在Python代码中使用

**ML投票系统**
```python
import pandas as pd
from ml_voting_system import MLVotingSystem

# 加载数据
estimates_df = pd.read_csv('fan_vote_estimates.csv')
processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')

# 创建ML投票系统（选择模型类型）
ml_system = MLVotingSystem(
    estimates_df=estimates_df,
    processed_df=processed_df,
    factor_analysis_path='factor_impact_analysis.json',
    model_type='lgb'  # 'mlp', 'rf', 'xgb', 'lgb', 'gbdt', 'sgd'
)

# 训练模型
training_results = ml_system.train(test_size=0.2, random_state=42)

# 应用ML系统到所有周次
ml_system_results = ml_system.apply_to_all_weeks()

# 比较ML系统与原始系统
comparison = ml_system.compare_with_original_systems(ml_system_results)

print(f"ML系统准确率: {comparison['ml_system_accuracy']:.2%}")
print(f"准确率提升: {comparison['accuracy_improvement']:.2%}")
```

## 系统参数

### 可调整参数

在 `FairnessAdjustedVotingSystem` 类中，可以调整以下参数：

```python
# 启用/禁用各项调整
new_system.age_adjustment_enabled = True  # 年龄调整
new_system.pro_dancer_adjustment_enabled = True  # 专业舞者调整
new_system.industry_adjustment_enabled = True  # 行业调整
new_system.region_adjustment_enabled = True  # 地区调整
new_system.dynamic_weight_enabled = True  # 动态权重

# 权重参数
new_system.base_judge_weight = 0.5  # 基础评委权重
new_system.base_fan_weight = 0.5  # 基础粉丝权重
new_system.adjustment_strength = 0.3  # 调整强度（0-1）
```

### 参数说明

- **adjustment_strength**：调整强度，范围0-1
  - 0：不调整
  - 0.3：适度调整（默认）
  - 1.0：最大调整

- **base_judge_weight / base_fan_weight**：基础权重
  - 默认各50%，可以根据需要调整

## 输出文件

运行脚本后会生成以下文件：

1. **`new_voting_system_results.csv`**
   - 新系统的详细结果
   - 包含每个选手的调整后评分、综合得分、权重等信息

2. **`new_system_comparison.json`**
   - 与原始系统的比较结果
   - 包含准确率、改进幅度等统计信息

3. **`new_system_theoretical_analysis.md`**
   - 详细的理论分析报告
   - 包含数学公式、系统优势等

4. **`stage5_new_system_report.txt`**
   - 综合报告
   - 包含系统概述、性能、优势等

## 代码结构

### 系统1：`new_voting_system_designer.py`

**核心类**：`FairnessAdjustedVotingSystem`

**主要方法**：

1. **`__init__()`**：初始化系统，加载影响因素分析结果
2. **`_build_factor_lookup_tables()`**：构建影响因素查找表（专业舞者、年龄、行业、地区）
3. **`_calculate_age_adjustment()`**：计算年龄调整因子
4. **`_calculate_pro_dancer_adjustment()`**：计算专业舞者调整因子
5. **`_calculate_industry_adjustment()`**：计算行业调整因子
6. **`_calculate_region_adjustment()`**：计算地区调整因子
7. **`_calculate_dynamic_weights()`**：计算动态权重（基于变异系数）
8. **`calculate_fairness_adjusted_scores()`**：
   - 计算公平性调整后的评分
   - **同时计算排名法和百分比法的综合得分**
   - 根据季次自动选择对应方法
9. **`apply_to_all_weeks()`**：对所有周次应用新系统
10. **`compare_with_original_systems()`**：比较新系统与原始系统

**关键实现**：
- 在 `calculate_fairness_adjusted_scores()` 中，同时计算：
  ```python
  # 排名法
  combined_ranks = judge_weight * judge_ranks + fan_weight * fan_ranks
  
  # 百分比法
  combined_percents = judge_weight * judge_percents + fan_weight * fan_percents
  ```
- 根据季次选择对应的淘汰预测（`eliminated_idx_rank` 或 `eliminated_idx_percent`）

### 系统2：`ml_voting_system.py`

**核心类**：`MLVotingSystem`

**主要方法**：

1. **`__init__()`**：初始化系统，设置模型类型
2. **`_prepare_features()`**：
   - 提取12个特征（评委评分、粉丝投票、排名、百分比、相对值、年龄、专业舞者、行业、地区）
   - 标准化和编码处理
3. **`_build_model()`**：构建ML模型（MLP、RF、XGB、LGB、GBDT、SGD）
4. **`train()`**：训练模型，使用train_test_split分割数据
5. **`predict_eliminated()`**：预测该周被淘汰的选手（使用概率）
6. **`apply_to_all_weeks()`**：对所有周次应用ML系统
7. **`compare_with_original_systems()`**：比较ML系统与原始系统

**关键实现**：
- 特征工程综合考量所有影响因素
- 使用 `predict_proba()` 获取淘汰概率
- 选择概率最高的选手作为淘汰预测

### 系统2（防过拟合版本）：`ml_voting_system_robust.py`

**核心类**：`RobustMLVotingSystem`（继承自`MLVotingSystem`）

**主要改进**：

1. **时间序列交叉验证**：
   - 使用按季次分割的交叉验证
   - 避免随机分割导致的数据泄露

2. **增加正则化**：
   - 降低模型复杂度（max_depth、n_estimators）
   - 增加L1/L2正则化强度

3. **过拟合风险分析**：
   - 比较训练集和验证集准确率
   - 计算过拟合风险指标

### 运行脚本

**`scripts/run_stage5_ml_auto.py`**：
- 自动测试所有ML模型（MLP、RF、XGB、LGB、GBDT、SGD）
- 创建集成模型（Voting Classifier）
- 生成模型比较报告

**`scripts/run_stage5_ml_robust.py`**：
- 运行防过拟合版本的ML系统
- 使用时间序列交叉验证（按季次分割）
- 增加正则化，降低模型复杂度
- 生成过拟合风险分析报告

**`scripts/analyze_stage5_model.py`**：
- 深度分析模型的内部机制
- 提取特征重要性、模型权重、决策规则
- 解释高准确率的原因
- 生成详细的算法分析报告

## 系统优势总结

### 系统1：公平性调整系统

**1. 综合考量所有影响因素**：
- ✅ 年龄影响（相关系数：-0.24评委，-0.26粉丝）
- ✅ 专业舞者影响（60位专业舞者的历史表现）
- ✅ 行业影响（各行业的平均表现）
- ✅ 地区影响（各国家/地区的平均表现）

**2. 集成排名法和百分比法**：
- ✅ 根据季次自动选择方法
- ✅ 同时计算两种方法的综合得分
- ✅ 保持与原始系统相同的淘汰机制

**3. 动态权重调整**：
- ✅ 根据评委和粉丝的分散程度动态调整权重
- ✅ 避免极端情况（权重限制在[0.3, 0.7]）

**4. 公平性提升**：
- ✅ 减少年龄偏见
- ✅ 平衡专业舞者影响
- ✅ 减少地区/行业偏见

**5. 可解释性**：
- ✅ 所有调整因子可量化
- ✅ 调整过程可追溯

### 系统2：ML投票系统

**1. 综合考量所有影响因素**：
- ✅ 评委评分和粉丝投票（多种表示：标准化、排名、百分比、相对值）
- ✅ 年龄特征（标准化）
- ✅ 专业舞者特征（编码为数值）
- ✅ 行业特征（编码为数值）
- ✅ 地区特征（编码为数值）

**2. 数据驱动**：
- ✅ 使用机器学习模型自动学习最优组合方式
- ✅ 不需要手动设计权重和调整因子

**3. 支持多种模型**：
- ✅ MLP、RF、XGB、LGB、GBDT、SGD
- ✅ 集成学习（Voting Classifier）

**4. 高准确率**：
- ✅ 测试集准确率：99.56%（防过拟合版本）
- ✅ 预测准确率：97.99%（防过拟合版本）

**5. 防过拟合**：
- ✅ 时间序列交叉验证
- ✅ 增加正则化
- ✅ 降低模型复杂度

### 两种系统的对比

| 特性 | 系统1：公平性调整 | 系统2：ML投票 |
|------|------------------|--------------|
| **影响因素** | ✅ 全部考量 | ✅ 全部考量 |
| **排名法/百分比法** | ✅ 集成（自动选择） | ❌ 不直接使用 |
| **动态权重** | ✅ 基于变异系数 | ✅ 模型自动学习 |
| **可解释性** | ✅ 高（公式明确） | ⚠️ 中等（模型可解释性） |
| **准确率** | 76.25% | 97.99%（防过拟合） |
| **公平性** | ✅ 明确调整 | ⚠️ 通过数据学习 |
| **灵活性** | ✅ 可调整参数 | ✅ 可切换模型 |

## 注意事项

1. **数据依赖**：确保阶段2、3、4已完成，相关数据文件存在
2. **参数调整**：根据实际效果调整 `adjustment_strength` 等参数
3. **计算时间**：处理所有周次可能需要一些时间，请耐心等待
4. **结果验证**：建议检查 `new_system_comparison.json` 中的准确率提升情况

## 算法深度分析

### 分析模型的内部机制

如果你想了解ML投票系统的具体算法、权重、特征重要性等细节，可以使用专门的分析脚本：

```bash
python scripts/analyze_stage5_model.py
```

这个脚本会：

1. **特征重要性分析**：
   - 显示所有特征的重要性排序
   - 按特征类别统计重要性
   - 分析粉丝投票 vs 评委评分的相对重要性

2. **模型权重分析**：
   - 对于线性模型（SGD）：提取系数和截距
   - 对于MLP：分析第一层权重
   - 对于树模型：显示特征重要性

3. **决策规则分析**：
   - 识别最重要的决策特征
   - 分析模型的决策逻辑

4. **高准确率原因分析**：
   - 解释为什么准确率这么高
   - 分析哪些特征贡献最大
   - 说明模型的优势

**输出文件**：
- `stage5_model_analysis_<model_type>.txt` - 详细分析报告
- `stage5_model_analysis_<model_type>.json` - 分析数据（JSON格式）

**示例输出**：
```
阶段5：ML投票系统算法深度分析报告
======================================================================

一、模型概述
模型类型: LGB
训练准确率: 99.65%
交叉验证准确率: 99.56%
预测准确率: 97.99%

二、特征重要性分析
1. fan_votes_normalized: 0.119000 (19.83%)
2. fan_relative: 0.077000 (12.83%)
3. fan_rank_normalized: 0.073000 (12.17%)
4. judge_percent: 0.054000 (9.00%)
5. fan_percent: 0.042000 (7.00%)
...

三、关键发现
1. 最重要的特征类别: 基础特征（评委+粉丝）
2. 粉丝投票相关特征的总重要性: 0.311000 (51.83%)
3. 评委评分相关特征的总重要性: 0.289000 (48.17%)
```

## 下一步

完成阶段5后，可以：

1. **分析模型的内部机制**：
   ```bash
   python scripts/analyze_stage5_model.py
   ```

2. 分析新系统的具体改进案例
3. 调整系统参数，优化性能
4. 准备阶段6：报告撰写

## 参考资料

- 阶段4影响因素分析：`factor_impact_analysis.json`
- 阶段4报告：`stage4_factor_impact_report.txt`
- 理论分析：`new_system_theoretical_analysis.md`
