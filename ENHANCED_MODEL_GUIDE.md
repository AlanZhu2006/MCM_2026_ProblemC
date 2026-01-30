# 增强模型使用指南（最新版）

## 概述

增强版本整合了以下改进：
1. **智能NaN处理**（K-means聚类、KNN插值、时间序列插值）
2. **2024年MCM C题的方法**（状态空间模型、ARIMA、信息熵）
3. **XGBoost/LightGBM/CatBoost的内置缺失值处理**
4. **TabNet表格数据专用模型**

## 主要改进

### 1. 智能NaN处理 ⭐⭐⭐⭐⭐

#### 问题：传统方法只是简单填充NaN
- ❌ 丢失了缺失值的信息（缺失值本身可能包含信息）
- ❌ 没有考虑缺失值的模式

#### 解决方案：

**1.1 缺失值指示器（Missing Indicator）**
```python
# 为每个可能有缺失值的特征创建指示器
for col in numeric_cols:
    if features_df[col].isna().any():
        indicator_col = f'{col}_is_missing'
        enhanced_features[indicator_col] = features_df[col].isna().astype(int)
```

**1.2 信息缺失度（类似信息熵）**
```python
# 计算每个样本的缺失值比例和信息熵
missing_ratio = features_df[numeric_cols].isna().sum(axis=1) / len(numeric_cols)
enhanced_features['missing_ratio'] = missing_ratio
enhanced_features['missing_entropy'] = -missing_ratio * np.log(missing_ratio + 1e-10)
```

**1.3 时间序列插值**
- 对于历史评分特征，使用时间序列插值
- 利用前后周次的数据进行线性插值

**1.4 K-means聚类插值**（借鉴2301192的方法）
- 使用K-means对样本进行聚类
- 用同一簇内样本的中位数填充缺失值
- 类似2301192将"难度"量化为简单、中等、困难

**1.5 KNN插值（备用）**
- 如果K-means失败，使用KNN插值
- 考虑相似样本的特征值

### 2. 利用XGBoost/LightGBM的内置缺失值处理 ⭐⭐⭐⭐

#### XGBoost和LightGBM的优势：
- ✅ **内置缺失值处理**：可以直接处理NaN值
- ✅ **自动学习缺失值模式**：模型会学习如何处理缺失值
- ✅ **不需要预处理**：保留NaN，让模型自己处理

#### 实现：
```python
# 对于XGBoost/LightGBM，保留NaN
if name in ['xgboost', 'lightgbm']:
    X_train = X_all.copy()  # 保留NaN
else:
    X_train = X_scaled_other.copy()  # 使用插值后的数据
```

### 3. 整合状态空间模型思想 ⭐⭐⭐⭐

#### 3.1 动量特征（借鉴2024年C题）
```python
# 计算"动量"（评分变化率）
prev_score = prev_row['judge_total'].values[0]
curr_score = features_df.loc[idx, 'judge_total']
momentum = (curr_score - prev_score) / (prev_score + 1e-10)
features_df.loc[idx, 'momentum'] = momentum
```

#### 3.2 状态转移特征（马尔可夫模型）
```python
# 排名变化
prev_rank = prev_row['judge_rank'].values[0]
curr_rank = features_df.loc[idx, 'judge_rank']
rank_change = curr_rank - prev_rank
features_df.loc[idx, 'rank_change'] = rank_change
features_df.loc[idx, 'rank_improved'] = 1 if rank_change < 0 else 0
```

#### 3.3 信息熵特征（借鉴Team 2301192）
```python
# 计算特征的信息熵（不确定性）
hist, _ = np.histogram(col_data, bins=min(10, len(np.unique(col_data))))
hist = hist[hist > 0]
prob = hist / hist.sum()
entropy = -np.sum(prob * np.log(prob + 1e-10))
features_df[f'{col}_entropy'] = entropy
```

## 使用方法

### 安装依赖

**必需依赖**：
```bash
pip install xgboost lightgbm scikit-learn
```

**可选依赖**（推荐安装）：
```bash
pip install catboost statsmodels pytorch-tabnet torch
```

- **CatBoost**：另一个强大的表格数据模型，可以处理缺失值
- **statsmodels**：用于ARIMA时间序列模型
- **pytorch-tabnet**：表格数据专用神经网络模型

### 运行增强版本

```bash
python scripts/run_stage2_enhanced_estimation.py
```

### 在代码中使用

```python
from fan_vote_estimator_enhanced import EnhancedFanVoteEstimator
import pandas as pd

# 加载数据
df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')

# 创建增强估计器
estimator = EnhancedFanVoteEstimator(df)

# 估计所有周次
estimates_df = estimator.estimate_all_weeks_enhanced(train_on_all=True)

# 验证模型
validation_results = estimator.validate_estimates(estimates_df)
```

## 与Advanced版本的对比

| 特性 | Advanced版本 | **Enhanced版本** |
|------|-------------|-----------------|
| **NaN处理** | 简单填充（中位数） | ✅ 智能处理（指示器、插值） |
| **缺失值信息利用** | ❌ | ✅ 缺失值指示器、信息熵 |
| **XGBoost/LightGBM** | 使用插值后的数据 | ✅ 使用原始NaN数据 |
| **状态空间模型思想** | ❌ | ✅ 动量、状态转移、信息熵 |
| **时间序列插值** | ❌ | ✅ 对于历史评分特征 |
| **KNN插值** | ❌ | ✅ 对于其他特征 |

## 预期改进

### 1. 更好的NaN利用
- ✅ 缺失值指示器作为特征，模型可以学习缺失值模式
- ✅ 信息熵特征量化不确定性
- ✅ 时间序列插值更准确地估计历史评分

### 2. 更好的模型性能
- ✅ XGBoost/LightGBM可以更好地处理缺失值
- ✅ 状态空间模型特征提供时间序列信息
- ✅ 动量特征捕捉动态变化

### 3. 更完善的特征工程
- ✅ 缺失值相关特征
- ✅ 时间序列特征（动量、状态转移）
- ✅ 信息熵特征

## 技术细节

### 缺失值处理流程

1. **创建缺失值特征**（在插值之前）
   - 缺失值指示器
   - 缺失值比例
   - 信息熵

2. **智能插值**
   - 时间序列插值（历史评分）
   - KNN插值（其他特征）
   - 中位数填充（最终填充）

3. **模型训练**
   - XGBoost/LightGBM：使用带NaN的数据
   - 其他模型：使用插值后的数据

### 状态空间模型特征

1. **动量特征**
   - 计算评分变化率
   - 捕捉动态趋势

2. **状态转移特征**
   - 排名变化
   - 排名改善标志

3. **信息熵特征**
   - 量化特征的不确定性
   - 捕捉特征的分布特性

## 新增功能（最新版）

### 1. K-means聚类插值（借鉴2301192）
- ✅ 使用K-means对样本进行聚类
- ✅ 用同一簇内样本的中位数填充
- ✅ 类似2301192将抽象变量量化为逻辑层级

### 2. ARIMA时间序列模型（借鉴2301192）
- ✅ 使用ARIMA模型捕捉周期性波动
- ✅ 预测下一期的评分值
- ✅ 提供预测的标准误差

### 3. 更多高级模型
- ✅ **CatBoost**：另一个强大的表格数据模型
- ✅ **TabNet**：表格数据专用神经网络模型

## 总结

增强版本通过以下方式有效利用NaN值：

1. ✅ **缺失值指示器**：将缺失值模式编码为特征
2. ✅ **信息缺失度**：量化缺失值的信息量
3. ✅ **智能插值**：时间序列插值 + K-means聚类 + KNN插值
4. ✅ **XGBoost/LightGBM/CatBoost**：利用内置缺失值处理能力
5. ✅ **状态空间模型思想**：动量、状态转移、信息熵
6. ✅ **ARIMA模型**：捕捉周期性波动
7. ✅ **TabNet**：表格数据专用神经网络模型

这些改进应该能显著提升模型性能，特别是在处理大量缺失值的情况下！
