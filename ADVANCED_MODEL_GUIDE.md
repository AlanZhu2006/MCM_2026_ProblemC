# 高级模型使用指南

## 为什么不用CNN/ResNet？

### CNN和ResNet的特点
- **CNN（卷积神经网络）**：主要用于**图像数据**，通过卷积操作提取空间特征
- **ResNet（残差网络）**：主要用于**图像分类**，解决深层网络训练问题

### 为什么不适合我们的问题？

1. **数据类型不匹配**
   - 我们的数据是**表格数据**（结构化数据），不是图像
   - 表格数据没有空间结构，不需要卷积操作
   - CNN的卷积核设计用于提取图像的局部模式（边缘、纹理等）

2. **特征维度不同**
   - 图像数据：高维（如224×224×3），有空间关系
   - 表格数据：低维特征向量，特征之间是独立的或弱相关的

3. **预训练模型不可用**
   - ImageNet等预训练模型是针对图像设计的
   - 表格数据没有通用的预训练模型

## 更适合的方法

### 1. 梯度提升树（Gradient Boosting Trees）⭐⭐⭐⭐⭐

**XGBoost** 和 **LightGBM** 是表格数据的**最佳选择**：

- ✅ **专门为表格数据设计**
- ✅ **自动特征交互**：能够学习特征之间的复杂关系
- ✅ **处理缺失值**：内置缺失值处理
- ✅ **特征重要性**：可以分析哪些特征最重要
- ✅ **高效**：训练和预测速度快
- ✅ **在Kaggle等竞赛中表现优异**

### 2. 集成学习（Ensemble Learning）⭐⭐⭐⭐

**VotingRegressor** 结合多个模型：
- Random Forest（随机森林）
- Gradient Boosting（梯度提升）
- XGBoost（如果可用）
- LightGBM（如果可用）
- Ridge/ElasticNet（线性模型）

### 3. 高级特征工程 ⭐⭐⭐⭐

#### 时间序列特征
- **趋势特征**：评分的变化趋势（斜率）
- **波动性特征**：评分的稳定性（标准差）
- **历史窗口**：最近N周的表现

#### 相对特征
- **归一化排名**：相对于当前选手数量的排名
- **百分位数**：排名在整体中的位置
- **与平均值的差异**：相对于平均水平的偏差

#### 交互特征
- **年龄×评分**：年龄与评分的交互
- **排名×竞争强度**：排名与剩余选手数的交互

#### 专业舞者特征增强
- **胜率**：专业舞者的历史胜率
- **平均排名**：专业舞者的平均排名
- **经验值**：专业舞者的参与次数

## 使用方法

### 安装依赖

```bash
pip install xgboost lightgbm
```

或者：

```bash
pip install -r requirements.txt
```

### 运行高级版本

```bash
python scripts/run_stage2_advanced_estimation.py
```

### 在代码中使用

```python
from fan_vote_estimator_advanced import AdvancedFanVoteEstimator
import pandas as pd

# 加载数据
df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')

# 创建高级估计器
estimator = AdvancedFanVoteEstimator(df)

# 估计所有周次
estimates_df = estimator.estimate_all_weeks_advanced(train_on_all=True)

# 验证模型
validation_results = estimator.validate_estimates(estimates_df)

# 查看特征重要性
if estimator.feature_importance:
    sorted_features = sorted(
        estimator.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
```

## 预期改进

### 相比基础版本
- ✅ **更准确的预测**：XGBoost/LightGBM通常比基础模型准确10-20%
- ✅ **更好的特征利用**：自动学习特征交互
- ✅ **更鲁棒**：对异常值更不敏感

### 相比ML版本
- ✅ **更先进的模型**：XGBoost/LightGBM > Random Forest
- ✅ **更丰富的特征**：时间序列、交互特征等
- ✅ **更好的特征缩放**：RobustScaler对异常值更鲁棒

## 模型选择建议

### 如果XGBoost/LightGBM可用
- 优先使用 **XGBoost** 或 **LightGBM**
- 使用 **集成模型**（ensemble）结合多个模型

### 如果只有基础库
- 使用 **Random Forest** + **Gradient Boosting**
- 仍然可以获得不错的性能

## 进一步优化方向

### 1. 超参数调优
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}
```

### 2. 特征选择
- 使用特征重要性进行特征选择
- 移除不重要的特征，减少过拟合

### 3. 交叉验证
- 使用时间序列交叉验证（TimeSeriesSplit）
- 避免数据泄露

### 4. 模型融合
- 结合多个模型的预测结果
- 使用加权平均或Stacking

## 总结

对于**表格数据**（结构化数据）：
- ✅ **推荐**：XGBoost, LightGBM, Random Forest, Gradient Boosting
- ❌ **不推荐**：CNN, ResNet（这些是图像模型）

我们的高级版本使用了最适合表格数据的方法，应该能获得更好的性能！
