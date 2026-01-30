# ML模型优化总结

## 问题诊断

准确率下降的主要原因：

1. **训练目标错误**：原版本使用"是否被淘汰"（0/1）作为目标，但模型是回归器，预测值不在合理范围内
2. **预测转换逻辑简单**：直接将预测值转换为排名，没有考虑约束条件
3. **缺少约束保证**：没有确保被淘汰选手的综合得分最差

## 优化方案

### 1. 改进训练策略 ✅

**原方法**：
- 使用"是否被淘汰"（0/1）作为目标
- 模型预测值可能不在[0,1]范围内

**新方法**：
- 使用基础方法生成"伪标签"（归一化的粉丝排名 [0,1]）
- 模型学习预测归一化的粉丝排名
- 更符合回归任务的特点

```python
# 使用基础方法生成训练标签
base_estimate = super().estimate_fan_votes_rank_method(season, week, features_df)
fan_ranks = base_estimate['fan_ranks']
y_rank = (fan_ranks - 1) / (n_contestants - 1)  # 归一化到[0,1]
```

### 2. 混合预测方法 ✅

**原方法**：
- 直接使用ML预测值转换为排名
- 可能不满足约束条件

**新方法**：
- ML预测作为初始值
- 使用优化算法微调，确保约束条件满足
- 结合ML的预测能力和优化的约束保证

```python
# 1. ML预测归一化排名
predicted_rank_norm = ensemble_model.predict(X_scaled)

# 2. 转换为实际排名
fan_ranks = (predicted_rank_norm * (n_contestants - 1) + 1).astype(int)

# 3. 优化微调，确保约束
result = minimize(
    objective,  # 最小化与ML预测的差异
    fan_ranks,  # ML预测作为初始值
    constraints=[constraint_eliminated],  # 确保约束满足
    ...
)
```

### 3. 改进模型参数 ✅

**优化**：
- 增加树的数量（100 → 150）
- 调整深度和正则化参数
- 选择最佳3个模型进行集成（而不是固定3个）

```python
RandomForestRegressor(
    n_estimators=150,  # 增加
    max_depth=12,      # 调整
    min_samples_leaf=2,  # 添加
    ...
)
```

### 4. 改进目标函数 ✅

**原方法**：
- 只考虑与ML预测的差异

**新方法**：
- 同时考虑与ML预测的差异
- 考虑与评委排名的相关性（粉丝投票应该与评委评分相关）
- 平衡两个目标

```python
def objective(fan_ranks_opt):
    # ML预测的差异
    ml_penalty = np.sum((fan_ranks_opt - fan_ranks) ** 2)
    # 与评委排名的相关性
    judge_penalty = np.sum((fan_ranks_opt - judge_ranks) ** 2) * 0.3
    return ml_penalty + judge_penalty
```

### 5. 确保约束满足 ✅

**改进**：
- 在优化前检查初始值是否满足约束
- 如果不满足，先调整初始值
- 优化过程中强制满足约束

```python
# 确保初始值满足约束
combined_ranks_init = judge_ranks + x0
if combined_ranks_init[eliminated_idx] < np.max(combined_ranks_init):
    # 调整被淘汰选手的粉丝排名
    max_combined = np.max(combined_ranks_init)
    needed_rank = max_combined - judge_ranks[eliminated_idx] + 1
    x0[eliminated_idx] = min(needed_rank, n_contestants)
```

## 预期改进效果

### 准确率提升
- **原ML版本**: ~45-50%（可能更低）
- **优化后预期**: 55-65%
- **改进幅度**: +10-15%

### 优势
1. **更好的训练目标**：学习归一化排名，更合理
2. **约束保证**：确保被淘汰选手的综合得分最差
3. **混合方法**：结合ML预测和优化调整
4. **更好的泛化**：使用基础方法生成标签，减少过拟合

## 使用建议

### 运行优化版本
```bash
python scripts/run_stage2_ml_estimation.py
```

### 对比结果
- 对比基础版本和优化ML版本的准确率
- 分析哪些周次预测更准确
- 查看特征重要性，了解模型学到了什么

### 进一步优化
如果准确率仍然不够高，可以：
1. 调整目标函数中的权重（ml_penalty vs judge_penalty）
2. 尝试不同的模型组合
3. 添加更多特征
4. 使用时间序列特征（考虑历史连续性）

## 关键改进点总结

| 方面 | 原方法 | 优化方法 | 影响 |
|------|--------|---------|------|
| 训练目标 | 是否被淘汰 (0/1) | 归一化粉丝排名 [0,1] | ⭐⭐⭐ |
| 预测转换 | 直接转换 | ML预测 + 优化微调 | ⭐⭐⭐ |
| 约束保证 | 无保证 | 强制满足 | ⭐⭐⭐⭐ |
| 模型参数 | 固定参数 | 优化参数 | ⭐⭐ |
| 集成策略 | 固定3个模型 | 选择最佳3个 | ⭐⭐ |

## 注意事项

1. **训练时间**：优化版本需要先运行基础方法生成标签，训练时间可能更长
2. **内存使用**：需要同时存储基础估计和ML预测
3. **可解释性**：混合方法可能不如纯优化方法可解释

## 下一步

1. 运行优化版本，查看准确率
2. 如果准确率提升，可以进一步微调参数
3. 如果准确率仍然不够，考虑使用更复杂的特征工程
