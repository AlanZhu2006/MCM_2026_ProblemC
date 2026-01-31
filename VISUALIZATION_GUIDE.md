# 数据可视化指南

## 概述

本指南说明如何为2026 MCM Problem C生成数据可视化图表，用于最终报告。

## 可视化脚本

**文件**: `scripts/generate_visualizations.py`

**功能**: 生成5个阶段的完整可视化图表

## 使用方法

### 1. 安装依赖

确保已安装所有必需的Python包：

```bash
pip install -r requirements.txt
```

### 2. 运行脚本

```bash
python scripts/generate_visualizations.py
```

### 3. 输出位置

所有图表将保存在 `visualizations/` 目录下：

- `stage2_fan_vote_estimation.png` - Stage 2粉丝投票估计可视化
- `stage3_voting_comparison.png` - Stage 3投票方法比较可视化
- `stage4_factor_impact.png` - Stage 4影响因素分析可视化
- `stage5_ml_system.png` - Stage 5 ML系统可视化
- `overall_summary.png` - 总体摘要图表

## 生成的图表说明

### Stage 2: 粉丝投票估计

**图表内容**:
1. 每季的周数分布（柱状图）
2. 粉丝投票分布（箱线图）
3. 评委评分 vs 粉丝投票散点图
4. 不确定性分析或模型性能

**用途**: 展示粉丝投票估计的分布和模型性能

### Stage 3: 投票方法比较

**图表内容**:
1. 两种方法的准确率对比（柱状图）
2. 按季次统计的不一致率（折线图）
3. 争议案例分析（水平柱状图）
4. 方法差异 vs 选手数量（柱状图）

**用途**: 展示排名法和百分比法的差异和影响

### Stage 4: 影响因素分析

**图表内容**:
1. 年龄对评委评分的影响（散点图+趋势线）
2. 表现最好的10位专业舞者（水平柱状图）
3. 表现最好的10个行业（水平柱状图）
4. 表现最好的10个地区（水平柱状图）

**用途**: 展示各种影响因素对选手表现的影响

### Stage 5: ML系统

**图表内容**:
1. 特征重要性（Top 10，水平柱状图）
2. 特征类别重要性分布（饼图）
3. 粉丝投票 vs 评委评分的重要性对比（饼图）
4. 原始系统 vs ML系统准确率对比（柱状图）

**用途**: 展示ML系统的内部机制和性能优势

### 总体摘要

**图表内容**:
1. 各阶段完成情况（水平柱状图）
2. 关键指标汇总（水平柱状图）
3. 数据覆盖范围（柱状图）
4. 系统性能提升（柱状图+箭头标注）

**用途**: 提供项目的整体概览和关键成果

## 数据要求

脚本需要以下文件存在：

### 必需文件
- `2026_MCM_Problem_C_Data_processed.csv` - 预处理数据
- `fan_vote_estimates.csv` - 粉丝投票估计
- `voting_method_comparison.csv` - 投票方法比较结果
- `controversial_cases_analysis.csv` - 争议案例分析
- `factor_impact_analysis.json` - 影响因素分析
- `stage5_model_analysis_lgb.json` - ML模型分析

### 可选文件
- `fan_vote_uncertainty.csv` - 不确定性分析
- `validation_results.json` - 验证结果

## 故障排除

### 问题1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'pandas'`

**解决**: 安装依赖包
```bash
pip install -r requirements.txt
```

### 问题2: 文件不存在

**错误**: `警告: XXX文件不存在`

**解决**: 
1. 确保已运行相应的Stage脚本生成数据文件
2. 检查文件路径是否正确
3. 脚本会自动处理缺失文件，但相关图表可能不完整

### 问题3: 中文显示问题

**错误**: 图表中中文显示为方框

**解决**: 
- Windows: 确保系统已安装中文字体（SimHei, Microsoft YaHei）
- Linux/Mac: 可能需要安装中文字体包

### 问题4: 内存不足

**错误**: 处理大数据集时内存不足

**解决**: 
- 脚本已对大数据集进行抽样（如散点图最多1000个点）
- 如果仍有问题，可以修改脚本中的抽样大小

## 自定义图表

如果需要自定义图表，可以修改 `scripts/generate_visualizations.py` 中的相应函数：

- `plot_stage2_fan_vote_estimation()` - Stage 2图表
- `plot_stage3_voting_comparison()` - Stage 3图表
- `plot_stage4_factor_impact()` - Stage 4图表
- `plot_stage5_ml_system()` - Stage 5图表
- `plot_overall_summary()` - 总体摘要图表

## 图表质量

所有图表以300 DPI保存，适合打印和报告使用。

## 注意事项

1. **数据完整性**: 确保所有Stage都已运行并生成输出文件
2. **文件路径**: 脚本使用相对路径，确保在项目根目录运行
3. **字体支持**: 图表使用中文字体，确保系统支持
4. **内存使用**: 大数据集可能消耗较多内存，建议关闭其他程序

## 报告使用建议

1. **选择关键图表**: 不是所有图表都需要放入报告，选择最能说明问题的
2. **图表说明**: 每个图表应配有清晰的标题和说明文字
3. **图表编号**: 在报告中为图表编号，便于引用
4. **图表大小**: 根据报告布局调整图表大小
5. **颜色选择**: 确保打印时颜色对比度足够

## 示例报告结构

```
1. 引言
   - 总体摘要图表

2. Stage 2: 粉丝投票估计
   - Stage 2可视化图表

3. Stage 3: 投票方法比较
   - Stage 3可视化图表

4. Stage 4: 影响因素分析
   - Stage 4可视化图表

5. Stage 5: 新投票系统
   - Stage 5可视化图表

6. 结论
   - 总体摘要图表（再次使用）
```

---

**最后更新**: 2026-01-31
