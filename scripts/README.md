# Scripts目录结构说明

## 核心运行脚本（scripts/）

这些脚本是项目运行的核心，必须保留：

1. **run_stage1_preprocessing.py** - Stage 1: 数据预处理
2. **run_stage2_fan_vote_estimation.py** - Stage 2: 粉丝投票估计
3. **run_stage3_comparison.py** - Stage 3: 投票方法比较
4. **run_stage4_factor_analysis.py** - Stage 4: 影响因素分析
5. **run_stage5_ml_auto.py** - Stage 5: ML投票系统（自动测试所有模型）
6. **run_stage5_ml_robust.py** - Stage 5: ML投票系统（防过拟合版本）
7. **analyze_stage5_model.py** - Stage 5: 模型深度分析工具
8. **generate_visualizations.py** - 生成所有可视化图表

## 优化脚本（scripts/optimization/）

这些脚本已经完成优化工作，已归档到此目录：

1. **quality_checker.py** - 质量检查
2. **enhance_uncertainty_analysis.py** - 增强不确定性分析
3. **sensitivity_analysis.py** - 敏感性分析
4. **enhance_controversial_cases.py** - 争议案例深度分析
5. **enhance_math_formulas.py** - 数学公式增强
6. **additional_analysis.py** - 额外分析
7. **optimize_visualizations.py** - 优化可视化
8. **integrate_optimizations.py** - 整合优化结果到LaTeX
9. **run_all_optimizations.py** - 运行所有优化脚本

## 使用说明

### 运行核心阶段脚本

```bash
# Stage 1-5
python scripts/run_stage1_preprocessing.py
python scripts/run_stage2_fan_vote_estimation.py
python scripts/run_stage3_comparison.py
python scripts/run_stage4_factor_analysis.py
python scripts/run_stage5_ml_auto.py

# 生成可视化
python scripts/generate_visualizations.py

# 分析模型
python scripts/analyze_stage5_model.py
```

### 重新运行优化（如果需要）

```bash
# 运行所有优化
python scripts/optimization/run_all_optimizations.py

# 或单独运行某个优化
python scripts/optimization/quality_checker.py
python scripts/optimization/enhance_uncertainty_analysis.py
# ... 等等
```

## 注意事项

- 优化脚本已经完成工作，结果已整合到LaTeX文件中
- 如果需要重新生成优化结果，可以运行optimization目录中的脚本
- 核心运行脚本必须保留，不要删除
