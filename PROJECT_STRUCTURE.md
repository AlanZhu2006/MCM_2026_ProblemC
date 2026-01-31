# 项目结构说明

## 核心文件结构

```
C_MCM_C/
├── 数据文件
│   ├── 2026_MCM_Problem_C_Data.csv              # 原始数据
│   ├── 2026_MCM_Problem_C_Data_processed.csv     # 预处理后的数据
│   └── 2026_MCM_Problem_C.pdf                   # 题目PDF
│
├── 核心代码文件（Stage 1-5）
│   ├── loader.py                                 # 数据加载模块
│   ├── preprocess_dwts.py                        # Stage 1: 数据预处理
│   ├── fan_vote_estimator.py                     # Stage 2: 粉丝投票估计
│   ├── voting_method_comparator.py               # Stage 3: 投票方法比较
│   ├── factor_impact_analyzer.py                 # Stage 4: 影响因素分析
│   ├── new_voting_system_designer.py             # Stage 5: 公平性调整投票系统
│   ├── ml_voting_system.py                       # Stage 5: ML投票系统
│   └── ml_voting_system_robust.py                # Stage 5: ML投票系统（防过拟合）
│
├── 运行脚本（scripts/）
│   ├── run_stage1_preprocessing.py               # Stage 1 运行脚本
│   ├── run_stage2_fan_vote_estimation.py        # Stage 2 运行脚本
│   ├── run_stage3_comparison.py                  # Stage 3 运行脚本
│   ├── run_stage4_factor_analysis.py             # Stage 4 运行脚本
│   ├── run_stage5_ml_auto.py                     # Stage 5: ML系统（自动测试所有模型）
│   ├── run_stage5_ml_robust.py                   # Stage 5: ML系统（防过拟合版本）
│   ├── analyze_stage5_model.py                   # Stage 5: 模型深度分析工具
│   ├── generate_visualizations.py                # 生成所有可视化图表
│   └── optimization/                             # 已完成的优化脚本（归档）
│       ├── quality_checker.py                    # 质量检查
│       ├── enhance_uncertainty_analysis.py       # 增强不确定性分析
│       ├── sensitivity_analysis.py                # 敏感性分析
│       ├── enhance_controversial_cases.py         # 争议案例深度分析
│       ├── enhance_math_formulas.py               # 数学公式增强
│       ├── additional_analysis.py                 # 额外分析
│       ├── optimize_visualizations.py             # 优化可视化
│       ├── integrate_optimizations.py             # 整合优化结果
│       └── run_all_optimizations.py               # 运行所有优化脚本
│
├── 指南文档
│   ├── README.md                                  # 项目总体说明
│   ├── STAGE1_GUIDE.md                           # Stage 1 使用指南
│   ├── STAGE2_GUIDE.md                           # Stage 2 使用指南
│   ├── STAGE3_GUIDE.md                           # Stage 3 使用指南
│   ├── STAGE4_GUIDE.md                           # Stage 4 使用指南
│   ├── STAGE5_GUIDE.md                           # Stage 5 使用指南
│   ├── PROJECT_STRUCTURE.md                      # 本文件
│   └── new_system_theoretical_analysis.md        # Stage 5 理论分析
│
├── 输出文件（Stage 1-5 核心输出）
│   ├── Stage 2 输出
│   │   ├── fan_vote_estimates.csv                # 估计的粉丝投票
│   │   ├── fan_vote_uncertainty.csv              # 不确定性分析
│   │   └── validation_results.json               # 模型验证结果
│   │
│   ├── Stage 3 输出
│   │   ├── voting_method_comparison.csv           # 投票方法比较结果
│   │   ├── controversial_cases_analysis.csv     # 争议案例分析
│   │   └── method_differences_analysis.json       # 方法差异分析
│   │
│   ├── Stage 4 输出
│   │   ├── factor_impact_analysis.json           # 影响因素分析
│   │   ├── pro_dancer_impact.csv                  # 专业舞者影响
│   │   ├── industry_impact.csv                    # 行业影响
│   │   └── region_impact.csv                      # 地区影响
│   │
│   └── Stage 5 输出
│       ├── stage5_model_analysis_<model>.txt      # 模型深度分析报告
│       └── stage5_model_analysis_<model>.json     # 模型分析数据
│
├── 配置文件
│   ├── requirements.txt                           # Python依赖包
│   └── 数据说明                                    # 数据说明文件
│
└── notebooks/
    └── C_problemC_template.ipynb                  # Jupyter笔记本模板
```

## 快速开始

### 按阶段运行

```bash
# Stage 1: 数据预处理
python scripts/run_stage1_preprocessing.py

# Stage 2: 粉丝投票估计
python scripts/run_stage2_fan_vote_estimation.py

# Stage 3: 投票方法比较
python scripts/run_stage3_comparison.py

# Stage 4: 影响因素分析
python scripts/run_stage4_factor_analysis.py

# Stage 5: 新投票系统设计
python scripts/run_stage5_new_system.py          # 公平性调整系统
python scripts/run_stage5_ml_auto.py            # ML系统（自动测试）
python scripts/run_stage5_ml_robust.py          # ML系统（防过拟合）
```

### 直接运行核心文件

```bash
# Stage 2: 粉丝投票估计（可直接运行）
python fan_vote_estimator.py
```

## 文件说明

### 核心代码文件

- **`loader.py`**: 从CSV文件加载原始数据
- **`preprocess_dwts.py`**: Stage 1 - 数据预处理，计算每周评分和排名
- **`fan_vote_estimator.py`**: Stage 2 - 粉丝投票估计模型（准确率90%）
- **`voting_method_comparator.py`**: Stage 3 - 比较排名法和百分比法
- **`factor_impact_analyzer.py`**: Stage 4 - 分析影响因素（专业舞者、年龄、行业、地区）
- **`new_voting_system_designer.py`**: Stage 5 - 公平性调整的投票系统
- **`ml_voting_system.py`**: Stage 5 - 基于机器学习的投票系统
- **`ml_voting_system_robust.py`**: Stage 5 - ML投票系统（防过拟合版本）

### 输出文件说明

**Stage 2 输出**：
- `fan_vote_estimates.csv`: 所有季、所有周的粉丝投票估计值
- `fan_vote_uncertainty.csv`: 不确定性分析（置信区间、标准差等）
- `validation_results.json`: 模型验证结果（准确率、正确/错误预测等）

**Stage 3 输出**：
- `voting_method_comparison.csv`: 两种方法的比较结果
- `controversial_cases_analysis.csv`: 争议案例分析（Jerry Rice, Bobby Bones等）
- `method_differences_analysis.json`: 方法差异的统计分析

**Stage 4 输出**：
- `factor_impact_analysis.json`: 影响因素的综合分析结果
- `pro_dancer_impact.csv`: 专业舞者影响统计
- `industry_impact.csv`: 行业影响统计
- `region_impact.csv`: 地区影响统计

**Stage 5 输出**：
- `new_voting_system_results.csv`: 新系统的详细结果
- `new_system_comparison.json`: 与原始系统的比较结果

## 依赖

主要依赖包（见 `requirements.txt`）：
- pandas, numpy
- scipy (用于优化算法)
- scikit-learn (用于机器学习)
- xgboost, lightgbm, catboost (用于梯度提升)
- 其他科学计算库

## 注意事项

1. **运行顺序**：建议按Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5的顺序运行
2. **数据依赖**：每个阶段依赖前一阶段的输出文件
3. **计算时间**：处理所有34季可能需要一些时间，请耐心等待
4. **输出文件**：运行脚本会自动生成输出文件，可以重新运行生成
