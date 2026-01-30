# 项目结构说明

## 核心文件

### 主要代码文件
- **`fan_vote_estimator.py`** - 粉丝投票估计模型（核心文件，可直接运行）
  - 实现了多初值SLSQP优化、全局优化回退、严格约束保证等策略
  - 准确率：约90%
  - 可直接运行：`python fan_vote_estimator.py`

- **`preprocess_dwts.py`** - 数据预处理模块
  - 处理原始数据，计算每周评分和排名

- **`loader.py`** - 数据加载模块
  - 从CSV文件加载原始数据

### 运行脚本
- **`scripts/run_stage1_preprocessing.py`** - 阶段1预处理脚本
- **`scripts/run_stage2_fan_vote_estimation.py`** - 阶段2粉丝投票估计脚本

### 核心文档
- **`README.md`** - 项目总体说明
- **`CHANGELOG_FAN_VOTE_ESTIMATOR.md`** - 模型改进日志
- **`STAGE1_GUIDE.md`** - 阶段1使用指南
- **`STAGE2_GUIDE.md`** - 阶段2使用指南

## 快速开始

### 方法1：直接运行核心文件
```bash
python fan_vote_estimator.py
```

### 方法2：使用脚本
```bash
# 阶段1：数据预处理
python scripts/run_stage1_preprocessing.py

# 阶段2：粉丝投票估计
python scripts/run_stage2_fan_vote_estimation.py
```

## 输出文件

运行后会生成以下文件：
- `fan_vote_estimates.csv` - 估计的粉丝投票数据
- `fan_vote_uncertainty.csv` - 不确定性分析结果
- `validation_results.json` - 模型验证结果（包含准确率）

## 依赖

主要依赖包（见 `requirements.txt`）：
- pandas, numpy
- scipy (用于优化算法)
- 其他基础科学计算库
