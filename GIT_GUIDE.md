# Git 提交指南

## 已创建 .gitignore 文件

`.gitignore` 文件会自动忽略以下类型的文件：

### 被忽略的文件（不应提交）

1. **Python 缓存文件**
   - `__pycache__/` 目录
   - `*.pyc` 文件

2. **处理后的数据文件**（通常很大）
   - `*_processed.csv`
   - `*_estimates.csv`
   - `*_uncertainty.csv`
   - `2026_MCM_Problem_C_Data_processed.csv`
   - `fan_vote_estimates.csv`
   - `fan_vote_uncertainty.csv`

3. **报告和输出文件**
   - `*_report.txt`
   - `*_report.json`
   - `validation_results.json`
   - `stage*.txt`

4. **IDE 和系统文件**
   - `.vscode/`, `.idea/`
   - `.DS_Store`, `Thumbs.db`

## 应该提交的文件

### 核心代码文件 ✅
- `loader.py` - 数据加载模块
- `preprocess_dwts.py` - 数据预处理模块
- `fan_vote_estimator.py` - 粉丝投票估计模型
- `check_project_structure.py` - 项目结构检查脚本

### 运行脚本 ✅
- `scripts/run_stage1_preprocessing.py` - 阶段1运行脚本
- `scripts/run_stage2_fan_vote_estimation.py` - 阶段2运行脚本

### 文档文件 ✅
- `README.md` - 项目说明
- `STAGE1_GUIDE.md` - 阶段1使用指南
- `STAGE2_GUIDE.md` - 阶段2使用指南
- `GIT_GUIDE.md` - 本文件

### 配置文件 ✅
- `requirements.txt` - Python依赖列表
- `.gitignore` - Git忽略规则

### 原始数据文件 ✅（如果不大）
- `2026_MCM_Problem_C_Data.csv` - 原始数据（如果文件不大可以提交）

## 当前 Git 状态说明

### 修改的文件（Modified）
- `loader.py` - 已更新（这是正常的，因为我们重新创建了它）

### 删除的文件（Deleted）
- `models.py` - 旧文件，已被新文件替代
- `preprocess.py` - 旧文件，已被 `preprocess_dwts.py` 替代
- `scripts/run_problemC.py` - 旧文件，已被新的阶段脚本替代

这些删除是正常的，因为我们在重构项目结构。

### 新文件（Untracked）
以下新文件应该被提交：
- `.gitignore` ✅
- `STAGE1_GUIDE.md` ✅
- `STAGE2_GUIDE.md` ✅
- `check_project_structure.py` ✅
- `fan_vote_estimator.py` ✅
- `preprocess_dwts.py` ✅
- `requirements.txt` ✅
- `scripts/run_stage1_preprocessing.py` ✅
- `scripts/run_stage2_fan_vote_estimation.py` ✅

## 建议的提交步骤

```bash
# 1. 添加所有应该提交的文件
git add .gitignore
git add loader.py
git add preprocess_dwts.py
git add fan_vote_estimator.py
git add check_project_structure.py
git add requirements.txt
git add STAGE1_GUIDE.md
git add STAGE2_GUIDE.md
git add scripts/run_stage1_preprocessing.py
git add scripts/run_stage2_fan_vote_estimation.py

# 2. 删除旧文件
git rm models.py
git rm preprocess.py
git rm scripts/run_problemC.py

# 3. 提交
git commit -m "重构项目：添加阶段1和阶段2的完整实现

- 添加数据预处理模块 (preprocess_dwts.py)
- 添加粉丝投票估计模型 (fan_vote_estimator.py)
- 添加阶段1和阶段2的运行脚本
- 添加使用指南文档
- 添加.gitignore忽略输出文件
- 更新loader.py
- 删除旧的模板文件"
```

## 注意事项

1. **不要提交大文件**
   - 处理后的CSV文件通常很大，不应提交
   - 如果原始数据文件很大，也不应提交

2. **不要提交输出文件**
   - 报告文件、估计结果等都是运行后生成的
   - 可以通过代码重新生成

3. **提交前检查**
   ```bash
   git status
   ```
   确保只提交代码和文档文件

4. **如果误提交了大文件**
   ```bash
   # 从Git历史中删除（但保留本地文件）
   git rm --cached 大文件.csv
   git commit -m "Remove large file"
   ```
