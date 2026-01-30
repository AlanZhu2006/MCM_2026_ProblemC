# 阶段1：数据探索与预处理 - 使用指南

## 快速开始

### 方法1：使用Python脚本（推荐）

```bash
python scripts/run_stage1_preprocessing.py
```

这将自动执行所有5个预处理任务：
1. ✅ 加载并检查数据完整性
2. ✅ 处理缺失值（N/A）
3. ✅ 处理被淘汰选手的0分
4. ✅ 计算每周的评委总分和排名
5. ✅ 识别每季的周数和选手数量

### 方法2：在Python代码中使用

```python
from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor

# 加载数据
df = load_data()

# 创建预处理器
preprocessor = DWTSDataPreprocessor(df)

# 执行所有任务
preprocessor.check_data_integrity()
preprocessor.handle_missing_values()
preprocessor.handle_eliminated_contestants()
processed_df = preprocessor.calculate_weekly_scores_and_ranks()
season_info = preprocessor.identify_season_info()

# 获取处理后的数据
final_df = preprocessor.get_processed_data()

# 获取摘要报告
report = preprocessor.generate_summary_report()
```

### 方法3：在Jupyter Notebook中使用

```python
# 在notebooks/C_problemC_template.ipynb中
from preprocess_dwts import DWTSDataPreprocessor
from loader import load_data

df = load_data()
preprocessor = DWTSDataPreprocessor(df)

# 逐步执行任务并查看结果
# ...
```

## 输出文件

运行脚本后会生成：

1. **`2026_MCM_Problem_C_Data_processed.csv`**
   - 处理后的完整数据
   - 包含新增的列：
     - `weekX_total_score`: 每周的评委总分
     - `weekX_judge_rank`: 每周的评委排名
     - `weekX_judge_percent`: 每周的评委百分比（用于百分比法）

2. **`stage1_preprocessing_report.txt`**
   - 文本格式的摘要报告
   - 包含所有预处理任务的统计信息

## 任务详解

### 任务1：检查数据完整性
- 检查数据形状、内存使用
- 验证关键列是否存在
- 检查数据范围（季数、排名等）
- 识别重复行

### 任务2：处理缺失值（N/A）
- 统计所有缺失值
- 识别评分列中的N/A（表示未进行的周次或不存在第4位评委）
- 保留N/A值（这是数据的一部分，不是错误）

### 任务3：处理被淘汰选手的0分
- 识别所有0分（表示选手已被淘汰）
- 分析0分模式
- 标记这些0分以便后续正确处理

### 任务4：计算每周的评委总分和排名
- 为每周计算所有评委的总分
- 按季分组计算排名
- 计算百分比（用于百分比法比较）

### 任务5：识别每季的周数和选手数量
- 统计每季的选手数量
- 识别每季实际进行的周数
- 分析排名范围

## 注意事项

1. **N/A值的处理**：
   - N/A值保留在数据中，因为它们表示：
     - 未进行的周次（如第1季只有6周，第7-11周为N/A）
     - 不存在第4位评委的情况
   - 在计算总分时，N/A值会被忽略

2. **0分的处理**：
   - 0分表示选手已被淘汰，不是真实评分
   - 在计算总分和排名时，0分会被忽略（使用np.nan）

3. **评分列命名**：
   - 代码会自动识别 `weekX_judgeY_score` 格式的列
   - 支持多种命名格式（week1_judge1_score, week_1_judge_1_score等）

4. **内存使用**：
   - 处理后的数据会包含更多列（每周的总分、排名、百分比）
   - 如果内存不足，可以考虑分批处理

## 下一步

完成阶段1后，可以：
- 查看处理后的数据文件
- 查看摘要报告了解数据特征
- 进入阶段2：开发粉丝投票估计模型

## 常见问题

**Q: 如果数据文件找不到怎么办？**
A: 确保 `2026_MCM_Problem_C_Data.csv` 在项目根目录，或修改 `loader.py` 中的路径。

**Q: 处理时间很长怎么办？**
A: 这是正常的，因为需要处理34季的数据。如果太慢，可以考虑优化代码或使用更快的硬件。

**Q: 如何查看详细的处理结果？**
A: 运行脚本后查看控制台输出，或打开生成的 `stage1_preprocessing_report.txt` 文件。
