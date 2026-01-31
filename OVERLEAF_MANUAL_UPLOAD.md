# Overleaf手动上传指南

## 📋 需要上传的文件清单

### 第一步：上传模板文件（最重要！）

**在项目根目录上传：**
1. ✅ `MCM_Latex2026/mcmthesis.cls` → 上传后重命名为 `mcmthesis.cls`（在根目录）

### 第二步：上传主文件

**在项目根目录上传：**
2. ✅ `main.tex`

### 第三步：创建sections文件夹并上传章节文件

**创建文件夹 `sections/`，然后上传：**
3. ✅ `sections/summary_sheet.tex`
4. ✅ `sections/introduction.tex`
5. ✅ `sections/stage1_preprocessing.tex`
6. ✅ `sections/stage2_fan_vote_estimation.tex`
7. ✅ `sections/stage3_voting_comparison.tex`
8. ✅ `sections/stage4_factor_impact.tex`
9. ✅ `sections/stage5_new_system.tex`
10. ✅ `sections/conclusions.tex`
11. ✅ `sections/memo_to_producers.tex`
12. ✅ `sections/references.tex`
13. ✅ `sections/ai_use_report.tex`

### 第四步：创建visualizations文件夹并上传图片

**创建文件夹 `visualizations/`，然后上传：**
14. ✅ `visualizations/stage2_fan_vote_estimation.png`
15. ✅ `visualizations/stage3_voting_comparison.png`
16. ✅ `visualizations/stage4_factor_impact.png`
17. ✅ `visualizations/stage5_ml_system.png`
18. ✅ `visualizations/overall_summary.png`
19. ✅ `visualizations/uncertainty_analysis.png`
20. ✅ `visualizations/confidence_intervals.png`
21. ✅ `visualizations/controversial_cases_detailed.png`
22. ✅ `visualizations/parameter_sensitivity.png`
23. ✅ `visualizations/data_sensitivity.png`

## 📁 最终文件结构

上传后，Overleaf项目应该是这样的结构：

```
your-project/
├── main.tex                    ← 主文件
├── mcmthesis.cls              ← 模板文件（重要！）
├── sections/
│   ├── summary_sheet.tex
│   ├── introduction.tex
│   ├── stage1_preprocessing.tex
│   ├── stage2_fan_vote_estimation.tex
│   ├── stage3_voting_comparison.tex
│   ├── stage4_factor_impact.tex
│   ├── stage5_new_system.tex
│   ├── conclusions.tex
│   ├── memo_to_producers.tex
│   ├── references.tex
│   └── ai_use_report.tex
└── visualizations/
    ├── stage2_fan_vote_estimation.png
    ├── stage3_voting_comparison.png
    ├── stage4_factor_impact.png
    ├── stage5_ml_system.png
    ├── overall_summary.png
    ├── uncertainty_analysis.png
    ├── confidence_intervals.png
    ├── controversial_cases_detailed.png
    ├── parameter_sensitivity.png
    └── data_sensitivity.png
```

## 🚀 详细上传步骤

### 步骤1：上传mcmthesis.cls（最关键！）

1. 在Overleaf项目根目录
2. 点击 "Upload" 按钮
3. 选择文件：`MCM_Latex2026/mcmthesis.cls`
4. **重要**：上传后确保文件名为 `mcmthesis.cls`（在根目录，不在MCM_Latex2026文件夹里）

### 步骤2：上传main.tex

1. 在项目根目录
2. 点击 "Upload"
3. 选择 `main.tex`
4. 上传后，右键点击 `main.tex` → "Set as Main Document"

### 步骤3：创建sections文件夹

1. 在项目根目录，点击 "New Folder"
2. 命名为 `sections`
3. 进入 `sections` 文件夹

### 步骤4：上传所有.tex文件到sections文件夹

在 `sections/` 文件夹内，逐个上传：
- `summary_sheet.tex`
- `introduction.tex`
- `stage1_preprocessing.tex`
- `stage2_fan_vote_estimation.tex`
- `stage3_voting_comparison.tex`
- `stage4_factor_impact.tex`
- `stage5_new_system.tex`
- `conclusions.tex`
- `memo_to_producers.tex`
- `references.tex`
- `ai_use_report.tex`

**提示**：可以按住Ctrl键多选文件一起上传！

### 步骤5：创建visualizations文件夹

1. 回到项目根目录
2. 点击 "New Folder"
3. 命名为 `visualizations`
4. 进入 `visualizations` 文件夹

### 步骤6：上传所有图片到visualizations文件夹

在 `visualizations/` 文件夹内，上传所有 `.png` 文件：
- `stage2_fan_vote_estimation.png`
- `stage3_voting_comparison.png`
- `stage4_factor_impact.png`
- `stage5_ml_system.png`
- `overall_summary.png`
- `uncertainty_analysis.png`
- `confidence_intervals.png`
- `controversial_cases_detailed.png`
- `parameter_sensitivity.png`
- `data_sensitivity.png`

**提示**：可以按住Ctrl键多选所有PNG文件一起上传！

### 步骤7：设置和编译

1. 确保 `main.tex` 是主文档（右键 → "Set as Main Document"）
2. 点击 "Recompile" 按钮
3. 检查是否有错误

## ⚠️ 重要检查点

### 检查1：mcmthesis.cls位置
- ✅ 必须在项目根目录
- ✅ 文件名必须是 `mcmthesis.cls`
- ❌ 不要在 `MCM_Latex2026/` 子文件夹里

### 检查2：文件夹结构
- ✅ `sections/` 文件夹在根目录
- ✅ `visualizations/` 文件夹在根目录
- ✅ 所有 `.tex` 文件在 `sections/` 里
- ✅ 所有 `.png` 文件在 `visualizations/` 里

### 检查3：文件路径
- ✅ `main.tex` 中使用 `\input{sections/introduction}` 是正确的
- ✅ 图片路径使用 `visualizations/stage2_fan_vote_estimation.png` 是正确的

## 🔧 如果上传后有问题

### 问题1：找不到mcmthesis.cls
**错误信息**：`File 'mcmthesis.cls' not found`

**解决**：
1. 检查 `mcmthesis.cls` 是否在项目根目录
2. 如果文件在 `MCM_Latex2026/` 文件夹里，需要移动到根目录
3. 在Overleaf中，可以拖拽文件到根目录

### 问题2：找不到sections文件
**错误信息**：`File 'sections/introduction.tex' not found`

**解决**：
1. 检查 `sections/` 文件夹是否存在
2. 检查所有 `.tex` 文件是否在 `sections/` 文件夹里
3. 检查文件名是否正确

### 问题3：图片不显示
**错误信息**：图片显示为占位符

**解决**：
1. 检查 `visualizations/` 文件夹是否存在
2. 检查所有 `.png` 文件是否在 `visualizations/` 文件夹里
3. 检查图片文件名是否与代码中的完全一致（区分大小写）

## 📝 快速检查清单

上传完成后，检查：
- [ ] `mcmthesis.cls` 在根目录
- [ ] `main.tex` 在根目录
- [ ] `sections/` 文件夹存在，包含11个 `.tex` 文件
- [ ] `visualizations/` 文件夹存在，包含10个 `.png` 文件
- [ ] `main.tex` 已设置为主文档
- [ ] 编译成功，无错误
- [ ] 所有图片正常显示
- [ ] 页码显示正确（Page X of Y）

---

**完成！现在可以开始上传了！**
