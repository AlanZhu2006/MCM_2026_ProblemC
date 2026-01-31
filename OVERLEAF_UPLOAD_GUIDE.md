# Overleaf上传指南

## 📋 需要上传的文件清单

### 必需文件（必须上传）

#### 1. 主LaTeX文件
- ✅ `main.tex` - 主文档文件

#### 2. 章节文件（sections文件夹）
- ✅ `sections/summary_sheet.tex`
- ✅ `sections/introduction.tex`
- ✅ `sections/stage1_preprocessing.tex`
- ✅ `sections/stage2_fan_vote_estimation.tex`
- ✅ `sections/stage3_voting_comparison.tex`
- ✅ `sections/stage4_factor_impact.tex`
- ✅ `sections/stage5_new_system.tex`
- ✅ `sections/conclusions.tex`
- ✅ `sections/memo_to_producers.tex`
- ✅ `sections/references.tex`
- ✅ `sections/ai_use_report.tex`

#### 3. 模板文件（MCM_Latex2026文件夹）
- ✅ `MCM_Latex2026/mcmthesis.cls` - **重要！必须上传**
- ✅ `MCM_Latex2026/mcmthesis.dtx` (可选，但建议上传)

#### 4. 图片文件（visualizations文件夹）
- ✅ `visualizations/stage2_fan_vote_estimation.png`
- ✅ `visualizations/stage3_voting_comparison.png`
- ✅ `visualizations/stage4_factor_impact.png`
- ✅ `visualizations/stage5_ml_system.png`
- ✅ `visualizations/overall_summary.png`
- ✅ `visualizations/uncertainty_analysis.png`
- ✅ `visualizations/confidence_intervals.png`
- ✅ `visualizations/controversial_cases_detailed.png`
- ✅ `visualizations/parameter_sensitivity.png`
- ✅ `visualizations/data_sensitivity.png`

## 🔧 上传前需要修改的内容

### 1. 检查main.tex中的路径

如果Overleaf项目结构不同，可能需要修改：

**当前main.tex使用：**
```latex
\input{sections/introduction}
```

**如果Overleaf中sections文件夹在根目录，保持不变即可。**

### 2. 检查mcmthesis.cls路径

**当前main.tex使用：**
```latex
\documentclass{mcmthesis}
```

**需要确保：**
- `mcmthesis.cls` 文件在Overleaf项目的根目录
- 或者修改为：`\documentclass{./MCM_Latex2026/mcmthesis}`（如果放在子文件夹）

### 3. 检查图片路径

**当前sections文件使用：**
```latex
\includegraphics[width=0.8\textwidth]{visualizations/stage2_fan_vote_estimation.png}
```

**需要确保：**
- `visualizations/` 文件夹在Overleaf项目根目录
- 图片路径正确

## 📤 上传步骤

### 方法1：上传到已有项目（推荐）

1. **登录Overleaf**
   - 访问 https://www.overleaf.com
   - 登录你的账户

2. **打开已有项目**
   - 找到你的项目并打开

3. **上传文件**
   - 点击左侧菜单的 "Upload" 按钮
   - 或者直接拖拽文件到文件树中

4. **上传顺序建议：**
   ```
   1. 先上传 mcmthesis.cls（到根目录）
   2. 上传 main.tex（到根目录）
   3. 创建 sections/ 文件夹，上传所有 .tex 文件
   4. 创建 visualizations/ 文件夹，上传所有 .png 文件
   ```

5. **设置主文件**
   - 右键点击 `main.tex`
   - 选择 "Set as Main Document"

6. **编译**
   - 点击 "Recompile" 按钮
   - 检查是否有错误

### 方法2：使用ZIP上传（更快）

1. **创建ZIP文件**
   - 选择以下文件/文件夹：
     - `main.tex`
     - `sections/` 文件夹
     - `visualizations/` 文件夹
     - `MCM_Latex2026/mcmthesis.cls`（提取到根目录）

2. **在Overleaf中**
   - 点击 "Upload" → "Upload .zip file"
   - 选择创建的ZIP文件
   - Overleaf会自动解压

## ⚠️ 重要注意事项

### 1. 文件路径结构

Overleaf项目结构应该是：
```
your-project/
├── main.tex
├── mcmthesis.cls          ← 必须！
├── sections/
│   ├── summary_sheet.tex
│   ├── introduction.tex
│   └── ... (其他.tex文件)
└── visualizations/
    ├── stage2_fan_vote_estimation.png
    └── ... (其他.png文件)
```

### 2. 编译设置

- **编译器**：pdfLaTeX（默认）
- **主文档**：main.tex
- **如果中文有问题**：切换到 XeLaTeX 或 LuaLaTeX

### 3. 团队编号

当前设置为：**2603215**

如需修改，编辑：
- `main.tex` 第3行：`tcn = 2603215`
- `sections/summary_sheet.tex` 中的团队编号

### 4. 常见问题排查

**问题1：找不到mcmthesis.cls**
- 解决：确保 `mcmthesis.cls` 在项目根目录

**问题2：图片不显示**
- 解决：检查 `visualizations/` 文件夹路径是否正确

**问题3：编译错误**
- 解决：查看编译日志，检查缺失的包或文件

**问题4：页码显示"??"**
- 解决：确保已添加 `\usepackage{lastpage}` 和 `\label{LastPage}`

## ✅ 上传后检查清单

- [ ] `main.tex` 已上传到根目录
- [ ] `mcmthesis.cls` 已上传到根目录
- [ ] `sections/` 文件夹已创建，所有 `.tex` 文件已上传
- [ ] `visualizations/` 文件夹已创建，所有 `.png` 文件已上传
- [ ] `main.tex` 已设置为主文档
- [ ] 编译成功，无错误
- [ ] 所有图表正常显示
- [ ] 目录正确生成
- [ ] 页码显示正确（Page X of Y）
- [ ] 总页数不超过25页（不包括AI使用报告）

## 🚀 快速上传脚本

如果需要，我可以帮你创建一个ZIP文件，包含所有必需文件。

---

**完成上传后，记得在Overleaf中编译并检查！**
