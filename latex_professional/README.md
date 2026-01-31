# 专业LaTeX模板使用说明

## 模板特点

本模板基于优秀数学建模报告（如2410482）的格式和结构创建，具有以下特点：

1. **专业格式**：使用标准数学建模报告格式
2. **清晰结构**：章节层次分明，易于阅读
3. **美观排版**：优化的表格、图表和公式显示
4. **完整内容**：包含所有必需部分

## 文件结构

```
latex_professional/
├── main.tex                    # 主LaTeX文件
├── sections/                   # 所有章节文件
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
└── visualizations/            # 图表目录
    ├── stage2_fan_vote_estimation.png
    ├── stage3_voting_comparison.png
    ├── stage4_factor_impact.png
    ├── stage5_ml_system.png
    └── overall_summary.png
```

## 使用方法

1. **编译LaTeX**
   ```bash
   pdflatex main.tex
   pdflatex main.tex
   ```

2. **上传到Overleaf**
   - 将整个 `latex_professional` 文件夹打包为ZIP
   - 上传到Overleaf
   - 设置 `main.tex` 为主文件
   - 编译

## 模板改进

相比基础模板，本模板包含以下改进：

1. **页眉页脚**：添加了专业的页眉页脚
2. **标题格式**：优化了章节标题格式
3. **数学环境**：添加了定义、定理等数学环境
4. **表格优化**：使用更专业的表格格式
5. **算法展示**：支持算法伪代码显示
6. **代码高亮**：支持代码高亮显示

## 注意事项

- 确保所有图表文件在 `visualizations/` 目录
- 编译需要两次以生成正确的目录
- 团队编号已在摘要表中设置为 2603215

---

**完成！现在可以使用专业模板了！**
