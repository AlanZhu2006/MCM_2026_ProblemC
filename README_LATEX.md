# LaTeX报告使用说明

## 快速开始

### 1. 检查文件结构

确保以下文件存在：
```
main.tex
sections/
  ├── summary_sheet.tex
  ├── introduction.tex
  ├── stage1_preprocessing.tex
  ├── stage2_fan_vote_estimation.tex
  ├── stage3_voting_comparison.tex
  ├── stage4_factor_impact.tex
  ├── stage5_new_system.tex
  ├── conclusions.tex
  ├── memo_to_producers.tex
  ├── references.tex
  └── ai_use_report.tex
visualizations/
  ├── stage2_fan_vote_estimation.png
  ├── stage3_voting_comparison.png
  ├── stage4_factor_impact.png
  ├── stage5_ml_system.png
  └── overall_summary.png
```

### 2. 修改团队编号

在以下文件中将 `[Your Team Number]` 替换为你的实际团队编号：
- `main.tex` (第24行)
- `sections/summary_sheet.tex` (第12行)

### 3. 编译LaTeX

**Windows**:
```bash
compile_latex.bat
```

**Linux/Mac**:
```bash
pdflatex main.tex
pdflatex main.tex
```

**注意**: 需要编译两次以生成正确的目录和交叉引用。

### 4. 检查输出

编译后会生成 `main.pdf`，检查：
- [ ] 摘要表在第1页
- [ ] 目录正确生成
- [ ] 所有图表正常显示
- [ ] 页码正确
- [ ] 总页数不超过25页（不包括AI使用报告）

## 报告结构

1. **摘要表** (1页) - `sections/summary_sheet.tex`
2. **目录** (自动生成)
3. **引言** - `sections/introduction.tex`
4. **Stage 1: 数据预处理** - `sections/stage1_preprocessing.tex`
5. **Stage 2: 粉丝投票估计** - `sections/stage2_fan_vote_estimation.tex`
6. **Stage 3: 投票方法比较** - `sections/stage3_voting_comparison.tex`
7. **Stage 4: 影响因素分析** - `sections/stage4_factor_impact.tex`
8. **Stage 5: 新投票系统提案** - `sections/stage5_new_system.tex`
9. **结论与建议** - `sections/conclusions.tex`
10. **备忘录** (1-2页) - `sections/memo_to_producers.tex`
11. **参考文献** - `sections/references.tex`
12. **AI使用报告** (如使用) - `sections/ai_use_report.tex`

## 自定义内容

### 添加更多内容

在相应的 `.tex` 文件中添加内容即可。

### 修改图表

1. 将新图表放入 `visualizations/` 目录
2. 在相应章节文件中添加：
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{visualizations/your_figure.png}
\caption{Your Caption}
\label{fig:yourlabel}
\end{figure}
```

### 调整页面布局

在 `main.tex` 中修改 `\geometry` 设置。

## 常见问题

### Q: 编译错误 "File not found"
A: 确保所有文件都在正确的位置，路径正确。

### Q: 中文显示问题
A: 当前使用UTF-8编码。如果仍有问题，可以改用XeLaTeX编译。

### Q: 目录不显示
A: 需要编译两次LaTeX文件。

### Q: 图表太大/太小
A: 调整 `\includegraphics` 中的 `width` 参数。

## 提交前检查清单

- [ ] 团队编号已填写
- [ ] 所有章节内容完整
- [ ] 所有图表已插入并正常显示
- [ ] 目录正确生成
- [ ] 页码正确
- [ ] 所有交叉引用正确
- [ ] 参考文献完整
- [ ] AI使用报告已包含（如使用AI）
- [ ] 总页数不超过25页（不包括AI使用报告）
- [ ] PDF文件可以正常打开和打印

## 技术支持

如果遇到问题：
1. 检查LaTeX日志文件（.log）
2. 确保所有必需的LaTeX包已安装
3. 参考 `LATEX_GUIDE.md` 获取详细说明

---

**祝提交顺利！**
