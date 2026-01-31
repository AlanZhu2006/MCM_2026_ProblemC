# LaTeX报告编译指南

## 文件结构

```
.
├── main.tex                    # 主LaTeX文件
├── sections/                   # 章节文件目录
│   ├── summary_sheet.tex      # 摘要表
│   ├── introduction.tex       # 引言
│   ├── stage1_preprocessing.tex
│   ├── stage2_fan_vote_estimation.tex
│   ├── stage3_voting_comparison.tex
│   ├── stage4_factor_impact.tex
│   ├── stage5_new_system.tex
│   ├── conclusions.tex
│   ├── memo_to_producers.tex  # 备忘录
│   ├── references.tex         # 参考文献
│   └── ai_use_report.tex      # AI使用报告
├── visualizations/            # 图表目录
│   ├── stage2_fan_vote_estimation.png
│   ├── stage3_voting_comparison.png
│   ├── stage4_factor_impact.png
│   ├── stage5_ml_system.png
│   └── overall_summary.png
└── compile_latex.bat          # Windows编译脚本
```

## 编译方法

### Windows

```bash
# 方法1: 使用批处理文件
compile_latex.bat

# 方法2: 手动编译
pdflatex main.tex
pdflatex main.tex
```

### Linux/Mac

```bash
pdflatex main.tex
pdflatex main.tex
```

**注意**: 需要编译两次以生成正确的目录和交叉引用。

## 插入图表

在相应的章节文件中，使用以下命令插入图表：

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{visualizations/stage2_fan_vote_estimation.png}
\caption{Stage 2: 粉丝投票估计可视化}
\label{fig:stage2}
\end{figure}
```

## 自定义设置

### 修改团队编号

在 `main.tex` 中修改：
```latex
\author{Team Control Number: [Your Team Number]}
```

在 `sections/summary_sheet.tex` 中修改：
```latex
\textbf{Team Control Number} & [Your Team Number] \\
```

### 调整页面布局

在 `main.tex` 中修改 `\geometry` 设置：
```latex
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
```

### 添加更多图表

1. 将图表文件放入 `visualizations/` 目录
2. 在相应章节文件中使用 `\includegraphics` 命令
3. 重新编译

## 常见问题

### 问题1: 找不到图表文件

**解决**: 确保图表文件在 `visualizations/` 目录中，路径正确。

### 问题2: 中文显示问题

**解决**: 确保使用UTF-8编码，XeLaTeX或LuaLaTeX编译（如果需要）。

### 问题3: 目录不显示

**解决**: 需要编译两次LaTeX文件。

### 问题4: 参考文献格式

**解决**: 当前使用 `thebibliography` 环境，可以改为BibTeX如果需要。

## 检查清单

在提交前，请检查：

- [ ] 团队编号已填写
- [ ] 所有图表已插入
- [ ] 目录正确生成
- [ ] 页码正确
- [ ] 所有交叉引用正确
- [ ] 参考文献完整
- [ ] AI使用报告已包含（如使用AI）
- [ ] 总页数不超过25页（不包括AI使用报告）

## 最终输出

编译后将生成 `main.pdf`，这就是你的最终提交文件。

---

**注意**: 确保所有必需的数据和图表文件都在正确的位置，否则编译会失败。
