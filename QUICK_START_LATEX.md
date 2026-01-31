# LaTeX报告快速开始指南

## 🚀 3步完成

### 步骤1: 修改团队编号

在以下两个文件中将 `[Your Team Number]` 替换为你的实际团队编号：

1. **`main.tex`** (第35行)
2. **`sections/summary_sheet.tex`** (第12行)

### 步骤2: 编译LaTeX

**Windows**:
```bash
compile_latex.bat
```

**Linux/Mac**:
```bash
pdflatex main.tex
pdflatex main.tex
```

**重要**: 需要编译**两次**以生成正确的目录和交叉引用。

### 步骤3: 检查PDF

打开生成的 `main.pdf`，检查：
- ✅ 摘要表在第1页
- ✅ 目录正确
- ✅ 所有图表显示正常
- ✅ 总页数不超过25页（不包括AI使用报告）

## 📁 文件结构

```
.
├── main.tex                    # 主文件（编译这个）
├── sections/                   # 所有章节文件
│   ├── summary_sheet.tex       # 摘要表
│   ├── introduction.tex
│   ├── stage1_preprocessing.tex
│   ├── stage2_fan_vote_estimation.tex
│   ├── stage3_voting_comparison.tex
│   ├── stage4_factor_impact.tex
│   ├── stage5_new_system.tex
│   ├── conclusions.tex
│   ├── memo_to_producers.tex   # 备忘录
│   ├── references.tex
│   └── ai_use_report.tex
├── visualizations/            # 图表目录
│   ├── stage2_fan_vote_estimation.png
│   ├── stage3_voting_comparison.png
│   ├── stage4_factor_impact.png
│   ├── stage5_ml_system.png
│   └── overall_summary.png
└── compile_latex.bat          # 编译脚本
```

## ✅ 报告包含内容

1. **摘要表** (1页) - 问题概述、方法、发现、结论
2. **目录** - 自动生成
3. **引言** - 问题背景和研究目标
4. **Stage 1** - 数据预处理
5. **Stage 2** - 粉丝投票估计（90%准确率）
6. **Stage 3** - 投票方法比较（排名法60.5% vs 百分比法97.0%）
7. **Stage 4** - 影响因素分析
8. **Stage 5** - 新投票系统提案（97.99%准确率）
9. **结论** - 主要发现和建议
10. **备忘录** (1-2页) - 给制作人的建议
11. **参考文献** - 所有引用
12. **AI使用报告** - 如使用AI

## ⚠️ 常见问题

### Q: 编译失败？
A: 
- 检查是否安装了LaTeX（如TeX Live, MiKTeX）
- 检查所有文件是否在正确位置
- 查看 `.log` 文件了解错误详情

### Q: 图表不显示？
A:
- 确保 `visualizations/` 目录存在
- 确保所有PNG文件都在该目录
- 检查路径是否正确

### Q: 页数太多？
A:
- AI使用报告不计入25页限制
- 可以调整图表大小
- 可以精简部分文字

## 📝 提交前检查

- [ ] 团队编号已修改
- [ ] 编译成功，无错误
- [ ] PDF可以正常打开
- [ ] 所有内容完整
- [ ] 总页数≤25页（不包括AI使用报告）

---

**完成！现在可以提交 `main.pdf` 了！**
