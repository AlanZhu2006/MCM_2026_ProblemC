# Overleaf导入说明

## 文件结构

```
latex_overleaf/
├── main.tex                    # 主LaTeX文件（编译这个）
├── sections/                    # 所有章节文件
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
└── visualizations/            # 所有图表
    ├── stage2_fan_vote_estimation.png
    ├── stage3_voting_comparison.png
    ├── stage4_factor_impact.png
    ├── stage5_ml_system.png
    └── overall_summary.png
```

## 导入Overleaf步骤

1. **登录Overleaf**
   - 访问 https://www.overleaf.com
   - 登录你的账户

2. **创建新项目**
   - 点击 "New Project" → "Upload Project"
   - 或者创建空白项目后上传文件

3. **上传文件**
   - 方法1: 上传整个 `latex_overleaf` 文件夹（推荐）
     - 在Overleaf中点击 "Upload" → 选择整个 `latex_overleaf` 文件夹
   - 方法2: 逐个上传文件
     - 先上传 `main.tex`
     - 然后上传 `sections/` 文件夹中的所有 `.tex` 文件
     - 最后上传 `visualizations/` 文件夹中的所有 `.png` 文件

4. **设置主文件**
   - 在Overleaf中，确保 `main.tex` 被设置为 "Main document"
   - 点击左侧菜单中的 "Menu" → "Main document" → 选择 `main.tex`

5. **编译**
   - 点击 "Recompile" 按钮
   - 等待编译完成
   - 检查是否有错误

## 注意事项

1. **团队编号**: 已在 `main.tex` 和 `sections/summary_sheet.tex` 中设置为 2603215
   - 如需修改，请在Overleaf中直接编辑

2. **图表路径**: 所有图表路径使用相对路径 `visualizations/xxx.png`
   - 确保 `visualizations/` 文件夹在Overleaf项目中

3. **编译设置**: Overleaf默认使用pdfLaTeX
   - 如果中文显示有问题，可以在设置中切换到XeLaTeX

4. **文件大小**: 图表文件可能较大
   - Overleaf免费账户有文件大小限制
   - 如果超过限制，可以压缩图片或使用付费账户

## 检查清单

上传后检查：
- [ ] main.tex 已上传
- [ ] sections/ 文件夹中的所有 .tex 文件已上传
- [ ] visualizations/ 文件夹中的所有 .png 文件已上传
- [ ] main.tex 被设置为主文件
- [ ] 编译成功，无错误
- [ ] 所有图表正常显示
- [ ] 目录正确生成
- [ ] 总页数不超过25页（不包括AI使用报告）

## 常见问题

### Q: 编译失败？
A: 
- 检查所有文件是否都已上传
- 检查文件路径是否正确
- 查看编译日志了解错误详情

### Q: 图表不显示？
A:
- 确保 visualizations/ 文件夹已上传
- 检查图片文件名是否正确
- 确保图片格式为PNG

### Q: 中文显示问题？
A:
- 在Overleaf设置中切换到XeLaTeX编译器
- 或使用LuaLaTeX编译器

---

**完成！现在可以将 latex_overleaf 文件夹上传到Overleaf了！**
