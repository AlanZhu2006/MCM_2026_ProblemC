# Overleaf上传指南

## 📦 ZIP文件已创建

已成功创建 `latex_overleaf.zip` 文件（约2.01 MB），包含所有必需的LaTeX文件和图表。

## 🚀 快速上传步骤

### 步骤1: 登录Overleaf
1. 访问 https://www.overleaf.com
2. 登录你的账户（如果没有账户，先注册）

### 步骤2: 上传ZIP文件
1. 点击左上角的 **"New Project"** 按钮
2. 选择 **"Upload Project"**
3. 点击 **"Select a .zip file"** 或拖拽文件
4. 选择项目根目录下的 **`latex_overleaf.zip`** 文件
5. 点击 **"Upload"**

### 步骤3: 等待解压
- Overleaf会自动解压ZIP文件
- 解压后会显示所有文件
- 确保文件结构正确：
  ```
  main.tex
  sections/
    ├── summary_sheet.tex
    ├── introduction.tex
    ├── ...
  visualizations/
    ├── stage2_fan_vote_estimation.png
    ├── ...
  ```

### 步骤4: 设置主文件
1. 点击左侧菜单中的 **"Menu"**（三条横线图标）
2. 选择 **"Main document"**
3. 确保 **`main.tex`** 被选中
4. 点击 **"OK"**

### 步骤5: 编译文档
1. 点击右上角的 **"Recompile"** 按钮
2. 等待编译完成（可能需要几秒钟）
3. 检查是否有错误

## ✅ 检查清单

上传后请检查：
- [ ] ZIP文件已成功上传
- [ ] 所有文件都已解压
- [ ] `main.tex` 存在
- [ ] `sections/` 文件夹包含11个 `.tex` 文件
- [ ] `visualizations/` 文件夹包含5个 `.png` 文件
- [ ] `main.tex` 被设置为主文件
- [ ] 编译成功，无错误
- [ ] PDF预览正常显示
- [ ] 所有图表正常显示
- [ ] 目录正确生成

## ⚠️ 常见问题

### Q: 上传失败？
A: 
- 检查ZIP文件大小（应约2.01 MB）
- 确保网络连接正常
- 尝试使用其他浏览器
- 检查Overleaf账户是否有足够的存储空间

### Q: 编译失败？
A:
- 检查所有文件是否都已上传
- 查看编译日志了解错误详情
- 确保 `main.tex` 被设置为主文件
- 检查文件路径是否正确

### Q: 图表不显示？
A:
- 确保 `visualizations/` 文件夹已上传
- 检查图片文件名是否正确
- 确保图片格式为PNG
- 检查图片路径：`visualizations/xxx.png`

### Q: 中文显示问题？
A:
- 在Overleaf设置中切换到 **XeLaTeX** 编译器
- 或使用 **LuaLaTeX** 编译器
- 点击 "Menu" → "Compiler" → 选择 "XeLaTeX"

### Q: 团队编号需要修改？
A:
- 在Overleaf中直接编辑 `main.tex` 文件
- 找到第35行：`\author{Team Control Number: [Your Team Number]}`
- 修改为你的实际团队编号
- 同时修改 `sections/summary_sheet.tex` 第14行

## 📝 文件内容

ZIP文件包含：
- **主文件**: `main.tex` (1个)
- **章节文件**: `sections/*.tex` (11个)
- **图表文件**: `visualizations/*.png` (5个)
- **说明文件**: `README_OVERLEAF.md` (1个)

总计：18个文件，约2.01 MB

## 🎯 下一步

1. ✅ ZIP文件已创建：`latex_overleaf.zip`
2. ⏳ 上传到Overleaf
3. ⏳ 设置主文件
4. ⏳ 编译文档
5. ⏳ 检查结果
6. ⏳ 修改团队编号（如需要）
7. ⏳ 最终检查并提交

---

**完成！现在可以将 `latex_overleaf.zip` 上传到Overleaf了！**
