"""
准备Overleaf导入文件夹
将所有LaTeX文件和图表复制到latex_overleaf目录
"""

import os
import shutil
from pathlib import Path

def prepare_overleaf_folder():
    """准备Overleaf文件夹"""
    print("=" * 70)
    print("准备Overleaf导入文件夹")
    print("=" * 70)
    
    # 创建目录
    overleaf_dir = Path('latex_overleaf')
    sections_dir = overleaf_dir / 'sections'
    visualizations_dir = overleaf_dir / 'visualizations'
    
    sections_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n创建目录: {overleaf_dir}")
    print(f"创建目录: {sections_dir}")
    print(f"创建目录: {visualizations_dir}")
    
    # 复制主文件
    print("\n复制主文件...")
    if Path('main.tex').exists():
        shutil.copy('main.tex', overleaf_dir / 'main.tex')
        print("main.tex")
    else:
        print("⚠️  main.tex 不存在，使用已创建的版本")
    
    # 复制所有章节文件
    print("\n复制章节文件...")
    source_sections = Path('sections')
    if source_sections.exists():
        for tex_file in source_sections.glob('*.tex'):
            shutil.copy(tex_file, sections_dir / tex_file.name)
            print(f"  {tex_file.name}")
    else:
        print("  sections目录不存在")
    
    # 复制所有图表文件
    print("\n复制图表文件...")
    source_viz = Path('visualizations')
    if source_viz.exists():
        for img_file in source_viz.glob('*.png'):
            shutil.copy(img_file, visualizations_dir / img_file.name)
            print(f"  {img_file.name}")
    else:
        print("  visualizations目录不存在")
    
    # 创建README
    readme_content = """# Overleaf导入说明

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
"""
    
    readme_path = overleaf_dir / 'README_OVERLEAF.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\nREADME_OVERLEAF.md 已创建")
    
    # 统计文件
    print("\n" + "=" * 70)
    print("文件统计:")
    print("=" * 70)
    
    tex_files = list(sections_dir.glob('*.tex'))
    png_files = list(visualizations_dir.glob('*.png'))
    
    print(f"\n章节文件: {len(tex_files)} 个")
    for f in tex_files:
        print(f"  - {f.name}")
    
    print(f"\n图表文件: {len(png_files)} 个")
    for f in png_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    if png_files:
        total_size = sum(f.stat().st_size for f in png_files) / (1024 * 1024)
        print(f"\n总图表大小: {total_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("Overleaf文件夹准备完成！")
    print(f"文件夹位置: {overleaf_dir.absolute()}")
    print("=" * 70)
    print("\n下一步:")
    print("1. 检查 latex_overleaf/ 文件夹")
    print("2. 将整个文件夹上传到Overleaf")
    print("3. 在Overleaf中设置 main.tex 为主文件")
    print("4. 编译并检查结果")

if __name__ == '__main__':
    prepare_overleaf_folder()
