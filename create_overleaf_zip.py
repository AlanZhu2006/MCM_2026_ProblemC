"""
创建Overleaf导入的ZIP文件
"""

import os
import shutil
from pathlib import Path
import zipfile

def create_overleaf_zip():
    """创建Overleaf ZIP文件"""
    print("=" * 70)
    print("创建Overleaf ZIP文件")
    print("=" * 70)
    
    overleaf_dir = Path('latex_overleaf')
    zip_path = Path('latex_overleaf.zip')
    
    if not overleaf_dir.exists():
        print(f"错误: {overleaf_dir} 文件夹不存在")
        print("请先运行 prepare_overleaf.py")
        return
    
    # 删除旧的ZIP文件（如果存在）
    if zip_path.exists():
        zip_path.unlink()
        print(f"删除旧的ZIP文件: {zip_path}")
    
    # 创建ZIP文件
    print(f"\n正在创建ZIP文件: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加主文件
        if (overleaf_dir / 'main.tex').exists():
            zipf.write(overleaf_dir / 'main.tex', 'main.tex')
            print("  main.tex")
        
        # 添加README（可选）
        if (overleaf_dir / 'README_OVERLEAF.md').exists():
            zipf.write(overleaf_dir / 'README_OVERLEAF.md', 'README_OVERLEAF.md')
            print("  README_OVERLEAF.md")
        
        # 添加所有章节文件
        sections_dir = overleaf_dir / 'sections'
        if sections_dir.exists():
            for tex_file in sections_dir.glob('*.tex'):
                arcname = f'sections/{tex_file.name}'
                zipf.write(tex_file, arcname)
                print(f"  {arcname}")
        
        # 添加所有图表文件
        visualizations_dir = overleaf_dir / 'visualizations'
        if visualizations_dir.exists():
            for img_file in visualizations_dir.glob('*.png'):
                arcname = f'visualizations/{img_file.name}'
                zipf.write(img_file, arcname)
                print(f"  {arcname}")
    
    # 获取文件大小
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("ZIP文件创建完成！")
    print("=" * 70)
    print(f"\n文件位置: {zip_path.absolute()}")
    print(f"文件大小: {zip_size_mb:.2f} MB")
    print("\n下一步:")
    print("1. 登录 Overleaf: https://www.overleaf.com")
    print("2. 点击 'New Project' → 'Upload Project'")
    print("3. 选择 'latex_overleaf.zip' 文件上传")
    print("4. Overleaf会自动解压并创建项目")
    print("5. 确保 main.tex 被设置为主文件")
    print("6. 点击 'Recompile' 编译文档")

if __name__ == '__main__':
    create_overleaf_zip()
