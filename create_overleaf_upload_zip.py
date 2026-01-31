"""
创建Overleaf上传用的ZIP文件
包含所有必需的LaTeX文件和图片
"""
import os
import zipfile
from pathlib import Path

def create_overleaf_zip():
    """创建Overleaf上传用的ZIP文件"""
    
    # 输出文件名
    zip_filename = 'overleaf_upload.zip'
    
    # 需要包含的文件和文件夹
    files_to_include = {
        # 主文件
        'main.tex': 'main.tex',
        
        # 模板文件（提取到根目录）
        'MCM_Latex2026/mcmthesis.cls': 'mcmthesis.cls',
        
        # sections文件夹
        'sections/': 'sections/',
        
        # visualizations文件夹
        'visualizations/': 'visualizations/',
    }
    
    print("=" * 70)
    print("创建Overleaf上传ZIP文件")
    print("=" * 70)
    print()
    
    # 创建ZIP文件
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for source_path, zip_path in files_to_include.items():
            source = Path(source_path)
            
            if source.is_file():
                # 单个文件
                if source.exists():
                    zipf.write(source, zip_path)
                    print(f"[OK] 添加文件: {source_path} -> {zip_path}")
                else:
                    print(f"[WARN] 文件不存在: {source_path}")
            
            elif source.is_dir():
                # 文件夹
                if source.exists():
                    # 遍历文件夹中的所有文件
                    for file_path in source.rglob('*'):
                        if file_path.is_file():
                            # 计算在ZIP中的相对路径
                            rel_path = file_path.relative_to(source.parent)
                            zip_path_full = zip_path / rel_path.relative_to(source.name) if zip_path.endswith('/') else zip_path / rel_path
                            
                            zipf.write(file_path, zip_path_full)
                            print(f"[OK] 添加文件: {file_path} -> {zip_path_full}")
                else:
                    print(f"[WARN] 文件夹不存在: {source_path}")
    
    # 检查ZIP文件大小
    zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
    print()
    print(f"[OK] ZIP文件创建成功: {zip_filename}")
    print(f"  文件大小: {zip_size:.2f} MB")
    print()
    print("=" * 70)
    print("上传到Overleaf:")
    print("1. 登录 https://www.overleaf.com")
    print("2. 打开你的项目")
    print("3. 点击 'Upload' → 'Upload .zip file'")
    print(f"4. 选择: {zip_filename}")
    print("5. 等待解压完成")
    print("6. 设置 main.tex 为主文档")
    print("7. 点击 'Recompile'")
    print("=" * 70)
    
    return zip_filename

if __name__ == '__main__':
    create_overleaf_zip()
