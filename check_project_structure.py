"""
项目结构检查脚本
用于检查项目文件是否完整
"""

import os
import sys

def check_project_structure():
    """检查项目结构"""
    print("=" * 70)
    print("项目结构检查")
    print("=" * 70)
    print()
    
    # 项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"项目根目录: {project_root}")
    print()
    
    # 必需的文件
    required_files = {
        '数据文件': '2026_MCM_Problem_C_Data.csv',
        '问题PDF': '2026_MCM_Problem_C.pdf',
        '数据加载模块': 'loader.py',
        '预处理模块': 'preprocess_dwts.py',
        'README': 'README.md',
    }
    
    # 可选的文件
    optional_files = {
        '运行脚本': 'scripts/run_stage1_preprocessing.py',
        '使用指南': 'STAGE1_GUIDE.md',
        'Notebook模板': 'notebooks/C_problemC_template.ipynb',
    }
    
    print("必需文件检查:")
    print("-" * 70)
    all_required_exist = True
    for name, filepath in required_files.items():
        full_path = os.path.join(project_root, filepath)
        exists = os.path.exists(full_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {name:20s} : {filepath}")
        if not exists:
            all_required_exist = False
    
    print()
    print("可选文件检查:")
    print("-" * 70)
    for name, filepath in optional_files.items():
        full_path = os.path.join(project_root, filepath)
        exists = os.path.exists(full_path)
        status = "[OK]" if exists else "[OPTIONAL]"
        print(f"{status} {name:20s} : {filepath}")
    
    print()
    print("=" * 70)
    
    if all_required_exist:
        print("[SUCCESS] 所有必需文件都存在！")
        print()
        print("可以运行以下命令开始阶段1预处理:")
        print("  python scripts/run_stage1_preprocessing.py")
    else:
        print("[ERROR] 缺少一些必需文件，请检查！")
    
    print("=" * 70)
    
    # 检查Python模块导入
    print()
    print("Python模块导入测试:")
    print("-" * 70)
    
    try:
        sys.path.insert(0, project_root)
        from loader import load_data
        print("[OK] loader模块导入成功")
        
        from preprocess_dwts import DWTSDataPreprocessor
        print("[OK] preprocess_dwts模块导入成功")
        
        print()
        print("[SUCCESS] 所有模块导入正常！")
        
    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {e}")
        print("  请检查文件是否存在且语法正确")
    except Exception as e:
        print(f"[WARNING] 导入时出现其他错误: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    check_project_structure()
