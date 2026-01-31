"""
运行所有优化脚本
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """运行脚本"""
    print("\n" + "=" * 70)
    print(f"运行: {description}")
    print("=" * 70)
    
    script_path = Path('scripts') / script_name
    
    if not script_path.exists():
        print(f"警告: {script_path} 不存在，跳过")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print("[OK] 成功完成")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("[WARNING] 执行完成但有警告")
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("运行所有优化脚本")
    print("=" * 70)
    
    optimizations = [
        ('quality_checker.py', '质量检查'),
        ('enhance_uncertainty_analysis.py', '增强不确定性分析'),
        ('sensitivity_analysis.py', '敏感性分析'),
        ('enhance_controversial_cases.py', '争议案例深度分析'),
        ('enhance_math_formulas.py', '数学公式增强'),
    ]
    
    results = {}
    
    for script, description in optimizations:
        success = run_script(script, description)
        results[description] = success
    
    # 总结
    print("\n" + "=" * 70)
    print("优化执行总结")
    print("=" * 70)
    
    for description, success in results.items():
        status = "[OK] 成功" if success else "[WARNING] 需要检查"
        print(f"{description}: {status}")
    
    print("\n" + "=" * 70)
    print("所有优化脚本执行完成！")
    print("=" * 70)
    print("\n注意: 某些脚本可能需要pandas等库，如果失败请安装依赖:")
    print("  pip install pandas numpy matplotlib seaborn")

if __name__ == '__main__':
    main()
