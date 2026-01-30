"""
数据加载模块
用于加载2026_MCM_Problem_C_Data.csv数据文件
"""

import sys
import os
import pandas as pd


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    加载DWTS数据文件
    
    Parameters:
    -----------
    path : str, optional
        数据文件的完整路径。如果为None，将在常见位置搜索文件。
    
    Returns:
    --------
    pd.DataFrame
        加载的数据框
    
    Raises:
    -------
    FileNotFoundError
        如果找不到数据文件
    """
    # 尝试多个常见位置
    candidates = []
    
    if path:
        candidates.append(path)
    
    # 添加其他可能的路径
    current_dir = os.getcwd()
    candidates.extend([
        os.path.join(current_dir, "2026_MCM_Problem_C_Data.csv"),
        os.path.join(current_dir, "..", "2026_MCM_Problem_C_Data.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026_MCM_Problem_C_Data.csv"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "2026_MCM_Problem_C_Data.csv"
        ),
    ])
    
    # 尝试每个候选路径
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding='utf-8')
                print(f"✓ 成功从以下路径加载数据: {p}")
                return df
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                try:
                    df = pd.read_csv(p, encoding='latin-1')
                    print(f"✓ 成功从以下路径加载数据 (使用latin-1编码): {p}")
                    return df
                except Exception as e:
                    print(f"⚠️  尝试加载 {p} 时出错: {e}")
                    continue
            except Exception as e:
                print(f"⚠️  尝试加载 {p} 时出错: {e}")
                continue
    
    # 如果所有路径都失败，抛出错误
    error_msg = (
        "无法找到 2026_MCM_Problem_C_Data.csv 文件。\n"
        "已尝试以下位置:\n" + "\n".join(f"  - {p}" for p in candidates)
    )
    raise FileNotFoundError(error_msg)


if __name__ == "__main__":
    # 测试加载功能
    try:
        df = load_data()
        print(f"\n数据加载成功!")
        print(f"数据形状: {df.shape}")
        print(f"列数: {len(df.columns)}")
        print(f"\n前5列: {list(df.columns[:5])}")
        print(f"\n前3行:")
        print(df.head(3))
    except FileNotFoundError as e:
        print(f"错误: {e}")
