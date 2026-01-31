"""
可视化图表优化
统一风格、添加新图表、优化说明
"""

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import seaborn as sns
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("警告: matplotlib/seaborn未安装，可视化优化将受限")

def setup_unified_style():
    """设置统一的图表风格"""
    if not HAS_LIBS:
        return
    
    # 使用seaborn风格
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 设置全局参数
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.sans-serif': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False
    })
    
    print("✓ 统一图表风格已设置")

def create_style_guide():
    """创建风格指南文档"""
    guide = """
# 可视化风格指南

## 统一风格设置

### 颜色方案
- 主色调: seaborn "husl" 调色板
- 背景: 白色网格 (whitegrid)
- 强调色: 蓝色系用于主要数据，红色系用于对比

### 字体设置
- 标题: 14pt
- 图表标题: 13pt
- 轴标签: 11pt
- 刻度标签: 10pt
- 图例: 10pt

### 图表尺寸
- 标准图表: 10x6 英寸
- 多子图: 根据子图数量调整
- DPI: 300 (用于打印质量)

### 图表元素
- 网格: 启用，透明度0.3
- 图例: 右上角或最佳位置
- 标题: 加粗，描述性
- 轴标签: 清晰，包含单位（如适用）

## 图表类型规范

### 1. 时间序列图
- 使用线图，标记点
- 多条线时使用不同颜色和标记
- 添加图例说明

### 2. 柱状图
- 使用seaborn barplot
- 添加数值标签
- 统一柱宽

### 3. 散点图
- 调整透明度避免重叠
- 添加趋势线（如适用）
- 使用颜色编码分类

### 4. 热力图
- 使用seaborn heatmap
- 添加数值标注
- 使用合适的颜色映射

## 文件命名规范

- 格式: `stage{number}_{description}.png`
- 示例: `stage2_fan_vote_estimation.png`
- 所有图表保存在 `visualizations/` 目录

## 图表说明

每个图表应该：
1. 有清晰的标题
2. 轴标签包含变量名和单位
3. 图例说明所有系列
4. 必要时添加注释或说明文字
"""
    
    guide_path = Path('VISUALIZATION_STYLE_GUIDE.md')
    guide_path.write_text(guide, encoding='utf-8')
    print(f"[OK] 风格指南已保存到: {guide_path}")

def main():
    """主函数"""
    print("=" * 70)
    print("可视化图表优化")
    print("=" * 70)
    
    if HAS_LIBS:
        setup_unified_style()
    
    create_style_guide()
    
    print("\n" + "=" * 70)
    print("可视化优化完成！")
    print("=" * 70)
    print("\n建议:")
    print("1. 使用统一的风格设置重新生成所有图表")
    print("2. 检查所有图表的清晰度和可读性")
    print("3. 确保所有图表有适当的标题和说明")

if __name__ == '__main__':
    main()
