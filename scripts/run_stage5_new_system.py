"""
阶段5：新投票系统设计 - 运行脚本
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_voting_system_designer import FairnessAdjustedVotingSystem
from loader import load_data
from preprocess_dwts import DWTSDataPreprocessor


def convert_to_native(obj):
    """将numpy/pandas类型转换为Python原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                new_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
            else:
                new_key = str(key) if not isinstance(key, (str, int, float, bool)) else key
            result[new_key] = convert_to_native(value)
        return result
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def generate_theoretical_analysis() -> str:
    """生成理论分析报告"""
    analysis = """
# 新投票系统理论分析

## 一、系统设计理念

### 1.1 公平性调整的投票系统（Fairness-Adjusted Voting System）

基于阶段4的影响因素分析，我们发现当前投票系统存在以下不公平因素：

1. **年龄偏见**：年龄越大，评分和投票都越低（相关系数：-0.24 和 -0.26）
2. **专业舞者影响**：某些专业舞者（如Derek Hough）明显提升选手表现
3. **地区差异**：某些地区（如South Korea）表现明显更好
4. **行业差异**：某些行业（如Musician）表现更好

### 1.2 核心改进策略

#### 策略1：影响因素标准化
- **年龄标准化**：根据年龄调整评分，使不同年龄段的选手在同等水平下获得相似评分
- **专业舞者平衡**：根据专业舞者的历史表现，对评分进行调整
- **行业/地区平衡**：对行业和地区因素进行适度调整

#### 策略2：动态权重调整
- 根据评委和粉丝的偏好差异，动态调整权重比例
- 当粉丝投票更分散时，适当降低粉丝权重
- 当评委评分更分散时，适当降低评委权重

#### 策略3：综合评分方法
- 使用调整后的评分计算综合得分
- 保持与原始系统相同的淘汰机制（排名法/百分比法）

## 二、数学公式

### 2.1 年龄调整

对于选手 i，年龄调整因子：

$$A_{judge,i} = 1 - \\rho_{age,judge} \\cdot z_{age,i} \\cdot \\alpha$$
$$A_{fan,i} = 1 - \\rho_{age,fan} \\cdot z_{age,i} \\cdot \\alpha$$

其中：
- $\\rho_{age,judge} = -0.24$（年龄与评委评分的相关系数）
- $\\rho_{age,fan} = -0.26$（年龄与粉丝投票的相关系数）
- $z_{age,i} = \\frac{age_i - \\bar{age}}{\\sigma_{age}}$（标准化年龄）
- $\\alpha = 0.1$（调整强度）

### 2.2 专业舞者调整

对于专业舞者 j，调整因子：

$$D_j = 0.5 + 0.5 \\cdot \\frac{1}{placement_j + 0.5} \\cdot \\frac{1}{2}$$

其中 $placement_j$ 是该专业舞者的平均排名。

### 2.3 动态权重

评委和粉丝的权重根据变异系数动态调整：

$$CV_{judge} = \\frac{\\sigma_{judge}}{\\mu_{judge}}$$
$$CV_{fan} = \\frac{\\sigma_{fan}}{\\mu_{fan}}$$

$$w_{judge} = w_{base} \\cdot (1 + (CV_{fan} - CV_{judge}) \\cdot 0.2)$$
$$w_{fan} = w_{base} \\cdot (1 + (CV_{judge} - CV_{fan}) \\cdot 0.2)$$

归一化：$w_{judge} + w_{fan} = 1$

### 2.4 综合得分

**排名法**（适用于第1-2季，第28-34季）：

$$R_{combined,i} = w_{judge} \\cdot R_{judge,i} + w_{fan} \\cdot R_{fan,i}$$

其中 $R_{judge,i}$ 和 $R_{fan,i}$ 是基于调整后评分的排名。

**百分比法**（适用于第3-27季）：

$$P_{combined,i} = w_{judge} \\cdot P_{judge,i} + w_{fan} \\cdot P_{fan,i}$$

其中 $P_{judge,i}$ 和 $P_{fan,i}$ 是基于调整后评分的百分比。

## 三、系统优势

### 3.1 公平性提升

1. **减少年龄偏见**：通过年龄标准化，使不同年龄段的选手在同等水平下获得相似评分
2. **平衡专业舞者影响**：减少因专业舞者能力差异导致的不公平
3. **减少地区/行业偏见**：适度调整地区和行业因素的影响

### 3.2 准确性提升

1. **动态权重**：根据实际情况动态调整评委和粉丝的权重，提高预测准确性
2. **综合考虑**：同时考虑多个影响因素，提供更全面的评估

### 3.3 可解释性

1. **透明调整**：所有调整因子都可以量化解释
2. **可追溯性**：可以追踪每个调整因子的影响

## 四、参数设置

- **调整强度** ($\\alpha$): 0.3（可调整，范围0-1）
- **基础权重**: 评委和粉丝各50%
- **动态权重调整系数**: 0.2（可调整）

## 五、预期效果

1. **公平性**：减少因年龄、专业舞者、地区、行业等因素导致的不公平
2. **准确性**：通过动态权重和综合调整，提高淘汰预测的准确性
3. **稳定性**：通过标准化处理，提高系统的稳定性

"""
    return analysis


def main():
    """主函数"""
    print("=" * 70)
    print("阶段5：新投票系统设计")
    print("=" * 70)
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据...")
    try:
        raw_df = load_data()
        print(f"✓ 已加载原始数据: {len(raw_df)} 行")
        
        # 加载预处理后的数据
        processed_df = pd.read_csv('2026_MCM_Problem_C_Data_processed.csv')
        print(f"✓ 已加载预处理数据: {len(processed_df)} 行")
        
        # 加载阶段2的估计结果
        estimates_df = pd.read_csv('fan_vote_estimates.csv')
        print(f"✓ 已加载粉丝投票估计: {len(estimates_df)} 行")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return
    
    # 步骤2: 初始化新投票系统
    print("\n步骤2: 初始化新投票系统...")
    try:
        new_system = FairnessAdjustedVotingSystem(
            estimates_df=estimates_df,
            processed_df=processed_df,
            factor_analysis_path='factor_impact_analysis.json'
        )
        print("✓ 新投票系统初始化完成")
        print(f"  - 年龄调整: {'启用' if new_system.age_adjustment_enabled else '禁用'}")
        print(f"  - 专业舞者调整: {'启用' if new_system.pro_dancer_adjustment_enabled else '禁用'}")
        print(f"  - 行业调整: {'启用' if new_system.industry_adjustment_enabled else '禁用'}")
        print(f"  - 地区调整: {'启用' if new_system.region_adjustment_enabled else '禁用'}")
        print(f"  - 动态权重: {'启用' if new_system.dynamic_weight_enabled else '禁用'}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 应用新系统到所有周次
    print("\n步骤3: 应用新系统到所有周次...")
    try:
        new_system_results = new_system.apply_to_all_weeks()
        print(f"✓ 已处理 {len(new_system_results)} 条记录")
        print(f"  - 涉及 {new_system_results['season'].nunique()} 季")
        print(f"  - 涉及 {new_system_results.groupby(['season', 'week']).ngroups} 个周次")
        
        # 保存结果
        new_system_results.to_csv('new_voting_system_results.csv', index=False)
        print("✓ 已保存新系统结果到: new_voting_system_results.csv")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 比较新系统与原始系统
    print("\n步骤4: 比较新系统与原始系统...")
    try:
        comparison = new_system.compare_with_original_systems(new_system_results)
        comparison_native = convert_to_native(comparison)
        
        print("\n比较结果:")
        print(f"  - 总周次数: {comparison['total_weeks']}")
        print(f"  - 原始系统准确率: {comparison['original_system_accuracy']:.2%}")
        print(f"  - 新系统准确率: {comparison['new_system_accuracy']:.2%}")
        print(f"  - 准确率提升: {comparison['accuracy_improvement']:.2%}")
        print(f"  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})")
        
        # 保存比较结果
        with open('new_system_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_native, f, indent=2, ensure_ascii=False)
        print("✓ 已保存比较结果到: new_system_comparison.json")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 生成理论分析报告
    print("\n步骤5: 生成理论分析报告...")
    try:
        theoretical_analysis = generate_theoretical_analysis()
        
        with open('new_system_theoretical_analysis.md', 'w', encoding='utf-8') as f:
            f.write(theoretical_analysis)
        print("✓ 已保存理论分析到: new_system_theoretical_analysis.md")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤6: 生成综合报告
    print("\n步骤6: 生成综合报告...")
    try:
        report = f"""
阶段5：新投票系统设计 - 综合报告
======================================================================

一、系统概述
----------------------------------------------------------------------
新投票系统名称: 公平性调整的投票系统（Fairness-Adjusted Voting System）

核心改进:
1. 影响因素标准化（年龄、专业舞者、行业、地区）
2. 动态权重调整（根据评委和粉丝的分散程度）
3. 综合评分方法（保持与原始系统相同的淘汰机制）

二、系统性能
----------------------------------------------------------------------
总周次数: {comparison['total_weeks']}

原始系统:
  - 准确率: {comparison['original_system_accuracy']:.2%}
  - 正确预测数: {comparison['original_correct_count']}

新系统:
  - 准确率: {comparison['new_system_accuracy']:.2%}
  - 正确预测数: {comparison['new_correct_count']}
  - 准确率提升: {comparison['accuracy_improvement']:.2%}

系统差异:
  - 不同预测数: {comparison['different_predictions']} ({comparison['different_predictions_rate']:.2%})

三、系统优势
----------------------------------------------------------------------
1. 公平性提升
   - 减少年龄偏见（年龄标准化）
   - 平衡专业舞者影响
   - 减少地区/行业偏见

2. 准确性提升
   - 动态权重调整
   - 综合考虑多个影响因素

3. 可解释性
   - 所有调整因子可量化
   - 调整过程可追溯

四、详细分析
----------------------------------------------------------------------
详细的理论分析和数学公式请参考: new_system_theoretical_analysis.md

五、输出文件
----------------------------------------------------------------------
1. new_voting_system_results.csv - 新系统的详细结果
2. new_system_comparison.json - 与原始系统的比较结果
3. new_system_theoretical_analysis.md - 理论分析报告
4. stage5_new_system_report.txt - 本报告

======================================================================
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('stage5_new_system_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✓ 已保存综合报告到: stage5_new_system_report.txt")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("阶段5完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
