# 粉丝投票估计模型：准确率改进建议

当前基线准确率约 **51.84%**（155/299 周正确）。以下按优先级列出可落地的改进方向。

---

## 一、高优先级（易实施、收益大）

### 1. 用全局优化替代 SLSQP，避免局部最优

**现状**：`fan_vote_estimator.py` 用 `minimize(..., method='SLSQP')`，容易陷入局部最优，导致约束满足但预测错。

**改法**：在 SLSQP 失败或结果不可信时，用 `differential_evolution` 做一次全局搜索，或对同一周做多次不同初值 SLSQP，选「约束满足且目标函数更优」的解。

```python
# 在 estimate_fan_votes_rank_method / estimate_fan_votes_percent_method 中
# 当 result.success 为 False 时，不要直接走启发式，先尝试：

from scipy.optimize import differential_evolution

# 包装目标函数与约束（用 penalty 把约束写进目标）
def objective_with_penalty(x):
    obj = np.sum((x - expected_fan_ranks) ** 2)
    penalty = 1e6 * max(0, -constraint_eliminated(x))  # 约束违反时加大惩罚
    return obj + penalty

# 全局搜索（仅当 SLSQP 失败时调用，避免太慢）
bounds = [(1, n_contestants)] * n_contestants
result_global = differential_evolution(
    objective_with_penalty, bounds,
    maxiter=300, popsize=15, seed=42,
    polish=True, atol=1e-6
)
if constraint_eliminated(result_global.x) >= 0:  # 约束满足
    fan_ranks = result_global.x
else:
    fan_ranks = self._heuristic_fan_ranks(...)  # 再回退启发式
```

百分比法同理，把 `constraint_sum` 和 `constraint_eliminated` 用 penalty 写进目标后再做 `differential_evolution`。

---

### 2. 多初值 + 选「最符合淘汰结果」的解

**现状**：排名法里 `x0 = expected_fan_ranks + np.random.normal(0, 0.5, n_contestants)` 只跑一次，初值敏感时容易选到差的局部解。

**改法**：对同一 (season, week) 用 5～10 个不同随机种子得到多个 `x0`，每个跑一次 SLSQP；只保留「约束满足」的解，若有多个则选目标函数最小的；若全部不满足约束，再走全局优化或启发式。

---

### 3. 强化约束：保证被淘汰者严格最差

**现状**：约束是 `eliminated_combined - np.max(combined_ranks) >= 0`，即「被淘汰者 ≥ 最大值」。若出现并列第一差，验证时按「综合排名最高者」判定淘汰，可能和真实规则不一致。

**改法**：  
- 在目标里加一项「让被淘汰者比第二名差一截」，例如加 `margin`：  
  `margin = min(0, (eliminated_combined - np.max(combined_ranks[非淘汰]))`，  
  目标 += `1e4 * margin^2`，这样优化会倾向拉开差距。  
- 或在优化后做后处理：若发现「被淘汰者综合得分不是严格最差」，在该周微调粉丝排名/百分比（只动被淘汰者与最差一名），使被淘汰者严格最差，再重新算综合得分用于验证。

---

### 4. 按季/按规则细分权重与预期

**现状**：`weights = {'judge': 0.5, 'history': 0.3, 'partner': 0.2}` 和 `expected_fan_ranks`/`expected_fan_percents` 全季统一。

**改法**：  
- 按 `determine_voting_method(season)` 分支（rank / percent）用不同权重。  
- 若有历史验证结果，可按季做网格搜索或简单交叉验证，选该季准确率更高的权重。  
- 对第 28 季及以后，若规则中有「评委从最低两名中选一人淘汰」，可在特征或约束中显式区分「本周最低两人」，再在约束里只要求「被淘汰者在最低两人中且综合最差」。

---

## 二、中优先级（特征与数据）

### 5. 丰富特征，尤其是「与淘汰相关」的

**现状**：已有评委排名、历史、舞伴等，但和「谁被淘汰」的直接信号不够。

**改法**：  
- **本周危险度**：当前评委排名最后 1～2 名、或连续多周排名靠后的选手，给一个「危险度」特征。  
- **历史淘汰模式**：同一季前几周「评委差但没被淘汰」的次数，或「评委好却被淘汰」的次数（若数据能推断）。  
- **当周人数、周次**：`contestants_remaining`、`week` 已部分有，可显式加入 `week / 预计总周数` 等，因为后期观众习惯会变。  
- 在 Advanced/Enhanced 里已有的趋势、波动、行业、年龄等继续保留并统一用上，避免同一套逻辑在不同入口特征不一致。

---

### 6. 识别并特殊处理「多淘汰」「评委选择」周

**现状**：`get_eliminated_contestant` 在多人同时消失时只取第一个，可能和实际规则不符。

**改法**：  
- 在预处理或 `get_eliminated_contestant` 里检测「本周消失人数 > 1」的周，打标为 `multi_elimination`。  
- 对这些周：若规则是「取综合最差的一位」，则约束改为「被淘汰者集合中至少一人综合最差」；若规则是「评委二选一」，则约束改为「综合最差两人中的某一人」，再在验证时用同一规则。  
- 避免把「双淘汰」强行当成「单淘汰」来拟合，否则会拉低准确率。

---

### 7. 启发式改进：保证约束满足且更平滑

**现状**：`_heuristic_fan_ranks` 里对非淘汰者用 `judge_ranks[i] * 0.8 + np.random.uniform(0.5, 2)`，随机性可能导致综合排名违反「被淘汰者最差」。

**改法**：  
- 启发式里先设 `fan_ranks[eliminated_idx] = n_contestants`，再给其余人赋值时，保证 `judge_ranks[i] + fan_ranks[i]` 始终严格小于 `judge_ranks[eliminated_idx] + n_contestants`（必要时整体缩放或微调）。  
- 去掉或固定随机种子，使同一 (season, week) 多次运行结果一致，便于复现和调试。

---

## 三、低优先级（模型与评估）

### 8. 用「淘汰预测」做目标训练 ML 模型

**现状**：Advanced/Enhanced 用回归预测排名或百分比，再优化微调；最终验证却是「是否猜对被淘汰者」。

**改法**：  
- 增加一个分类目标：对每一周，用特征训练一个「谁会被淘汰」的分类器（如 XGBoost/LightGBM），输出概率。  
- 将该概率作为「预期受欢迎度」的修正项参与 expected_fan_ranks/expected_fan_percents，或用于筛选「多初值」中优先尝试的初值。  
- 或直接做两阶段：先分类预测淘汰者，再仅对该周用优化反推粉丝投票，使该周的优化约束与分类一致。

---

### 9. 后处理一致性检查

**现状**：优化结果直接用于验证，没有再做一次「约束是否满足」的检查。

**改法**：  
- 在 `estimate_fan_votes_rank_method` / `estimate_fan_votes_percent_method` 返回前，用当前 `fan_ranks`/`fan_percents` 算一遍综合得分，检查被淘汰者是否的确最差。  
- 若不满足，则在该周标记 `optimization_success=False` 并尝试一轮全局优化或更强的启发式，再写入 `estimates_df`。  
- 在验证阶段可统计 `optimization_success=False` 的周的比例和这些周的准确率，便于定位问题。

---

### 10. 按季/按方法分析错误案例

**改法**：  
- 在 `validate_estimates` 的 `details` 里保留 `voting_method`、`season`、`week`。  
- 跑完验证后，按 `(voting_method, season)` 聚合准确率，找出「准确率明显低于平均」的季或方法。  
- 针对这些季：检查数据是否有多淘汰、缺失、异常；检查权重和约束是否与该季规则一致；再考虑单独调参或单独规则。

---

## 四、实施顺序建议

| 步骤 | 内容 | 预期效果 |
|------|------|----------|
| 1 | 多初值 SLSQP + 选满足约束且目标更优的解 | +2～5% |
| 2 | SLSQP 失败时用 differential_evolution 替代直接启发式 | +2～4% |
| 3 | 启发式保证约束满足 + 后处理检查 | +1～3% |
| 4 | 按季/按规则调权重与预期、加强约束 margin | +1～3% |
| 5 | 多淘汰/评委选择周的特殊处理 | +1～2% |
| 6 | 特征工程（危险度、历史淘汰模式等） | +2～5% |

整体有望在现有约 52% 基础上提升到 **60～70%**；若再结合「淘汰分类器 + 优化」的混合方案，有机会再往上走一点。实施时建议每改一项就跑一遍 `validate_estimates`，看准确率与 `optimization_success` 比例的变化，便于定位有效改动。
