# 粉丝投票估计模型改动说明

本文档陈述对 `fan_vote_estimator.py` 所做的修改，旨在提高淘汰预测准确率。

---

## 一、改动目标

- 减少因**局部最优**导致的错误预测（多初值 SLSQP + 全局优化回退）。
- 保证**约束严格满足**：被淘汰者综合得分必须严格最差（带 margin，并做后处理校验）。
- 启发式**可复现且满足约束**（固定种子、逻辑保证被淘汰者最差）。

---

## 二、新增模块级常量

在文件顶部、`class FanVoteEstimator` 之前增加：

| 常量 | 取值 | 含义 |
|------|------|------|
| `N_RESTARTS` | `8` | 多初值 SLSQP 的随机初值个数。 |
| `USE_DIFFERENTIAL_EVOLUTION` | `True` | 当所有 SLSQP 都无可行解时，是否尝试 `differential_evolution` 全局优化。 |
| `CONSTRAINT_MARGIN_RANK` | `0.1` | 排名法约束 margin：被淘汰者综合排名需比「第二名最差」至少高 0.1。 |
| `CONSTRAINT_MARGIN_PERCENT` | `0.1` | 百分比法约束 margin：被淘汰者综合百分比需比「第二名最好」至少低 0.1。 |

可通过修改上述常量调节行为（例如关闭全局优化以提速）。

---

## 三、排名法 `estimate_fan_votes_rank_method` 的改动

### 3.1 约束条件

- **原约束**：`eliminated_combined - np.max(combined_ranks) >= 0`（被淘汰者综合排名 ≥ 最大值即可，允许并列）。
- **现约束**：`constraint_eliminated(fan_ranks) = eliminated_combined - second_worst - CONSTRAINT_MARGIN_RANK >= 0`，其中 `second_worst` 为综合排名的**第二大的值**（`np.partition(..., -2)[-2]`）。
- **效果**：要求被淘汰者综合排名**严格**最差，且比第二名最差至少多 0.1，减少「并列最差」导致验证判错。

### 3.2 多初值 SLSQP

- **原逻辑**：单一随机初值 `x0`，跑一次 SLSQP，失败则直接走启发式。
- **现逻辑**：
  - 用 `seed = 0 .. N_RESTARTS-1` 各生成一个 `x0`，共跑 `N_RESTARTS` 次 SLSQP。
  - 只保留**成功且满足** `constraint_eliminated(result.x) >= 0` 的解。
  - 在这些可行解中，取**目标函数值最小**的作为最终 `fan_ranks`。
- **效果**：降低对初值的依赖，减少陷入「坏」的局部最优。

### 3.3 全局优化回退（differential_evolution）

- **触发条件**：多初值 SLSQP 均未得到可行解，且 `USE_DIFFERENTIAL_EVOLUTION == True`，且 `n_contestants <= 12`（避免维度过高过慢）。
- **做法**：
  - 构造惩罚目标：`obj_penalty(x) = objective(x) + 1e6 * max(0, -constraint_eliminated(x))^2`。
  - 使用 `scipy.optimize.differential_evolution` 最小化 `obj_penalty`（maxiter=300, popsize=15, seed=42）。
  - 若 `constraint_eliminated(res_de.x) >= 0`，则采用 `res_de.x` 作为 `fan_ranks`；否则回退到启发式。
- **效果**：在 SLSQP 全部失败时仍有机会得到满足约束的解。

### 3.4 后处理

- 在得到 `fan_ranks`（无论来自 SLSQP、differential_evolution 还是启发式）之后，调用 **`_ensure_eliminated_worst_rank`**。
- 若当前 `fan_ranks` 下被淘汰者**不是**综合排名最大，则对 `fan_ranks` 做小幅调整（提高被淘汰者排名、降低当前综合最差者排名），使被淘汰者综合排名严格最大。
- **效果**：保证返回给验证流程的解一定满足「被淘汰者综合最差」，避免因数值或启发式边界情况导致验证错误。

---

## 四、百分比法 `estimate_fan_votes_percent_method` 的改动

### 4.1 约束条件

- **原约束**：`np.min(combined_percents) - eliminated_combined >= 0`（被淘汰者综合百分比为最小即可）。
- **现约束**：`constraint_eliminated(fan_percents) = second_best - eliminated_combined - CONSTRAINT_MARGIN_PERCENT >= 0`，其中 `second_best` 为综合百分比的**第二小的值**（`np.partition(..., 1)[1]`）。
- **效果**：被淘汰者综合百分比严格最低，且与「第二名最好」至少差 0.1。

### 4.2 多初值 SLSQP

- 与排名法相同：`N_RESTARTS` 个随机初值，只保留可行解，取目标函数最小的解作为 `fan_percents`。

### 4.3 全局优化回退

- 惩罚目标：`obj_penalty(x) = objective(x) + 1e6 * constraint_sum(x)^2 + 1e6 * max(0, -constraint_eliminated(x))^2`（同时惩罚总和偏离 100 与约束违反）。
- 使用 `differential_evolution` 求解；解需再归一化到总和 100，并检查 `constraint_eliminated`。满足则采用，否则回退启发式。

### 4.4 后处理

- 调用 **`_ensure_eliminated_worst_percent`**：若被淘汰者不是综合百分比最低，则微调 `fan_percents`（降低被淘汰者百分比、提高当前最低者百分比）并重新归一化到 100，使被淘汰者严格最低。

---

## 五、启发式方法的改动

### 5.1 `_heuristic_fan_ranks`

- **原逻辑**：被淘汰者粉丝排名设为 `n_contestants`；其他人用 `judge_ranks[i]*0.8 + np.random.uniform(0.5, 2)`，再 clip，**不保证**综合排名一定使被淘汰者最差。
- **现逻辑**：
  - 新增参数 `seed: Optional[int] = 42`，若传入则 `np.random.seed(seed)`，保证可复现。
  - 先设被淘汰者粉丝排名为 `n_contestants`，再为其他人赋值时**显式保证** `judge_ranks[i] + fan_ranks[i] < judge_ranks[eliminated_idx] + n_contestants`（即综合排名严格小于被淘汰者）。
  - 若赋值后发现 `np.argmax(combined) != eliminated_idx`，则对当前综合最差者与被淘汰者做一次微调，确保被淘汰者综合最差。
- **效果**：启发式结果必然满足「被淘汰者综合排名最大」，且同一输入得到同一输出。

### 5.2 `_heuristic_fan_percents`

- **原逻辑**：按评委百分比缩放、被淘汰者设较低值、归一化到 100，**不保证**综合百分比最低的一定是被淘汰者。
- **现逻辑**：
  - 先按原思路得到一版 `fan_percents`，检查 `np.argmin(judge_percents + fan_percents) == eliminated_idx`。
  - 若不等于，则重新分配：被淘汰者占 100/n*0.35 左右，其余按评委百分比比例分配，再归一化到 100，并保证非负。
- **效果**：启发式下被淘汰者综合百分比严格最低，且总和为 100。

---

## 六、新增辅助方法

### 6.1 `_ensure_eliminated_worst_rank(fan_ranks, judge_ranks, eliminated_idx, n_contestants)`

- **作用**：后处理，确保在给定 `judge_ranks` 下，`fan_ranks` 使得被淘汰者综合排名（`judge_ranks + fan_ranks`）严格最大。
- **做法**：若 `np.argmax(combined) != eliminated_idx`，则适当增大被淘汰者的 `fan_ranks`、减小当前综合最大者的 `fan_ranks`，再 clip 到 [1, n_contestants]，保证差距至少 0.5。

### 6.2 `_ensure_eliminated_worst_percent(fan_percents, judge_percents, eliminated_idx, n_contestants)`

- **作用**：后处理，确保被淘汰者综合百分比严格最低。
- **做法**：若 `np.argmin(combined) != eliminated_idx`，则降低被淘汰者百分比、提高当前最低者百分比，再归一化到 100，并保证不小于 `CONSTRAINT_MARGIN_PERCENT` 的差距。

---

## 七、返回值与调用方式

- **返回值**：仍为包含 `fan_ranks`/`fan_percents`、`fan_votes`、`combined_ranks`/`combined_percents`、`eliminated_idx`、`optimization_success` 的字典；`optimization_success` 表示是否来自 SLSQP 或 differential_evolution 的可行解（启发式时为 `False`）。
- **调用方式**：与原先一致，无需改调用代码；`FanVoteEstimator(df)` 与 `estimate_all_weeks()`、`validate_estimates()` 等用法不变。

---

## 八、预期效果与可调参数

- **预期**：通过多初值、全局回退与约束/后处理保证，淘汰预测准确率有望在原有约 52% 基础上有所提升（具体以 `validate_estimates` 为准）。
- **可调**：
  - 增大 `N_RESTARTS` 可进一步提高找到可行解的概率，但会增加单周耗时。
  - 将 `USE_DIFFERENTIAL_EVOLUTION` 设为 `False` 可关闭全局优化，加快运行。
  - 若发现约束过严导致可行解很少，可适当减小 `CONSTRAINT_MARGIN_RANK` / `CONSTRAINT_MARGIN_PERCENT`（例如设为 0）。

---

## 九、涉及文件

- **仅修改**：`fan_vote_estimator.py`。
- **未修改**：`fan_vote_estimator_advanced.py`、`fan_vote_estimator_enhanced.py` 等；若它们继承 `FanVoteEstimator` 并调用 `estimate_fan_votes_rank_method` / `estimate_fan_votes_percent_method`，将自动使用上述改动。
