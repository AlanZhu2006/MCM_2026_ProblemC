"""
数学公式严格化
完善约束条件、理论分析、公式推导
"""

from pathlib import Path

def enhance_stage2_formulas():
    """增强Stage 2的数学公式"""
    print("=" * 70)
    print("增强Stage 2数学公式")
    print("=" * 70)
    
    enhancements = """
\\subsubsection{Mathematical Formulation}

\\textbf{Problem Definition}: Given judge scores $\\mathbf{s} = [s_1, s_2, \\ldots, s_n]$ and the eliminated contestant $i_e$, find fan votes $\\mathbf{v} = [v_1, v_2, \\ldots, v_n]$ such that the elimination result is consistent.

\\textbf{For Rank-Based Method} (Seasons 1-2, 28-34):
\\begin{align}
\\min_{\\mathbf{v}} \\quad & f(\\mathbf{v}, \\mathbf{s}) = \\sum_{i=1}^{n} w_i \\left( v_i - \\hat{v}_i(\\mathbf{s}) \\right)^2 \\\\
\\text{s.t.} \\quad & R_{\\text{judge}}(i_e) + R_{\\text{fan}}(i_e) \\geq R_{\\text{judge}}(j) + R_{\\text{fan}}(j) + \\epsilon \\quad \\forall j \\neq i_e \\\\
& \\sum_{i=1}^{n} v_i > 0 \\\\
& v_i \\geq 0 \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
\\end{align}

where:
\\begin{itemize}
    \\item $\\hat{v}_i(\\mathbf{s})$ is the expected fan vote for contestant $i$ based on features
    \\item $w_i$ are weights based on feature importance
    \\item $\\epsilon > 0$ is a margin to ensure strict inequality (default: 0.1)
    \\item $R_{\\text{fan}}(i) = \\text{rank}(v_i)$ where rank is computed in descending order
\\end{itemize}

\\textbf{For Percent-Based Method} (Seasons 3-27):
\\begin{align}
\\min_{\\mathbf{v}} \\quad & f(\\mathbf{v}, \\mathbf{s}) = \\sum_{i=1}^{n} w_i \\left( v_i - \\hat{v}_i(\\mathbf{s}) \\right)^2 \\\\
\\text{s.t.} \\quad & P_{\\text{judge}}(i_e) + P_{\\text{fan}}(i_e) \\leq P_{\\text{judge}}(j) + P_{\\text{fan}}(j) - \\epsilon \\quad \\forall j \\neq i_e \\\\
& \\sum_{i=1}^{n} v_i > 0 \\\\
& v_i \\geq 0 \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
\\end{align}

where:
\\begin{itemize}
    \\item $P_{\\text{judge}}(i) = \\frac{s_i}{\\sum_{j=1}^{n} s_j}$ is the judge percentage
    \\item $P_{\\text{fan}}(i) = \\frac{v_i}{\\sum_{j=1}^{n} v_j}$ is the fan percentage
    \\item $\\epsilon > 0$ is a margin (default: 0.1)
\\end{itemize}

\\subsubsection{Expected Fan Vote Function}

The expected fan vote $\\hat{v}_i(\\mathbf{s})$ is computed as:
\\begin{equation}
\\hat{v}_i(\\mathbf{s}) = \\alpha_0 + \\alpha_1 s_i + \\alpha_2 H_i + \\alpha_3 A_i + \\alpha_4 I_i + \\alpha_5 P_i
\\end{equation}

where:
\\begin{itemize}
    \\item $s_i$ = judge score for contestant $i$
    \\item $H_i$ = historical performance (average of previous weeks)
    \\item $A_i$ = age factor (normalized)
    \\item $I_i$ = industry factor (encoded)
    \\item $P_i$ = professional dancer factor (encoded)
    \\item $\\alpha_j$ are coefficients learned from data
\\end{itemize}

\\subsubsection{Optimization Algorithm Properties}

\\textbf{Convexity}: The objective function $f(\\mathbf{v}, \\mathbf{s})$ is convex in $\\mathbf{v}$ (sum of squared terms), but the constraints are non-convex due to ranking operations.

\\textbf{Feasibility}: The problem is always feasible because:
\\begin{itemize}
    \\item We can always find $\\mathbf{v}$ such that the eliminated contestant has the worst combined score
    \\item The non-negativity constraints are always satisfiable
\\end{itemize}

\\textbf{Solution Uniqueness}: The problem may have multiple solutions (inverse problem), so we use:
\\begin{itemize}
    \\item Multi-start optimization (8 different initial guesses)
    \\item Selection of the solution closest to expected values
\\end{itemize}

\\subsubsection{Convergence Analysis}

The SLSQP algorithm converges to a local minimum. We ensure global optimality by:
\\begin{enumerate}
    \\item Running multiple optimizations from different starting points
    \\item Using differential evolution as a global optimizer if SLSQP fails
    \\item Post-processing to ensure constraint satisfaction
\\end{enumerate}
"""
    
    return enhancements

def enhance_stage5_formulas():
    """增强Stage 5的数学公式"""
    print("\n增强Stage 5数学公式")
    
    enhancements = """
\\subsubsection{Mathematical Formulation}

\\textbf{Feature Vector}: For each contestant $i$ at week $t$, we construct a 12-dimensional feature vector:
\\begin{equation}
\\mathbf{x}_i^{(t)} = [x_{i,1}^{(t)}, x_{i,2}^{(t)}, \\ldots, x_{i,12}^{(t)}]^T
\\end{equation}

where the features are:
\\begin{align}
x_{i,1}^{(t)} &= \\frac{s_i^{(t)} - \\min_j s_j^{(t)}}{\\max_j s_j^{(t)} - \\min_j s_j^{(t)}} \\quad \\text{(normalized judge score)} \\\\
x_{i,2}^{(t)} &= \\frac{v_i^{(t)} - \\min_j v_j^{(t)}}{\\max_j v_j^{(t)} - \\min_j v_j^{(t)}} \\quad \\text{(normalized fan vote)} \\\\
x_{i,3}^{(t)} &= \\frac{R_{\\text{judge}}(i) - 1}{n-1} \\quad \\text{(normalized judge rank)} \\\\
x_{i,4}^{(t)} &= \\frac{R_{\\text{fan}}(i) - 1}{n-1} \\quad \\text{(normalized fan rank)} \\\\
x_{i,5}^{(t)} &= \\frac{s_i^{(t)}}{\\sum_{j=1}^{n} s_j^{(t)}} \\quad \\text{(judge percentage)} \\\\
x_{i,6}^{(t)} &= \\frac{v_i^{(t)}}{\\sum_{j=1}^{n} v_j^{(t)}} \\quad \\text{(fan percentage)} \\\\
x_{i,7}^{(t)} &= \\frac{s_i^{(t)} - \\bar{s}^{(t)}}{\\sigma_s^{(t)}} \\quad \\text{(judge relative to mean)} \\\\
x_{i,8}^{(t)} &= \\frac{v_i^{(t)} - \\bar{v}^{(t)}}{\\sigma_v^{(t)}} \\quad \\text{(fan relative to mean)} \\\\
x_{i,9}^{(t)} &= \\frac{A_i - \\min_j A_j}{\\max_j A_j - \\min_j A_j} \\quad \\text{(normalized age)} \\\\
x_{i,10}^{(t)} &= \\text{encode}(I_i) \\quad \\text{(industry encoded)} \\\\
x_{i,11}^{(t)} &= \\text{encode}(P_i) \\quad \\text{(pro dancer encoded)} \\\\
x_{i,12}^{(t)} &= \\text{encode}(R_i) \\quad \\text{(region encoded)}
\\end{align}

\\textbf{Model}: We use LightGBM, which learns a function $f: \\mathbb{R}^{12} \\rightarrow [0,1]$ such that:
\\begin{equation}
P(\\text{eliminated}_i | \\mathbf{x}_i^{(t)}) = f(\\mathbf{x}_i^{(t)})
\\end{equation}

\\textbf{Decision Rule}:
\\begin{equation}
i_{\\text{eliminated}} = \\arg\\max_{i \\in \\mathcal{C}^{(t)}} P(\\text{eliminated}_i | \\mathbf{x}_i^{(t)})
\\end{equation}

where $\\mathcal{C}^{(t)}$ is the set of contestants remaining at week $t$.

\\subsubsection{Model Training}

The model is trained to minimize the cross-entropy loss:
\\begin{equation}
\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\log(f(\\mathbf{x}_i)) + (1-y_i) \\log(1-f(\\mathbf{x}_i)) \\right]
\\end{equation}

where $y_i \\in \\{0, 1\\}$ indicates whether contestant $i$ was eliminated.

\\textbf{Regularization}: We use:
\\begin{itemize}
    \\item L2 regularization on leaf values
    \\item Minimum child samples to prevent overfitting
    \\item Feature subsampling (0.8)
    \\item Data subsampling (0.8)
\\end{itemize}

\\subsubsection{Feature Importance}

The importance of feature $j$ is computed as:
\\begin{equation}
I_j = \\sum_{t=1}^{T} \\sum_{l=1}^{L_t} w_{t,l} \\cdot \\mathbb{1}[\\text{split at } x_j]
\\end{equation}

where:
\\begin{itemize}
    \\item $T$ = number of trees
    \\item $L_t$ = number of leaves in tree $t$
    \\item $w_{t,l}$ = weight of leaf $l$ in tree $t$
    \\item $\\mathbb{1}[\\cdot]$ = indicator function
\\end{itemize}
"""
    
    return enhancements

def create_enhanced_formulas_file():
    """创建增强的数学公式文件"""
    print("\n" + "=" * 70)
    print("创建增强的数学公式")
    print("=" * 70)
    
    stage2_formulas = enhance_stage2_formulas()
    stage5_formulas = enhance_stage5_formulas()
    
    # 保存到文件
    output_path = Path('enhanced_math_formulas.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% Enhanced Mathematical Formulas\n")
        f.write("% Stage 2 Enhancements\n")
        f.write(stage2_formulas)
        f.write("\n\n% Stage 5 Enhancements\n")
        f.write(stage5_formulas)
    
    print(f"\n[OK] 增强的数学公式已保存到: {output_path}")
    print("\n这些公式可以手动添加到相应的LaTeX章节文件中")

def main():
    """主函数"""
    create_enhanced_formulas_file()
    print("\n" + "=" * 70)
    print("数学公式增强完成！")
    print("=" * 70)
    print("\n注意: 请手动将这些公式添加到相应的LaTeX文件中")

if __name__ == '__main__':
    main()
