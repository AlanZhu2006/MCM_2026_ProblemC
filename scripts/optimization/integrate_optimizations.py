"""
整合所有优化结果到LaTeX文件
"""

from pathlib import Path
import re

def integrate_uncertainty_analysis():
    """整合不确定性分析到Stage 2"""
    print("=" * 70)
    print("整合不确定性分析到Stage 2")
    print("=" * 70)
    
    stage2_file = Path('sections/stage2_fan_vote_estimation.tex')
    if not stage2_file.exists():
        print("错误: Stage 2文件不存在")
        return False
    
    content = stage2_file.read_text(encoding='utf-8')
    
    # 查找不确定性分析部分
    uncertainty_section = r"""
\subsubsection{Uncertainty Source Analysis}

We analyzed the sources of uncertainty in our estimates:

\\begin{itemize}
    \\item \\textbf{Week-to-Week Variation}: Uncertainty varies significantly across different weeks
    \\item \\textbf{Seasonal Patterns}: Some seasons show higher average uncertainty than others
    \\item \\textbf{Contestant-Specific}: Uncertainty depends on the contestant's characteristics and performance
    \\item \\textbf{Data Quality}: Weeks with missing data or extreme values show higher uncertainty
\\end{itemize}

\\subsubsection{Uncertainty Visualization}

Figure~\\ref{fig:uncertainty} shows comprehensive uncertainty analysis, including:
\\begin{itemize}
    \\item Distribution of uncertainty (standard deviation)
    \\item Confidence interval width distribution
    \\item Uncertainty trends over time
    \\item Seasonal comparison of uncertainty
    \\item Relationship between uncertainty and fan vote magnitude
    \\item Coefficient of variation distribution
\\end{itemize}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{visualizations/uncertainty_analysis.png}
\\caption{Comprehensive Uncertainty Analysis}
\\label{fig:uncertainty}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{visualizations/confidence_intervals.png}
\\caption{95\\% Confidence Intervals for Fan Vote Estimates (Sample Weeks)}
\\label{fig:confidence_intervals}
\\end{figure}
"""
    
    # 在不确定性分析部分后添加
    if '\\subsubsection{Uncertainty Results}' in content:
        # 在Uncertainty Results后添加
        pattern = r'(\\subsubsection{Uncertainty Results}.*?)(\\subsection)'
        replacement = r'\\1' + uncertainty_section + r'\n\n\\2'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        stage2_file.write_text(content, encoding='utf-8')
        print("✓ 不确定性分析已整合到Stage 2")
        return True
    else:
        print("⚠️  未找到Uncertainty Results部分，请手动添加")
        return False

def integrate_sensitivity_analysis():
    """整合敏感性分析到Stage 2"""
    print("\n整合敏感性分析到Stage 2")
    
    stage2_file = Path('sections/stage2_fan_vote_estimation.tex')
    if not stage2_file.exists():
        return False
    
    content = stage2_file.read_text(encoding='utf-8')
    
    sensitivity_section = r"""
\subsection{Sensitivity Analysis}

\\subsubsection{Parameter Sensitivity}

We tested different parameter configurations to assess model robustness:
\\begin{itemize}
    \\item Number of optimization restarts (4, 8, 12)
    \\item Constraint margins (0.05, 0.1, 0.15, 0.2)
    \\item Results show the model is robust to parameter variations
\\end{itemize}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.7\\textwidth]{visualizations/parameter_sensitivity.png}
\\caption{Parameter Sensitivity Analysis}
\\label{fig:parameter_sensitivity}
\\end{figure}

\\subsubsection{Data Sensitivity}

We analyzed the impact of:
\\begin{itemize}
    \\item Missing data (0\\%, 10\\%, 20\\%, 30\\% missing)
    \\item Outliers (0\\%, 5\\%, 10\\%, 15\\% outliers)
    \\item Results show the model is relatively robust to data quality issues
\\end{itemize}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{visualizations/data_sensitivity.png}
\\caption{Data Sensitivity Analysis}
\\label{fig:data_sensitivity}
\\end{figure}
"""
    
    # 在模型验证部分后添加
    if '\\subsection{Model Validation}' in content:
        pattern = r'(\\subsection{Model Validation}.*?)(\\subsection)'
        replacement = r'\\1' + sensitivity_section + r'\n\n\\2'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        stage2_file.write_text(content, encoding='utf-8')
        print("✓ 敏感性分析已整合到Stage 2")
        return True
    else:
        print("⚠️  未找到Model Validation部分，请手动添加")
        return False

def integrate_controversial_cases():
    """整合争议案例分析到Stage 3"""
    print("\n整合争议案例分析到Stage 3")
    
    stage3_file = Path('sections/stage3_voting_comparison.tex')
    if not stage3_file.exists():
        return False
    
    content = stage3_file.read_text(encoding='utf-8')
    
    cases_visualization = r"""
\subsubsection{Detailed Case Visualizations}

Figure~\\ref{fig:controversial_detailed} shows detailed time-series analysis for each controversial case, including judge scores, fan votes, and rank progression throughout their respective seasons.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{visualizations/controversial_cases_detailed.png}
\\caption{Detailed Analysis of Controversial Cases}
\\label{fig:controversial_detailed}
\\end{figure}

Figure~\\ref{fig:controversial_comparison} provides a comprehensive comparison across all four controversial cases, highlighting their similarities and differences.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{visualizations/controversial_cases_comparison.png}
\\caption{Comparison of Controversial Cases}
\\label{fig:controversial_comparison}
\\end{figure}
"""
    
    # 在争议案例分析部分后添加
    if '\\subsection{Controversial Cases Analysis}' in content:
        pattern = r'(\\subsection{Controversial Cases Analysis}.*?)(\\subsection)'
        replacement = r'\\1' + cases_visualization + r'\n\n\\2'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        stage3_file.write_text(content, encoding='utf-8')
        print("✓ 争议案例可视化已整合到Stage 3")
        return True
    else:
        print("⚠️  未找到Controversial Cases Analysis部分，请手动添加")
        return False

def integrate_math_formulas():
    """整合增强的数学公式"""
    print("\n整合增强的数学公式")
    
    # 读取增强的公式
    formulas_file = Path('enhanced_math_formulas.tex')
    if not formulas_file.exists():
        print("⚠️  enhanced_math_formulas.tex 不存在")
        return False
    
    formulas_content = formulas_file.read_text(encoding='utf-8')
    
    # 提取Stage 2的公式
    stage2_match = re.search(r'% Stage 2 Enhancements(.*?)(?=% Stage 5|$)', formulas_content, re.DOTALL)
    if stage2_match:
        stage2_formulas = stage2_match.group(1).strip()
        
        # 添加到Stage 2文件
        stage2_file = Path('sections/stage2_fan_vote_estimation.tex')
        if stage2_file.exists():
            content = stage2_file.read_text(encoding='utf-8')
            
            # 在Mathematical Model部分后添加
            if '\\subsection{Mathematical Model}' in content:
                # 在Objective Function之前插入
                pattern = r'(\\subsubsection{Objective Function})'
                replacement = stage2_formulas + r'\n\n\\1'
                content = re.sub(pattern, replacement, content)
                
                stage2_file.write_text(content, encoding='utf-8')
                print("✓ Stage 2数学公式已整合")
    
    # 提取Stage 5的公式
    stage5_match = re.search(r'% Stage 5 Enhancements(.*?)$', formulas_content, re.DOTALL)
    if stage5_match:
        stage5_formulas = stage5_match.group(1).strip()
        
        # 添加到Stage 5文件
        stage5_file = Path('sections/stage5_new_system.tex')
        if stage5_file.exists():
            content = stage5_file.read_text(encoding='utf-8')
            
            # 在Model Architecture部分后添加
            if '\\subsubsection{Model Architecture}' in content:
                pattern = r'(\\subsubsection{Model Architecture}.*?)(\\subsection)'
                replacement = r'\\1' + stage5_formulas + r'\n\n\\2'
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                
                stage5_file.write_text(content, encoding='utf-8')
                print("✓ Stage 5数学公式已整合")
    
    return True

def main():
    """主函数"""
    print("=" * 70)
    print("整合所有优化结果到LaTeX文件")
    print("=" * 70)
    
    results = {
        'uncertainty': integrate_uncertainty_analysis(),
        'sensitivity': integrate_sensitivity_analysis(),
        'controversial_cases': integrate_controversial_cases(),
        'math_formulas': integrate_math_formulas()
    }
    
    print("\n" + "=" * 70)
    print("整合完成总结")
    print("=" * 70)
    
    for item, success in results.items():
        status = "[OK]" if success else "[WARNING]"
        print(f"{item}: {status}")
    
    print("\n" + "=" * 70)
    print("整合完成！")
    print("=" * 70)
    print("\n注意: 请检查LaTeX文件，确保所有内容正确整合")
    print("建议: 编译LaTeX检查是否有错误")

if __name__ == '__main__':
    main()
