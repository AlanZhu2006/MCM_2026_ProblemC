@echo off
REM 使用Python 3.11运行争议案例分析脚本
echo ======================================================================
echo 运行争议案例深度分析（使用Python 3.11）
echo ======================================================================
echo.

REM 优先尝试使用python3.11（已检测到存在）
echo 使用 python3.11 运行脚本...
python3.11 scripts/enhance_controversial_cases.py
if %errorlevel% == 0 goto :success

REM 如果失败，尝试其他方式
echo.
echo 尝试使用 py -3.11...
py -3.11 scripts/enhance_controversial_cases.py
if %errorlevel% == 0 goto :success

echo.
echo 尝试使用默认 python...
python scripts/enhance_controversial_cases.py
if %errorlevel% == 0 goto :success

echo.
echo ======================================================================
echo 错误: 脚本运行失败
echo ======================================================================
echo.
echo 请检查:
echo 1. Python 3.11是否已安装pandas, matplotlib, seaborn
echo 2. 运行: python3.11 -c "import pandas; print('OK')"
echo.
pause
exit /b 1

:success
echo.
echo ======================================================================
echo 脚本运行成功！
echo ======================================================================
echo.
echo 生成的图表文件在 visualizations/ 目录
echo 生成的分析报告在项目根目录
echo.
pause
