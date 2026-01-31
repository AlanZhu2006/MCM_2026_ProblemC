@echo off
REM 使用Python 3.11运行所有优化脚本
echo ======================================================================
echo 运行所有优化脚本（使用Python 3.11）
echo ======================================================================
echo.

REM 尝试使用python3.11（最常见）
echo 尝试使用 python3.11...
python3.11 scripts/run_all_optimizations.py
if %errorlevel% == 0 goto :success

REM 尝试使用py launcher指定Python 3.11
echo.
echo 尝试使用 py -3.11...
py -3.11 scripts/run_all_optimizations.py
if %errorlevel% == 0 goto :success

REM 尝试查找Python 3.11的常见路径
echo.
echo 尝试使用常见路径的Python 3.11...
if exist "C:\Python311\python.exe" (
    C:\Python311\python.exe scripts/run_all_optimizations.py
    if %errorlevel% == 0 goto :success
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" scripts/run_all_optimizations.py
    if %errorlevel% == 0 goto :success
)

REM 最后尝试使用默认python
echo.
echo 尝试使用默认 python...
python scripts/run_all_optimizations.py
if %errorlevel% == 0 goto :success

echo.
echo ======================================================================
echo 错误: 无法找到可用的Python 3.11环境
echo ======================================================================
echo.
echo 请手动指定Python 3.11路径，例如:
echo   C:\Python311\python.exe scripts/run_all_optimizations.py
echo.
pause
exit /b 1

:success
echo.
echo ======================================================================
echo 所有优化脚本运行成功！
echo ======================================================================
pause
