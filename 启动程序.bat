@echo off
title 手势粒子交互系统

echo ================================================
echo    手势控制粒子交互系统
echo ================================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 把 Python 加到环境变量里
set "PATH=C:\Users\dongz\AppData\Local\Python\pythoncore-3.14-64;%PATH%"
set "PATH=C:\Users\dongz\AppData\Local\Python\pythoncore-3.14-64\Scripts;%PATH%"

:: 验证 Python 是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 找不到 Python！
    echo 请确认 Python 已安装在这个位置：
    echo C:\Users\dongz\AppData\Local\Python\pythoncore-3.14-64
    echo.
    pause
    exit /b 1
)

echo Python 已就绪:
python --version
echo.

:: 检查模型文件
if not exist "hand_landmarker.task" (
    echo [错误] 找不到模型文件 hand_landmarker.task
    pause
    exit /b 1
)

echo 正在打开摄像头...
echo 按 Q 键可以退出程序
echo ================================================
echo.

:: 运行程序
python gesture_particles.py

:: 程序退出后显示
echo.
echo 程序已退出。
pause
