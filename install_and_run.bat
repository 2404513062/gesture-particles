@echo off
chcp 65001 >nul
title 手势粒子交互系统 - 安装与启动

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       🌟 手势控制粒子交互系统 - 一键安装启动 🌟         ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python！
    echo    请先从 https://www.python.org/downloads/ 下载安装 Python
    echo    安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo ✅ Python 已检测到
python --version
echo.

:: 安装依赖
echo 📦 正在安装依赖包 (可能需要几分钟)...
pip install opencv-python mediapipe numpy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/  --trusted-host mirrors.tuna.tsinghua.edu.cn
if %errorlevel% neq 0 (
    echo ⚠️ 清华源安装失败，尝试使用默认源...
    pip install opencv-python mediapipe numpy
)

echo.
echo ✅ 依赖安装完成！
echo.

:: 启动程序
echo 🚀 正在启动手势粒子交互系统...
echo.
python gesture_particles.py

:: 程序退出后
echo.
echo 程序已退出。按任意键关闭此窗口...
pause >nul
