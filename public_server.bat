@echo off
chcp 65001 >nul
title 手势粒子系统 - 公网部署

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║     🌟 手势粒子交互系统 - 公网链接生成器                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo    [1/2] 启动本地 HTTP 服务器...

cd /d "%~dp0"
start /B python -m http.server 8000 >nul 2>&1

echo    [2/2] 创建公网隧道...
echo.
echo    正在连接 serveo.net...
echo.

ssh -o StrictHostKeyChecking=no -R 80:localhost:8000 serveo.net

echo.
echo    隧道已断开。按任意键退出...
pause >nul
