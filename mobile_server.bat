@echo off
chcp 65001 >nul
title 手势粒子系统 - 手机版服务器

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║     🌟 手势粒子交互系统 - 手机版服务器                    ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM 获取本机 IP 地址
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set ip=%%a
    goto :found
)
:found
set ip=%ip:~1%

echo    📱 在手机浏览器中打开以下地址：
echo.
echo    👉 http://%ip%:8000/gesture_particles.html
echo.
echo    ⚠️  确保手机和电脑连接同一个 WiFi！
echo    ⚠️  首次打开需加载模型（~16MB），建议在 WiFi 下使用
echo.
echo    按 Ctrl+C 停止服务器
echo ═══════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"
python -m http.server 8000

pause
