@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3 launch_demo.py %*
) else (
  python launch_demo.py %*
)
set "demoExitCode=%errorlevel%"
if not "%demoExitCode%"=="0" echo Demo failed. See the error above.
if not defined LPQAOA_DEMO_NO_PAUSE pause
exit /b %demoExitCode%
