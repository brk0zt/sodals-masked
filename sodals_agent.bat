@echo off
setlocal
set "WORK_DIR=%CD%"
if "%WORK_DIR:~-1%"=="\" set "WORK_DIR=%WORK_DIR%."
if exist "C:\Python\Python313\python.exe" (
    "C:\Python\Python313\python.exe" -B "%~dp0sodals_chat.py" --agent --cwd "%WORK_DIR%" %*
) else (
    python -B "%~dp0sodals_chat.py" --agent --cwd "%WORK_DIR%" %*
)
exit /b %ERRORLEVEL%
