@echo off
if exist "C:\Python\Python313\python.exe" (
    "C:\Python\Python313\python.exe" "%~dp0sodals_chat.py" %*
) else (
    python "%~dp0sodals_chat.py" %*
)
