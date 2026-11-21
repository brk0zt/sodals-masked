@echo off
setlocal
set "ROOT=%~dp0"
set "PY_EXE=C:\Python\Python313\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

echo ========================================================================
echo   SODALS Offline Claude Code Launcher (No Internet / No GGUF Crash)
echo ========================================================================
where claude >nul 2>nul
if errorlevel 1 (
    echo [HATA] 'claude' CLI PATH uzerinde bulunamadi. Lutfen Claude Code CLI kurun.
    exit /b 1
)

"%PY_EXE%" -c "import sys, urllib.request; op=urllib.request.build_opener(urllib.request.ProxyHandler({})); sys.exit(0 if op.open('http://127.0.0.1:11435/api/version', timeout=2).status == 200 else 1)" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [1/2] Mevcut SODALS SafeTensors+LoRA Daemon :11435 uzerinde zaten calisiyor.
) else (
    echo [1/2] Starting SODALS SafeTensors+LoRA Server (Anthropic /v1/messages) on :11435 ...
    start "SODALS-Daemon-11435" cmd /k ""%PY_EXE%" -B "%ROOT%sodals_chat.py" --server --host 127.0.0.1 --port 11435"
)

echo [2/2] Waiting for SODALS daemon readiness on http://127.0.0.1:11435/api/version (up to 90s) ...
"%PY_EXE%" -c "import time, urllib.request; op=urllib.request.build_opener(urllib.request.ProxyHandler({})); url='http://127.0.0.1:11435/api/version'; t0=time.time(); ok=False; exec('while time.time()-t0 < 90:\n try:\n  if op.open(url, timeout=2).status == 200: ok=True; break\n except Exception:\n  time.sleep(1.5)'); print('  [OK] Daemon hazir!' if ok else '  [UYARI] Daemon 90sn icinde hazir olmadi; claude yine de baslatiliyor.')"
set "ANTHROPIC_BASE_URL=http://127.0.0.1:11435"
set "ANTHROPIC_AUTH_TOKEN=sodals-offline"
set "ANTHROPIC_API_KEY=sodals-offline"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
claude --model sodals-neurosymbolic-9b %*
