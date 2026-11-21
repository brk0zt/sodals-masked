@echo off
setlocal
set "ROOT=%~dp0"
if exist "C:\Python\Python313\python.exe" (
    set "PY=C:\Python\Python313\python.exe"
) else (
    set "PY=python"
)

if "%~1"=="" goto :help
if /I "%~1"=="help" goto :help
if /I "%~1"=="agent" goto :agent
if /I "%~1"=="provision" goto :vm
if /I "%~1"=="start" goto :vm
if /I "%~1"=="stop" goto :vm
if /I "%~1"=="status" goto :vm
if /I "%~1"=="reset" goto :vm
if /I "%~1"=="dashboard" goto :vm
goto :help

:vm
"%PY%" -B "%ROOT%sodals_os\vm_manager.py" %*
exit /b %ERRORLEVEL%

:agent
"%PY%" -B "%ROOT%sodals_os\vm_manager.py" start
if errorlevel 1 exit /b 1
set "EXTRA_ARGS="
shift
:collect_args
if "%~1"=="" goto :run_agent
set "EXTRA_ARGS=%EXTRA_ARGS% %1"
shift
goto :collect_args
:run_agent
"%PY%" -B "%ROOT%sodals_chat.py" --os-agent%EXTRA_ARGS%
exit /b %ERRORLEVEL%

:help
echo.
echo ====================================================================
echo   SODALS-OS — Izole Mini Isletim Sistemi (Alpine Linux QEMU VM)
echo ====================================================================
echo   Kullanim:
echo     sodals_os.bat start      - Izole Mini-OS sanal makinesini baslatir
echo     sodals_os.bat dashboard  - Canli Web Kontrol Panelini tarayicida acar
echo     sodals_os.bat agent      - Mini-OS'i baslatir ve SODALS Otonom Ajanini acar
echo     sodals_os.bat status     - VM durumu, RAM ve izolasyon denetimini gosterir
echo     sodals_os.bat stop       - Mini-OS'i guvenli (ACPI) olarak kapatir
echo     sodals_os.bat reset      - Mini-OS diskini temiz baslangic anina (snapshot) dondurur
echo     sodals_os.bat provision  - Ilk kurulumu (Alpine Linux + daemon) yapar
echo.
exit /b 0
