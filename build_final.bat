@echo off
set "VSLANG=1033"
chcp 437 >nul

echo Killing potential locking processes...
taskkill /F /IM candle_inference_full.exe /T 2>nul
taskkill /F /IM candle_inference_stdin.exe /T 2>nul
taskkill /F /IM candle_inference_final.exe /T 2>nul

echo Rebuilding...
cargo build --release --bin candle_inference_full --features cuda

if %ERRORLEVEL% EQU 0 (
    echo Build successful. Copying to final binary...
    copy /Y target\release\candle_inference_full.exe target\release\candle_inference_final.exe
    exit /b 0
) else (
    echo Build FAILED.
    exit /b 1
)
