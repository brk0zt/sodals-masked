@echo off
echo ==============================================================
echo       SODALS CLOSED-BOX DOCKER BENCHMARK VALIDATION
echo ==============================================================
echo Step 1: Building isolated validation container...
docker build -f Dockerfile.validation -t sodals-validation-sandbox .

echo.
echo Step 2: Running validation sandbox in air-gapped mode...
echo (Network disabled to prevent any outbound leakage or cheating)

if not exist run_logs mkdir run_logs

:: Run the container air-gapped (--network none) with local model mounting
:: You can also mount custom blind evaluation datasets by adding:
::   -v %cd%/custom_sre:/app/sre
::   -v %cd%/custom_finops:/app/finops
::   -v %cd%/custom_fsiq:/app/FailureSensorIQ
docker run --network none --gpus all --name sodals-sandbox ^
  -v %cd%/run_logs:/app/run_logs ^
  -v %cd%/models:/app/models ^
  sodals-validation-sandbox

echo.
echo Validation complete. Logs written to %cd%\run_logs\
echo ==============================================================
