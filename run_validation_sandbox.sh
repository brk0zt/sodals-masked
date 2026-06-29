#!/bin/bash
echo "=============================================================="
echo "      SODALS CLOSED-BOX DOCKER BENCHMARK VALIDATION"
echo "=============================================================="
echo "Step 1: Building isolated validation container..."
docker build -f Dockerfile.validation -t sodals-validation-sandbox .

echo ""
echo "Step 2: Running validation sandbox in air-gapped mode..."
echo "(Network disabled to prevent any outbound leakage or cheating)"

# Ensure run_logs directory exists locally
mkdir -p run_logs

# Run the container air-gapped (--network none) with local model mounting
# You can also mount custom blind evaluation datasets by adding:
#   -v $(pwd)/custom_sre:/app/sre
#   -v $(pwd)/custom_finops:/app/finops
#   -v $(pwd)/custom_fsiq:/app/FailureSensorIQ
docker run --network none --gpus all --name sodals-sandbox \
  -v "$(pwd)/run_logs:/app/run_logs" \
  -v "$(pwd)/models:/app/models" \
  sodals-validation-sandbox

echo ""
echo "Validation complete. Logs written to $(pwd)/run_logs/"
echo "=============================================================="
