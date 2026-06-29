#!/usr/bin/env bash
# API Gateway Testing Script
# Usage: ./test_api.sh
# Requires: curl, jq (for pretty JSON formatting)

BASE_URL="http://localhost:8080"
API_KEY="sodals-secret-key"

echo "🧪 SODALS API Gateway Testing Suite"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Health Check (no auth required)
echo -e "${BLUE}📋 Test 1: Health Check${NC}"
echo "GET /health"
response=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✅ Status: $http_code${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Status: $http_code${NC}"
fi
echo ""

# Test 2: Solve - Basic Request
echo -e "${BLUE}📋 Test 2: Solve Endpoint - Basic${NC}"
echo "POST /v1/solve (with auth)"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
    -H "X-API-KEY: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is the meaning of life?",
        "high_stakes": false
    }')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo "Request body:"
cat << 'EOF' | jq .
{
    "prompt": "What is the meaning of life?",
    "high_stakes": false
}
EOF
echo ""

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✅ Status: $http_code${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Status: $http_code${NC}"
    echo "Response:"
    echo "$body" | jq . 2>/dev/null || echo "$body"
fi
echo ""

# Test 3: Solve - High Stakes
echo -e "${BLUE}📋 Test 3: Solve Endpoint - High Stakes${NC}"
echo "POST /v1/solve (high_stakes=true)"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
    -H "X-API-KEY: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Prove that pi is irrational",
        "high_stakes": true,
        "user_id": "test_user"
    }')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo "Request body:"
cat << 'EOF' | jq .
{
    "prompt": "Prove that pi is irrational",
    "high_stakes": true,
    "user_id": "test_user"
}
EOF
echo ""

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✅ Status: $http_code${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Status: $http_code${NC}"
fi
echo ""

# Test 4: Memorize
echo -e "${BLUE}📋 Test 4: Memorize Endpoint${NC}"
echo "POST /v1/memorize"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/memorize" \
    -H "X-API-KEY: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "SODALS is an edge AI service designed for commercial deployment",
        "user_id": "test_user",
        "category": "knowledge"
    }')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✅ Status: $http_code${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Status: $http_code${NC}"
fi
echo ""

# Test 5: Error - Missing API Key
echo -e "${BLUE}📋 Test 5: Missing API Key (Expected: 401)${NC}"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test"}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "401" ]; then
    echo -e "${GREEN}✅ Correctly returned 401 Unauthorized${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Expected 401 but got: $http_code${NC}"
fi
echo ""

# Test 6: Error - Empty Prompt
echo -e "${BLUE}📋 Test 6: Empty Prompt (Expected: 400)${NC}"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
    -H "X-API-KEY: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"prompt": ""}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "400" ]; then
    echo -e "${GREEN}✅ Correctly returned 400 Bad Request${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Expected 400 but got: $http_code${NC}"
fi
echo ""

# Test 7: Error - Invalid API Key
echo -e "${BLUE}📋 Test 7: Invalid API Key (Expected: 401)${NC}"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
    -H "X-API-KEY: wrong-key" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test"}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "401" ]; then
    echo -e "${GREEN}✅ Correctly returned 401 Unauthorized${NC}"
    echo "$body" | jq . 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Expected 401 but got: $http_code${NC}"
fi
echo ""

# Test 8: Performance - Multiple Requests
echo -e "${BLUE}📋 Test 8: Performance - 5 Sequential Requests${NC}"
total_time=0
for i in {1..5}; do
    echo -n "Request $i... "
    start_time=$(date +%s%N)
    response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/solve" \
        -H "X-API-KEY: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"Test query $i\"}")
    end_time=$(date +%s%N)
    http_code=$(echo "$response" | tail -n1)
    elapsed=$((($end_time - $start_time) / 1000000)) # Convert to ms
    total_time=$(($total_time + $elapsed))
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅${NC} ($elapsed ms)"
    else
        echo -e "${RED}❌${NC} (Status: $http_code)"
    fi
done
avg_time=$(($total_time / 5))
echo -e "Average time per request: ${YELLOW}${avg_time} ms${NC}"
echo ""

echo "✅ Testing Complete!"
echo ""
echo "Summary:"
echo "--------"
echo "✓ Health endpoint (no auth)"
echo "✓ Solve endpoint with basic prompt"
echo "✓ Solve endpoint with high-stakes flag"
echo "✓ Memorize endpoint"
echo "✓ Authentication validation"
echo "✓ Input validation (empty payload)"
echo "✓ Performance testing"
