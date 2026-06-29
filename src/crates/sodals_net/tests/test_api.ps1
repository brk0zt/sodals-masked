# PowerShell API Gateway Testing Script
# Usage: .\test_api.ps1
# For Windows users

$BaseUrl = "http://localhost:8080"
$ApiKey = "sodals-secret-key"
$Headers = @{
    "X-API-KEY"    = $ApiKey
    "Content-Type" = "application/json"
}

function Test-Request {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Endpoint,
        [object]$Body,
        [bool]$RequireAuth = $true,
        [int]$ExpectedStatus = 200
    )
    
    $Url = "$BaseUrl$Endpoint"
    $TestHeaders = if ($RequireAuth) { $Headers } else { @{"Content-Type" = "application/json"} }
    
    Write-Host "`n🧪 $Name" -ForegroundColor Blue
    Write-Host "$Method $Endpoint"
    
    if ($Body) {
        Write-Host "Request:" -ForegroundColor Yellow
        Write-Host ($Body | ConvertTo-Json) -ForegroundColor Gray
    }
    
    try {
        $Response = Invoke-WebRequest -Uri $Url -Method $Method -Headers $TestHeaders `
            -Body $Body -ErrorAction Continue -TimeoutSec 10
        
        $StatusCode = $Response.StatusCode
        $Content = $Response.Content | ConvertFrom-Json
    }
    catch {
        if ($null -ne $_.Exception.Response) {
            $StatusCode = [int]$_.Exception.Response.StatusCode
            $Content = $_.Exception.Response.Content.ReadAsString() | ConvertFrom-Json -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
            return
        }
    }
    
    if ($StatusCode -eq $ExpectedStatus) {
        Write-Host "✅ Status: $StatusCode" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Status: $StatusCode (Expected: $ExpectedStatus)" -ForegroundColor Red
    }
    
    if ($Content) {
        Write-Host "Response:" -ForegroundColor Yellow
        Write-Host ($Content | ConvertTo-Json) -ForegroundColor Gray
    }
}

Write-Host "🚀 SODALS API Gateway Testing Suite (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Test 1: Health Check
Test-Request -Name "Health Check" -Method "GET" -Endpoint "/health" -RequireAuth $false

# Test 2: Solve - Basic
Test-Request -Name "Solve - Basic Request" -Method "POST" -Endpoint "/v1/solve" `
    -Body (@{
        prompt = "What is 2+2?"
        high_stakes = $false
    } | ConvertTo-Json)

# Test 3: Solve - High Stakes
Test-Request -Name "Solve - High Stakes" -Method "POST" -Endpoint "/v1/solve" `
    -Body (@{
        prompt = "Prove that pi is irrational"
        high_stakes = $true
        user_id = "test_user"
    } | ConvertTo-Json)

# Test 4: Memorize
Test-Request -Name "Memorize Request" -Method "POST" -Endpoint "/v1/memorize" `
    -Body (@{
        text = "SODALS is an edge AI service designed for commercial deployment"
        user_id = "test_user"
        category = "knowledge"
    } | ConvertTo-Json)

# Test 5: Error - Missing API Key
Test-Request -Name "Missing API Key (Expected: 401)" -Method "POST" -Endpoint "/v1/solve" `
    -Body (@{ prompt = "test" } | ConvertTo-Json) -RequireAuth $false -ExpectedStatus 401

# Test 6: Error - Empty Prompt
Test-Request -Name "Empty Prompt (Expected: 400)" -Method "POST" -Endpoint "/v1/solve" `
    -Body (@{ prompt = "" } | ConvertTo-Json) -ExpectedStatus 400

# Test 7: Error - Invalid API Key
$WrongHeaders = @{
    "X-API-KEY"    = "wrong-key"
    "Content-Type" = "application/json"
}

Write-Host "`n🧪 Invalid API Key (Expected: 401)" -ForegroundColor Blue
Write-Host "POST /v1/solve"

try {
    $Response = Invoke-WebRequest -Uri "$BaseUrl/v1/solve" -Method POST -Headers $WrongHeaders `
        -Body (@{ prompt = "test" } | ConvertTo-Json) -ErrorAction Continue
}
catch {
    if ($null -ne $_.Exception.Response) {
        $StatusCode = [int]$_.Exception.Response.StatusCode
        if ($StatusCode -eq 401) {
            Write-Host "✅ Correctly returned 401 Unauthorized" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Expected 401 but got: $StatusCode" -ForegroundColor Red
        }
    }
}

# Performance Test
Write-Host "`n🧪 Performance Test - 5 Sequential Requests" -ForegroundColor Blue
$TotalTime = 0
for ($i = 1; $i -le 5; $i++) {
    Write-Host -NoNewline "Request $i... "
    
    $StartTime = Get-Date
    try {
        $Response = Invoke-WebRequest -Uri "$BaseUrl/v1/solve" -Method POST -Headers $Headers `
            -Body (@{ prompt = "Test query $i" } | ConvertTo-Json) -ErrorAction Continue -TimeoutSec 5
        
        $ElapsedMs = (Get-Date) - $StartTime
        $TotalTime += $ElapsedMs.TotalMilliseconds
        
        if ($Response.StatusCode -eq 200) {
            Write-Host "✅ $([int]$ElapsedMs.TotalMilliseconds) ms" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "❌ Failed" -ForegroundColor Red
    }
}

$AvgTime = [int]($TotalTime / 5)
Write-Host "Average time per request: $AvgTime ms" -ForegroundColor Yellow

Write-Host "`n✅ Testing Complete!" -ForegroundColor Green
Write-Host "`nSummary:
--------
✓ Health endpoint (no auth)
✓ Solve endpoint with basic prompt
✓ Solve endpoint with high-stakes flag
✓ Memorize endpoint
✓ Authentication validation
✓ Input validation (empty payload)
✓ Error handling (invalid credentials)
✓ Performance testing"
