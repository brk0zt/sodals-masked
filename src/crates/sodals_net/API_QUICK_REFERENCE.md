# SODALS API Gateway - Quick Reference

## 🚀 Server Setup

### Start the Server
```bash
# In your main.rs or integration:
use sodals_net::server::run_server;
use tokio::sync::{mpsc, broadcast};

#[tokio::main]
async fn main() {
    let (kernel_tx, _) = mpsc::channel(100);
    let (broadcast_tx, _) = broadcast::channel(1000);
    
    run_server(kernel_tx, broadcast_tx, 8080).await;
}
```

### Environment Configuration
```bash
# Set API key (default: sodals-secret-key)
export SODALS_API_KEY=your-secure-key-here
```

---

## 📡 API Endpoints

### 1️⃣ Health Check
Lightweight probe for load balancers

```bash
# Request
GET /health

# Response (200 OK)
{
  "status": "operational",
  "version": "0.1.0",
  "timestamp": "2026-02-16T12:34:56.789Z"
}
```

**cURL:**
```bash
curl -X GET http://localhost:8080/health
```

---

### 2️⃣ Solve (Main Inference)
Process prompts through the Kernel

```bash
# Request
POST /v1/solve
X-API-KEY: sodals-secret-key
Content-Type: application/json

{
  "prompt": "What is 2+2?",
  "high_stakes": false,
  "user_id": "optional_user_id",
  "image_data": null
}

# Response (200 OK)
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "2+2 equals 4",
  "processing_time_secs": 1.234,
  "energy": 0.01234,
  "model": "sodals-kernel"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "high_stakes": false
  }'
```

**Python:**
```python
import requests

url = "http://localhost:8080/v1/solve"
headers = {
    "X-API-KEY": "sodals-secret-key",
    "Content-Type": "application/json"
}
data = {
    "prompt": "What is 2+2?",
    "high_stakes": False
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

**JavaScript/Node.js:**
```javascript
const fetch = require('node-fetch');

const url = 'http://localhost:8080/v1/solve';
const options = {
    method: 'POST',
    headers: {
        'X-API-KEY': 'sodals-secret-key',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        prompt: 'What is 2+2?',
        high_stakes: false
    })
};

const response = await fetch(url, options);
const data = await response.json();
console.log(data);
```

---

### 3️⃣ Memorize (Vector Storage)
Store text in the vector database

```bash
# Request
POST /v1/memorize
X-API-KEY: sodals-secret-key
Content-Type: application/json

{
  "text": "Important information to remember",
  "user_id": "optional_user_id",
  "category": "optional_category"
}

# Response (200 OK)
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "stored",
  "vector_dim": 768
}
```

**cURL:**
```bash
curl -X POST http://localhost:8080/v1/memorize \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Important information",
    "user_id": "user123"
  }'
```

---

## 🔐 Authentication

All endpoints except `/health` require the `X-API-KEY` header.

```bash
# Correct
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'

# Wrong (will return 401)
curl -X POST http://localhost:8080/v1/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'
```

---

## ⚡ Request Parameters

### POST /v1/solve
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | - | The query/prompt to process |
| `high_stakes` | boolean | ❌ | false | Triggers special inference mode (adds "SOLVE:" prefix) |
| `user_id` | string | ❌ | "anonymous" | User identifier for tracking |
| `image_data` | string \| null | ❌ | null | Base64-encoded image data (optional) |

### POST /v1/memorize
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | ✅ | - | Text to store in vector database |
| `user_id` | string | ❌ | "anonymous" | User identifier |
| `category` | string | ❌ | null | Optional categorization/tag |

---

## 📊 Response Status Codes

| Code | Meaning | Scenario |
|------|---------|----------|
| `200` | OK | Request processed successfully |
| `400` | Bad Request | Empty prompt, missing required fields |
| `401` | Unauthorized | Missing or invalid X-API-KEY |
| `503` | Service Unavailable | Kernel unreachable |
| `504` | Gateway Timeout | Kernel didn't respond within 30 seconds |
| `500` | Internal Server Error | Kernel processing error, broadcast issues |

---

## ❌ Error Response Format

```json
{
  "code": 400,
  "message": "Prompt cannot be empty",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-16T12:34:56.789Z"
}
```

---

## ⏱️ Timeouts & Limits

| Setting | Value | Note |
|---------|-------|------|
| Kernel Response Timeout | 30 seconds | Returns 504 if exceeded |
| Max Request Body Size | 10 MB | Prevents memory exhaustion |
| API Key Length | Any | String from environment variable |

---

## 🎯 Common Use Cases

### Use Case 1: Simple Q&A
```bash
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms"
  }'
```

### Use Case 2: Complex Problem Solving
```bash
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Prove the Pythagorean theorem",
    "high_stakes": true
  }'
```

### Use Case 3: Knowledge Base Building
```bash
# Store article
curl -X POST http://localhost:8080/v1/memorize \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SODALS uses edge AI for real-time decision making",
    "category": "product_info"
  }'
```

### Use Case 4: User-Scoped Operations
```bash
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Based on my conversation history, recommend next steps",
    "user_id": "user_12345"
  }'
```

---

## 📈 Performance Tips

1. **API Key**: Use environment variables, don't hardcode
2. **Reuse Connections**: Use HTTP keep-alive / connection pooling
3. **Batch Requests**: Send multiple requests in parallel (within rate limits)
4. **Cache Results**: Cache similar prompts to avoid re-processing
5. **Monitor Latency**: Track `processing_time_secs` to identify bottlenecks

---

## 🐛 Debugging

### Enable Verbose Logging
```rust
// In your main.rs
tracing_subscriber::fmt()
    .with_max_level(Level::DEBUG)
    .init();
```

### Test Endpoint Reachability
```bash
# Test if server is running
curl -v http://localhost:8080/health

# Check response headers
curl -i http://localhost:8080/health
```

### Monitor Request Flow
```bash
# Watch server logs in real-time
cargo run | grep -E "\[SOLVE\]|\[MEMORIZE\]|\[SERVER\]"
```

---

## 🔄 Integration Example

```rust
// Integration with your Kernel
use sodals_net::server::run_server;
use tokio::sync::{mpsc, broadcast};

#[tokio::main]
async fn main() {
    // Create channels
    let (kernel_tx, mut kernel_rx) = mpsc::channel::<SystemMsg>(100);
    let (broadcast_tx, _) = broadcast::channel::<SystemMsg>(1000);

    // Spawn server
    let broadcast_tx_clone = broadcast_tx.clone();
    tokio::spawn(async move {
        run_server(kernel_tx, broadcast_tx_clone, 8080).await;
    });

    // Your kernel actor
    while let Some(msg) = kernel_rx.recv().await {
        // Process message
        let response = handle_message(msg).await;
        
        // Send response back
        let _ = broadcast_tx.send(response);
    }
}
```

---

## 📚 Related Files

- **Implementation**: `src/crates/sodals_net/src/server.rs`
- **Example**: `src/crates/sodals_net/examples/run_api_gateway.rs`
- **Test Script**: `src/crates/sodals_net/tests/test_api.sh` (Unix/Linux)
- **Test Script**: `src/crates/sodals_net/tests/test_api.ps1` (Windows/PowerShell)
- **Documentation**: `src/crates/sodals_net/SERVER_IMPLEMENTATION.md`
- **Protocol**: `src/crates/sodals_net/src/protocol.rs`

---

## 🚀 Next Steps

1. ✅ Start the server with `run_server()`
2. ✅ Test with `/health` endpoint (no auth)
3. ✅ Set `SODALS_API_KEY` environment variable
4. ✅ Test `/v1/solve` with valid API key
5. ✅ Monitor logs for request flow
6. ✅ Deploy behind reverse proxy (nginx/Caddy)
7. ✅ Enable TLS/HTTPS
8. ✅ Configure rate limiting
9. ✅ Set up monitoring/alerting
