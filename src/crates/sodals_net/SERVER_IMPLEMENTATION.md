# SODALS REST API Gateway - Implementation Summary

## Overview
This document summarizes the implementation of `server.rs` for the SODALS Edge AI Service REST API Gateway.

## File Location
`src/crates/sodals_net/src/server.rs`

## Architecture

The server implements a complete REST API gateway that:
1. Accepts JSON requests from clients
2. Authenticates requests via `X-API-KEY` header (Middleware)
3. Forwards requests to the Kernel actor via mpsc channel
4. Waits for Kernel responses via broadcast channel with request ID filtering
5. Implements 30-second timeout for Kernel responses
6. Returns structured JSON responses or appropriate HTTP error codes

## Key Components

### 1. Application State
```rust
pub struct AppState {
    pub kernel_tx: mpsc::Sender<SystemMsg>,
    pub broadcast_tx: broadcast::Sender<SystemMsg>,
    pub api_key: String,
    pub config_debug: bool,
}
```
- Holds communication channels to Kernel
- Stores API key from environment variables
- Shared via `Arc<AppState>` across handlers

### 2. Data Transfer Objects (DTOs)

**Request Models:**
- `SolveRequest`: Prompt + optional high_stakes flag, user_id, image_data
- `MemorizeRequest`: Text to store + optional user_id and category

**Response Models:**
- `SolveResponse`: Request ID, response content, processing time, energy estimate, model name
- `MemorizeResponse`: Request ID, status, optional vector dimension
- `HealthResponse`: Service status, version, timestamp
- `ErrorResponse`: HTTP-like error code, message, request ID, timestamp

### 3. Authentication Middleware

**`validate_api_key` middleware:**
- Checks `X-API-KEY` header on all endpoints except `/health`
- Validates against API key from environment or uses default
- Returns `401 Unauthorized` if key is missing/invalid
- Logs failed authentication attempts

### 4. HTTP Endpoints

#### `GET /health`
- **Purpose**: Liveness probe for load balancers
- **Auth**: Not required
- **Response**: 
  ```json
  {
    "status": "operational",
    "version": "0.1.0",
    "timestamp": "2026-02-16T..."
  }
  ```

#### `POST /v1/solve` (Main Inference)
- **Purpose**: Process prompts via Kernel
- **Auth**: Required (`X-API-KEY`)
- **Request**:
  ```json
  {
    "prompt": "What is 2+2?",
    "high_stakes": false,
    "user_id": "user123",
    "image_data": null
  }
  ```
- **Response**:
  ```json
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "response": "2+2 equals 4",
    "processing_time_secs": 1.234,
    "energy": 0.01234,
    "model": "sodals-kernel"
  }
  ```
- **Error Handling**:
  - `400 Bad Request`: Empty prompt
  - `503 Service Unavailable`: Kernel unreachable
  - `504 Gateway Timeout`: No response within 30 seconds
  - `500 Internal Server Error`: Kernel processing error

#### `POST /v1/memorize` (Vector Storage)
- **Purpose**: Store text in vector database
- **Auth**: Required (`X-API-KEY`)
- **Request**:
  ```json
  {
    "text": "Important information to remember",
    "user_id": "user123",
    "category": "notes"
  }
  ```
- **Response**:
  ```json
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "stored",
    "vector_dim": 768
  }
  ```

## Flow Diagram: Request Processing

```
CLIENT REQUEST
    ↓
[Middleware: Validate X-API-KEY]
    ↓
[Generate unique req_id (u64 from UUID)]
    ↓
[Create SystemMsg (LlmRequest, MemorizeRequest, etc.)]
    ↓
[Send to Kernel via mpsc::send()]
    ↓
[Subscribe to broadcast channel]
    ↓
[Loop: Wait for SystemMsg with matching req_id]
    ↓
[Timeout after 30 seconds if no match]
    ↓
[Extract content from CortexResponse]
    ↓
[Return JSON response to client]
```

## Request ID Generation

```rust
let req_id: u64 = rand::thread_rng().gen();
```

- Uses `uuid::Uuid::new_v4()` for unique IDs
- Extracts lower 64 bits for use as `req_id`
- Ensures Kernel responses can be correctly matched

## Broadcast Channel Filtering

The `wait_for_response()` helper function:
1. Subscribes to broadcast channel
2. Loops indefinitely on incoming messages
3. Checks `req_id` field of `SystemMsg::CortexResponse`
4. Returns content if IDs match
5. Ignores other message types and continues waiting
6. Handles lagged/closed broadcast errors

## Error Handling Strategy

| Scenario | HTTP Status | Message |
|----------|------------|---------|
| Invalid API key | 401 | "Invalid or missing X-API-KEY header" |
| Empty prompt | 400 | "Prompt cannot be empty" |
| Kernel unreachable | 503 | "Kernel service unavailable" |
| Timeout (>30s) | 504 | "Kernel did not respond within 30 seconds" |
| Kernel error | 500 | "Kernel error: ..." |
| Broadcast lagged | 500 | "Receiver lagged, message queue overflow" |
| Broadcast closed | 500 | "Broadcast channel closed" |

## Configuration

**Environment Variables:**
- `SODALS_API_KEY`: API key for authentication (default: "sodals-secret-key")

**Hard-coded Defaults:**
- Port: Must be passed to `run_server()`
- Request timeout: 30 seconds
- Max request body size: 10MB
- Vector dimension (memorize): 768

## Deployment

To start the server:

```rust
use sodals_net::server::run_server;
use tokio::sync::{mpsc, broadcast};

#[tokio::main]
async fn main() {
    let (kernel_tx, _) = mpsc::channel(100);
    let (broadcast_tx, _) = broadcast::channel(1000);
    
    run_server(kernel_tx, broadcast_tx, 8080).await;
}
```

**Server Features:**
- CORS enabled (permissive)
- Structured logging via `tracing` crate
- Graceful error responses with timestamps
- Load balancer friendly (/health endpoint)

## Testing

Included unit tests verify:
- `SolveRequest` JSON deserialization
- `SolveResponse` JSON serialization
- `MemorizeRequest` validation

To run tests:
```bash
cargo test -p sodals_net
```

## Security Considerations

1. **API Key Validation**
   - Required for all endpoints except `/health`
   - Should be rotated regularly
   - In production, use environment variables from secrets manager

2. **Request Limits**
   - 10MB max body size prevents memory exhaustion
   - 30s timeout prevents resource starvation

3. **Error Messages**
   - Avoid exposing internal implementation details
   - Return generic messages for critical errors

4. **Future Enhancements**
   - Rate limiting per API key
   - Request signing with HMAC-SHA256
   - Audit logging for compliance
   - TLS/mTLS for secure communication
   - JWT token authentication
   - Request payload encryption

## Dependencies Added

```toml
uuid = { version = "1.0", features = ["v4", "serde"] }
```

Already present:
- axum 0.7 (with WebSocket support)
- tokio (with full features for non-wasm)
- serde/serde_json
- chrono
- tower-http (with CORS)
- tracing

## Integration with Kernel

The server acts as a bridge between HTTP clients and the Kernel actor:

1. **Send Path**: HTTP request → `SystemMsg::LlmRequest` → Kernel mpsc channel
2. **Receive Path**: Kernel response → Broadcast channel → Filter by req_id → HTTP response

The Kernel is expected to:
- Receive `SystemMsg::LlmRequest` with a unique `req_id`
- Process the request
- Send back `SystemMsg::CortexResponse` with the same `req_id`
- Similarly for memorization and other operations

## Production Readiness

For production deployment:
- [ ] Configure SODALS_API_KEY from environment secrets
- [ ] Enable TLS/HTTPS via reverse proxy (nginx/Caddy)
- [ ] Set up monitoring and alerting for API latency
- [ ] Implement rate limiting middleware
- [ ] Add request tracing/correlation IDs
- [ ] Configure structured logging to centralized system
- [ ] Add database backend for persistent API keys
- [ ] Implement admin dashboard for key rotation

## Performance Notes

- **Latency**: Depends on Kernel processing time (typically <5s for LLM)
- **Throughput**: Limited by Kernel capacity and broadcast channel size (currently 1000)
- **Memory**: O(N) where N = concurrent requests waiting on broadcast
- **Broadcast Overflow**: If receiver lags behind sender, channel can overflow (lagged error)

## Future Enhancements

1. **Request Queueing**: Add priority queue for high-stakes requests
2. **Response Caching**: Cache similar prompts
3. **Streaming Responses**: Use Server-Sent Events (SSE) for token streaming
4. **WebSocket Support**: Real-time bidirectional communication
5. **Batch Processing**: Accept multiple requests in one call
6. **Admin Endpoints**: Dashboard, metrics, diagnostics
