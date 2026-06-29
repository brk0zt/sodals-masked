# SODALS API Gateway - Integration Checklist

## Pre-Integration Setup
- [ ] Rust 1.75+ installed (`rustc --version`)
- [ ] Tokio async runtime available
- [ ] All dependencies in `Cargo.toml` (uuid added)
- [ ] `src/crates/sodals_net/src/server.rs` created
- [ ] `server` module exported in `lib.rs`
- [ ] Code compiles without errors (`cargo check`)

## Dependencies Verification
- [x] `axum` (0.7) - Web framework
- [x] `tokio` (1.36+) - Async runtime
- [x] `serde`/`serde_json` (1.0) - Serialization
- [x] `uuid` (1.0) - Request IDs
- [x] `chrono` (0.4) - Timestamps
- [x] `tracing` (0.1) - Logging
- [x] `tower-http` (0.5) - CORS middleware
- [ ] Are all features enabled in Cargo.toml? `features = ["..."]`

## Architecture Integration

### Step 1: Create Communication Channels
```rust
// In your main.rs or kernel initialization
use tokio::sync::{mpsc, broadcast};
use sodals_net::protocol::SystemMsg;

let (kernel_tx, kernel_rx) = mpsc::channel::<SystemMsg>(100);
let (broadcast_tx, _) = broadcast::channel::<SystemMsg>(1000);
```
- [ ] mpsc channel created for Kernel input
- [ ] broadcast channel created for Kernel output
- [ ] Channels are cloneable and Arc-wrapped where needed

### Step 2: Spawn Kernel Actor
```rust
// Your kernel should:
// 1. Receive SystemMsg from mpsc
// 2. Process the message
// 3. Send response via broadcast
```
- [ ] Kernel receives `SystemMsg::LlmRequest`
- [ ] Kernel processes and generates response
- [ ] Kernel sends `SystemMsg::CortexResponse` with matching `req_id`
- [ ] Kernel receives `SystemMsg::MemorizeRequest`
- [ ] Kernel stores to vector database

### Step 3: Start API Server
```rust
use sodals_net::server::run_server;

let server_handle = tokio::spawn(async move {
    run_server(kernel_tx, broadcast_tx, 8080).await;
});
```
- [ ] `run_server()` called with correct parameters
- [ ] Server is spawned on separate Tokio task
- [ ] Port is available (not in use)
- [ ] Server logs appear in tracing output

## Configuration

### API Key Setup
- [ ] Environment variable `SODALS_API_KEY` defined
- [ ] API key is secure (32+ characters)
- [ ] Default fallback exists for development
- [ ] API key is validated in middleware

### Logging Configuration
```rust
tracing_subscriber::fmt()
    .with_target(true)
    .with_thread_ids(true)
    .with_level(true)
    .init();
```
- [ ] Logging initialized before server starts
- [ ] Log level set appropriately (info/debug)
- [ ] Tracing works for all components

### Environment Variables
- [x] `SODALS_API_KEY` - API key
- [ ] `RUST_LOG` - Log level (optional)
- [ ] Any custom variables needed?

## Endpoint Testing

### Health Endpoint
```bash
curl -X GET http://localhost:8080/health
```
- [ ] Returns 200 OK
- [ ] Response includes status, version, timestamp
- [ ] No authentication required
- [ ] Load balancer compatible

### Solve Endpoint
```bash
curl -X POST http://localhost:8080/v1/solve \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'
```
- [ ] Returns 200 OK on success
- [ ] Request ID in response
- [ ] Processing time calculated
- [ ] Response contains model name
- [ ] Request routed to Kernel
- [ ] Response filtered by req_id
- [ ] Timeout works (504 after 30s)
- [ ] API key validation works (401 without key)
- [ ] Empty prompt rejected (400)

### Memorize Endpoint
```bash
curl -X POST http://localhost:8080/v1/memorize \
  -H "X-API-KEY: sodals-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'
```
- [ ] Returns 200 OK on success
- [ ] Request ID in response
- [ ] Vector dimension included
- [ ] Text stored in Kernel/vector DB
- [ ] User ID tracking works
- [ ] Empty text rejected (400)

## Error Handling

Test each error scenario:

| Error Type | HTTP Code | Testing |
|-----------|-----------|---------|
| Missing API Key | 401 | [ ] Tested |
| Invalid API Key | 401 | [ ] Tested |
| Empty prompt | 400 | [ ] Tested |
| Empty text | 400 | [ ] Tested |
| Kernel unavailable | 503 | [ ] Tested |
| Timeout (>30s) | 504 | [ ] Tested |
| Kernel error | 500 | [ ] Tested |

## Performance Testing

- [ ] Single request latency measured < 5s (typical)
- [ ] 10 sequential requests processed correctly
- [ ] 5 concurrent requests handled (ab or hey)
- [ ] Memory usage stable over time
- [ ] No resource leaks detected
- [ ] Error recovery works (after exception)

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 100 -H "X-API-KEY: sodals-secret-key" http://localhost:8080/health

# Using hey
hey -n 1000 -c 100 -H "X-API-KEY: sodals-secret-key" http://localhost:8080/health
```
- [ ] 1000 requests @ 100 concurrency passes
- [ ] Server remains responsive
- [ ] No connection errors
- [ ] Response times consistent

## Security Checklist

- [ ] API key is not hardcoded
- [ ] X-API-KEY header validated on protected endpoints
- [ ] CORS headers configured (permissive for testing)
- [ ] Request body size limited (10MB)
- [ ] Input validation on prompt/text fields
- [ ] Error messages don't leak sensitive info
- [ ] Request IDs are unique (UUID-based)
- [ ] Broadcast channel doesn't lag (no lost messages)
- [ ] No authorization bypass possible
- [ ] Rate limiting available (nginx/reverse proxy)

## Deployment Readiness

### Docker
- [ ] Dockerfile created
- [ ] Healthcheck configured
- [ ] Non-root user enabled
- [ ] Image builds successfully
- [ ] Container starts without errors
- [ ] Port exposed correctly
- [ ] Environment variables injectable

### Reverse Proxy (Nginx/Caddy)
- [ ] TLS/HTTPS configured
- [ ] Rate limiting rules set
- [ ] Security headers added
- [ ] Proxy passes to correct backend
- [ ] Health check proxied correctly
- [ ] Request timeout configured (>30s)

### Kubernetes
- [ ] Deployment manifest created
- [ ] Service configured
- [ ] HPA configured if needed
- [ ] Liveness probe working
- [ ] Readiness probe working
- [ ] Secrets injected correctly
- [ ] StatefulSet unnecessary (stateless design)

## Documentation

- [ ] [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) - Client guide
- [ ] [SERVER_IMPLEMENTATION.md](SERVER_IMPLEMENTATION.md) - Architecture
- [ ] [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Deployment guide
- [ ] [examples/run_api_gateway.rs](examples/run_api_gateway.rs) - Working example
- [ ] [tests/test_api.sh](tests/test_api.sh) - Test suite (Unix/Linux)
- [ ] [tests/test_api.ps1](tests/test_api.ps1) - Test suite (Windows)

## Monitoring & Observability

- [ ] Logging configured (structured logs)
- [ ] Key metrics tracked (latency, errors, throughput)
- [ ] Health endpoint monitored by load balancer
- [ ] Error rates tracked
- [ ] Request IDs propagated in logs
- [ ] Performance trends monitored
- [ ] Alert thresholds defined

## Integration Testing

```rust
#[tokio::test]
async fn test_end_to_end() {
    // Setup channels
    let (kernel_tx, _) = mpsc::channel(100);
    let (broadcast_tx, _) = broadcast::channel(1000);
    
    // Start server
    let server = run_server(kernel_tx, broadcast_tx, 8081).await;
    
    // Make request
    let response = http_client.post("/v1/solve").send().await;
    
    // Verify response
    assert_eq!(response.status(), 200);
}
```
- [ ] Integration test written
- [ ] Full request-response cycle tested
- [ ] Test passes consistently
- [ ] Error scenarios tested

## Production Checklist

- [ ] API key rotated from default
- [ ] TLS/HTTPS enabled
- [ ] Rate limiting configured
- [ ] Monitoring/alerting set up
- [ ] Logging centralized
- [ ] Backup/recovery plan documented
- [ ] Incident response plan ready
- [ ] Load testing completed
- [ ] Security audit performed
- [ ] Performance baselines established
- [ ] Rollback plan documented
- [ ] Runbooks created for ops team

## Final Verification

- [ ] All tests pass (`cargo test`)
- [ ] No warnings in `cargo check` (related to server.rs)
- [ ] Code review approved
- [ ] Documentation complete
- [ ] Performance acceptable
- [ ] Security requirements met
- [ ] Ready for production deployment

## Handoff Checklist

- [ ] Source code in version control
- [ ] All dependencies documented
- [ ] Deployment procedures documented
- [ ] Monitoring configured
- [ ] Emergency contacts list updated
- [ ] On-call procedures documented
- [ ] Training completed for ops team
- [ ] Support plan established

---

## Next Phase: Future Enhancements

- [ ] Add request caching for similar prompts
- [ ] Implement WebSocket support for streaming
- [ ] Add batch processing endpoint
- [ ] Implement request queuing/prioritization
- [ ] Add admin dashboard
- [ ] Integration with async streaming responses
- [ ] JWT token authentication
- [ ] Request signing with HMAC
- [ ] Multi-tenant support
- [ ] API rate limiting per key

---

**Completion Date**: _________________

**Tested By**: _________________

**Approved By**: _________________

**Notes/Issues**:
```
[Space for notes]
```
