# Docker Deployment for SODALS API Gateway

## Production-Ready Docker Setup

### Dockerfile
```dockerfile
# Multi-stage build for optimal image size
FROM rust:1.75 as builder

WORKDIR /app

# Copy workspace
COPY . .

# Build only the sodals_net crate with release optimizations
RUN cargo build --release -p sodals_net \
    --no-default-features \
    --features "full" 2>&1 | grep -E "error|warning" || true

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/sodals_net /usr/local/bin/

# Create non-root user for security
RUN useradd -m -u 1000 sodals && \
    chown -R sodals:sodals /app

USER sodals

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV SODALS_API_KEY=change-in-production
ENV RUST_LOG=sodals_net=info,axum=debug

# Expose API port
EXPOSE 8080

# Run the server
CMD ["sodals_net"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  # SODALS API Gateway
  api-gateway:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: sodals-api-gateway
    ports:
      - "8080:8080"
    environment:
      SODALS_API_KEY: ${SODALS_API_KEY:-sodals-secret-key}
      RUST_LOG: sodals_net=info,axum=debug
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    networks:
      - sodals-network
    restart: unless-stopped
    labels:
      - "com.sodals.service=api-gateway"
      - "com.sodals.version=0.1.0"

  # Optional: Nginx reverse proxy with TLS
  nginx:
    image: nginx:alpine
    container_name: sodals-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api-gateway
    networks:
      - sodals-network
    restart: unless-stopped

  # Optional: Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: sodals-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - sodals-network
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

networks:
  sodals-network:
    driver: bridge

volumes:
  prometheus-data:
```

### Environment Variables (.env)
```bash
# Production Configuration
SODALS_API_KEY=your-secure-api-key-here-change-this
RUST_LOG=sodals_net=info,axum=warn
RUST_BACKTRACE=1

# Kernel Configuration (if in same container)
KERNEL_CAPACITY=100
BROADCAST_CAPACITY=1000

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_TRACES_EXPORTER=otlp
```

### nginx.conf (Reverse Proxy with TLS)
```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req_zone $http_x_api_key zone=key_limit:10m rate=1000r/s;

    # Upstream backend
    upstream sodals_backend {
        server api-gateway:8080;
        keepalive 32;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name _;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location / {
            return 301 https://$host$request_uri;
        }
    }

    # HTTPS Server
    server {
        listen 443 ssl http2;
        server_name sodals.example.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/certs/cert.pem;
        ssl_certificate_key /etc/nginx/certs/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security Headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # API Gateway
        location /health {
            limit_req zone=api_limit burst=10 nodelay;
            proxy_pass http://sodals_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            access_log off;
        }

        location /v1/ {
            limit_req zone=key_limit burst=50 nodelay;
            
            # API Key validation
            if ($http_x_api_key = "") {
                return 401;
            }

            proxy_pass http://sodals_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 35s;
        }

        # Metrics (internal only)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            proxy_pass http://sodals_backend;
        }
    }
}
```

## Deployment Instructions

### Option 1: Docker Compose (Quick Start)
```bash
# Clone or navigate to project
cd sodals_core

# Set environment variables
export SODALS_API_KEY="your-secure-key-here"

# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f api-gateway

# Test the API
curl -X GET http://localhost:8080/health
curl -X GET https://sodals.example.com/health
```

### Option 2: Kubernetes Deployment

Create `sodals-api-gateway-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sodals-api-gateway
  labels:
    app: sodals-api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sodals-api-gateway
  template:
    metadata:
      labels:
        app: sodals-api-gateway
    spec:
      containers:
      - name: api-gateway
        image: sodals/api-gateway:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: api
        env:
        - name: SODALS_API_KEY
          valueFrom:
            secretKeyRef:
              name: sodals-secrets
              key: api-key
        - name: RUST_LOG
          value: "sodals_net=info,axum=warn"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2

---
apiVersion: v1
kind: Service
metadata:
  name: sodals-api-gateway-service
spec:
  selector:
    app: sodals-api-gateway
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sodals-api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sodals-api-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Deploy to Kubernetes:
```bash
# Create secret
kubectl create secret generic sodals-secrets \
  --from-literal=api-key=your-secure-key-here

# Deploy
kubectl apply -f sodals-api-gateway-deployment.yaml

# Check status
kubectl get pods -l app=sodals-api-gateway
kubectl logs -l app=sodals-api-gateway -f
```

## Monitoring & Observability

### Health Check
```bash
curl -X GET https://sodals.example.com/health \
  -H "Authorization: Bearer monitoring-token"
```

### Metrics Endpoint
```bash
# Access Prometheus metrics (internal only)
curl -X GET http://localhost:9090/api/v1/series?match[]=sodals_*
```

### Log Aggregation
```bash
# Ship logs to centralized system
docker-compose logs api-gateway | logstash-forwarder -c logstash.conf
```

## Security Best Practices

1. ✅ Use strong API keys (32+ chars, alphanumeric + special)
2. ✅ Rotate API keys regularly
3. ✅ Enable TLS/HTTPS (use LetsEncrypt for free certs)
4. ✅ Implement rate limiting
5. ✅ Use non-root user in container
6. ✅ Enable security headers (HSTS, CSP, etc.)
7. ✅ Monitor failed authentication attempts
8. ✅ Use network isolation (private subnets, VPC)
9. ✅ Regular security audits
10. ✅ Keep dependencies updated

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs sodals-api-gateway

# Check configuration
docker inspect sodals-api-gateway

# Verify image
docker run -it --rm sodals/api-gateway:latest /bin/bash
```

### High latency
```bash
# Check Kernel capacity
# Increase BROADCAST_CAPACITY if lagging

# Monitor container resources
docker stats sodals-api-gateway

# Consider horizontal scaling
docker-compose up -d --scale api-gateway=3
```

### API Key issues
```bash
# Verify key is set
docker exec sodals-api-gateway env | grep SODALS_API_KEY

# Test with curl
curl -v -H "X-API-KEY: your-key" http://localhost:8080/health
```

## Performance Optimization

- Set appropriate worker threads: `RAYON_NUM_THREADS=8`
- Tune broadcast channel capacity based on load
- Use connection pooling in clients
- Enable HTTP/2 multiplexing
- Consider adding Redis cache layer
- Implement request batching
