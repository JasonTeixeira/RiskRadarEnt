# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Environment

### Core Technologies
- Python 3.11+
- Docker 24.0+
- Docker Compose 2.20+
- Node.js 18+ (for dashboard)
- kubectl 1.28+ (for Kubernetes deployment)

### Service Architecture
RiskRadar Enterprise is a distributed system built with:
- FastAPI-based microservices
- Event-driven processing via Kafka/Redpanda
- TimescaleDB for time-series data
- Redis for caching
- Kubernetes for orchestration

## Common Development Commands

### Environment Setup
```bash
# Install dependencies and create virtual environment
make install

# Start infrastructure services (Kafka, TimescaleDB, Redis)
make infra-up

# Run database migrations
make db-migrate

# Start application services
make services-up
```

### Testing
```bash
# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Run performance tests
make test-performance

# Run all tests with coverage
make test-coverage
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Run linters (ruff + mypy)
make lint

# Run type checking
make typecheck

# Run security scans
make security-scan
```

### Service Management
```bash
# Build all services
make build

# Build specific service
make build-service SERVICE=risk-api

# View service logs
make services-logs

# Restart services
make services-restart
```

### Monitoring
```bash
# Start monitoring stack
make monitoring-up

Access points:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686
```

## Project Structure Overview

```
RiskRadar-Enterprise/
├── services/              # Microservices
│   ├── risk-api/         # Risk calculation API
│   ├── risk-compute-worker/  # Async workers
│   ├── data-ingestion/   # Data collectors
├── libs/                 # Shared libraries
│   ├── python/          # Python packages
│   ├── proto/           # Protocol buffers
│   ├── schemas/         # Data schemas
├── infrastructure/       # IaC and deployment
│   ├── terraform/       # Cloud resources
│   ├── kubernetes/      # K8s manifests
│   ├── helm/            # Helm charts
├── monitoring/          # Observability setup
└── tests/               # Test suites
```

## Key Development Guidelines

1. **Service Independence**: Each service should be independently deployable and maintainable.

2. **Event-Driven Architecture**: Use Kafka/Redpanda for inter-service communication and event streaming.

3. **Type Safety**: All Python code must use type hints and pass mypy --strict.

4. **Testing**: Maintain high test coverage with unit, integration, and performance tests.

5. **Observability**: Include metrics, traces, and structured logging in all services.

## Performance Expectations

- Risk Calculation: 45ms p50, 120ms p99
- Portfolio Update: 15ms p50, 50ms p99
- Market Data Fetch: 100ms p50, 300ms p99
- Event Processing: 5ms p50, 15ms p99

## Documentation Standards

Critical documentation is maintained in:
- [Architecture Guide](docs/architecture/README.md)
- [API Reference](docs/api/README.md)
- [Deployment Guide](docs/deployment/README.md)
- [Runbooks](docs/runbooks/README.md)
- [ADRs](docs/adr/README.md)
