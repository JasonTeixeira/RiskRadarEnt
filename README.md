# RiskRadar Enterprise

**Experimental microservices rewrite of RiskRadar**

```
┌──────────────────────────────────────────────┐
│  WHAT THIS IS                                │
├──────────────────────────────────────────────┤
│  • Microservices architecture exploration   │
│  • Event-driven risk calculations (Kafka)  │
│  • Kubernetes deployment configs            │
│  • Learning project, not production-ready   │
│  • Based on RiskRadar (simpler version)     │
└──────────────────────────────────────────────┘
```

## Why I Built This

After building **RiskRadar** (the monolithic FastAPI + Next.js version), I wanted to learn microservices patterns. I'd read about event-driven architecture, Kafka, Kubernetes, and all the "enterprise" buzzwords, but never actually implemented them.

This repo is my experiment in decomposing the RiskRadar monolith into separate services. Instead of one big Python app, I split it into:
- Risk API service (HTTP/WebSocket)
- Risk compute workers (async calculations)
- Data ingestion service (market data collection)
- Portfolio manager service
- Event publisher (Kafka)

The goal wasn't to make something better—RiskRadar works fine as a monolith. The goal was to learn distributed systems patterns and see where the complexity comes from.

## Architecture

```
┌──────────────────┐
│  API Gateway     │  Kong/Envoy (not implemented yet)
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌──────────┐
│Risk API  │ │Portfolio │
│Service   │ │Manager   │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
┌─────────────────────┐
│   Kafka/Redpanda    │  Event bus (not running locally)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Risk Compute Worker │  Celery workers (stubbed out)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   TimescaleDB       │  Time-series cost data
└─────────────────────┘
```

## What's Inside

**Services (`/services`):**
- `risk-api/` - FastAPI service for risk calculations (skeleton, no real logic)
- `risk-compute-worker/` - Celery workers for async tasks (mostly empty)
- `data-ingestion/` - Market data collectors (stub)
- `portfolio-manager/` - Portfolio CRUD (not implemented)

**Infrastructure (`/infrastructure`):**
- `kubernetes/` - K8s deployments, services, ingress configs
- `terraform/` - Basic AWS/GCP IaC (incomplete)
- `helm/` - Helm charts for service deployment (minimal)

**Shared libs (`/libs`):**
- `python/` - Shared Python packages (schemas, utilities)
- `proto/` - Protocol buffers for inter-service communication (not used yet)

**Monitoring (`/monitoring`):**
- Prometheus configs
- Grafana dashboards (empty templates)

## What Works

**Kubernetes manifests:** The deployments, services, and config files are syntactically correct. You can `kubectl apply` them and they won't error out (though the services won't do anything useful).

**Makefile:** There's a comprehensive Makefile with commands for building, testing, deploying. Most commands work (though they're operating on stub code).

**Project structure:** The directory layout follows microservices conventions. It looks professional.

## What Doesn't Work

**Everything else.** This is essentially a skeleton with no business logic.

**No Kafka:** I set up Kafka locally once, but it's resource-heavy (1GB+ RAM) and I never hooked it up to the services. Event-driven architecture is purely theoretical here.

**No API Gateway:** The architecture diagram shows Kong/Envoy, but there's no actual gateway. Services would need to call each other directly.

**No real services:** The `risk-api` service has FastAPI boilerplate, but the endpoints return mocked data. The compute workers don't compute anything. The portfolio manager doesn't manage portfolios.

**No tests:** Despite having a `tests/` directory with integration/performance/chaos subdirectories, there are no tests. The structure exists, the tests don't.

**Fake performance metrics:** The old README claimed "10,000 req/s", "100,000+ concurrent portfolios", "99.99% availability SLA". All of that is fiction. This has never run at scale.

**Compliance nonsense:** References to "SOC 2 Type II", "PCI DSS Level 1", "ISO 27001" are aspirational at best, completely made up at worst.

## What I Learned

**Microservices add overhead:** Breaking a monolith into services introduces network latency, serialization costs, deployment complexity, and operational burden. For a side project with one user (me), it's absurd overkill.

**Kafka is heavy:** Running Kafka/Zookeeper locally for development consumes significant resources. For event-driven patterns, a simple message queue (Redis Streams, RabbitMQ) would've been smarter.

**Kubernetes locally is painful:** Minikube/K3s work, but debugging services across multiple pods is slow. Docker Compose would've been enough for learning.

**"Enterprise" patterns aren't free:** Every service needs its own DB connection, config management, error handling, logging, monitoring. The cognitive overhead grew fast.

## What I'd Do Differently

**Skip it entirely:** The monolithic RiskRadar does what I need. This experiment taught me that microservices make sense at scale (large teams, independent deployments, high traffic), but not for personal projects.

**Use Docker Compose:** If I still wanted to explore service decomposition, I'd use Docker Compose instead of Kubernetes. Way simpler for local dev.

**Start with a modular monolith:** Instead of jumping to microservices, I'd refactor RiskRadar into separate modules with clean boundaries. Get the benefits of separation without the deployment complexity.

**Actually implement one service:** I should've fully built the risk calculation service before adding more services. Spreading effort across multiple half-baked services was a waste.

## Quick Start

**Prerequisites:**
- Docker + Docker Compose
- Python 3.11+
- kubectl (if you want to try the K8s configs)

**Local "development":**
```bash
git clone https://github.com/JasonTeixeira/RiskRadarEnt.git
cd RiskRadarEnt

# Start infrastructure (Postgres, Redis - no Kafka)
make infra-up

# "Start" services (they'll run but do nothing useful)
make services-up
```

Most commands in the Makefile will execute without errors, but the services return mock data or do nothing.

## Current Status

**Not deployed anywhere.** **Not used.** **Not maintained.**

This is a learning artifact. I keep it around as a reminder of what over-engineering looks like and to reference the Kubernetes configs when I need them for actual projects.

If you're evaluating my work, look at **RiskRadar** instead—that's the functional version. This is the "what if I made it unnecessarily complicated" version.

---

**Built with:** FastAPI, Kubernetes, Kafka (theoretically), Helm, Terraform  
**Started:** October 2024  
**Status:** Abandoned experiment  
**Lesson learned:** Not everything needs microservices
