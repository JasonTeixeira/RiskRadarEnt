# RiskRadar Enterprise Developer Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [API Development](#api-development)
8. [Database Guidelines](#database-guidelines)
9. [Deployment Process](#deployment-process)
10. [Troubleshooting](#troubleshooting)

## Getting Started

Welcome to RiskRadar Enterprise! This guide will help you set up your development environment and understand our development practices.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - Primary development language
- **Docker Desktop** - Container runtime
- **kubectl** - Kubernetes CLI
- **Poetry** - Python dependency management
- **Git** - Version control
- **PostgreSQL 15** - Database (or use Docker)
- **Redis 7** - Cache (or use Docker)
- **Node.js 18+** - For frontend tools (optional)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/riskradar-enterprise.git
cd riskradar-enterprise

# Install Python dependencies
poetry install

# Set up pre-commit hooks
poetry run pre-commit install

# Copy environment variables
cp .env.example .env
# Edit .env with your local configuration

# Start infrastructure services
docker-compose up -d postgres redis rabbitmq

# Run database migrations
poetry run alembic upgrade head

# Start the development server
poetry run uvicorn app.main:app --reload --port 8000

# Run tests
poetry run pytest
```

## Development Environment Setup

### 1. macOS Setup

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.11 postgresql@15 redis docker kubectl helm k9s

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Linux Setup (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install PostgreSQL 15
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install postgresql-15

# Install Redis
sudo apt install redis-server

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Windows Setup (WSL2)

```powershell
# Install WSL2
wsl --install

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Follow Linux setup instructions inside WSL2
```

### 4. Docker Development Environment

```yaml
# docker-compose.yml for local development
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: riskradar
      POSTGRES_USER: riskradar
      POSTGRES_PASSWORD: localdev123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: riskradar
      RABBITMQ_DEFAULT_PASS: localdev123
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: marketdata
      POSTGRES_USER: riskradar
      POSTGRES_PASSWORD: localdev123
    ports:
      - "5433:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  timescale_data:
```

### 5. Environment Variables

Create a `.env` file in the project root:

```bash
# Application
APP_NAME=RiskRadar-Enterprise
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true

# Database
DATABASE_URL=postgresql://riskradar:localdev123@localhost:5432/riskradar
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10

# RabbitMQ
RABBITMQ_URL=amqp://riskradar:localdev123@localhost:5672/
CELERY_BROKER_URL=${RABBITMQ_URL}
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# AWS (for S3, optional for local dev)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET=riskradar-dev

# Monitoring (optional for local dev)
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=false
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
```

## Project Structure

```
RiskRadar-Enterprise/
├── services/
│   ├── risk-api/           # Main API service
│   │   ├── app/
│   │   │   ├── api/        # API endpoints
│   │   │   ├── core/       # Core functionality
│   │   │   ├── models/     # Database models
│   │   │   ├── schemas/    # Pydantic schemas
│   │   │   ├── services/   # Business logic
│   │   │   └── utils/      # Utilities
│   │   ├── tests/          # Test files
│   │   └── alembic/        # Database migrations
│   ├── market-data/        # Market data service
│   ├── risk-engine/        # Risk calculation engine
│   └── notification/       # Notification service
├── infrastructure/
│   ├── kubernetes/         # K8s manifests
│   ├── terraform/          # IaC configuration
│   └── helm/              # Helm charts
├── .github/
│   └── workflows/         # CI/CD pipelines
├── docs/                  # Documentation
├── scripts/              # Utility scripts
└── docker-compose.yml    # Local development
```

## Development Workflow

### 1. Branch Strategy

We follow GitFlow with the following branches:

- **main** - Production-ready code
- **develop** - Integration branch for features
- **feature/** - Feature development branches
- **hotfix/** - Emergency fixes for production
- **release/** - Release preparation branches

```bash
# Create a feature branch
git checkout develop
git pull origin develop
git checkout -b feature/RISK-123-add-new-metric

# Work on your feature
git add .
git commit -m "feat(risk): add Omega ratio calculation"

# Push and create PR
git push origin feature/RISK-123-add-new-metric
```

### 2. Commit Convention

We follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation
- **style**: Code style changes
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding tests
- **chore**: Maintenance tasks

Examples:
```bash
git commit -m "feat(api): add Monte Carlo simulation endpoint"
git commit -m "fix(auth): resolve JWT token expiration issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "perf(calc): optimize VaR calculation using vectorization"
```

### 3. Pull Request Process

1. Create feature branch from `develop`
2. Write code and tests
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Run tests: `pytest`
5. Update documentation if needed
6. Create pull request to `develop`
7. Request code review
8. Address review feedback
9. Merge after approval

### 4. Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] Error handling is appropriate
- [ ] Logging is sufficient
- [ ] Database migrations are included

## Code Standards

### Python Style Guide

We follow PEP 8 with the following tools:

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "C90", "I", "N", "B", "S"]
ignore = ["E501", "B008"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Code Examples

```python
# Good example
from typing import List, Optional
from decimal import Decimal
from datetime import datetime

from pydantic import BaseModel, Field
from fastapi import HTTPException, status

from app.core.auth import get_current_user
from app.models.portfolio import Portfolio
from app.schemas.risk import RiskMetrics


class PortfolioService:
    """Service for managing portfolios."""
    
    def __init__(self, db_session: Session) -> None:
        """Initialize portfolio service.
        
        Args:
            db_session: Database session
        """
        self.db = db_session
    
    async def calculate_risk(
        self,
        portfolio_id: str,
        metrics: List[str],
        confidence_level: float = 0.95
    ) -> RiskMetrics:
        """Calculate risk metrics for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            metrics: List of metrics to calculate
            confidence_level: Confidence level for VaR/CVaR
            
        Returns:
            Calculated risk metrics
            
        Raises:
            HTTPException: If portfolio not found
        """
        portfolio = await self._get_portfolio(portfolio_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio {portfolio_id} not found"
            )
        
        # Calculate metrics...
        return RiskMetrics(...)
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── performance/   # Performance tests
├── fixtures/      # Test fixtures
└── conftest.py    # Pytest configuration
```

### Writing Tests

```python
# tests/unit/test_risk_calculator.py
import pytest
from decimal import Decimal
from app.services.risk import RiskCalculator


class TestRiskCalculator:
    """Test suite for RiskCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return RiskCalculator()
    
    @pytest.fixture
    def sample_returns(self):
        """Sample return data."""
        return [0.01, -0.02, 0.03, -0.01, 0.02]
    
    def test_calculate_var(self, calculator, sample_returns):
        """Test VaR calculation."""
        var = calculator.calculate_var(
            returns=sample_returns,
            confidence_level=0.95
        )
        assert isinstance(var, Decimal)
        assert var > 0
    
    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_var_confidence_levels(
        self, 
        calculator, 
        sample_returns, 
        confidence_level
    ):
        """Test VaR with different confidence levels."""
        var = calculator.calculate_var(sample_returns, confidence_level)
        assert var > 0
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_risk_calculator.py

# Run tests matching pattern
poetry run pytest -k "test_var"

# Run with verbose output
poetry run pytest -v

# Run integration tests only
poetry run pytest tests/integration/

# Run with parallel execution
poetry run pytest -n auto
```

## API Development

### Creating New Endpoints

```python
# app/api/v1/endpoints/portfolios.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.portfolio import (
    Portfolio, 
    PortfolioCreate, 
    PortfolioUpdate
)
from app.services.portfolio import PortfolioService
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Portfolio, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_in: PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Portfolio:
    """Create a new portfolio.
    
    Args:
        portfolio_in: Portfolio creation data
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Created portfolio
    """
    service = PortfolioService(db)
    portfolio = await service.create(
        data=portfolio_in,
        user_id=current_user.id
    )
    return portfolio


@router.get("/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(
    portfolio_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Portfolio:
    """Get portfolio by ID.
    
    Args:
        portfolio_id: Portfolio identifier
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Portfolio details
        
    Raises:
        HTTPException: If portfolio not found
    """
    service = PortfolioService(db)
    portfolio = await service.get(
        portfolio_id=portfolio_id,
        user_id=current_user.id
    )
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    return portfolio
```

### API Documentation

FastAPI automatically generates OpenAPI documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Database Guidelines

### Migrations with Alembic

```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "Add risk_calculations table"

# Apply migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1

# View migration history
poetry run alembic history

# Show current revision
poetry run alembic current
```

### Database Best Practices

1. **Use UUIDs for primary keys**
```python
from sqlalchemy.dialects.postgresql import UUID
import uuid

id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
```

2. **Add indexes for frequently queried columns**
```python
__table_args__ = (
    Index('idx_portfolio_org', 'organization_id'),
    Index('idx_portfolio_status', 'status'),
)
```

3. **Use proper column types**
```python
from sqlalchemy import Numeric, DateTime

amount = Column(Numeric(20, 4), nullable=False)
created_at = Column(DateTime(timezone=True), server_default=func.now())
```

4. **Implement soft deletes**
```python
deleted_at = Column(DateTime(timezone=True), nullable=True)

def soft_delete(self):
    self.deleted_at = datetime.utcnow()
```

## Deployment Process

### Local Testing

```bash
# Build Docker image
docker build -t riskradar-api:local .

# Run container
docker run -p 8000:8000 --env-file .env riskradar-api:local

# Test with docker-compose
docker-compose up

# Run integration tests against containers
docker-compose run --rm api pytest tests/integration/
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace risk-dev

# Apply configurations
kubectl apply -f infrastructure/kubernetes/configmap.yaml -n risk-dev
kubectl apply -f infrastructure/kubernetes/secrets.yaml -n risk-dev
kubectl apply -f infrastructure/kubernetes/deployment.yaml -n risk-dev
kubectl apply -f infrastructure/kubernetes/service.yaml -n risk-dev

# Check deployment status
kubectl get pods -n risk-dev
kubectl logs -f deployment/risk-api -n risk-dev

# Port forward for testing
kubectl port-forward service/risk-api 8000:80 -n risk-dev
```

### CI/CD Pipeline

Our GitHub Actions pipeline:

1. **On Pull Request**:
   - Run linters (black, ruff, mypy)
   - Run security checks (bandit, safety)
   - Run unit tests with coverage
   - Build Docker image
   - Run integration tests

2. **On Merge to Develop**:
   - Deploy to staging environment
   - Run smoke tests
   - Run performance tests

3. **On Merge to Main**:
   - Deploy to production (blue-green)
   - Run health checks
   - Monitor metrics
   - Rollback if needed

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection string
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check logs
docker-compose logs postgres
```

#### 2. Redis Connection Error

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli ping

# Clear cache if needed
redis-cli FLUSHALL
```

#### 3. Import Errors

```bash
# Reinstall dependencies
poetry install

# Update dependencies
poetry update

# Clear cache
poetry cache clear pypi --all

# Rebuild virtual environment
poetry env remove python
poetry install
```

#### 4. Migration Issues

```bash
# Check current revision
poetry run alembic current

# Show SQL for migration
poetry run alembic upgrade head --sql

# Force revision
poetry run alembic stamp head

# Recreate database
dropdb riskradar && createdb riskradar
poetry run alembic upgrade head
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Or with ipdb
import ipdb; ipdb.set_trace()

# FastAPI debug mode
app = FastAPI(debug=True)

# SQLAlchemy query logging
engine = create_engine(DATABASE_URL, echo=True)
```

### Performance Profiling

```python
# Profile with cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)

# Memory profiling
from memory_profiler import profile

@profile
def calculate_risk():
    # ... code ...
    pass

# Line profiling
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(calculate_risk)
lp.enable()
calculate_risk()
lp.disable()
lp.print_stats()
```

## Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Internal Resources
- [API Documentation](/docs/api/openapi.yaml)
- [Architecture Guide](/docs/architecture/ARCHITECTURE.md)
- [Security Guidelines](/docs/security/SECURITY.md)
- [Deployment Guide](/docs/deployment/DEPLOYMENT.md)

### Support Channels
- **Slack**: #riskradar-dev
- **Email**: dev-team@riskradar.com
- **Wiki**: https://wiki.riskradar.com
- **JIRA**: https://jira.riskradar.com

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License

Copyright © 2024 RiskRadar Enterprise. All rights reserved.
