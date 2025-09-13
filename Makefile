# RiskRadar Enterprise Makefile
.PHONY: help install test lint format clean build deploy

# Variables
PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
HELM := helm
TERRAFORM := terraform

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(GREEN)RiskRadar Enterprise - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-30s$(NC) %s\n", $$1, $$2}'

# Environment Setup
install: ## Install all dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-hooks: ## Install git hooks
	@echo "$(GREEN)Installing git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Git hooks installed!$(NC)"

# Development
dev: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment is running!$(NC)"

dev-down: ## Stop development environment
	@echo "$(YELLOW)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml down
	@echo "$(GREEN)Development environment stopped!$(NC)"

dev-logs: ## Show development logs
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f

# Infrastructure
infra-up: ## Start infrastructure services
	@echo "$(GREEN)Starting infrastructure services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.infra.yml up -d
	@echo "$(GREEN)Infrastructure services are running!$(NC)"

infra-down: ## Stop infrastructure services
	@echo "$(YELLOW)Stopping infrastructure services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.infra.yml down
	@echo "$(GREEN)Infrastructure services stopped!$(NC)"

infra-status: ## Check infrastructure status
	$(DOCKER_COMPOSE) -f docker-compose.infra.yml ps

# Services
services-up: ## Start all application services
	@echo "$(GREEN)Starting application services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Application services are running!$(NC)"

services-down: ## Stop all application services
	@echo "$(YELLOW)Stopping application services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Application services stopped!$(NC)"

services-restart: ## Restart all application services
	@echo "$(YELLOW)Restarting application services...$(NC)"
	$(DOCKER_COMPOSE) restart
	@echo "$(GREEN)Application services restarted!$(NC)"

services-logs: ## Show service logs
	$(DOCKER_COMPOSE) logs -f

# Database
db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	$(PYTHON_VENV) scripts/migration/migrate.py up
	@echo "$(GREEN)Migrations completed!$(NC)"

db-rollback: ## Rollback database migrations
	@echo "$(YELLOW)Rolling back database migrations...$(NC)"
	$(PYTHON_VENV) scripts/migration/migrate.py down
	@echo "$(GREEN)Rollback completed!$(NC)"

db-seed: ## Seed database with test data
	@echo "$(GREEN)Seeding database...$(NC)"
	$(PYTHON_VENV) scripts/migration/seed.py
	@echo "$(GREEN)Database seeded!$(NC)"

# Testing
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ -v --cov=services --cov-report=html
	@echo "$(GREEN)Tests completed!$(NC)"

test-unit: ## Run unit tests
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTHON_VENV) -m pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTHON_VENV) -m pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(PYTHON_VENV) -m locust -f tests/performance/load/locustfile.py --headless -u 100 -r 10 -t 60s
	@echo "$(GREEN)Performance tests completed!$(NC)"

test-coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ --cov=services --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# Code Quality
lint: ## Run linters
	@echo "$(GREEN)Running linters...$(NC)"
	$(PYTHON_VENV) -m ruff check services/ libs/python/
	$(PYTHON_VENV) -m mypy services/ libs/python/
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	$(PYTHON_VENV) -m black services/ libs/python/ tests/
	$(PYTHON_VENV) -m isort services/ libs/python/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

typecheck: ## Run type checking
	@echo "$(GREEN)Running type checks...$(NC)"
	$(PYTHON_VENV) -m mypy services/ libs/python/ --strict
	@echo "$(GREEN)Type checking completed!$(NC)"

security-scan: ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	$(PYTHON_VENV) -m bandit -r services/ libs/python/
	$(PYTHON_VENV) -m safety check
	trivy fs .
	@echo "$(GREEN)Security scan completed!$(NC)"

# Build
build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build --parallel
	@echo "$(GREEN)Docker images built!$(NC)"

build-service: ## Build specific service (usage: make build-service SERVICE=risk-api)
	@echo "$(GREEN)Building $(SERVICE) Docker image...$(NC)"
	docker build -t riskradar/$(SERVICE):latest ./services/$(SERVICE)
	@echo "$(GREEN)$(SERVICE) image built!$(NC)"

# Deployment
deploy-local: ## Deploy to local Kubernetes
	@echo "$(GREEN)Deploying to local Kubernetes...$(NC)"
	$(KUBECTL) apply -k infrastructure/kubernetes/overlays/local
	@echo "$(GREEN)Local deployment completed!$(NC)"

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	$(HELM) upgrade --install riskradar ./infrastructure/helm/charts/riskradar \
		--namespace staging \
		--values ./infrastructure/helm/values/staging.yaml
	@echo "$(GREEN)Staging deployment completed!$(NC)"

deploy-production: ## Deploy to production environment
	@echo "$(RED)Deploying to PRODUCTION...$(NC)"
	@read -p "Are you sure? (y/N) " confirm && [ $$confirm = y ]
	$(HELM) upgrade --install riskradar ./infrastructure/helm/charts/riskradar \
		--namespace production \
		--values ./infrastructure/helm/values/production.yaml
	@echo "$(GREEN)Production deployment completed!$(NC)"

rollback: ## Rollback deployment
	@echo "$(YELLOW)Rolling back deployment...$(NC)"
	$(HELM) rollback riskradar
	@echo "$(GREEN)Rollback completed!$(NC)"

# Monitoring
monitoring-up: ## Start monitoring stack
	@echo "$(GREEN)Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Monitoring stack is running!$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"

monitoring-down: ## Stop monitoring stack
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml down
	@echo "$(GREEN)Monitoring stack stopped!$(NC)"

# Kafka/Redpanda
kafka-up: ## Start Kafka/Redpanda
	@echo "$(GREEN)Starting Kafka/Redpanda...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.kafka.yml up -d
	@echo "$(GREEN)Kafka/Redpanda is running!$(NC)"
	@echo "Redpanda Console: http://localhost:8080"

kafka-down: ## Stop Kafka/Redpanda
	@echo "$(YELLOW)Stopping Kafka/Redpanda...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.kafka.yml down
	@echo "$(GREEN)Kafka/Redpanda stopped!$(NC)"

kafka-topics: ## List Kafka topics
	docker exec -it riskradar-redpanda rpk topic list

# Utilities
clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	@echo "$(GREEN)Cleanup completed!$(NC)"

logs: ## Show all logs
	$(DOCKER_COMPOSE) logs -f

ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

shell: ## Open shell in service (usage: make shell SERVICE=risk-api)
	docker exec -it riskradar-$(SERVICE) /bin/bash

version: ## Show version information
	@echo "$(GREEN)RiskRadar Enterprise Version Information:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell $(DOCKER_COMPOSE) --version)"
	@echo "Kubectl: $(shell $(KUBECTL) version --client --short 2>/dev/null || echo 'Not installed')"
	@echo "Helm: $(shell $(HELM) version --short 2>/dev/null || echo 'Not installed')"
	@echo "Terraform: $(shell $(TERRAFORM) version -json 2>/dev/null | jq -r '.terraform_version' || echo 'Not installed')"

# Performance
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTHON_VENV) scripts/testing/benchmark.py
	@echo "$(GREEN)Benchmarks completed!$(NC)"

profile: ## Profile service performance
	@echo "$(GREEN)Profiling service performance...$(NC)"
	$(PYTHON_VENV) -m cProfile -o profile.stats services/risk-api/app/main.py
	$(PYTHON_VENV) -m pstats profile.stats
	@echo "$(GREEN)Profiling completed!$(NC)"

# Documentation
docs-build: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && mkdocs build
	@echo "$(GREEN)Documentation built!$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	cd docs && mkdocs serve

# CI/CD
ci-validate: ## Validate CI configuration
	@echo "$(GREEN)Validating CI configuration...$(NC)"
	yamllint .github/workflows/
	@echo "$(GREEN)CI configuration is valid!$(NC)"

# Terraform
tf-init: ## Initialize Terraform
	@echo "$(GREEN)Initializing Terraform...$(NC)"
	cd infrastructure/terraform && $(TERRAFORM) init
	@echo "$(GREEN)Terraform initialized!$(NC)"

tf-plan: ## Plan Terraform changes
	@echo "$(GREEN)Planning Terraform changes...$(NC)"
	cd infrastructure/terraform && $(TERRAFORM) plan
	@echo "$(GREEN)Terraform plan completed!$(NC)"

tf-apply: ## Apply Terraform changes
	@echo "$(RED)Applying Terraform changes...$(NC)"
	@read -p "Are you sure? (y/N) " confirm && [ $$confirm = y ]
	cd infrastructure/terraform && $(TERRAFORM) apply
	@echo "$(GREEN)Terraform apply completed!$(NC)"

# Quick commands
up: infra-up services-up monitoring-up ## Start everything
down: services-down infra-down monitoring-down ## Stop everything
restart: down up ## Restart everything
status: ## Show status of all services
	@echo "$(GREEN)Service Status:$(NC)"
	$(DOCKER_COMPOSE) ps
	@echo "\n$(GREEN)Infrastructure Status:$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.infra.yml ps
	@echo "\n$(GREEN)Monitoring Status:$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml ps
