"""
Enterprise Health Check and Monitoring Endpoints
Comprehensive health, readiness, and liveness checks with detailed diagnostics
"""

import asyncio
import os
import platform
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge, Histogram
import httpx

from app.core.database import get_db, Database
from app.core.config import settings
from app.core.redis import RedisCache
from app.core.events import EventBus

router = APIRouter()

# Metrics
health_check_counter = Counter(
    'health_check_total',
    'Total health check requests',
    ['endpoint', 'status']
)
health_check_duration = Histogram(
    'health_check_duration_seconds',
    'Health check duration',
    ['check_type']
)
service_uptime = Gauge(
    'service_uptime_seconds',
    'Service uptime in seconds'
)

# Service start time
SERVICE_START_TIME = datetime.utcnow()


class HealthStatus:
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Component health check result"""
    def __init__(
        self,
        name: str,
        status: str,
        latency_ms: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.name = name
        self.status = status
        self.latency_ms = latency_ms
        self.details = details or {}
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2)
        }
        if self.details:
            result["details"] = self.details
        if self.error:
            result["error"] = self.error
        return result


async def check_database_health(db: AsyncSession) -> ComponentHealth:
    """Check database connectivity and performance"""
    start_time = time.time()
    
    try:
        # Basic connectivity check
        result = await db.execute(text("SELECT 1"))
        
        # Check database stats
        db_stats = await db.execute(text("""
            SELECT 
                pg_database_size(current_database()) as db_size,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                (SELECT count(*) FROM pg_stat_activity) as total_connections,
                pg_postmaster_start_time() as server_start_time
        """))
        stats = db_stats.first()
        
        # Check TimescaleDB health
        timescale_check = await db.execute(text("""
            SELECT count(*) as hypertable_count 
            FROM timescaledb_information.hypertables
        """))
        hypertables = timescale_check.scalar()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine health status
        status = HealthStatus.HEALTHY
        if stats.active_connections > 50:
            status = HealthStatus.DEGRADED
        elif stats.active_connections > 80:
            status = HealthStatus.UNHEALTHY
        
        return ComponentHealth(
            name="database",
            status=status,
            latency_ms=latency_ms,
            details={
                "db_size_mb": round(stats.db_size / 1024 / 1024, 2),
                "active_connections": stats.active_connections,
                "total_connections": stats.total_connections,
                "hypertables": hypertables,
                "server_uptime": str(datetime.utcnow() - stats.server_start_time)
            }
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            error=str(e)
        )


async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity and performance"""
    start_time = time.time()
    
    try:
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Ping test
        await redis_client.ping()
        
        # Get Redis info
        info = await redis_client.info()
        memory_info = await redis_client.info("memory")
        
        # Performance test
        test_key = f"health_check_{time.time()}"
        await redis_client.setex(test_key, 10, "test")
        await redis_client.get(test_key)
        await redis_client.delete(test_key)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine health status
        used_memory_mb = int(memory_info.get("used_memory", 0)) / 1024 / 1024
        max_memory_mb = int(memory_info.get("maxmemory", 0)) / 1024 / 1024
        
        status = HealthStatus.HEALTHY
        if max_memory_mb > 0:
            memory_usage_pct = (used_memory_mb / max_memory_mb) * 100
            if memory_usage_pct > 80:
                status = HealthStatus.DEGRADED
            elif memory_usage_pct > 90:
                status = HealthStatus.UNHEALTHY
        
        await redis_client.close()
        
        return ComponentHealth(
            name="redis",
            status=status,
            latency_ms=latency_ms,
            details={
                "version": info.get("redis_version"),
                "used_memory_mb": round(used_memory_mb, 2),
                "max_memory_mb": round(max_memory_mb, 2) if max_memory_mb > 0 else "unlimited",
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days")
            }
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            error=str(e)
        )


async def check_kafka_health() -> ComponentHealth:
    """Check Kafka/Redpanda connectivity"""
    start_time = time.time()
    
    try:
        # Check Redpanda admin API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://{settings.KAFKA_BOOTSTRAP_SERVERS.replace(':9092', ':9644')}/v1/brokers",
                timeout=5.0
            )
            
            if response.status_code == 200:
                brokers = response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                return ComponentHealth(
                    name="kafka",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    details={
                        "brokers": len(brokers),
                        "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS
                    }
                )
    
    except Exception as e:
        pass  # Try alternative check
    
    # Fallback: basic connectivity check
    try:
        from aiokafka import AIOKafkaProducer
        
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id="health-check"
        )
        await producer.start()
        await producer.stop()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="kafka",
            status=HealthStatus.HEALTHY,
            latency_ms=latency_ms,
            details={"bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS}
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ComponentHealth(
            name="kafka",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            error=str(e)
        )


async def check_celery_health() -> ComponentHealth:
    """Check Celery workers health"""
    start_time = time.time()
    
    try:
        from celery import Celery
        
        app = Celery('risk_compute_worker')
        app.config_from_object('celeryconfig')
        
        # Get worker stats
        stats = app.control.inspect().stats()
        active = app.control.inspect().active()
        
        if stats:
            worker_count = len(stats)
            total_active_tasks = sum(len(tasks) for tasks in active.values()) if active else 0
            
            latency_ms = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            if worker_count < 2:
                status = HealthStatus.DEGRADED
            elif worker_count == 0:
                status = HealthStatus.UNHEALTHY
            
            return ComponentHealth(
                name="celery",
                status=status,
                latency_ms=latency_ms,
                details={
                    "workers": worker_count,
                    "active_tasks": total_active_tasks
                }
            )
        else:
            raise Exception("No Celery workers available")
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ComponentHealth(
            name="celery",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            error=str(e)
        )


async def check_external_services() -> List[ComponentHealth]:
    """Check external service dependencies"""
    checks = []
    
    # Market data providers
    providers = {
        "yahoo_finance": "https://query1.finance.yahoo.com/v1/test",
        "alpha_vantage": "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo",
    }
    
    async with httpx.AsyncClient() as client:
        for name, url in providers.items():
            start_time = time.time()
            try:
                response = await client.get(url, timeout=5.0)
                latency_ms = (time.time() - start_time) * 1000
                
                status = HealthStatus.HEALTHY
                if response.status_code != 200:
                    status = HealthStatus.DEGRADED
                
                checks.append(ComponentHealth(
                    name=name,
                    status=status,
                    latency_ms=latency_ms,
                    details={"status_code": response.status_code}
                ))
            
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                checks.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    error=str(e)
                ))
    
    return checks


def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics"""
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    
    # Disk metrics
    disk = psutil.disk_usage('/')
    
    # Network metrics
    network = psutil.net_io_counters()
    
    # Process metrics
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    
    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "cores": cpu_count,
            "load_average": os.getloadavg() if platform.system() != 'Windows' else [0, 0, 0]
        },
        "memory": {
            "total_mb": round(memory.total / 1024 / 1024, 2),
            "used_mb": round(memory.used / 1024 / 1024, 2),
            "available_mb": round(memory.available / 1024 / 1024, 2),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
            "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "percent": disk.percent
        },
        "network": {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        },
        "process": {
            "pid": process.pid,
            "memory_mb": round(process_memory.rss / 1024 / 1024, 2),
            "threads": process.num_threads(),
            "connections": len(process.connections()),
            "cpu_percent": process.cpu_percent()
        }
    }


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    Returns 200 if service is alive, 503 if not
    """
    health_check_counter.labels(endpoint='liveness', status='success').inc()
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": (datetime.utcnow() - SERVICE_START_TIME).total_seconds()
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe endpoint
    Checks if service is ready to accept traffic
    """
    
    start_time = time.time()
    checks = []
    
    # Check critical dependencies
    with health_check_duration.labels(check_type='readiness').time():
        # Database check
        db_health = await check_database_health(db)
        checks.append(db_health)
        
        # Redis check
        redis_health = await check_redis_health()
        checks.append(redis_health)
        
        # Kafka check (non-blocking)
        kafka_health = await check_kafka_health()
        checks.append(kafka_health)
    
    # Determine overall readiness
    critical_checks = [c for c in checks if c.name in ["database", "redis"]]
    is_ready = all(c.status != HealthStatus.UNHEALTHY for c in critical_checks)
    
    total_latency = (time.time() - start_time) * 1000
    
    if is_ready:
        health_check_counter.labels(endpoint='readiness', status='ready').inc()
        return {
            "status": "ready",
            "checks": [c.to_dict() for c in checks],
            "total_latency_ms": round(total_latency, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        health_check_counter.labels(endpoint='readiness', status='not_ready').inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "checks": [c.to_dict() for c in checks],
                "total_latency_ms": round(total_latency, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/startup")
async def startup_check():
    """
    Kubernetes startup probe endpoint
    Used during initial container startup
    """
    
    # Check if service has been up for at least 10 seconds
    uptime = (datetime.utcnow() - SERVICE_START_TIME).total_seconds()
    
    if uptime < 10:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "starting",
                "uptime_seconds": uptime,
                "required_uptime": 10
            }
        )
    
    return {
        "status": "started",
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health")
async def comprehensive_health_check(
    db: AsyncSession = Depends(get_db),
    include_details: bool = True
):
    """
    Comprehensive health check endpoint with detailed diagnostics
    """
    
    start_time = time.time()
    
    with health_check_duration.labels(check_type='comprehensive').time():
        # Run all health checks in parallel
        checks_coroutines = [
            check_database_health(db),
            check_redis_health(),
            check_kafka_health(),
            check_celery_health()
        ]
        
        component_checks = await asyncio.gather(*checks_coroutines, return_exceptions=True)
        
        # Handle exceptions
        checks = []
        for check in component_checks:
            if isinstance(check, Exception):
                checks.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    error=str(check)
                ))
            else:
                checks.append(check)
        
        # Check external services (non-critical)
        external_checks = await check_external_services()
        checks.extend(external_checks)
    
    # Determine overall health
    critical_components = ["database", "redis", "kafka", "celery"]
    critical_checks = [c for c in checks if c.name in critical_components]
    
    unhealthy_critical = [c for c in critical_checks if c.status == HealthStatus.UNHEALTHY]
    degraded_critical = [c for c in critical_checks if c.status == HealthStatus.DEGRADED]
    
    if unhealthy_critical:
        overall_status = HealthStatus.UNHEALTHY
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif degraded_critical:
        overall_status = HealthStatus.DEGRADED
        status_code = status.HTTP_200_OK
    else:
        overall_status = HealthStatus.HEALTHY
        status_code = status.HTTP_200_OK
    
    # Update metrics
    health_check_counter.labels(endpoint='health', status=overall_status).inc()
    service_uptime.set((datetime.utcnow() - SERVICE_START_TIME).total_seconds())
    
    total_latency = (time.time() - start_time) * 1000
    
    response = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("VERSION", "unknown"),
        "environment": settings.ENVIRONMENT,
        "uptime": str(datetime.utcnow() - SERVICE_START_TIME),
        "checks": [c.to_dict() for c in checks],
        "total_latency_ms": round(total_latency, 2)
    }
    
    if include_details:
        response["system_metrics"] = get_system_metrics()
        response["configuration"] = {
            "workers": settings.WORKERS,
            "max_portfolio_size": settings.MAX_PORTFOLIO_SIZE,
            "cache_enabled": settings.ENABLE_CACHE,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED
        }
    
    if status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=status_code, detail=response)
    
    return response


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format
    """
    
    # Generate metrics
    metrics_output = generate_latest()
    
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
        headers={"Content-Type": CONTENT_TYPE_LATEST}
    )


@router.get("/debug/config")
async def debug_configuration():
    """
    Debug endpoint to view current configuration (only in non-production)
    """
    
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints are disabled in production"
        )
    
    return {
        "environment": settings.ENVIRONMENT,
        "service_name": settings.SERVICE_NAME,
        "port": settings.PORT,
        "workers": settings.WORKERS,
        "database": {
            "url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "hidden",
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW
        },
        "redis": {
            "url": settings.REDIS_URL.split('@')[1] if '@' in settings.REDIS_URL else "hidden",
            "db": settings.REDIS_DB,
            "pool_size": settings.REDIS_POOL_SIZE
        },
        "kafka": {
            "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "consumer_group": settings.KAFKA_CONSUMER_GROUP
        },
        "features": {
            "realtime_updates": settings.ENABLE_REALTIME_UPDATES,
            "historical_analysis": settings.ENABLE_HISTORICAL_ANALYSIS,
            "stress_testing": settings.ENABLE_STRESS_TESTING,
            "monte_carlo": settings.ENABLE_MONTE_CARLO,
            "machine_learning": settings.ENABLE_MACHINE_LEARNING
        },
        "rate_limiting": {
            "enabled": settings.RATE_LIMIT_ENABLED,
            "per_minute": settings.RATE_LIMIT_PER_MINUTE,
            "per_hour": settings.RATE_LIMIT_PER_HOUR
        }
    }


@router.post("/debug/test-error")
async def test_error_handling():
    """
    Test error handling and monitoring (only in non-production)
    """
    
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints are disabled in production"
        )
    
    # Simulate an error for testing monitoring/alerting
    raise Exception("This is a test error for monitoring validation")
