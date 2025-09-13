"""
Risk API Service - Main Application
Enterprise-grade FastAPI service for real-time risk calculations
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs" / "python"))

from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.ratelimit import RateLimitMiddleware
from app.api.v1 import health, portfolios, risk
from app.core.config import settings
from app.core.database import Database
from app.core.events import EventBus
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics
from app.core.redis import RedisCache

# Setup logging
logger = setup_logging(settings.SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Risk API Service...")
    
    # Initialize database
    await Database.connect()
    logger.info("Database connected")
    
    # Initialize Redis cache
    await RedisCache.connect()
    logger.info("Redis cache connected")
    
    # Initialize event bus
    await EventBus.connect()
    logger.info("Event bus connected")
    
    # Setup metrics
    setup_metrics(app)
    logger.info("Metrics initialized")
    
    # Startup complete
    logger.info(f"Risk API Service started successfully on port {settings.PORT}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Risk API Service...")
    
    await EventBus.disconnect()
    await RedisCache.disconnect()
    await Database.disconnect()
    
    logger.info("Risk API Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="RiskRadar Risk API",
        description="Enterprise-grade API for portfolio risk calculations and analysis",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENABLE_OPENAPI else None,
        lifespan=lifespan,
    )
    
    # Add middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )
    
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Add exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": request.state.request_id if hasattr(request.state, "request_id") else None,
                }
            },
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": "Validation error",
                    "details": exc.errors(),
                    "request_id": request.state.request_id if hasattr(request.state, "request_id") else None,
                }
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "Internal server error",
                    "request_id": request.state.request_id if hasattr(request.state, "request_id") else None,
                }
            },
        )
    
    # Add routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(risk.router, prefix="/api/v1/risk", tags=["risk"])
    app.include_router(portfolios.router, prefix="/api/v1/portfolios", tags=["portfolios"])
    
    # Add Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        workers=1 if settings.ENVIRONMENT == "development" else settings.WORKERS,
    )
