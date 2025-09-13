"""
Configuration module using Pydantic Settings
"""

from typing import List, Optional
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Service configuration
    SERVICE_NAME: str = "risk-api"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    PORT: int = Field(default=8000, ge=1, le=65535)
    WORKERS: int = Field(default=4, ge=1)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Security
    SECRET_KEY: str = Field(default="change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    API_KEY_HEADER: str = "X-API-Key"
    ENABLE_OPENAPI: bool = True
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://riskradar:riskradar@localhost:5432/riskradar"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, ge=1)
    DATABASE_MAX_OVERFLOW: int = Field(default=40, ge=0)
    DATABASE_POOL_TIMEOUT: int = Field(default=30, ge=1)
    DATABASE_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_DB: int = Field(default=0, ge=0)
    REDIS_POOL_SIZE: int = Field(default=10, ge=1)
    REDIS_TTL: int = Field(default=3600, ge=1)
    
    # Kafka/Redpanda
    KAFKA_BOOTSTRAP_SERVERS: str = Field(default="localhost:9092")
    KAFKA_CONSUMER_GROUP: str = Field(default="risk-api-group")
    KAFKA_AUTO_OFFSET_RESET: str = Field(default="latest", pattern="^(earliest|latest)$")
    KAFKA_MAX_BATCH_SIZE: int = Field(default=100, ge=1)
    KAFKA_ENABLE_IDEMPOTENCE: bool = True
    
    # Event topics
    RISK_CALCULATION_TOPIC: str = "risk.calculations"
    PORTFOLIO_UPDATE_TOPIC: str = "portfolio.updates"
    ALERT_TOPIC: str = "risk.alerts"
    AUDIT_TOPIC: str = "audit.logs"
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1)
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, ge=1)
    RATE_LIMIT_BURST: int = Field(default=10, ge=1)
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = Field(default=9090, ge=1, le=65535)
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_SERVICE_NAME: str = Field(default="risk-api")
    OTEL_TRACES_EXPORTER: str = Field(default="otlp")
    OTEL_METRICS_EXPORTER: str = Field(default="prometheus")
    
    # Risk calculation settings
    DEFAULT_CONFIDENCE_LEVEL: float = Field(default=0.95, ge=0.0, le=1.0)
    DEFAULT_TIME_HORIZON: int = Field(default=1, ge=1)
    MAX_PORTFOLIO_SIZE: int = Field(default=10000, ge=1)
    CALCULATION_TIMEOUT: int = Field(default=30, ge=1)
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = Field(default=300, ge=1)
    
    # Market data providers
    MARKET_DATA_PROVIDERS: List[str] = Field(
        default=["yahoo", "alphavantage", "polygon"]
    )
    YAHOO_FINANCE_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    
    # Feature flags
    ENABLE_REALTIME_UPDATES: bool = True
    ENABLE_HISTORICAL_ANALYSIS: bool = True
    ENABLE_STRESS_TESTING: bool = True
    ENABLE_MONTE_CARLO: bool = True
    ENABLE_MACHINE_LEARNING: bool = False
    
    # Performance tuning
    USE_UVLOOP: bool = True
    ENABLE_PROFILING: bool = False
    PROFILE_REQUESTS_SLOWER_THAN: float = Field(default=1.0, ge=0.0)
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("MARKET_DATA_PROVIDERS", pre=True)
    def parse_providers(cls, v):
        if isinstance(v, str):
            return [provider.strip() for provider in v.split(",")]
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("Database URL must be a PostgreSQL connection string")
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v):
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL must be a Redis connection string")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
