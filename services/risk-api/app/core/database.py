"""
Async Database Connection Management
Enterprise-grade database handling with connection pooling, retry logic, and monitoring
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base
from prometheus_client import Counter, Histogram, Gauge

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
db_connection_gauge = Gauge(
    'db_connections_active',
    'Number of active database connections'
)
db_query_histogram = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table']
)
db_error_counter = Counter(
    'db_errors_total',
    'Total number of database errors',
    ['error_type']
)


class Database:
    """
    Enterprise database manager with:
    - Connection pooling
    - Automatic retry logic
    - Health checking
    - Performance monitoring
    - Read replica support
    """
    
    _engine: Optional[AsyncEngine] = None
    _read_engine: Optional[AsyncEngine] = None
    _sessionmaker: Optional[async_sessionmaker] = None
    _read_sessionmaker: Optional[async_sessionmaker] = None
    
    @classmethod
    async def connect(cls) -> None:
        """Initialize database connections"""
        try:
            # Create main engine with connection pooling
            cls._engine = create_async_engine(
                settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
                echo=settings.DATABASE_ECHO,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Verify connections before using
                connect_args={
                    "server_settings": {
                        "application_name": settings.SERVICE_NAME,
                        "jit": "off"
                    },
                    "command_timeout": 60,
                    "statement_cache_size": 0,  # Disable statement caching for dynamic queries
                    "prepared_statement_cache_size": 0,
                }
            )
            
            # Create session maker
            cls._sessionmaker = async_sessionmaker(
                cls._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Initialize read replica if configured
            if hasattr(settings, 'DATABASE_READ_URL') and settings.DATABASE_READ_URL:
                cls._read_engine = create_async_engine(
                    settings.DATABASE_READ_URL.replace('postgresql://', 'postgresql+asyncpg://'),
                    echo=False,
                    pool_size=settings.DATABASE_POOL_SIZE // 2,
                    max_overflow=settings.DATABASE_MAX_OVERFLOW // 2,
                    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                )
                
                cls._read_sessionmaker = async_sessionmaker(
                    cls._read_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                    autocommit=False
                )
            
            # Test connection
            async with cls._engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            # Create TimescaleDB extension if not exists
            await cls._setup_timescaledb()
            
            logger.info("Database connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            db_error_counter.labels(error_type='connection').inc()
            raise
    
    @classmethod
    async def disconnect(cls) -> None:
        """Close database connections"""
        try:
            if cls._engine:
                await cls._engine.dispose()
                cls._engine = None
            
            if cls._read_engine:
                await cls._read_engine.dispose()
                cls._read_engine = None
            
            logger.info("Database disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting database: {e}")
            db_error_counter.labels(error_type='disconnection').inc()
    
    @classmethod
    async def _setup_timescaledb(cls) -> None:
        """Setup TimescaleDB extensions and hypertables"""
        async with cls._engine.begin() as conn:
            # Create extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            
            # Create hypertables for time-series data
            hypertables = [
                ("portfolio_values", "timestamp"),
                ("market_data", "timestamp"),
                ("risk_calculations", "calculated_at"),
                ("audit_logs", "timestamp"),
                ("api_requests", "timestamp"),
            ]
            
            for table, time_column in hypertables:
                try:
                    await conn.execute(
                        f"SELECT create_hypertable('{table}', '{time_column}', "
                        f"if_not_exists => TRUE, migrate_data => TRUE)"
                    )
                    
                    # Set compression policy (compress data older than 7 days)
                    await conn.execute(
                        f"ALTER TABLE {table} SET ("
                        f"timescaledb.compress, "
                        f"timescaledb.compress_segmentby = 'portfolio_id'"
                        f")"
                    )
                    
                    await conn.execute(
                        f"SELECT add_compression_policy('{table}', "
                        f"INTERVAL '7 days', if_not_exists => TRUE)"
                    )
                    
                except Exception as e:
                    # Table might not exist yet
                    logger.debug(f"Could not create hypertable for {table}: {e}")
    
    @classmethod
    @asynccontextmanager
    async def get_session(cls, read_only: bool = False) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup
        
        Args:
            read_only: Use read replica if available
        """
        if not cls._engine:
            await cls.connect()
        
        # Use read replica for read-only operations if available
        sessionmaker = cls._read_sessionmaker if (read_only and cls._read_sessionmaker) else cls._sessionmaker
        
        async with sessionmaker() as session:
            try:
                db_connection_gauge.inc()
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                db_error_counter.labels(error_type='session').inc()
                raise
            finally:
                await session.close()
                db_connection_gauge.dec()
    
    @classmethod
    async def execute_with_retry(
        cls,
        func,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ):
        """
        Execute database operation with retry logic
        
        Args:
            func: Async function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries
            exponential_backoff: Use exponential backoff for retries
        """
        last_exception = None
        delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                return await func()
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
                    if exponential_backoff:
                        delay *= 2
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts")
                    db_error_counter.labels(error_type='retry_exhausted').inc()
        
        raise last_exception
    
    @classmethod
    async def health_check(cls) -> dict:
        """
        Perform database health check
        
        Returns:
            Health status dictionary
        """
        try:
            if not cls._engine:
                return {
                    "status": "unhealthy",
                    "error": "Database not connected"
                }
            
            # Check main connection
            async with cls._engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                
            # Get pool stats
            pool = cls._engine.pool
            
            health = {
                "status": "healthy",
                "pool_size": pool.size() if hasattr(pool, 'size') else 'N/A',
                "connections_in_use": pool.checked_in() if hasattr(pool, 'checked_in') else 'N/A',
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else 'N/A',
            }
            
            # Check read replica if configured
            if cls._read_engine:
                try:
                    async with cls._read_engine.begin() as conn:
                        await conn.execute("SELECT 1")
                    health["read_replica"] = "healthy"
                except Exception as e:
                    health["read_replica"] = f"unhealthy: {str(e)}"
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @classmethod
    async def get_table_stats(cls) -> dict:
        """Get database table statistics"""
        stats = {}
        
        async with cls._engine.begin() as conn:
            # Get table sizes
            result = await conn.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                    n_live_tup AS row_count
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 20
            """)
            
            stats['tables'] = [dict(row) for row in result]
            
            # Get database size
            result = await conn.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
            """)
            stats['total_size'] = result.scalar()
            
            # Get connection stats
            result = await conn.execute("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            stats['connections'] = dict(result.first())
            
        return stats


# Create a dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    async with Database.get_session() as session:
        yield session


async def get_read_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for read-only database sessions"""
    async with Database.get_session(read_only=True) as session:
        yield session
