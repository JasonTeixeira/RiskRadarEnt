"""
Pytest configuration and fixtures for RiskRadar Enterprise tests.

This module provides shared fixtures, configuration, and utilities
for all test suites in the project.
"""

import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Generator, AsyncGenerator, Dict, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from faker import Faker
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from redis import Redis
import fakeredis

from app.main import app
from app.core.config import settings
from app.core.database import Base, get_db
from app.core.auth import create_access_token, get_password_hash
from app.models.user import User
from app.models.portfolio import Portfolio
from app.models.position import Position

# Initialize Faker for test data generation
fake = Faker()


# --------------------------------------------------------------------------
# Pytest Configuration
# --------------------------------------------------------------------------

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring database/redis"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests covering full workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )


# --------------------------------------------------------------------------
# Database Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_database_url():
    """Get test database URL."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://test:test@localhost:5432/riskradar_test"
    )


@pytest.fixture(scope="session")
def sync_engine(test_database_url):
    """Create synchronous database engine for tests."""
    engine = create_engine(test_database_url)
    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield engine
    # Drop all tables after tests
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(sync_engine) -> Generator[Session, None, None]:
    """Create a database session for tests."""
    connection = sync_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest_asyncio.fixture(scope="session")
async def async_engine(test_database_url):
    """Create async database engine for tests."""
    # Convert sync URL to async
    async_url = test_database_url.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(async_url)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for tests."""
    async_session_maker = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


# --------------------------------------------------------------------------
# Redis Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="function")
def redis_client() -> Generator[Redis, None, None]:
    """Create fake Redis client for tests."""
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()


@pytest_asyncio.fixture(scope="function")
async def async_redis_client():
    """Create async fake Redis client for tests."""
    import fakeredis.aioredis
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.close()


# --------------------------------------------------------------------------
# FastAPI Test Client Fixtures
# --------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def test_client(async_session, async_redis_client) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with dependency overrides."""
    
    # Override database dependency
    async def override_get_db():
        yield async_session
    
    # Override Redis dependency
    async def override_get_redis():
        yield async_redis_client
    
    app.dependency_overrides[get_db] = override_get_db
    # app.dependency_overrides[get_redis] = override_get_redis
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Authentication Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def test_user_data() -> Dict[str, Any]:
    """Generate test user data."""
    return {
        "email": fake.email(),
        "username": fake.user_name(),
        "full_name": fake.name(),
        "password": "TestPassword123!",
        "is_active": True,
        "is_superuser": False,
        "organization_id": str(uuid4())
    }


@pytest.fixture
def test_user(db_session, test_user_data) -> User:
    """Create test user in database."""
    user = User(
        id=uuid4(),
        email=test_user_data["email"],
        username=test_user_data["username"],
        full_name=test_user_data["full_name"],
        hashed_password=get_password_hash(test_user_data["password"]),
        is_active=test_user_data["is_active"],
        is_superuser=test_user_data["is_superuser"],
        organization_id=test_user_data["organization_id"],
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_token(test_user) -> str:
    """Generate authentication token for test user."""
    access_token = create_access_token(
        data={"sub": test_user.email, "user_id": str(test_user.id)}
    )
    return access_token


@pytest.fixture
def auth_headers(auth_token) -> Dict[str, str]:
    """Generate authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def authenticated_client(test_client, auth_headers) -> AsyncClient:
    """Create authenticated test client."""
    test_client.headers.update(auth_headers)
    return test_client


# --------------------------------------------------------------------------
# Portfolio and Position Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def test_portfolio_data(test_user) -> Dict[str, Any]:
    """Generate test portfolio data."""
    return {
        "name": f"Test Portfolio {fake.company()}",
        "code": f"TP-{fake.random_number(digits=6)}",
        "description": fake.text(max_nb_chars=200),
        "currency": "USD",
        "initial_value": Decimal("1000000.00"),
        "organization_id": test_user.organization_id,
        "owner_id": test_user.id,
        "status": "active"
    }


@pytest.fixture
def test_portfolio(db_session, test_portfolio_data) -> Portfolio:
    """Create test portfolio in database."""
    portfolio = Portfolio(
        id=uuid4(),
        **test_portfolio_data,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)
    return portfolio


@pytest.fixture
def test_positions(db_session, test_portfolio) -> list[Position]:
    """Create test positions for portfolio."""
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    positions = []
    
    for symbol in symbols:
        position = Position(
            id=uuid4(),
            portfolio_id=test_portfolio.id,
            symbol=symbol,
            name=f"{symbol} Inc.",
            quantity=Decimal(str(fake.random_int(100, 1000))),
            average_price=Decimal(str(fake.random_int(50, 500))),
            current_price=Decimal(str(fake.random_int(50, 500))),
            asset_class="equity",
            currency="USD",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        positions.append(position)
        db_session.add(position)
    
    db_session.commit()
    return positions


# --------------------------------------------------------------------------
# Market Data Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    data = []
    base_price = 100
    dates = [datetime.utcnow() - timedelta(days=i) for i in range(30, 0, -1)]
    
    for date in dates:
        open_price = base_price + fake.random.uniform(-5, 5)
        close_price = open_price + fake.random.uniform(-3, 3)
        high_price = max(open_price, close_price) + fake.random.uniform(0, 2)
        low_price = min(open_price, close_price) - fake.random.uniform(0, 2)
        
        data.append({
            "date": date,
            "open": Decimal(str(round(open_price, 2))),
            "high": Decimal(str(round(high_price, 2))),
            "low": Decimal(str(round(low_price, 2))),
            "close": Decimal(str(round(close_price, 2))),
            "volume": fake.random_int(1000000, 10000000)
        })
        
        base_price = close_price
    
    return data


@pytest.fixture
def sample_returns():
    """Generate sample return data for risk calculations."""
    # Generate realistic daily returns (mean ~0.0005, std ~0.02)
    import numpy as np
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # One year of daily returns
    return returns.tolist()


# --------------------------------------------------------------------------
# Mock External Services
# --------------------------------------------------------------------------

@pytest.fixture
def mock_market_data_provider(mocker):
    """Mock market data provider."""
    mock = mocker.Mock()
    mock.get_quote.return_value = {
        "symbol": "AAPL",
        "price": Decimal("150.00"),
        "bid": Decimal("149.95"),
        "ask": Decimal("150.05"),
        "volume": 50000000,
        "timestamp": datetime.utcnow()
    }
    return mock


@pytest.fixture
def mock_celery_task(mocker):
    """Mock Celery task execution."""
    mock = mocker.patch("app.tasks.risk_calculation.calculate_risk_task.delay")
    mock.return_value.id = str(uuid4())
    mock.return_value.state = "SUCCESS"
    mock.return_value.result = {
        "var_95": 0.025,
        "cvar_95": 0.035,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.15
    }
    return mock


# --------------------------------------------------------------------------
# Performance Testing Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.utcnow()
        
        def stop(self):
            self.end_time = datetime.utcnow()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
        
        def assert_max_duration(self, max_seconds: float):
            assert self.elapsed is not None, "Timer not stopped"
            assert self.elapsed <= max_seconds, \
                f"Operation took {self.elapsed}s, expected max {max_seconds}s"
    
    return Timer()


# --------------------------------------------------------------------------
# Cleanup Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_test_data(request, db_session):
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup is handled by transaction rollback in db_session fixture


# --------------------------------------------------------------------------
# Async Event Loop Configuration
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
