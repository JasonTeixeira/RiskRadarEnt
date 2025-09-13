"""
Database Models for RiskRadar Enterprise
Using SQLAlchemy with async support and TimescaleDB extensions
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, Text, DECIMAL, Enum, Table, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


class PortfolioStatus(str, PyEnum):
    """Portfolio status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"


class PositionType(str, PyEnum):
    """Position type enumeration"""
    LONG = "long"
    SHORT = "short"


class AssetClass(str, PyEnum):
    """Asset class enumeration"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    ALTERNATIVE = "alternative"


class RiskCalculationStatus(str, PyEnum):
    """Risk calculation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertSeverity(str, PyEnum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Association tables for many-to-many relationships
portfolio_users = Table(
    'portfolio_users',
    Base.metadata,
    Column('portfolio_id', UUID(as_uuid=True), ForeignKey('portfolios.id')),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('role', String(50), default='viewer'),  # owner, manager, viewer
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)


class User(Base):
    """User model with enterprise features"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    
    # Enterprise features
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    department = Column(String(100))
    role = Column(String(50), default='analyst')  # admin, risk_manager, analyst, viewer
    
    # Security
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255))
    
    # API access
    api_key = Column(String(255), unique=True, index=True)
    api_tier = Column(String(50), default='standard')  # basic, standard, premium, enterprise
    rate_limit_override = Column(Integer)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    login_count = Column(Integer, default=0)
    failed_login_count = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    portfolios = relationship("Portfolio", secondary=portfolio_users, back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")
    api_requests = relationship("APIRequest", back_populates="user")
    
    __table_args__ = (
        Index('idx_user_org', 'organization_id'),
        Index('idx_user_active', 'is_active'),
    )


class Organization(Base):
    """Organization model for multi-tenancy"""
    __tablename__ = 'organizations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    
    # Billing and limits
    subscription_tier = Column(String(50), default='standard')
    max_portfolios = Column(Integer, default=100)
    max_users = Column(Integer, default=50)
    max_api_calls_per_month = Column(BigInteger, default=1000000)
    
    # Settings
    settings = Column(JSONB, default={})
    features = Column(ARRAY(String), default=[])
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="organization")
    portfolios = relationship("Portfolio", back_populates="organization")


class Portfolio(Base):
    """Portfolio model with comprehensive tracking"""
    __tablename__ = 'portfolios'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Organization and ownership
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Portfolio details
    currency = Column(String(3), default='USD')
    inception_date = Column(DateTime(timezone=True), default=func.now())
    status = Column(Enum(PortfolioStatus), default=PortfolioStatus.ACTIVE)
    
    # Value tracking
    initial_value = Column(DECIMAL(20, 2), nullable=False)
    current_value = Column(DECIMAL(20, 2))
    cash_balance = Column(DECIMAL(20, 2), default=0)
    
    # Risk parameters
    target_volatility = Column(Float)
    max_drawdown_limit = Column(Float)
    var_limit = Column(DECIMAL(20, 2))
    leverage_limit = Column(Float, default=1.0)
    
    # Benchmark
    benchmark_index = Column(String(50))
    benchmark_weights = Column(JSONB)
    
    # Metadata
    tags = Column(ARRAY(String))
    metadata = Column(JSONB, default={})
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_rebalanced = Column(DateTime(timezone=True))
    
    # Relationships
    organization = relationship("Organization", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    users = relationship("User", secondary=portfolio_users, back_populates="portfolios")
    risk_calculations = relationship("RiskCalculation", back_populates="portfolio")
    portfolio_values = relationship("PortfolioValue", back_populates="portfolio")
    alerts = relationship("Alert", back_populates="portfolio")
    
    __table_args__ = (
        Index('idx_portfolio_org', 'organization_id'),
        Index('idx_portfolio_status', 'status'),
        Index('idx_portfolio_owner', 'owner_id'),
        CheckConstraint('leverage_limit >= 0', name='check_leverage_positive'),
    )


class Position(Base):
    """Position model for portfolio holdings"""
    __tablename__ = 'positions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    
    # Asset identification
    symbol = Column(String(50), nullable=False)
    name = Column(String(255))
    asset_class = Column(Enum(AssetClass), nullable=False)
    sector = Column(String(100))
    country = Column(String(2))
    exchange = Column(String(50))
    
    # Position details
    position_type = Column(Enum(PositionType), default=PositionType.LONG)
    quantity = Column(DECIMAL(20, 8), nullable=False)
    average_price = Column(DECIMAL(20, 8), nullable=False)
    current_price = Column(DECIMAL(20, 8))
    
    # Values
    cost_basis = Column(DECIMAL(20, 2), nullable=False)
    market_value = Column(DECIMAL(20, 2))
    unrealized_pnl = Column(DECIMAL(20, 2))
    realized_pnl = Column(DECIMAL(20, 2), default=0)
    
    # Weights and exposure
    weight = Column(Float)
    target_weight = Column(Float)
    sector_exposure = Column(Float)
    country_exposure = Column(Float)
    
    # Risk metrics (cached)
    beta = Column(Float)
    volatility = Column(Float)
    var_contribution = Column(DECIMAL(20, 2))
    
    # Dates
    open_date = Column(DateTime(timezone=True), default=func.now())
    close_date = Column(DateTime(timezone=True))
    last_updated = Column(DateTime(timezone=True), default=func.now())
    
    # Metadata
    metadata = Column(JSONB, default={})
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    transactions = relationship("Transaction", back_populates="position")
    
    __table_args__ = (
        Index('idx_position_portfolio', 'portfolio_id'),
        Index('idx_position_symbol', 'symbol'),
        Index('idx_position_asset_class', 'asset_class'),
        UniqueConstraint('portfolio_id', 'symbol', 'position_type', name='uq_portfolio_symbol_type'),
    )


class Transaction(Base):
    """Transaction model for trade history"""
    __tablename__ = 'transactions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey('positions.id'))
    
    # Transaction details
    transaction_type = Column(String(20), nullable=False)  # buy, sell, dividend, fee, etc.
    symbol = Column(String(50), nullable=False)
    quantity = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)
    amount = Column(DECIMAL(20, 2), nullable=False)
    
    # Fees and costs
    commission = Column(DECIMAL(10, 2), default=0)
    fees = Column(DECIMAL(10, 2), default=0)
    tax = Column(DECIMAL(10, 2), default=0)
    
    # Settlement
    trade_date = Column(DateTime(timezone=True), nullable=False)
    settlement_date = Column(DateTime(timezone=True))
    
    # Metadata
    order_id = Column(String(100))
    broker = Column(String(50))
    notes = Column(Text)
    metadata = Column(JSONB, default={})
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    position = relationship("Position", back_populates="transactions")
    
    __table_args__ = (
        Index('idx_transaction_portfolio', 'portfolio_id'),
        Index('idx_transaction_date', 'trade_date'),
        Index('idx_transaction_symbol', 'symbol'),
    )


class RiskCalculation(Base):
    """Risk calculation results with versioning"""
    __tablename__ = 'risk_calculations'
    
    # Make this a TimescaleDB hypertable
    __table_args__ = (
        Index('idx_risk_calc_portfolio_time', 'portfolio_id', 'calculated_at'),
        Index('idx_risk_calc_status', 'status'),
        {'timescaledb_hypertable': {'time_column': 'calculated_at'}},
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    
    # Calculation details
    calculation_type = Column(String(50), nullable=False)  # realtime, batch, stress_test, etc.
    status = Column(Enum(RiskCalculationStatus), default=RiskCalculationStatus.PENDING)
    
    # Risk metrics
    var_95 = Column(DECIMAL(20, 2))
    var_99 = Column(DECIMAL(20, 2))
    cvar_95 = Column(DECIMAL(20, 2))
    cvar_99 = Column(DECIMAL(20, 2))
    expected_shortfall = Column(DECIMAL(20, 2))
    
    # Performance metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    information_ratio = Column(Float)
    
    # Risk measures
    volatility = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    max_drawdown = Column(Float)
    tracking_error = Column(Float)
    
    # Additional metrics
    skewness = Column(Float)
    kurtosis = Column(Float)
    downside_deviation = Column(Float)
    omega_ratio = Column(Float)
    
    # Detailed results
    metrics = Column(JSONB, nullable=False)
    position_risks = Column(JSONB)
    correlation_matrix = Column(JSONB)
    stress_test_results = Column(JSONB)
    
    # Calculation metadata
    confidence_level = Column(Float, default=0.95)
    time_horizon = Column(String(10), default='1d')
    calculation_method = Column(String(50))
    model_version = Column(String(20))
    
    # Performance
    calculation_time_ms = Column(Integer)
    data_points_used = Column(Integer)
    
    # Timestamps
    calculated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    valid_until = Column(DateTime(timezone=True))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="risk_calculations")


class PortfolioValue(Base):
    """Historical portfolio values - TimescaleDB hypertable"""
    __tablename__ = 'portfolio_values'
    
    # TimescaleDB hypertable configuration
    __table_args__ = (
        Index('idx_portfolio_value_time', 'portfolio_id', 'timestamp'),
        UniqueConstraint('portfolio_id', 'timestamp', name='uq_portfolio_timestamp'),
        {'timescaledb_hypertable': {'time_column': 'timestamp'}},
    )
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Values
    total_value = Column(DECIMAL(20, 2), nullable=False)
    cash_value = Column(DECIMAL(20, 2))
    securities_value = Column(DECIMAL(20, 2))
    
    # P&L
    daily_pnl = Column(DECIMAL(20, 2))
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    
    # Risk metrics snapshot
    current_var = Column(DECIMAL(20, 2))
    current_volatility = Column(Float)
    current_sharpe = Column(Float)
    
    # Metadata
    source = Column(String(50))  # realtime, eod, manual
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="portfolio_values")


class MarketData(Base):
    """Market data storage - TimescaleDB hypertable"""
    __tablename__ = 'market_data'
    
    # TimescaleDB hypertable
    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
        UniqueConstraint('symbol', 'timestamp', 'data_type', name='uq_symbol_time_type'),
        {'timescaledb_hypertable': {'time_column': 'timestamp'}},
    )
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Price data
    data_type = Column(String(20), nullable=False)  # price, volume, etc.
    open = Column(DECIMAL(20, 8))
    high = Column(DECIMAL(20, 8))
    low = Column(DECIMAL(20, 8))
    close = Column(DECIMAL(20, 8))
    volume = Column(BigInteger)
    
    # Additional data
    bid = Column(DECIMAL(20, 8))
    ask = Column(DECIMAL(20, 8))
    vwap = Column(DECIMAL(20, 8))
    
    # Source
    source = Column(String(50))
    quality_score = Column(Float)


class Alert(Base):
    """Risk alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'))
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # risk_breach, drawdown, concentration, etc.
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Trigger conditions
    metric_name = Column(String(100))
    threshold_value = Column(Float)
    actual_value = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    acknowledged_at = Column(DateTime(timezone=True))
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    # Metadata
    metadata = Column(JSONB, default={})
    
    # Timestamps
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_portfolio', 'portfolio_id'),
        Index('idx_alert_active', 'is_active'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_triggered', 'triggered_at'),
    )


class AuditLog(Base):
    """Comprehensive audit logging"""
    __tablename__ = 'audit_logs'
    
    # TimescaleDB hypertable
    __table_args__ = (
        Index('idx_audit_user_time', 'user_id', 'timestamp'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        {'timescaledb_hypertable': {'time_column': 'timestamp'}},
    )
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # User and action
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    action = Column(String(50), nullable=False)  # create, update, delete, view, calculate, etc.
    
    # Entity
    entity_type = Column(String(50), nullable=False)  # portfolio, position, user, etc.
    entity_id = Column(UUID(as_uuid=True))
    
    # Changes
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    
    # Request context
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    request_id = Column(UUID(as_uuid=True))
    session_id = Column(String(255))
    
    # Additional context
    metadata = Column(JSONB, default={})
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


class APIRequest(Base):
    """API request tracking for analytics and billing"""
    __tablename__ = 'api_requests'
    
    # TimescaleDB hypertable
    __table_args__ = (
        Index('idx_api_request_user_time', 'user_id', 'timestamp'),
        Index('idx_api_request_endpoint', 'endpoint'),
        {'timescaledb_hypertable': {'time_column': 'timestamp'}},
    )
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Request details
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    
    # Performance
    response_time_ms = Column(Integer)
    
    # Rate limiting
    rate_limit_remaining = Column(Integer)
    
    # Billing
    credits_used = Column(Integer, default=1)
    
    # Relationships
    user = relationship("User", back_populates="api_requests")


# Create indexes for better query performance
def create_indexes():
    """Additional indexes for performance optimization"""
    return [
        Index('idx_position_market_value', Position.market_value),
        Index('idx_risk_calc_var', RiskCalculation.var_95),
        Index('idx_portfolio_value_date_range', PortfolioValue.portfolio_id, PortfolioValue.timestamp),
        Index('idx_transaction_amount', Transaction.amount),
        Index('idx_alert_unresolved', Alert.is_resolved, Alert.severity),
    ]
