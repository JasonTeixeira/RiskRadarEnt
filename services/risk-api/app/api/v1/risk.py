"""
Risk API Endpoints - Enterprise Production Implementation
Comprehensive risk calculation and management endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram
import pandas as pd

from app.core.database import get_db, get_read_db
from app.core.auth import get_current_user, require_permissions
from app.core.cache import cache_result, invalidate_cache
from app.models.database import (
    Portfolio, Position, RiskCalculation, User,
    RiskCalculationStatus, RiskMetric as DBRiskMetric
)
from app.services.risk_service import RiskService
from app.services.event_publisher import EventPublisher
from app.schemas.risk import (
    RiskCalculationRequest, RiskCalculationResponse,
    StressTestRequest, StressTestResponse,
    MonteCarloRequest, MonteCarloResponse,
    RiskMetricsResponse, PortfolioRiskSummary
)
from risk_compute_worker.app.worker import (
    calculate_portfolio_risk,
    calculate_urgent_risk,
    run_stress_test,
    run_monte_carlo_simulation
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Metrics
api_requests = Counter(
    'risk_api_requests_total',
    'Total risk API requests',
    ['endpoint', 'method']
)
api_latency = Histogram(
    'risk_api_latency_seconds',
    'Risk API latency',
    ['endpoint']
)


# Request/Response Models
class RiskCalculationRequest(BaseModel):
    """Risk calculation request model"""
    portfolio_id: UUID
    metrics: List[str] = Field(
        default=['var', 'cvar', 'sharpe_ratio', 'volatility'],
        description="Risk metrics to calculate"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.90,
        le=0.99,
        description="Confidence level for VaR/CVaR"
    )
    time_horizon: str = Field(
        default='1d',
        pattern='^(1d|1w|1m|3m|1y)$',
        description="Time horizon for risk calculations"
    )
    benchmark_id: Optional[str] = Field(
        default=None,
        description="Benchmark for relative metrics"
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results if available"
    )
    priority: str = Field(
        default='normal',
        pattern='^(low|normal|high|urgent)$',
        description="Calculation priority"
    )
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = {
            'var', 'cvar', 'expected_shortfall', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown',
            'volatility', 'beta', 'alpha', 'treynor_ratio',
            'information_ratio', 'tracking_error', 'downside_deviation',
            'omega_ratio', 'kurtosis', 'skewness'
        }
        invalid = set(v) - valid_metrics
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}")
        return v


class BatchRiskRequest(BaseModel):
    """Batch risk calculation request"""
    portfolio_ids: List[UUID]
    metrics: List[str]
    parallel: bool = True
    max_parallel: int = Field(default=10, ge=1, le=50)


class RiskLimitsUpdate(BaseModel):
    """Risk limits update model"""
    var_limit: Optional[float] = None
    max_drawdown_limit: Optional[float] = None
    volatility_limit: Optional[float] = None
    leverage_limit: Optional[float] = Field(None, ge=0)
    concentration_limit: Optional[float] = Field(None, ge=0, le=1)


# API Endpoints
@router.post("/calculate", response_model=RiskCalculationResponse)
async def calculate_risk(
    request: RiskCalculationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate risk metrics for a portfolio
    
    - Supports 17 different risk metrics
    - Async processing with Celery for heavy calculations
    - Caching for performance optimization
    - Real-time WebSocket updates available
    """
    api_requests.labels(endpoint='calculate_risk', method='POST').inc()
    
    # Check portfolio access
    portfolio = await db.get(Portfolio, request.portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check cache if enabled
    if request.use_cache:
        cache_key = f"risk:{request.portfolio_id}:{':'.join(request.metrics)}:{request.confidence_level}:{request.time_horizon}"
        cached = await cache_result.get(cache_key)
        if cached:
            logger.info(f"Returning cached risk calculation for portfolio {request.portfolio_id}")
            return RiskCalculationResponse(**cached)
    
    # Determine if async processing is needed
    is_heavy_calculation = (
        len(request.metrics) > 5 or
        'monte_carlo' in request.metrics or
        'stress_test' in request.metrics or
        request.priority == 'low'
    )
    
    if is_heavy_calculation:
        # Queue for async processing
        if request.priority == 'urgent':
            task = calculate_urgent_risk.apply_async(
                args=[str(request.portfolio_id), 'api_request', request.metrics],
                priority=10
            )
        else:
            task = calculate_portfolio_risk.apply_async(
                args=[
                    str(request.portfolio_id),
                    request.metrics,
                    request.confidence_level,
                    request.time_horizon,
                    request.use_cache
                ],
                priority={'low': 1, 'normal': 5, 'high': 8}[request.priority]
            )
        
        # Create pending calculation record
        calc = RiskCalculation(
            portfolio_id=request.portfolio_id,
            calculation_type='async',
            status=RiskCalculationStatus.PENDING,
            confidence_level=request.confidence_level,
            time_horizon=request.time_horizon,
            calculation_method='celery_task',
            metrics={'task_id': task.id, 'requested_metrics': request.metrics}
        )
        db.add(calc)
        await db.commit()
        
        return RiskCalculationResponse(
            calculation_id=calc.id,
            status='pending',
            task_id=task.id,
            message="Calculation queued for processing",
            estimated_time=_estimate_calculation_time(request)
        )
    
    else:
        # Perform synchronous calculation for simple requests
        try:
            risk_service = RiskService(db)
            results = await risk_service.calculate_risk(
                portfolio_id=request.portfolio_id,
                metrics=request.metrics,
                confidence_level=request.confidence_level,
                time_horizon=request.time_horizon
            )
            
            # Store calculation results
            calc = RiskCalculation(
                portfolio_id=request.portfolio_id,
                calculation_type='realtime',
                status=RiskCalculationStatus.COMPLETED,
                **results
            )
            db.add(calc)
            await db.commit()
            
            # Cache results
            if request.use_cache:
                await cache_result.set(cache_key, results, ttl=300)
            
            # Publish event
            await EventPublisher.publish(
                'risk.calculated',
                {
                    'portfolio_id': str(request.portfolio_id),
                    'calculation_id': str(calc.id),
                    'metrics': request.metrics
                }
            )
            
            return RiskCalculationResponse(
                calculation_id=calc.id,
                status='completed',
                results=results
            )
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            raise HTTPException(status_code=500, detail="Calculation failed")


@router.get("/calculation/{calculation_id}", response_model=RiskCalculationResponse)
async def get_calculation_result(
    calculation_id: UUID,
    db: AsyncSession = Depends(get_read_db),
    current_user: User = Depends(get_current_user)
):
    """Get risk calculation results by ID"""
    
    calc = await db.get(RiskCalculation, calculation_id)
    if not calc:
        raise HTTPException(status_code=404, detail="Calculation not found")
    
    # Check access
    portfolio = await db.get(Portfolio, calc.portfolio_id)
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if calc.status == RiskCalculationStatus.PENDING:
        # Check Celery task status
        task_id = calc.metrics.get('task_id')
        if task_id:
            from celery.result import AsyncResult
            task = AsyncResult(task_id)
            
            if task.ready():
                if task.successful():
                    results = task.result
                    calc.status = RiskCalculationStatus.COMPLETED
                    calc.metrics.update(results)
                    await db.commit()
                else:
                    calc.status = RiskCalculationStatus.FAILED
                    calc.metrics['error'] = str(task.info)
                    await db.commit()
    
    return RiskCalculationResponse(
        calculation_id=calc.id,
        status=calc.status.value,
        results=calc.metrics if calc.status == RiskCalculationStatus.COMPLETED else None,
        error=calc.metrics.get('error') if calc.status == RiskCalculationStatus.FAILED else None
    )


@router.post("/batch", response_model=Dict[str, RiskCalculationResponse])
async def batch_calculate_risk(
    request: BatchRiskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Calculate risk for multiple portfolios"""
    
    # Verify access to all portfolios
    portfolios = await db.execute(
        select(Portfolio).where(Portfolio.id.in_(request.portfolio_ids))
    )
    portfolios = portfolios.scalars().all()
    
    if len(portfolios) != len(request.portfolio_ids):
        raise HTTPException(status_code=404, detail="Some portfolios not found")
    
    for portfolio in portfolios:
        if not await _check_portfolio_access(portfolio, current_user, db):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to portfolio {portfolio.id}"
            )
    
    # Queue batch calculation
    from risk_compute_worker.app.worker import batch_risk_calculation
    
    task = batch_risk_calculation.apply_async(
        args=[
            [str(pid) for pid in request.portfolio_ids],
            request.metrics,
            request.parallel
        ]
    )
    
    # Create calculation records
    results = {}
    for portfolio_id in request.portfolio_ids:
        calc = RiskCalculation(
            portfolio_id=portfolio_id,
            calculation_type='batch',
            status=RiskCalculationStatus.PENDING,
            metrics={'task_id': task.id, 'batch': True}
        )
        db.add(calc)
        
        results[str(portfolio_id)] = RiskCalculationResponse(
            calculation_id=calc.id,
            status='pending',
            task_id=task.id
        )
    
    await db.commit()
    return results


@router.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(
    request: StressTestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run stress test scenarios on portfolio
    
    Scenarios can include:
    - Market crash scenarios
    - Interest rate shocks
    - Currency devaluations
    - Sector-specific events
    - Custom user-defined scenarios
    """
    
    portfolio = await db.get(Portfolio, request.portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Queue stress test
    task = run_stress_test.apply_async(
        args=[str(request.portfolio_id), request.scenarios]
    )
    
    # Store request
    calc = RiskCalculation(
        portfolio_id=request.portfolio_id,
        calculation_type='stress_test',
        status=RiskCalculationStatus.PENDING,
        stress_test_results={'task_id': task.id, 'scenarios': request.scenarios}
    )
    db.add(calc)
    await db.commit()
    
    return StressTestResponse(
        calculation_id=calc.id,
        status='pending',
        task_id=task.id,
        message=f"Stress test queued with {len(request.scenarios)} scenarios"
    )


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(
    request: MonteCarloRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run Monte Carlo simulation for portfolio risk
    
    - Generates thousands of potential market scenarios
    - Calculates probability distributions of returns
    - Provides confidence intervals for risk metrics
    """
    
    portfolio = await db.get(Portfolio, request.portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Queue Monte Carlo simulation
    task = run_monte_carlo_simulation.apply_async(
        args=[
            str(request.portfolio_id),
            request.num_simulations,
            request.time_horizon,
            request.confidence_levels
        ]
    )
    
    calc = RiskCalculation(
        portfolio_id=request.portfolio_id,
        calculation_type='monte_carlo',
        status=RiskCalculationStatus.PENDING,
        metrics={
            'task_id': task.id,
            'num_simulations': request.num_simulations,
            'time_horizon': request.time_horizon
        }
    )
    db.add(calc)
    await db.commit()
    
    return MonteCarloResponse(
        calculation_id=calc.id,
        status='pending',
        task_id=task.id,
        estimated_time=request.num_simulations / 1000  # Rough estimate in seconds
    )


@router.get("/portfolio/{portfolio_id}/summary", response_model=PortfolioRiskSummary)
@cache_result(ttl=60)
async def get_portfolio_risk_summary(
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_read_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive risk summary for portfolio"""
    
    portfolio = await db.get(Portfolio, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get latest risk calculation
    latest_calc = await db.execute(
        select(RiskCalculation)
        .where(
            and_(
                RiskCalculation.portfolio_id == portfolio_id,
                RiskCalculation.status == RiskCalculationStatus.COMPLETED
            )
        )
        .order_by(RiskCalculation.calculated_at.desc())
        .limit(1)
    )
    latest_calc = latest_calc.scalar_one_or_none()
    
    if not latest_calc:
        raise HTTPException(
            status_code=404,
            detail="No risk calculations found for portfolio"
        )
    
    # Get risk trends (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    risk_trends = await db.execute(
        select(
            func.date(RiskCalculation.calculated_at).label('date'),
            func.avg(RiskCalculation.var_95).label('avg_var'),
            func.avg(RiskCalculation.volatility).label('avg_volatility'),
            func.avg(RiskCalculation.sharpe_ratio).label('avg_sharpe')
        )
        .where(
            and_(
                RiskCalculation.portfolio_id == portfolio_id,
                RiskCalculation.calculated_at >= thirty_days_ago,
                RiskCalculation.status == RiskCalculationStatus.COMPLETED
            )
        )
        .group_by(func.date(RiskCalculation.calculated_at))
        .order_by(func.date(RiskCalculation.calculated_at))
    )
    
    trends = [dict(row) for row in risk_trends]
    
    return PortfolioRiskSummary(
        portfolio_id=portfolio_id,
        current_metrics=latest_calc.metrics,
        var_95=latest_calc.var_95,
        cvar_95=latest_calc.cvar_95,
        volatility=latest_calc.volatility,
        sharpe_ratio=latest_calc.sharpe_ratio,
        max_drawdown=latest_calc.max_drawdown,
        calculated_at=latest_calc.calculated_at,
        risk_trends=trends,
        risk_score=_calculate_risk_score(latest_calc),
        alerts=await _get_active_alerts(portfolio_id, db)
    )


@router.put("/portfolio/{portfolio_id}/limits")
async def update_risk_limits(
    portfolio_id: UUID,
    limits: RiskLimitsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update risk limits for portfolio"""
    
    portfolio = await db.get(Portfolio, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check permissions (only managers and owners)
    if not await _check_portfolio_access(
        portfolio, current_user, db, required_role='manager'
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Update limits
    for field, value in limits.dict(exclude_unset=True).items():
        setattr(portfolio, field, value)
    
    portfolio.updated_at = datetime.utcnow()
    await db.commit()
    
    # Trigger risk breach check
    calculate_urgent_risk.delay(
        str(portfolio_id),
        'limits_updated',
        ['var', 'max_drawdown', 'volatility']
    )
    
    return {"message": "Risk limits updated successfully"}


@router.websocket("/ws/{portfolio_id}")
async def websocket_risk_updates(
    websocket: WebSocket,
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time risk updates
    
    Streams:
    - Risk calculation updates
    - Alert notifications
    - Market data changes affecting portfolio
    """
    
    await websocket.accept()
    
    try:
        # Authenticate WebSocket connection
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")
        user = await _authenticate_websocket(token, db)
        
        if not user:
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        # Check portfolio access
        portfolio = await db.get(Portfolio, portfolio_id)
        if not await _check_portfolio_access(portfolio, user, db):
            await websocket.close(code=1008, reason="Access denied")
            return
        
        # Subscribe to portfolio events
        async with EventPublisher.subscribe(f"portfolio.{portfolio_id}.*") as subscriber:
            while True:
                # Check for new events
                event = await subscriber.get_message()
                
                if event:
                    await websocket.send_json({
                        "type": event["type"],
                        "data": event["data"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Send heartbeat
                await asyncio.sleep(30)
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for portfolio {portfolio_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")


@router.get("/metrics/historical")
async def get_historical_metrics(
    portfolio_id: UUID,
    start_date: datetime = Query(..., description="Start date for historical data"),
    end_date: datetime = Query(default=None, description="End date (default: now)"),
    metrics: List[str] = Query(default=['var', 'volatility', 'sharpe_ratio']),
    interval: str = Query(default='daily', pattern='^(hourly|daily|weekly|monthly)$'),
    db: AsyncSession = Depends(get_read_db),
    current_user: User = Depends(get_current_user)
):
    """Get historical risk metrics for portfolio"""
    
    portfolio = await db.get(Portfolio, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not end_date:
        end_date = datetime.utcnow()
    
    # Query historical calculations
    query = select(RiskCalculation).where(
        and_(
            RiskCalculation.portfolio_id == portfolio_id,
            RiskCalculation.calculated_at >= start_date,
            RiskCalculation.calculated_at <= end_date,
            RiskCalculation.status == RiskCalculationStatus.COMPLETED
        )
    ).order_by(RiskCalculation.calculated_at)
    
    results = await db.execute(query)
    calculations = results.scalars().all()
    
    # Format response
    historical_data = []
    for calc in calculations:
        data_point = {
            'timestamp': calc.calculated_at.isoformat(),
            'calculation_id': str(calc.id)
        }
        
        for metric in metrics:
            if hasattr(calc, metric):
                data_point[metric] = getattr(calc, metric)
            elif metric in calc.metrics:
                data_point[metric] = calc.metrics[metric]
        
        historical_data.append(data_point)
    
    return {
        'portfolio_id': str(portfolio_id),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'interval': interval,
        'data_points': len(historical_data),
        'data': historical_data
    }


@router.post("/export/{portfolio_id}")
async def export_risk_report(
    portfolio_id: UUID,
    format: str = Query(default='pdf', pattern='^(pdf|excel|csv)$'),
    include_history: bool = Query(default=True),
    db: AsyncSession = Depends(get_read_db),
    current_user: User = Depends(get_current_user)
):
    """Export comprehensive risk report"""
    
    portfolio = await db.get(Portfolio, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    if not await _check_portfolio_access(portfolio, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Generate report (this would use a reporting service)
    report_data = await _generate_risk_report(
        portfolio_id, format, include_history, db
    )
    
    # Return file stream
    return StreamingResponse(
        report_data,
        media_type={
            'pdf': 'application/pdf',
            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'csv': 'text/csv'
        }[format],
        headers={
            'Content-Disposition': f'attachment; filename="risk_report_{portfolio_id}_{datetime.utcnow().date()}.{format}"'
        }
    )


# Helper functions
async def _check_portfolio_access(
    portfolio: Portfolio,
    user: User,
    db: AsyncSession,
    required_role: str = 'viewer'
) -> bool:
    """Check if user has access to portfolio"""
    
    # Admin always has access
    if user.role == 'admin':
        return True
    
    # Check organization
    if portfolio.organization_id != user.organization_id:
        return False
    
    # Check portfolio-specific access
    if portfolio.owner_id == user.id:
        return True
    
    # Check portfolio users table
    # Implementation would check the many-to-many relationship
    
    return True  # Simplified for now


async def _authenticate_websocket(token: str, db: AsyncSession) -> Optional[User]:
    """Authenticate WebSocket connection"""
    # Implementation would validate JWT token
    return None


def _estimate_calculation_time(request: RiskCalculationRequest) -> float:
    """Estimate calculation time in seconds"""
    base_time = 1.0
    time_per_metric = 0.5
    
    estimated = base_time + (len(request.metrics) * time_per_metric)
    
    if request.priority == 'urgent':
        estimated *= 0.5
    elif request.priority == 'low':
        estimated *= 2
    
    return estimated


def _calculate_risk_score(calc: RiskCalculation) -> float:
    """Calculate overall risk score (0-100)"""
    # Simplified scoring algorithm
    score = 50.0
    
    if calc.var_95:
        score += min(calc.var_95 / 1000, 20)
    
    if calc.volatility:
        score += min(calc.volatility * 100, 20)
    
    if calc.sharpe_ratio and calc.sharpe_ratio < 1:
        score += 10
    
    return min(max(score, 0), 100)


async def _get_active_alerts(portfolio_id: UUID, db: AsyncSession) -> List[Dict]:
    """Get active alerts for portfolio"""
    # Implementation would query alerts table
    return []


async def _generate_risk_report(
    portfolio_id: UUID,
    format: str,
    include_history: bool,
    db: AsyncSession
) -> bytes:
    """Generate risk report in specified format"""
    # Implementation would use reporting library
    return b"Report content"
