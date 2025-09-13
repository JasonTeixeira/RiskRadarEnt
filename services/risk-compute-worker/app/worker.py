"""
Risk Compute Worker - Celery-based async processing
Enterprise-grade background task processing for risk calculations
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure, task_retry
from kombu import Queue, Exchange
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import asyncio

# Add libs to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs" / "python"))

from risk_models.risk_engine import RiskEngine, RiskMetric, TimeHorizon

# Metrics
task_counter = Counter(
    'celery_task_total',
    'Total number of Celery tasks',
    ['task_name', 'status']
)
task_duration = Histogram(
    'celery_task_duration_seconds',
    'Celery task duration',
    ['task_name']
)
active_tasks = Gauge(
    'celery_active_tasks',
    'Number of active Celery tasks',
    ['task_name']
)

logger = logging.getLogger(__name__)

# Celery configuration
app = Celery('risk_compute_worker')

app.conf.update(
    broker_url='redis://redis:6379/0',
    result_backend='redis://redis:6379/0',
    
    # Task configuration
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    task_acks_late=True,
    
    # Worker configuration
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_disable_rate_limits=False,
    
    # Queue configuration
    task_default_queue='risk_calculations',
    task_queues=(
        Queue('risk_calculations', Exchange('risk', type='topic'), routing_key='risk.calculate'),
        Queue('risk_urgent', Exchange('risk', type='topic'), routing_key='risk.urgent'),
        Queue('risk_batch', Exchange('risk', type='topic'), routing_key='risk.batch'),
        Queue('stress_tests', Exchange('risk', type='topic'), routing_key='risk.stress'),
        Queue('monte_carlo', Exchange('risk', type='topic'), routing_key='risk.monte_carlo'),
    ),
    
    # Routing
    task_routes={
        'worker.calculate_portfolio_risk': {'queue': 'risk_calculations'},
        'worker.calculate_urgent_risk': {'queue': 'risk_urgent'},
        'worker.batch_risk_calculation': {'queue': 'risk_batch'},
        'worker.run_stress_test': {'queue': 'stress_tests'},
        'worker.run_monte_carlo': {'queue': 'monte_carlo'},
    },
    
    # Result backend configuration
    result_expires=3600,  # Results expire after 1 hour
    result_compression='gzip',
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'calculate-eod-risk': {
            'task': 'worker.calculate_eod_risk_all_portfolios',
            'schedule': timedelta(hours=24),
            'options': {'queue': 'risk_batch'}
        },
        'update-market-data': {
            'task': 'worker.update_market_data',
            'schedule': timedelta(minutes=15),
            'options': {'queue': 'risk_calculations'}
        },
        'check-risk-breaches': {
            'task': 'worker.check_risk_breaches',
            'schedule': timedelta(minutes=5),
            'options': {'queue': 'risk_urgent'}
        },
    },
)


class RiskCalculationTask(Task):
    """Base task class with enterprise features"""
    
    autoretry_for = (Exception,)
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def __init__(self):
        super().__init__()
        self.risk_engine = None
    
    def before_start(self, task_id, args, kwargs):
        """Initialize resources before task execution"""
        if not self.risk_engine:
            self.risk_engine = RiskEngine(
                max_workers=4,
                enable_cache=True,
                cache_ttl=300
            )
        active_tasks.labels(task_name=self.name).inc()
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion"""
        task_counter.labels(task_name=self.name, status='success').inc()
        active_tasks.labels(task_name=self.name).dec()
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {self.name} failed: {exc}\n{einfo}")
        task_counter.labels(task_name=self.name, status='failure').inc()
        active_tasks.labels(task_name=self.name).dec()
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {self.name} retrying: {exc}")
        task_counter.labels(task_name=self.name, status='retry').inc()


@app.task(base=RiskCalculationTask, name='worker.calculate_portfolio_risk')
def calculate_portfolio_risk(
    portfolio_id: str,
    metrics: List[str],
    confidence_level: float = 0.95,
    time_horizon: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate risk metrics for a portfolio
    
    Args:
        portfolio_id: Portfolio identifier
        metrics: List of risk metrics to calculate
        confidence_level: Confidence level for VaR/CVaR
        time_horizon: Time horizon for calculations
        use_cache: Whether to use cached results
    
    Returns:
        Dictionary containing calculated risk metrics
    """
    start_time = time.time()
    
    try:
        # Convert string metrics to enum
        risk_metrics = [RiskMetric(m) for m in metrics]
        horizon = TimeHorizon(time_horizon)
        
        # Fetch portfolio data (mock for now)
        positions = _fetch_portfolio_positions(portfolio_id)
        returns = _fetch_historical_returns(portfolio_id)
        
        # Initialize risk engine
        risk_engine = RiskEngine(enable_cache=use_cache)
        
        # Run async calculation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            risk_engine.calculate_portfolio_risk(
                positions=positions,
                returns=returns,
                metrics=risk_metrics,
                confidence_level=confidence_level,
                time_horizon=horizon
            )
        )
        
        # Add calculation metadata
        results['calculation_metadata'] = {
            'portfolio_id': portfolio_id,
            'task_id': calculate_portfolio_risk.request.id,
            'calculation_time': time.time() - start_time,
            'worker_hostname': calculate_portfolio_risk.request.hostname,
        }
        
        # Store results in database
        _store_risk_results(portfolio_id, results)
        
        # Send notification if risk breach detected
        _check_and_notify_risk_breaches(portfolio_id, results)
        
        task_duration.labels(task_name='calculate_portfolio_risk').observe(time.time() - start_time)
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating risk for portfolio {portfolio_id}: {e}")
        raise


@app.task(base=RiskCalculationTask, name='worker.calculate_urgent_risk')
def calculate_urgent_risk(
    portfolio_id: str,
    trigger_event: str,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    High-priority risk calculation for urgent events
    
    Args:
        portfolio_id: Portfolio identifier
        trigger_event: Event that triggered the calculation
        metrics: Specific metrics to calculate (all if None)
    """
    
    # Default to critical metrics if not specified
    if not metrics:
        metrics = ['var', 'cvar', 'max_drawdown', 'volatility']
    
    logger.info(f"Urgent risk calculation triggered by {trigger_event} for portfolio {portfolio_id}")
    
    # Calculate with high priority
    results = calculate_portfolio_risk(
        portfolio_id=portfolio_id,
        metrics=metrics,
        use_cache=False  # Don't use cache for urgent calculations
    )
    
    results['trigger_event'] = trigger_event
    results['priority'] = 'urgent'
    
    return results


@app.task(base=RiskCalculationTask, name='worker.batch_risk_calculation')
def batch_risk_calculation(
    portfolio_ids: List[str],
    metrics: List[str],
    parallel: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Batch risk calculation for multiple portfolios
    
    Args:
        portfolio_ids: List of portfolio identifiers
        metrics: Risk metrics to calculate
        parallel: Whether to process portfolios in parallel
    """
    
    results = {}
    
    if parallel:
        # Create subtasks for parallel execution
        job = app.group([
            calculate_portfolio_risk.s(pid, metrics)
            for pid in portfolio_ids
        ])
        
        # Execute and collect results
        job_results = job.apply_async()
        
        for pid, result in zip(portfolio_ids, job_results.get(timeout=300)):
            results[pid] = result
    else:
        # Sequential processing
        for pid in portfolio_ids:
            try:
                results[pid] = calculate_portfolio_risk(pid, metrics)
            except Exception as e:
                logger.error(f"Failed to calculate risk for portfolio {pid}: {e}")
                results[pid] = {'error': str(e)}
    
    return results


@app.task(base=RiskCalculationTask, name='worker.run_stress_test')
def run_stress_test(
    portfolio_id: str,
    scenarios: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Run stress test scenarios on portfolio
    
    Args:
        portfolio_id: Portfolio identifier
        scenarios: Stress test scenarios
    """
    
    try:
        # Fetch portfolio data
        positions = _fetch_portfolio_positions(portfolio_id)
        base_prices = _fetch_current_prices(portfolio_id)
        
        # Initialize risk engine
        risk_engine = RiskEngine()
        
        # Run stress test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            risk_engine.run_stress_test(
                positions=positions,
                scenarios=scenarios,
                base_prices=base_prices
            )
        )
        
        # Store results
        _store_stress_test_results(portfolio_id, results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running stress test for portfolio {portfolio_id}: {e}")
        raise


@app.task(base=RiskCalculationTask, name='worker.run_monte_carlo')
def run_monte_carlo_simulation(
    portfolio_id: str,
    num_simulations: int = 10000,
    time_horizon: int = 252,
    confidence_levels: List[float] = None
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for portfolio
    
    Args:
        portfolio_id: Portfolio identifier
        num_simulations: Number of simulation paths
        time_horizon: Time horizon in days
        confidence_levels: Confidence levels for VaR calculation
    """
    
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    
    try:
        # Fetch portfolio data
        positions = _fetch_portfolio_positions(portfolio_id)
        returns = _fetch_historical_returns(portfolio_id)
        
        # Initialize risk engine
        risk_engine = RiskEngine()
        
        # Run Monte Carlo simulation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            risk_engine.monte_carlo_simulation(
                positions=positions,
                returns=returns,
                num_simulations=num_simulations,
                time_horizon=time_horizon,
                confidence_levels=confidence_levels
            )
        )
        
        # Store results
        _store_monte_carlo_results(portfolio_id, results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo for portfolio {portfolio_id}: {e}")
        raise


@app.task(name='worker.calculate_eod_risk_all_portfolios')
def calculate_eod_risk_all_portfolios():
    """Calculate end-of-day risk for all active portfolios"""
    
    # Fetch all active portfolios
    portfolio_ids = _fetch_active_portfolio_ids()
    
    logger.info(f"Starting EOD risk calculation for {len(portfolio_ids)} portfolios")
    
    # Calculate in batches
    batch_size = 10
    for i in range(0, len(portfolio_ids), batch_size):
        batch = portfolio_ids[i:i+batch_size]
        batch_risk_calculation.delay(
            portfolio_ids=batch,
            metrics=['var', 'cvar', 'sharpe_ratio', 'volatility', 'max_drawdown']
        )
    
    return f"Scheduled EOD risk calculation for {len(portfolio_ids)} portfolios"


@app.task(name='worker.update_market_data')
def update_market_data():
    """Update market data from external providers"""
    
    try:
        # Fetch latest market data
        symbols = _fetch_tracked_symbols()
        
        for symbol in symbols:
            # Fetch and store market data
            data = _fetch_market_data(symbol)
            _store_market_data(symbol, data)
        
        return f"Updated market data for {len(symbols)} symbols"
        
    except Exception as e:
        logger.error(f"Error updating market data: {e}")
        raise


@app.task(name='worker.check_risk_breaches')
def check_risk_breaches():
    """Check for risk limit breaches across all portfolios"""
    
    breaches = []
    
    # Fetch portfolios with risk limits
    portfolios = _fetch_portfolios_with_risk_limits()
    
    for portfolio in portfolios:
        # Get latest risk metrics
        metrics = _fetch_latest_risk_metrics(portfolio['id'])
        
        # Check against limits
        for metric, value in metrics.items():
            limit = portfolio.get(f'{metric}_limit')
            if limit and value > limit:
                breach = {
                    'portfolio_id': portfolio['id'],
                    'metric': metric,
                    'value': value,
                    'limit': limit,
                    'breach_percentage': ((value - limit) / limit) * 100
                }
                breaches.append(breach)
                
                # Send alert
                _send_risk_breach_alert(breach)
    
    return f"Found {len(breaches)} risk breaches"


# Helper functions (these would connect to actual databases)
def _fetch_portfolio_positions(portfolio_id: str) -> pd.DataFrame:
    """Fetch portfolio positions from database"""
    # Mock implementation
    return pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'quantity': [100, 50, 75],
        'price': [150.0, 2800.0, 300.0],
        'weight': [0.3, 0.4, 0.3]
    })


def _fetch_historical_returns(portfolio_id: str) -> pd.DataFrame:
    """Fetch historical returns data"""
    # Mock implementation
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.02,
        index=dates,
        columns=['AAPL', 'GOOGL', 'MSFT']
    )
    return returns


def _fetch_current_prices(portfolio_id: str) -> pd.Series:
    """Fetch current asset prices"""
    # Mock implementation
    return pd.Series({
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 300.0
    })


def _fetch_active_portfolio_ids() -> List[str]:
    """Fetch all active portfolio IDs"""
    # Mock implementation
    return ['portfolio_1', 'portfolio_2', 'portfolio_3']


def _fetch_tracked_symbols() -> List[str]:
    """Fetch all tracked symbols"""
    # Mock implementation
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']


def _fetch_market_data(symbol: str) -> Dict:
    """Fetch market data for symbol"""
    # Mock implementation
    return {
        'open': 100.0,
        'high': 105.0,
        'low': 99.0,
        'close': 103.0,
        'volume': 1000000
    }


def _fetch_portfolios_with_risk_limits() -> List[Dict]:
    """Fetch portfolios that have risk limits set"""
    # Mock implementation
    return [
        {'id': 'portfolio_1', 'var_limit': 10000, 'volatility_limit': 0.25},
        {'id': 'portfolio_2', 'max_drawdown_limit': 0.20}
    ]


def _fetch_latest_risk_metrics(portfolio_id: str) -> Dict[str, float]:
    """Fetch latest risk metrics for portfolio"""
    # Mock implementation
    return {
        'var': 9500,
        'volatility': 0.22,
        'max_drawdown': 0.15
    }


def _store_risk_results(portfolio_id: str, results: Dict):
    """Store risk calculation results in database"""
    logger.info(f"Storing risk results for portfolio {portfolio_id}")


def _store_stress_test_results(portfolio_id: str, results: Dict):
    """Store stress test results"""
    logger.info(f"Storing stress test results for portfolio {portfolio_id}")


def _store_monte_carlo_results(portfolio_id: str, results: Dict):
    """Store Monte Carlo results"""
    logger.info(f"Storing Monte Carlo results for portfolio {portfolio_id}")


def _store_market_data(symbol: str, data: Dict):
    """Store market data"""
    logger.info(f"Storing market data for {symbol}")


def _check_and_notify_risk_breaches(portfolio_id: str, results: Dict):
    """Check for risk breaches and send notifications"""
    # Implementation would check against limits and send alerts
    pass


def _send_risk_breach_alert(breach: Dict):
    """Send risk breach alert"""
    logger.warning(f"Risk breach detected: {breach}")


# Signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **kw):
    """Handle task pre-run signal"""
    logger.debug(f"Task {task.name} starting with id {task_id}")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **kw):
    """Handle task post-run signal"""
    logger.debug(f"Task {task.name} completed with state {state}")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Handle task failure signal"""
    logger.error(f"Task {task_id} failed: {exception}")


if __name__ == '__main__':
    app.start()
