"""
Enterprise Risk Calculation Engine
High-performance risk metrics calculation with caching and parallel processing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, partial

logger = logging.getLogger(__name__)


class RiskMetric(str, Enum):
    """Available risk metrics"""
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"
    TREYNOR_RATIO = "treynor_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    DOWNSIDE_DEVIATION = "downside_deviation"
    OMEGA_RATIO = "omega_ratio"
    KURTOSIS = "kurtosis"
    SKEWNESS = "skewness"


class TimeHorizon(str, Enum):
    """Risk calculation time horizons"""
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    QUARTERLY = "3m"
    YEARLY = "1y"


class RiskEngine:
    """
    Enterprise-grade risk calculation engine with advanced features:
    - Parallel processing for large portfolios
    - Caching of intermediate results
    - Support for multiple risk metrics
    - Stress testing and scenario analysis
    - Monte Carlo simulations
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    async def calculate_portfolio_risk(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        metrics: List[RiskMetric],
        confidence_level: float = 0.95,
        time_horizon: TimeHorizon = TimeHorizon.DAILY,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive risk metrics for a portfolio
        
        Args:
            positions: DataFrame with columns ['symbol', 'quantity', 'price', 'weight']
            returns: Historical returns data
            metrics: List of risk metrics to calculate
            confidence_level: Confidence level for VaR/CVaR calculations
            time_horizon: Time horizon for risk calculations
            benchmark_returns: Benchmark returns for relative metrics
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        
        Returns:
            Dictionary containing calculated risk metrics
        """
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, returns)
        
        # Prepare tasks for parallel execution
        tasks = []
        for metric in metrics:
            task = self._calculate_metric(
                metric,
                portfolio_returns,
                confidence_level,
                time_horizon,
                benchmark_returns,
                risk_free_rate
            )
            tasks.append(task)
        
        # Execute calculations in parallel
        results = await asyncio.gather(*tasks)
        
        # Combine results
        risk_metrics = {}
        for metric, value in zip(metrics, results):
            risk_metrics[metric.value] = value
        
        # Add metadata
        risk_metrics['metadata'] = {
            'calculated_at': datetime.utcnow().isoformat(),
            'confidence_level': confidence_level,
            'time_horizon': time_horizon.value,
            'portfolio_size': len(positions),
            'calculation_method': 'parallel_processing'
        }
        
        return risk_metrics
    
    def _calculate_portfolio_returns(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted portfolio returns"""
        
        # Ensure weights sum to 1
        weights = positions['weight'].values
        weights = weights / weights.sum()
        
        # Calculate weighted returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        return portfolio_returns
    
    async def _calculate_metric(
        self,
        metric: RiskMetric,
        returns: pd.Series,
        confidence_level: float,
        time_horizon: TimeHorizon,
        benchmark_returns: Optional[pd.Series],
        risk_free_rate: float
    ) -> Union[float, Dict]:
        """Calculate a specific risk metric"""
        
        # Check cache
        cache_key = f"{metric}_{confidence_level}_{time_horizon}_{len(returns)}"
        if self.enable_cache and cache_key in self._cache:
            cached_time, cached_value = self._cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl):
                logger.debug(f"Using cached value for {metric}")
                return cached_value
        
        # Scale returns to time horizon
        scaled_returns = self._scale_returns(returns, time_horizon)
        
        # Calculate metric
        if metric == RiskMetric.VAR:
            value = self._calculate_var(scaled_returns, confidence_level)
        elif metric == RiskMetric.CVAR:
            value = self._calculate_cvar(scaled_returns, confidence_level)
        elif metric == RiskMetric.EXPECTED_SHORTFALL:
            value = self._calculate_expected_shortfall(scaled_returns, confidence_level)
        elif metric == RiskMetric.SHARPE_RATIO:
            value = self._calculate_sharpe_ratio(scaled_returns, risk_free_rate, time_horizon)
        elif metric == RiskMetric.SORTINO_RATIO:
            value = self._calculate_sortino_ratio(scaled_returns, risk_free_rate, time_horizon)
        elif metric == RiskMetric.CALMAR_RATIO:
            value = self._calculate_calmar_ratio(scaled_returns)
        elif metric == RiskMetric.MAX_DRAWDOWN:
            value = self._calculate_max_drawdown(scaled_returns)
        elif metric == RiskMetric.VOLATILITY:
            value = self._calculate_volatility(scaled_returns, time_horizon)
        elif metric == RiskMetric.BETA:
            value = self._calculate_beta(scaled_returns, benchmark_returns)
        elif metric == RiskMetric.ALPHA:
            value = self._calculate_alpha(scaled_returns, benchmark_returns, risk_free_rate)
        elif metric == RiskMetric.TREYNOR_RATIO:
            value = self._calculate_treynor_ratio(scaled_returns, benchmark_returns, risk_free_rate)
        elif metric == RiskMetric.INFORMATION_RATIO:
            value = self._calculate_information_ratio(scaled_returns, benchmark_returns)
        elif metric == RiskMetric.TRACKING_ERROR:
            value = self._calculate_tracking_error(scaled_returns, benchmark_returns)
        elif metric == RiskMetric.DOWNSIDE_DEVIATION:
            value = self._calculate_downside_deviation(scaled_returns, risk_free_rate)
        elif metric == RiskMetric.OMEGA_RATIO:
            value = self._calculate_omega_ratio(scaled_returns, risk_free_rate)
        elif metric == RiskMetric.KURTOSIS:
            value = float(stats.kurtosis(scaled_returns))
        elif metric == RiskMetric.SKEWNESS:
            value = float(stats.skew(scaled_returns))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = (datetime.utcnow(), value)
        
        return value
    
    def _scale_returns(self, returns: pd.Series, time_horizon: TimeHorizon) -> pd.Series:
        """Scale returns to specified time horizon"""
        
        scaling_factors = {
            TimeHorizon.DAILY: 1,
            TimeHorizon.WEEKLY: 5,
            TimeHorizon.MONTHLY: 21,
            TimeHorizon.QUARTERLY: 63,
            TimeHorizon.YEARLY: 252
        }
        
        factor = scaling_factors[time_horizon]
        
        if factor == 1:
            return returns
        
        # Aggregate returns over the time horizon
        scaled_returns = returns.rolling(window=factor).sum().dropna()
        
        return scaled_returns
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence_level)
        return float(returns[returns <= var].mean())
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (same as CVaR)"""
        return self._calculate_cvar(returns, confidence_level)
    
    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
        time_horizon: TimeHorizon
    ) -> float:
        """Calculate Sharpe Ratio"""
        
        # Annualize based on time horizon
        periods_per_year = {
            TimeHorizon.DAILY: 252,
            TimeHorizon.WEEKLY: 52,
            TimeHorizon.MONTHLY: 12,
            TimeHorizon.QUARTERLY: 4,
            TimeHorizon.YEARLY: 1
        }
        
        periods = periods_per_year[time_horizon]
        
        excess_returns = returns - risk_free_rate / periods
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float(np.sqrt(periods) * excess_returns.mean() / excess_returns.std())
    
    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
        time_horizon: TimeHorizon
    ) -> float:
        """Calculate Sortino Ratio"""
        
        periods_per_year = {
            TimeHorizon.DAILY: 252,
            TimeHorizon.WEEKLY: 52,
            TimeHorizon.MONTHLY: 12,
            TimeHorizon.QUARTERLY: 4,
            TimeHorizon.YEARLY: 1
        }
        
        periods = periods_per_year[time_horizon]
        
        excess_returns = returns - risk_free_rate / periods
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        return float(np.sqrt(periods) * excess_returns.mean() / downside_deviation)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar Ratio"""
        
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        # Annualized return
        annual_return = returns.mean() * 252
        
        return float(annual_return / abs(max_dd))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return float(drawdowns.min())
    
    def _calculate_volatility(self, returns: pd.Series, time_horizon: TimeHorizon) -> float:
        """Calculate Volatility (annualized)"""
        
        periods_per_year = {
            TimeHorizon.DAILY: 252,
            TimeHorizon.WEEKLY: 52,
            TimeHorizon.MONTHLY: 12,
            TimeHorizon.QUARTERLY: 4,
            TimeHorizon.YEARLY: 1
        }
        
        periods = periods_per_year[time_horizon]
        
        return float(returns.std() * np.sqrt(periods))
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Beta"""
        
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 1.0
        
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 1.0
        
        covariance = aligned['portfolio'].cov(aligned['benchmark'])
        benchmark_variance = aligned['benchmark'].var()
        
        if benchmark_variance == 0:
            return 1.0
        
        return float(covariance / benchmark_variance)
    
    def _calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> float:
        """Calculate Alpha (Jensen's Alpha)"""
        
        if benchmark_returns is None:
            return 0.0
        
        beta = self._calculate_beta(returns, benchmark_returns)
        
        # Annualized returns
        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        return float(alpha)
    
    def _calculate_treynor_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> float:
        """Calculate Treynor Ratio"""
        
        beta = self._calculate_beta(returns, benchmark_returns)
        
        if beta == 0:
            return 0.0
        
        # Annualized return
        portfolio_return = returns.mean() * 252
        
        return float((portfolio_return - risk_free_rate) / beta)
    
    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information Ratio"""
        
        if benchmark_returns is None:
            return 0.0
        
        # Calculate tracking error
        tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
        
        if tracking_error == 0:
            return 0.0
        
        # Active return
        active_return = (returns.mean() - benchmark_returns.mean()) * 252
        
        return float(active_return / tracking_error)
    
    def _calculate_tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Tracking Error"""
        
        if benchmark_returns is None:
            return 0.0
        
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        # Calculate active returns
        active_returns = aligned['portfolio'] - aligned['benchmark']
        
        # Annualized tracking error
        return float(active_returns.std() * np.sqrt(252))
    
    def _calculate_downside_deviation(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """Calculate Downside Deviation"""
        
        downside_returns = returns[returns < threshold]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return float(np.sqrt(np.mean((downside_returns - threshold) ** 2)) * np.sqrt(252))
    
    def _calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """Calculate Omega Ratio"""
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0
        
        return float(gains.sum() / losses.sum())
    
    async def run_stress_test(
        self,
        positions: pd.DataFrame,
        scenarios: Dict[str, Dict[str, float]],
        base_prices: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Run stress tests on portfolio
        
        Args:
            positions: Portfolio positions
            scenarios: Dictionary of scenarios with asset price changes
            base_prices: Current asset prices
        
        Returns:
            Stress test results for each scenario
        """
        
        results = {}
        
        for scenario_name, price_changes in scenarios.items():
            # Apply scenario to prices
            stressed_prices = base_prices.copy()
            for asset, change_pct in price_changes.items():
                if asset in stressed_prices.index:
                    stressed_prices[asset] *= (1 + change_pct)
            
            # Calculate portfolio value under stress
            base_value = (positions['quantity'] * positions['price']).sum()
            stressed_value = (positions['quantity'] * stressed_prices[positions['symbol']]).sum()
            
            # Calculate impact
            value_change = stressed_value - base_value
            pct_change = (value_change / base_value) * 100
            
            results[scenario_name] = {
                'base_value': float(base_value),
                'stressed_value': float(stressed_value),
                'value_change': float(value_change),
                'pct_change': float(pct_change),
                'scenario_severity': 'severe' if abs(pct_change) > 20 else 'moderate' if abs(pct_change) > 10 else 'mild'
            }
        
        return results
    
    async def monte_carlo_simulation(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        num_simulations: int = 10000,
        time_horizon: int = 252,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Run Monte Carlo simulation for portfolio risk
        
        Args:
            positions: Portfolio positions
            returns: Historical returns
            num_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            confidence_levels: Confidence levels for VaR calculation
        
        Returns:
            Monte Carlo simulation results
        """
        
        # Calculate portfolio statistics
        portfolio_returns = self._calculate_portfolio_returns(positions, returns)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        
        # Generate random paths
        dt = 1 / 252  # Daily time step
        
        # Run simulation in parallel
        loop = asyncio.get_event_loop()
        simulation_results = await loop.run_in_executor(
            self.executor,
            self._run_monte_carlo_paths,
            mu,
            sigma,
            dt,
            time_horizon,
            num_simulations
        )
        
        # Calculate statistics
        final_values = simulation_results[:, -1]
        
        results = {
            'mean_return': float(np.mean(final_values)),
            'std_return': float(np.std(final_values)),
            'min_return': float(np.min(final_values)),
            'max_return': float(np.max(final_values)),
            'paths_sample': simulation_results[:100].tolist()  # Sample paths for visualization
        }
        
        # Calculate VaR at different confidence levels
        for conf_level in confidence_levels:
            var_value = np.percentile(final_values, (1 - conf_level) * 100)
            cvar_value = final_values[final_values <= var_value].mean()
            
            results[f'var_{int(conf_level*100)}'] = float(var_value)
            results[f'cvar_{int(conf_level*100)}'] = float(cvar_value)
        
        # Calculate probability of loss
        prob_loss = (final_values < 0).mean()
        results['probability_of_loss'] = float(prob_loss)
        
        return results
    
    @staticmethod
    def _run_monte_carlo_paths(
        mu: float,
        sigma: float,
        dt: float,
        time_horizon: int,
        num_simulations: int
    ) -> np.ndarray:
        """Run Monte Carlo simulation paths (for parallel execution)"""
        
        # Generate random walks
        random_walks = np.random.normal(
            mu * dt,
            sigma * np.sqrt(dt),
            size=(num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns
        paths = np.cumprod(1 + random_walks, axis=1)
        
        # Convert to returns
        paths = paths - 1
        
        return paths
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self._cache.clear()
