"""
Base classes and interfaces for market data providers.

This module defines the abstract base classes and common functionality
for all market data providers in the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Protocol
import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    INDEX = "index"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"


class DataFrequency(str, Enum):
    """Market data frequency types."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class MarketDataType(str, Enum):
    """Types of market data."""
    PRICE = "price"
    VOLUME = "volume"
    OHLCV = "ohlcv"
    QUOTE = "quote"
    TRADE = "trade"
    ORDER_BOOK = "order_book"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    NEWS = "news"
    SENTIMENT = "sentiment"


@dataclass
class MarketQuote:
    """Market quote data."""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    last_price: Decimal
    volume: int
    exchange: Optional[str] = None
    currency: str = "USD"
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        if self.mid_price == 0:
            return Decimal(0)
        return (self.spread / self.mid_price) * 100


@dataclass
class OHLCV:
    """OHLCV (Open, High, Low, Close, Volume) data."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal] = None
    trades: Optional[int] = None
    currency: str = "USD"
    
    @property
    def range(self) -> Decimal:
        """Calculate price range."""
        return self.high - self.low
    
    @property
    def change(self) -> Decimal:
        """Calculate price change."""
        return self.close - self.open
    
    @property
    def change_percentage(self) -> Decimal:
        """Calculate percentage change."""
        if self.open == 0:
            return Decimal(0)
        return (self.change / self.open) * 100


class MarketDataError(Exception):
    """Base exception for market data errors."""
    pass


class ProviderError(MarketDataError):
    """Provider-specific error."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(ProviderError):
    """Authentication failed error."""
    pass


class DataNotFoundError(MarketDataError):
    """Requested data not found."""
    pass


class ProviderConfig(BaseModel):
    """Configuration for market data provider."""
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_secret: Optional[str] = Field(None, description="API secret for authentication")
    base_url: str = Field(..., description="Base URL for API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    rate_limit: Optional[int] = Field(None, description="Rate limit (requests per minute)")
    use_sandbox: bool = Field(False, description="Use sandbox/test environment")
    
    class Config:
        extra = "allow"  # Allow provider-specific extra fields


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter: Optional[asyncio.Semaphore] = None
        
        if config.rate_limit:
            self._rate_limiter = asyncio.Semaphore(config.rate_limit)
    
    @property
    def name(self) -> str:
        """Get provider name."""
        return self.__class__.__name__.replace("Provider", "")
    
    @asynccontextmanager
    async def client(self):
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers=self._get_headers()
            )
        try:
            yield self._client
        finally:
            pass  # Keep client alive for connection pooling
    
    async def close(self):
        """Close provider connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests.
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "User-Agent": f"RiskRadar-Enterprise/1.0 ({self.name}Provider)",
            "Accept": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            
        Returns:
            Response data
            
        Raises:
            ProviderError: If request fails
        """
        async with self.client() as client:
            try:
                # Apply rate limiting if configured
                if self._rate_limiter:
                    async with self._rate_limiter:
                        response = await client.request(
                            method=method,
                            url=endpoint,
                            params=params,
                            json=json
                        )
                else:
                    response = await client.request(
                        method=method,
                        url=endpoint,
                        params=params,
                        json=json
                    )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", 60)
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after} seconds"
                    )
                
                # Check for authentication errors
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed: {response.text}"
                    )
                
                # Check for not found
                if response.status_code == 404:
                    raise DataNotFoundError(
                        f"Data not found: {endpoint}"
                    )
                
                # Check for other errors
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    f"HTTP error {e.response.status_code}: {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise ProviderError(
                    f"Request error: {str(e)}"
                ) from e
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketQuote:
        """Get current quote for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Market quote
            
        Raises:
            DataNotFoundError: If symbol not found
            ProviderError: If request fails
        """
        pass
    
    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """Get current quotes for multiple symbols.
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary of symbol to market quote
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> List[OHLCV]:
        """Get historical OHLCV data.
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
            
        Raises:
            DataNotFoundError: If symbol not found
            ProviderError: If request fails
        """
        pass
    
    @abstractmethod
    async def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get asset information.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset information dictionary
        """
        pass
    
    @abstractmethod
    async def search_symbols(
        self,
        query: str,
        asset_class: Optional[AssetClass] = None
    ) -> List[Dict[str, Any]]:
        """Search for symbols.
        
        Args:
            query: Search query
            asset_class: Optional asset class filter
            
        Returns:
            List of matching symbols with metadata
        """
        pass
    
    async def is_market_open(self, exchange: str = "NYSE") -> bool:
        """Check if market is open.
        
        Args:
            exchange: Exchange code
            
        Returns:
            True if market is open
        """
        # Default implementation - override in providers
        now = datetime.now()
        weekday = now.weekday()
        
        # Basic NYSE hours (9:30 AM - 4:00 PM ET)
        if weekday >= 5:  # Weekend
            return False
        
        # This is simplified - real implementation would check holidays
        # and convert to exchange timezone
        hour = now.hour
        minute = now.minute
        
        if exchange in ("NYSE", "NASDAQ"):
            # Simplified check for US markets
            market_open = (hour == 9 and minute >= 30) or (10 <= hour < 16)
            return market_open
        
        return True  # Default to open for unknown exchanges
    
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate if symbols exist.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary of symbol to validation status
        """
        results = {}
        for symbol in symbols:
            try:
                await self.get_asset_info(symbol)
                results[symbol] = True
            except DataNotFoundError:
                results[symbol] = False
            except Exception as e:
                logger.warning(f"Error validating {symbol}: {e}")
                results[symbol] = False
        
        return results


class MarketDataAggregator:
    """Aggregates data from multiple providers."""
    
    def __init__(self, providers: List[MarketDataProvider]):
        """Initialize aggregator with providers.
        
        Args:
            providers: List of market data providers
        """
        self.providers = providers
        self._provider_weights = {p.name: 1.0 for p in providers}
    
    def set_provider_weight(self, provider_name: str, weight: float):
        """Set provider weight for aggregation.
        
        Args:
            provider_name: Provider name
            weight: Weight (0.0 to 1.0)
        """
        self._provider_weights[provider_name] = weight
    
    async def get_best_quote(self, symbol: str) -> MarketQuote:
        """Get best quote from all providers.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Best available quote
            
        Raises:
            DataNotFoundError: If no provider has data
        """
        quotes = []
        errors = []
        
        # Fetch from all providers in parallel
        tasks = [p.get_quote(symbol) for p in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for provider, result in zip(self.providers, results):
            if isinstance(result, Exception):
                errors.append((provider.name, result))
            else:
                quotes.append((provider, result))
        
        if not quotes:
            raise DataNotFoundError(
                f"No provider could fetch quote for {symbol}. "
                f"Errors: {errors}"
            )
        
        # Return quote with tightest spread
        best_quote = min(quotes, key=lambda x: x[1].spread)
        return best_quote[1]
    
    async def get_consensus_price(
        self,
        symbol: str,
        use_weights: bool = True
    ) -> Decimal:
        """Get consensus price from multiple providers.
        
        Args:
            symbol: Asset symbol
            use_weights: Use provider weights
            
        Returns:
            Weighted average price
        """
        quotes = []
        weights = []
        
        tasks = [p.get_quote(symbol) for p in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for provider, result in zip(self.providers, results):
            if not isinstance(result, Exception):
                quotes.append(result.mid_price)
                weight = self._provider_weights.get(provider.name, 1.0) if use_weights else 1.0
                weights.append(weight)
        
        if not quotes:
            raise DataNotFoundError(f"No provider could fetch quote for {symbol}")
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(q * w for q, w in zip(quotes, weights))
        
        return weighted_sum / total_weight
    
    async def close_all(self):
        """Close all provider connections."""
        await asyncio.gather(
            *[p.close() for p in self.providers],
            return_exceptions=True
        )
