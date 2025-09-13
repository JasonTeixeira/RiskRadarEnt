"""
Event bus integration for RiskRadar Enterprise.

This module provides Kafka/Redpanda integration for event-driven architecture,
including producers, consumers, and event schemas.
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Type
from enum import Enum
from uuid import uuid4
import traceback

from pydantic import BaseModel, Field, validator
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.serialization import StringSerializer, JSONSerializer
from prometheus_client import Counter, Histogram, Gauge
import orjson

from app.core.config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
events_published = Counter(
    "events_published_total",
    "Total number of events published",
    ["event_type", "topic"]
)
events_consumed = Counter(
    "events_consumed_total",
    "Total number of events consumed",
    ["event_type", "topic", "consumer_group"]
)
event_processing_duration = Histogram(
    "event_processing_duration_seconds",
    "Event processing duration",
    ["event_type", "consumer_group"]
)
event_processing_errors = Counter(
    "event_processing_errors_total",
    "Total number of event processing errors",
    ["event_type", "consumer_group", "error_type"]
)
kafka_lag = Gauge(
    "kafka_consumer_lag",
    "Kafka consumer lag",
    ["topic", "partition", "consumer_group"]
)


class EventType(str, Enum):
    """Event types in the system."""
    # Portfolio events
    PORTFOLIO_CREATED = "portfolio.created"
    PORTFOLIO_UPDATED = "portfolio.updated"
    PORTFOLIO_DELETED = "portfolio.deleted"
    
    # Position events
    POSITION_CREATED = "position.created"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"
    
    # Risk calculation events
    RISK_CALCULATION_REQUESTED = "risk.calculation.requested"
    RISK_CALCULATION_STARTED = "risk.calculation.started"
    RISK_CALCULATION_COMPLETED = "risk.calculation.completed"
    RISK_CALCULATION_FAILED = "risk.calculation.failed"
    
    # Alert events
    RISK_ALERT_TRIGGERED = "risk.alert.triggered"
    RISK_ALERT_RESOLVED = "risk.alert.resolved"
    
    # Market data events
    MARKET_DATA_UPDATED = "market.data.updated"
    MARKET_STATUS_CHANGED = "market.status.changed"
    
    # System events
    SERVICE_HEALTH_CHANGED = "service.health.changed"
    AUDIT_LOG_CREATED = "audit.log.created"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BaseEvent(BaseModel):
    """Base event model."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="risk-api")
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PortfolioEvent(BaseEvent):
    """Portfolio-related event."""
    portfolio_id: str
    portfolio_name: str
    portfolio_data: Dict[str, Any]


class RiskCalculationEvent(BaseEvent):
    """Risk calculation event."""
    calculation_id: str
    portfolio_id: str
    metrics: List[str]
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AlertEvent(BaseEvent):
    """Risk alert event."""
    alert_id: str
    portfolio_id: str
    alert_type: str
    severity: str
    message: str
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None


class MarketDataEvent(BaseEvent):
    """Market data event."""
    symbol: str
    asset_class: str
    price_data: Dict[str, Any]
    volume: Optional[int] = None
    exchange: Optional[str] = None


class EventProducer:
    """Kafka event producer."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        client_id: str = "risk-api-producer",
        enable_idempotence: bool = True
    ):
        """Initialize event producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            client_id: Producer client ID
            enable_idempotence: Enable idempotent producer
        """
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.client_id = client_id
        self.enable_idempotence = enable_idempotence
        self._producer: Optional[AIOKafkaProducer] = None
        self._running = False
    
    async def start(self):
        """Start the producer."""
        if self._producer is not None:
            return
        
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            enable_idempotence=self.enable_idempotence,
            acks="all",  # Wait for all replicas
            compression_type="snappy",
            max_batch_size=32768,
            linger_ms=10,
            value_serializer=lambda v: orjson.dumps(v),
            key_serializer=lambda k: k.encode() if k else None
        )
        
        await self._producer.start()
        self._running = True
        logger.info(f"Event producer started: {self.client_id}")
    
    async def stop(self):
        """Stop the producer."""
        if self._producer is None:
            return
        
        self._running = False
        await self._producer.stop()
        self._producer = None
        logger.info(f"Event producer stopped: {self.client_id}")
    
    async def publish(
        self,
        event: BaseEvent,
        topic: str = None,
        key: str = None,
        priority: EventPriority = EventPriority.NORMAL
    ):
        """Publish an event.
        
        Args:
            event: Event to publish
            topic: Topic to publish to (defaults to event type)
            key: Message key for partitioning
            priority: Event priority
        """
        if not self._running:
            await self.start()
        
        topic = topic or f"riskradar.{event.event_type.value}"
        key = key or event.correlation_id
        
        try:
            # Add priority to headers
            headers = [
                ("priority", priority.value.encode()),
                ("event_type", event.event_type.value.encode()),
                ("source", event.source.encode())
            ]
            
            # Send message
            await self._producer.send_and_wait(
                topic=topic,
                value=event.dict(),
                key=key,
                headers=headers
            )
            
            # Update metrics
            events_published.labels(
                event_type=event.event_type.value,
                topic=topic
            ).inc()
            
            logger.debug(
                f"Published event: {event.event_type.value} "
                f"to topic: {topic}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to publish event: {e}")
            raise
    
    async def publish_batch(
        self,
        events: List[BaseEvent],
        topic: str = None
    ):
        """Publish multiple events.
        
        Args:
            events: List of events to publish
            topic: Topic to publish to
        """
        tasks = [
            self.publish(event, topic)
            for event in events
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


class EventConsumer:
    """Kafka event consumer."""
    
    def __init__(
        self,
        topics: List[str],
        group_id: str,
        bootstrap_servers: str = None,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False
    ):
        """Initialize event consumer.
        
        Args:
            topics: Topics to subscribe to
            group_id: Consumer group ID
            bootstrap_servers: Kafka bootstrap servers
            auto_offset_reset: Where to start consuming
            enable_auto_commit: Enable auto commit
        """
        self.topics = topics
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[BaseEvent], None]
    ):
        """Register an event handler.
        
        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def start(self):
        """Start the consumer."""
        if self._consumer is not None:
            return
        
        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=self.enable_auto_commit,
            value_deserializer=lambda v: orjson.loads(v),
            key_deserializer=lambda k: k.decode() if k else None,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        
        await self._consumer.start()
        self._running = True
        
        # Start consumer loop
        self._tasks.append(
            asyncio.create_task(self._consume_loop())
        )
        
        logger.info(
            f"Event consumer started: {self.group_id} "
            f"on topics: {', '.join(self.topics)}"
        )
    
    async def stop(self):
        """Stop the consumer."""
        if self._consumer is None:
            return
        
        self._running = False
        
        # Cancel consumer tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        await self._consumer.stop()
        self._consumer = None
        
        logger.info(f"Event consumer stopped: {self.group_id}")
    
    async def _consume_loop(self):
        """Main consumer loop."""
        while self._running:
            try:
                # Fetch messages
                messages = await self._consumer.getmany(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        await self._process_message(record)
                    
                    # Update lag metrics
                    lag = self._consumer.highwater(topic_partition) - record.offset
                    kafka_lag.labels(
                        topic=topic_partition.topic,
                        partition=topic_partition.partition,
                        consumer_group=self.group_id
                    ).set(lag)
                
                # Commit offsets if not auto-committing
                if not self.enable_auto_commit and messages:
                    await self._consumer.commit()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, record):
        """Process a single message.
        
        Args:
            record: Kafka record
        """
        try:
            # Parse event
            event_data = record.value
            event_type = EventType(event_data.get("event_type"))
            
            # Create event object
            event = BaseEvent(**event_data)
            
            # Update metrics
            events_consumed.labels(
                event_type=event_type.value,
                topic=record.topic,
                consumer_group=self.group_id
            ).inc()
            
            # Call handlers
            handlers = self._handlers.get(event_type, [])
            
            with event_processing_duration.labels(
                event_type=event_type.value,
                consumer_group=self.group_id
            ).time():
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(
                            f"Handler error for {event_type.value}: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                        event_processing_errors.labels(
                            event_type=event_type.value,
                            consumer_group=self.group_id,
                            error_type=type(e).__name__
                        ).inc()
            
            logger.debug(
                f"Processed event: {event_type.value} "
                f"from topic: {record.topic}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to process message: {e}\n"
                f"Record: {record}"
            )
            event_processing_errors.labels(
                event_type="unknown",
                consumer_group=self.group_id,
                error_type=type(e).__name__
            ).inc()


class EventBus:
    """Central event bus for the application."""
    
    def __init__(self):
        """Initialize event bus."""
        self._producer: Optional[EventProducer] = None
        self._consumers: Dict[str, EventConsumer] = {}
        self._local_handlers: Dict[EventType, List[Callable]] = {}
    
    async def initialize(self):
        """Initialize the event bus."""
        # Create producer
        self._producer = EventProducer()
        await self._producer.start()
        
        # Create consumers for different event categories
        await self._create_consumers()
    
    async def shutdown(self):
        """Shutdown the event bus."""
        # Stop producer
        if self._producer:
            await self._producer.stop()
        
        # Stop all consumers
        for consumer in self._consumers.values():
            await consumer.stop()
    
    async def _create_consumers(self):
        """Create event consumers."""
        # Portfolio events consumer
        portfolio_consumer = EventConsumer(
            topics=["riskradar.portfolio.*"],
            group_id="risk-api-portfolio"
        )
        await portfolio_consumer.start()
        self._consumers["portfolio"] = portfolio_consumer
        
        # Risk calculation events consumer
        risk_consumer = EventConsumer(
            topics=["riskradar.risk.*"],
            group_id="risk-api-risk"
        )
        await risk_consumer.start()
        self._consumers["risk"] = risk_consumer
        
        # Alert events consumer
        alert_consumer = EventConsumer(
            topics=["riskradar.risk.alert.*"],
            group_id="risk-api-alerts"
        )
        await alert_consumer.start()
        self._consumers["alerts"] = alert_consumer
    
    async def publish(
        self,
        event: BaseEvent,
        priority: EventPriority = EventPriority.NORMAL
    ):
        """Publish an event.
        
        Args:
            event: Event to publish
            priority: Event priority
        """
        # Publish to Kafka
        if self._producer:
            await self._producer.publish(event, priority=priority)
        
        # Call local handlers
        await self._call_local_handlers(event)
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[BaseEvent], None],
        local_only: bool = False
    ):
        """Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Event handler
            local_only: Only handle local events
        """
        if local_only:
            if event_type not in self._local_handlers:
                self._local_handlers[event_type] = []
            self._local_handlers[event_type].append(handler)
        else:
            # Register with appropriate consumer
            for consumer in self._consumers.values():
                consumer.register_handler(event_type, handler)
    
    async def _call_local_handlers(self, event: BaseEvent):
        """Call local event handlers.
        
        Args:
            event: Event to handle
        """
        handlers = self._local_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    f"Local handler error for {event.event_type.value}: {e}"
                )


# Global event bus instance
event_bus = EventBus()


# Event publishing helpers
async def publish_portfolio_created(
    portfolio_id: str,
    portfolio_name: str,
    portfolio_data: Dict[str, Any],
    user_id: str = None,
    organization_id: str = None
):
    """Publish portfolio created event."""
    event = PortfolioEvent(
        event_type=EventType.PORTFOLIO_CREATED,
        portfolio_id=portfolio_id,
        portfolio_name=portfolio_name,
        portfolio_data=portfolio_data,
        user_id=user_id,
        organization_id=organization_id
    )
    await event_bus.publish(event)


async def publish_risk_calculation_requested(
    calculation_id: str,
    portfolio_id: str,
    metrics: List[str],
    user_id: str = None,
    organization_id: str = None
):
    """Publish risk calculation requested event."""
    event = RiskCalculationEvent(
        event_type=EventType.RISK_CALCULATION_REQUESTED,
        calculation_id=calculation_id,
        portfolio_id=portfolio_id,
        metrics=metrics,
        status="requested",
        user_id=user_id,
        organization_id=organization_id
    )
    await event_bus.publish(event, priority=EventPriority.HIGH)


async def publish_risk_alert(
    alert_id: str,
    portfolio_id: str,
    alert_type: str,
    severity: str,
    message: str,
    threshold_value: float = None,
    actual_value: float = None,
    organization_id: str = None
):
    """Publish risk alert event."""
    event = AlertEvent(
        event_type=EventType.RISK_ALERT_TRIGGERED,
        alert_id=alert_id,
        portfolio_id=portfolio_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        threshold_value=threshold_value,
        actual_value=actual_value,
        organization_id=organization_id
    )
    await event_bus.publish(event, priority=EventPriority.CRITICAL)
