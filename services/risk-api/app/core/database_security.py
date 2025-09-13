"""
Secure database operations with SQL injection prevention.

CRITICAL: This module ensures ALL database operations are safe from SQL injection
by enforcing parameterized queries and input validation.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Type
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from sqlalchemy import text, create_engine, event, pool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import Select, Insert, Update, Delete
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool, QueuePool
import sqlparse

from app.core.security_middleware import InputSanitizer, SQLInjectionPrevention

logger = logging.getLogger(__name__)


class SecureDatabaseError(Exception):
    """Base exception for secure database operations."""
    pass


class SQLInjectionAttemptError(SecureDatabaseError):
    """Raised when SQL injection attempt is detected."""
    pass


class SecureQueryBuilder:
    """
    CRITICAL: Secure query builder that prevents SQL injection.
    
    This class ensures all queries are parameterized and validated.
    """
    
    # Whitelist of allowed SQL functions
    ALLOWED_FUNCTIONS = {
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DATE', 'NOW', 
        'CURRENT_TIMESTAMP', 'COALESCE', 'CAST', 'EXTRACT'
    }
    
    # Whitelist of allowed operators
    ALLOWED_OPERATORS = {
        '=', '!=', '<>', '<', '>', '<=', '>=', 'LIKE', 'ILIKE', 
        'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL', 'BETWEEN', 'AND', 'OR'
    }
    
    @staticmethod
    def validate_identifier(identifier: str, identifier_type: str = "column") -> str:
        """
        Validate and sanitize database identifiers (table/column names).
        
        CRITICAL: This prevents SQL injection through identifier manipulation.
        """
        if not identifier:
            raise ValueError(f"Empty {identifier_type} name")
        
        # Only allow alphanumeric, underscore, and dot (for schema.table)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$', identifier):
            raise SQLInjectionAttemptError(
                f"Invalid {identifier_type} name: {identifier}. "
                f"Possible SQL injection attempt."
            )
        
        # Check against SQL keywords
        sql_keywords = {
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
            'ALTER', 'EXEC', 'EXECUTE', 'UNION', 'FROM', 'WHERE'
        }
        
        if identifier.upper() in sql_keywords:
            raise SQLInjectionAttemptError(
                f"SQL keyword used as {identifier_type}: {identifier}"
            )
        
        return identifier
    
    @staticmethod
    def build_select(
        table: str,
        columns: List[str] = None,
        where: Dict[str, Any] = None,
        order_by: List[str] = None,
        limit: int = None,
        offset: int = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a secure SELECT query with parameterized values.
        
        CRITICAL: All user input is parameterized to prevent SQL injection.
        """
        # Validate table name
        table = SecureQueryBuilder.validate_identifier(table, "table")
        
        # Validate columns
        if columns:
            columns = [
                SecureQueryBuilder.validate_identifier(col, "column") 
                for col in columns
            ]
            column_str = ", ".join(columns)
        else:
            column_str = "*"
        
        # Start building query
        query_parts = [f"SELECT {column_str} FROM {table}"]
        params = {}
        
        # Add WHERE clause with parameterized values
        if where:
            where_parts = []
            for i, (col, value) in enumerate(where.items()):
                col = SecureQueryBuilder.validate_identifier(col, "column")
                param_name = f"param_{i}"
                
                if value is None:
                    where_parts.append(f"{col} IS NULL")
                elif isinstance(value, list):
                    # Handle IN clause
                    in_params = []
                    for j, v in enumerate(value):
                        param_key = f"{param_name}_{j}"
                        params[param_key] = v
                        in_params.append(f":{param_key}")
                    where_parts.append(f"{col} IN ({', '.join(in_params)})")
                else:
                    where_parts.append(f"{col} = :{param_name}")
                    params[param_name] = value
            
            if where_parts:
                query_parts.append(f"WHERE {' AND '.join(where_parts)}")
        
        # Add ORDER BY clause
        if order_by:
            order_parts = []
            for order in order_by:
                # Handle "column DESC" format
                parts = order.split()
                col = SecureQueryBuilder.validate_identifier(parts[0], "column")
                direction = "ASC"
                if len(parts) > 1 and parts[1].upper() in ("ASC", "DESC"):
                    direction = parts[1].upper()
                order_parts.append(f"{col} {direction}")
            query_parts.append(f"ORDER BY {', '.join(order_parts)}")
        
        # Add LIMIT and OFFSET
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("Invalid limit value")
            query_parts.append(f"LIMIT :limit")
            params['limit'] = limit
        
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("Invalid offset value")
            query_parts.append(f"OFFSET :offset")
            params['offset'] = offset
        
        query = " ".join(query_parts)
        return query, params
    
    @staticmethod
    def build_insert(
        table: str,
        values: Dict[str, Any],
        returning: List[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a secure INSERT query with parameterized values.
        
        CRITICAL: Prevents SQL injection in INSERT statements.
        """
        # Validate table name
        table = SecureQueryBuilder.validate_identifier(table, "table")
        
        if not values:
            raise ValueError("No values to insert")
        
        # Validate column names and prepare parameters
        columns = []
        value_placeholders = []
        params = {}
        
        for col, value in values.items():
            col = SecureQueryBuilder.validate_identifier(col, "column")
            columns.append(col)
            param_name = f"val_{col}"
            value_placeholders.append(f":{param_name}")
            params[param_name] = value
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(value_placeholders)})"
        
        # Add RETURNING clause if needed
        if returning:
            returning_cols = [
                SecureQueryBuilder.validate_identifier(col, "column") 
                for col in returning
            ]
            query += f" RETURNING {', '.join(returning_cols)}"
        
        return query, params
    
    @staticmethod
    def build_update(
        table: str,
        values: Dict[str, Any],
        where: Dict[str, Any],
        returning: List[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a secure UPDATE query with parameterized values.
        
        CRITICAL: Prevents SQL injection in UPDATE statements.
        """
        # Validate table name
        table = SecureQueryBuilder.validate_identifier(table, "table")
        
        if not values:
            raise ValueError("No values to update")
        
        if not where:
            raise ValueError("UPDATE without WHERE clause is not allowed")
        
        # Build SET clause
        set_parts = []
        params = {}
        
        for col, value in values.items():
            col = SecureQueryBuilder.validate_identifier(col, "column")
            param_name = f"set_{col}"
            set_parts.append(f"{col} = :{param_name}")
            params[param_name] = value
        
        # Build WHERE clause
        where_parts = []
        for col, value in where.items():
            col = SecureQueryBuilder.validate_identifier(col, "column")
            param_name = f"where_{col}"
            if value is None:
                where_parts.append(f"{col} IS NULL")
            else:
                where_parts.append(f"{col} = :{param_name}")
                params[param_name] = value
        
        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        
        # Add RETURNING clause if needed
        if returning:
            returning_cols = [
                SecureQueryBuilder.validate_identifier(col, "column") 
                for col in returning
            ]
            query += f" RETURNING {', '.join(returning_cols)}"
        
        return query, params
    
    @staticmethod
    def build_delete(
        table: str,
        where: Dict[str, Any],
        returning: List[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a secure DELETE query with parameterized values.
        
        CRITICAL: Prevents SQL injection in DELETE statements.
        """
        # Validate table name
        table = SecureQueryBuilder.validate_identifier(table, "table")
        
        if not where:
            raise ValueError("DELETE without WHERE clause is not allowed")
        
        # Build WHERE clause
        where_parts = []
        params = {}
        
        for col, value in where.items():
            col = SecureQueryBuilder.validate_identifier(col, "column")
            param_name = f"where_{col}"
            if value is None:
                where_parts.append(f"{col} IS NULL")
            else:
                where_parts.append(f"{col} = :{param_name}")
                params[param_name] = value
        
        query = f"DELETE FROM {table} WHERE {' AND '.join(where_parts)}"
        
        # Add RETURNING clause if needed
        if returning:
            returning_cols = [
                SecureQueryBuilder.validate_identifier(col, "column") 
                for col in returning
            ]
            query += f" RETURNING {', '.join(returning_cols)}"
        
        return query, params


class SecureDatabase:
    """
    CRITICAL: Secure database wrapper that enforces security best practices.
    
    This class ensures all database operations are safe from SQL injection
    and implements additional security measures.
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_timeout: int = 30,
        echo: bool = False
    ):
        """Initialize secure database connection."""
        self.database_url = self._sanitize_database_url(database_url)
        
        # Create engine with security configurations
        self.engine = create_async_engine(
            self.database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            echo=echo,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            connect_args={
                "server_settings": {
                    "application_name": "RiskRadar-Secure",
                    "jit": "off"  # Disable JIT for security
                },
                "command_timeout": 60,
                "options": "-c statement_timeout=60000"  # 60 second timeout
            }
        )
        
        # Create session factory
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Set up query logging for audit
        self._setup_query_logging()
    
    def _sanitize_database_url(self, url: str) -> str:
        """Sanitize database URL to prevent injection."""
        # Basic validation of database URL format
        if not url.startswith(('postgresql://', 'postgresql+asyncpg://')):
            raise ValueError("Invalid database URL format")
        
        # Remove any suspicious characters that could be injection attempts
        if any(char in url for char in [';', '--', '/*', '*/', 'xp_', 'sp_']):
            raise SQLInjectionAttemptError("Suspicious characters in database URL")
        
        return url
    
    def _setup_query_logging(self):
        """Set up query logging for security audit."""
        @event.listens_for(Engine, "before_execute")
        def log_query(conn, clauseelement, multiparams, params, execution_options):
            """Log all queries for security audit."""
            logger.info(
                f"Executing query: {clauseelement}\n"
                f"Parameters: {params}"
            )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Get a secure database session.
        
        CRITICAL: All sessions are properly managed and closed.
        """
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_secure_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        fetch_one: bool = False
    ) -> Union[List[Dict], Dict, None]:
        """
        Execute a secure parameterized query.
        
        CRITICAL: This method ensures all queries are parameterized
        and validates input to prevent SQL injection.
        """
        # Validate query doesn't contain obvious injection attempts
        if not self._validate_query_safety(query):
            raise SQLInjectionAttemptError("Potentially unsafe query detected")
        
        # Sanitize parameters
        if params:
            params = self._sanitize_parameters(params)
        
        async with self.get_session() as session:
            try:
                # Use parameterized query with text()
                result = await session.execute(text(query), params or {})
                
                if fetch_one:
                    row = result.fetchone()
                    return dict(row) if row else None
                else:
                    rows = result.fetchall()
                    return [dict(row) for row in rows]
                    
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise SecureDatabaseError(f"Database query failed: {e}")
    
    def _validate_query_safety(self, query: str) -> bool:
        """
        Validate query for common SQL injection patterns.
        
        CRITICAL: This is a defense-in-depth measure.
        """
        query_lower = query.lower()
        
        # Check for multiple statements (semicolon not in string)
        if ';' in query and not self._is_in_string(query, ';'):
            logger.warning(f"Multiple statements detected in query: {query}")
            return False
        
        # Check for dangerous commands
        dangerous_commands = [
            'exec', 'execute', 'xp_cmdshell', 'sp_executesql',
            'drop table', 'drop database', 'truncate', 'alter table'
        ]
        
        for cmd in dangerous_commands:
            if cmd in query_lower:
                logger.warning(f"Dangerous command detected: {cmd}")
                return False
        
        # Check for comment injection
        if '--' in query or '/*' in query or '*/' in query:
            logger.warning(f"SQL comment detected in query")
            return False
        
        return True
    
    def _is_in_string(self, query: str, char: str) -> bool:
        """Check if character appears inside a string literal."""
        in_string = False
        escape_next = False
        
        for i, c in enumerate(query):
            if escape_next:
                escape_next = False
                continue
            
            if c == '\\':
                escape_next = True
                continue
            
            if c == "'":
                in_string = not in_string
            
            if c == char and in_string:
                return True
        
        return False
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize query parameters to prevent injection.
        
        CRITICAL: All user input must pass through this sanitization.
        """
        sanitized = {}
        
        for key, value in params.items():
            # Validate parameter name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid parameter name: {key}")
            
            # Sanitize value based on type
            if isinstance(value, str):
                # Check for SQL injection patterns
                if not InputSanitizer.validate_sql_input(value):
                    raise SQLInjectionAttemptError(
                        f"Potential SQL injection in parameter: {key}"
                    )
                # Additional string sanitization
                sanitized[key] = value.replace('\x00', '')  # Remove null bytes
                
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized[key] = value
                
            elif isinstance(value, (datetime, UUID)):
                sanitized[key] = value
                
            elif isinstance(value, list):
                # Sanitize list elements
                sanitized[key] = [
                    self._sanitize_single_value(item) for item in value
                ]
                
            else:
                # Don't allow other types for security
                raise ValueError(f"Unsupported parameter type: {type(value)}")
        
        return sanitized
    
    def _sanitize_single_value(self, value: Any) -> Any:
        """Sanitize a single parameter value."""
        if isinstance(value, str):
            if not InputSanitizer.validate_sql_input(value):
                raise SQLInjectionAttemptError("Potential SQL injection in value")
            return value.replace('\x00', '')
        elif isinstance(value, (int, float, bool, type(None), datetime, UUID)):
            return value
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")
    
    async def close(self):
        """Close database connection pool."""
        await self.engine.dispose()


# Global secure database instance
_secure_db: Optional[SecureDatabase] = None


def get_secure_db() -> SecureDatabase:
    """Get secure database instance."""
    global _secure_db
    if _secure_db is None:
        from app.core.config import settings
        _secure_db = SecureDatabase(settings.DATABASE_URL)
    return _secure_db


async def close_secure_db():
    """Close secure database connection."""
    global _secure_db
    if _secure_db:
        await _secure_db.close()
        _secure_db = None
