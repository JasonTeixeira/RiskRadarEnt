"""
Security middleware and input validation for RiskRadar Enterprise.

CRITICAL: This module addresses major security vulnerabilities including:
- Input sanitization to prevent injection attacks
- Request validation and rate limiting
- Security headers implementation
- CORS configuration
- Request signing verification
"""

import re
import hashlib
import hmac
import time
import bleach
import secrets
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json

from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import text
import html

from app.core.config import settings
from app.core.auth import verify_api_key

# CRITICAL SECURITY ISSUE: These patterns prevent SQL Injection
SQL_INJECTION_PATTERNS = [
    r"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)",
    r"(--|\#|\/\*|\*\/)",  # SQL comments
    r"(\bOR\b.*=.*)",  # OR conditions
    r"(\'|\"|;|\\x[0-9a-fA-F]{2})",  # Quotes and hex encoding
    r"(\b(sys|information_schema)\b)",  # System tables
    r"(xp_cmdshell|sp_executesql)",  # Dangerous procedures
]

# CRITICAL SECURITY ISSUE: These patterns prevent XSS attacks
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",  # Event handlers
    r"<iframe[^>]*>",
    r"<embed[^>]*>",
    r"<object[^>]*>",
    r"eval\s*\(",
    r"expression\s*\(",
]

# CRITICAL SECURITY ISSUE: Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$]",  # Shell metacharacters
    r"\$\([^)]*\)",  # Command substitution
    r"`[^`]*`",  # Backticks
    r"(nc|netcat|curl|wget|python|perl|ruby|php|bash|sh)\s",
]

# CRITICAL SECURITY ISSUE: Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",  # Directory traversal
    r"\.\.\\",  # Windows traversal
    r"%2e%2e",  # URL encoded traversal
    r"\x00",  # Null byte injection
]


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    CRITICAL SECURITY MIDDLEWARE
    
    This middleware provides essential security protections that are currently
    MISSING from the application, leaving it vulnerable to attacks.
    """
    
    def __init__(self, app, blocked_ips: Set[str] = None):
        super().__init__(app)
        self.blocked_ips = blocked_ips or set()
        self.request_counts: Dict[str, List[float]] = {}
        self.failed_auth_attempts: Dict[str, int] = {}
        
    async def dispatch(self, request: Request, call_next):
        # CRITICAL: Check if IP is blocked
        client_ip = request.client.host
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Access denied"}
            )
        
        # CRITICAL: Implement rate limiting per IP
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )
        
        # CRITICAL: Validate request size to prevent DoS
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Request too large"}
            )
        
        # CRITICAL: Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove sensitive headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        return response
    
    def _check_rate_limit(self, ip: str, window: int = 60, max_requests: int = 60) -> bool:
        """Check if IP has exceeded rate limit."""
        now = time.time()
        if ip not in self.request_counts:
            self.request_counts[ip] = []
        
        # Remove old requests outside window
        self.request_counts[ip] = [
            req_time for req_time in self.request_counts[ip]
            if now - req_time < window
        ]
        
        # Check if limit exceeded
        if len(self.request_counts[ip]) >= max_requests:
            return False
        
        # Add current request
        self.request_counts[ip].append(now)
        return True


class InputSanitizer:
    """
    CRITICAL: Input sanitization to prevent injection attacks.
    
    This class is MISSING from the current implementation, leaving
    the application vulnerable to SQL injection, XSS, and other attacks.
    """
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not value:
            return ""
        
        # CRITICAL: Truncate to prevent buffer overflow
        value = value[:max_length]
        
        # CRITICAL: HTML escape to prevent XSS
        value = html.escape(value)
        
        # CRITICAL: Remove null bytes
        value = value.replace('\x00', '')
        
        # CRITICAL: Strip dangerous HTML tags
        value = bleach.clean(value, tags=[], strip=True)
        
        return value
    
    @staticmethod
    def validate_sql_input(value: str) -> bool:
        """
        CRITICAL: Validate input for SQL injection patterns.
        
        Returns False if dangerous patterns detected.
        """
        if not value:
            return True
        
        value_lower = value.lower()
        
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_xss_input(value: str) -> bool:
        """
        CRITICAL: Validate input for XSS patterns.
        
        Returns False if dangerous patterns detected.
        """
        if not value:
            return True
        
        value_lower = value.lower()
        
        for pattern in XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_command_input(value: str) -> bool:
        """
        CRITICAL: Validate input for command injection patterns.
        
        Returns False if dangerous patterns detected.
        """
        if not value:
            return True
        
        for pattern in COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_path_input(value: str) -> bool:
        """
        CRITICAL: Validate input for path traversal patterns.
        
        Returns False if dangerous patterns detected.
        """
        if not value:
            return True
        
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        # Check for absolute paths
        if value.startswith('/') or value.startswith('\\'):
            return False
        
        return True
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """
        CRITICAL: Recursively sanitize dictionary input.
        
        This prevents nested injection attacks.
        """
        if max_depth <= 0:
            raise ValueError("Maximum recursion depth exceeded")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            if not isinstance(key, str):
                continue
            
            sanitized_key = InputSanitizer.sanitize_string(key, max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[sanitized_key] = InputSanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[sanitized_key] = InputSanitizer.sanitize_dict(
                    value, max_depth - 1
                )
            elif isinstance(value, list):
                sanitized[sanitized_key] = [
                    InputSanitizer.sanitize_string(item) if isinstance(item, str)
                    else item for item in value[:1000]  # Limit list size
                ]
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized[sanitized_key] = value
            # Ignore other types for security
        
        return sanitized


class RequestValidator:
    """
    CRITICAL: Request validation to prevent malicious requests.
    
    This is MISSING from the current implementation.
    """
    
    @staticmethod
    def validate_content_type(request: Request, allowed_types: List[str]) -> bool:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "").lower()
        
        for allowed in allowed_types:
            if allowed.lower() in content_type:
                return True
        
        return False
    
    @staticmethod
    def validate_request_signature(
        request: Request,
        body: bytes,
        secret_key: str
    ) -> bool:
        """
        CRITICAL: Validate request signature to prevent tampering.
        
        This is currently NOT IMPLEMENTED, allowing request tampering.
        """
        signature = request.headers.get("X-Signature")
        if not signature:
            return False
        
        # Calculate expected signature
        expected = hmac.new(
            secret_key.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected)
    
    @staticmethod
    def validate_api_version(request: Request, supported_versions: List[str]) -> bool:
        """Validate API version in request."""
        version = request.headers.get("X-API-Version", "v1")
        return version in supported_versions


class SessionManager:
    """
    CRITICAL: Session management to prevent session hijacking.
    
    This is MISSING from the current implementation.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=30)
        self.max_concurrent_sessions = 5
    
    def create_session(self, user_id: str, ip: str) -> str:
        """Create a new session with security controls."""
        # CRITICAL: Limit concurrent sessions
        user_sessions = [
            sid for sid, data in self.sessions.items()
            if data.get("user_id") == user_id
        ]
        
        if len(user_sessions) >= self.max_concurrent_sessions:
            # Remove oldest session
            oldest = min(
                user_sessions,
                key=lambda sid: self.sessions[sid]["created_at"]
            )
            del self.sessions[oldest]
        
        # Create new session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_id": user_id,
            "ip": ip,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        return session_id
    
    def validate_session(self, session_id: str, ip: str) -> Optional[str]:
        """
        CRITICAL: Validate session with security checks.
        
        Returns user_id if valid, None otherwise.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # CRITICAL: Check session timeout
        if datetime.utcnow() - session["last_activity"] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # CRITICAL: Check IP binding to prevent hijacking
        if session["ip"] != ip:
            # Possible session hijacking attempt
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        
        return session["user_id"]
    
    def destroy_session(self, session_id: str):
        """Destroy a session."""
        self.sessions.pop(session_id, None)


class SQLInjectionPrevention:
    """
    CRITICAL: SQL injection prevention utilities.
    
    The current implementation uses raw SQL in places without
    proper parameterization, making it vulnerable to SQL injection.
    """
    
    @staticmethod
    def safe_query(query: str, params: Dict[str, Any]) -> tuple:
        """
        CRITICAL: Create safe parameterized query.
        
        This should be used for ALL database queries.
        """
        # Use SQLAlchemy's text() with bound parameters
        safe_query = text(query)
        
        # Validate parameters
        safe_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Check for SQL injection patterns
                if not InputSanitizer.validate_sql_input(value):
                    raise ValueError(f"Potentially dangerous input in parameter: {key}")
                safe_params[key] = value
            else:
                safe_params[key] = value
        
        return safe_query, safe_params
    
    @staticmethod
    def validate_table_name(table_name: str) -> bool:
        """
        CRITICAL: Validate table name to prevent injection.
        
        Only allow alphanumeric and underscore.
        """
        return bool(re.match(r'^[a-zA-Z0-9_]+$', table_name))
    
    @staticmethod
    def validate_column_name(column_name: str) -> bool:
        """
        CRITICAL: Validate column name to prevent injection.
        
        Only allow alphanumeric and underscore.
        """
        return bool(re.match(r'^[a-zA-Z0-9_]+$', column_name))


# CRITICAL: MISSING - CSRF Protection
class CSRFProtection:
    """
    CRITICAL: CSRF protection is completely MISSING.
    
    This leaves the application vulnerable to CSRF attacks.
    """
    
    def __init__(self):
        self.tokens: Dict[str, str] = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session."""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token."""
        expected = self.tokens.get(session_id)
        if not expected:
            return False
        
        # Constant-time comparison
        return hmac.compare_digest(token, expected)


# CRITICAL: MISSING - File Upload Security
class FileUploadValidator:
    """
    CRITICAL: File upload validation is MISSING.
    
    This could allow malicious file uploads.
    """
    
    ALLOWED_EXTENSIONS = {'.pdf', '.csv', '.xlsx', '.json'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def validate_file(filename: str, content: bytes) -> bool:
        """
        CRITICAL: Validate uploaded file for security.
        """
        # Check file extension
        import os
        _, ext = os.path.splitext(filename.lower())
        if ext not in FileUploadValidator.ALLOWED_EXTENSIONS:
            return False
        
        # Check file size
        if len(content) > FileUploadValidator.MAX_FILE_SIZE:
            return False
        
        # Check for malicious content (basic check)
        # In production, use a proper antivirus scanner
        dangerous_patterns = [
            b'<%',  # ASP/JSP tags
            b'<?php',  # PHP tags
            b'<script',  # JavaScript
            b'\x00',  # Null bytes
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content:
                return False
        
        return True


# CRITICAL: MISSING - API Key Security
class APIKeySecurity:
    """
    CRITICAL: API key security enhancements are MISSING.
    
    Current implementation stores API keys in plain text.
    """
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage."""
        # Use a proper key derivation function
        salt = secrets.token_bytes(32)
        key_hash = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode(),
            salt,
            100000  # iterations
        )
        # Store salt with hash
        return salt.hex() + ':' + key_hash.hex()
    
    @staticmethod
    def verify_api_key_hash(api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        try:
            salt_hex, hash_hex = stored_hash.split(':')
            salt = bytes.fromhex(salt_hex)
            stored = bytes.fromhex(hash_hex)
            
            # Recompute hash
            computed = hashlib.pbkdf2_hmac(
                'sha256',
                api_key.encode(),
                salt,
                100000
            )
            
            # Constant-time comparison
            return hmac.compare_digest(computed, stored)
        except Exception:
            return False
