"""
CSRF protection and secure session management.

CRITICAL: This module implements CSRF tokens and secure sessions to prevent
cross-site request forgery and session hijacking attacks.
"""

import secrets
import hashlib
import hmac
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import redis.asyncio as redis
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.secrets_manager import get_secret

# Security configuration
SESSION_TIMEOUT = timedelta(minutes=30)
MAX_CONCURRENT_SESSIONS = 5
CSRF_TOKEN_LENGTH = 32
SESSION_ID_LENGTH = 32


class CSRFError(Exception):
    """CSRF validation error."""
    pass


class SessionError(Exception):
    """Session management error."""
    pass


class CSRFProtection:
    """
    CRITICAL: CSRF protection implementation.
    
    This class generates and validates CSRF tokens to prevent
    cross-site request forgery attacks.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize CSRF protection with secret key."""
        self.secret_key = secret_key or get_secret("CSRF_SECRET_KEY")
        if not self.secret_key:
            # Generate a new key if none exists
            self.secret_key = secrets.token_urlsafe(32)
    
    def generate_token(self, session_id: str) -> str:
        """
        Generate CSRF token for session.
        
        CRITICAL: Token is tied to session to prevent reuse.
        """
        # Generate random token
        random_token = secrets.token_urlsafe(CSRF_TOKEN_LENGTH)
        
        # Create HMAC signature tied to session
        signature = self._create_signature(session_id, random_token)
        
        # Combine token and signature
        csrf_token = f"{random_token}.{signature}"
        
        return csrf_token
    
    def validate_token(self, session_id: str, csrf_token: str) -> bool:
        """
        Validate CSRF token for session.
        
        CRITICAL: This prevents CSRF attacks by validating the token
        is tied to the current session.
        """
        if not csrf_token:
            return False
        
        try:
            # Split token and signature
            parts = csrf_token.split('.')
            if len(parts) != 2:
                return False
            
            random_token, provided_signature = parts
            
            # Recreate signature
            expected_signature = self._create_signature(session_id, random_token)
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(provided_signature, expected_signature)
            
        except Exception as e:
            logger.warning(f"CSRF token validation error: {e}")
            return False
    
    def _create_signature(self, session_id: str, random_token: str) -> str:
        """Create HMAC signature for CSRF token."""
        message = f"{session_id}:{random_token}".encode()
        signature = hmac.new(
            self.secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature


class SecureSessionManager:
    """
    CRITICAL: Secure session management.
    
    This class implements secure sessions with:
    - Session timeouts
    - IP binding to prevent hijacking
    - Concurrent session limits
    - Secure session storage in Redis
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize session manager with Redis client."""
        self.redis_client = redis_client
        self.session_timeout = SESSION_TIMEOUT
        self.max_concurrent_sessions = MAX_CONCURRENT_SESSIONS
        self.secret_key = get_secret("SESSION_SECRET_KEY") or secrets.token_urlsafe(32)
    
    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Create a new secure session.
        
        CRITICAL: Sessions are bound to IP and user agent to prevent hijacking.
        
        Returns:
            Tuple of (session_id, csrf_token)
        """
        # Check concurrent session limit
        await self._enforce_session_limit(user_id)
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(SESSION_ID_LENGTH)
        
        # Create session data
        session_data = {
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Generate CSRF token for session
        csrf_protection = CSRFProtection(self.secret_key)
        csrf_token = csrf_protection.generate_token(session_id)
        session_data["csrf_token"] = csrf_token
        
        # Store session in Redis with expiration
        if self.redis_client:
            session_key = f"session:{session_id}"
            await self.redis_client.setex(
                session_key,
                int(self.session_timeout.total_seconds()),
                json.dumps(session_data)
            )
            
            # Track user sessions
            user_sessions_key = f"user_sessions:{user_id}"
            await self.redis_client.sadd(user_sessions_key, session_id)
            await self.redis_client.expire(
                user_sessions_key,
                int(self.session_timeout.total_seconds())
            )
        
        return session_id, csrf_token
    
    async def validate_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str,
        update_activity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Validate session with security checks.
        
        CRITICAL: This prevents session hijacking by validating:
        - Session exists and not expired
        - IP address matches
        - User agent matches
        """
        if not self.redis_client:
            return None
        
        # Get session from Redis
        session_key = f"session:{session_id}"
        session_json = await self.redis_client.get(session_key)
        
        if not session_json:
            return None
        
        try:
            session_data = json.loads(session_json)
        except json.JSONDecodeError:
            return None
        
        # CRITICAL: Validate IP binding
        if session_data.get("ip_address") != ip_address:
            logger.warning(
                f"Session hijacking attempt detected for session {session_id}. "
                f"Expected IP: {session_data.get('ip_address')}, Got: {ip_address}"
            )
            await self.destroy_session(session_id)
            return None
        
        # CRITICAL: Validate user agent
        if session_data.get("user_agent") != user_agent:
            logger.warning(
                f"User agent mismatch for session {session_id}. "
                f"Possible session hijacking attempt."
            )
            # Less strict - log warning but don't destroy session
            # as user agents can change with browser updates
        
        # Check session timeout
        last_activity = datetime.fromisoformat(session_data["last_activity"])
        if datetime.utcnow() - last_activity > self.session_timeout:
            await self.destroy_session(session_id)
            return None
        
        # Update last activity if requested
        if update_activity:
            session_data["last_activity"] = datetime.utcnow().isoformat()
            await self.redis_client.setex(
                session_key,
                int(self.session_timeout.total_seconds()),
                json.dumps(session_data)
            )
        
        return session_data
    
    async def validate_csrf_token(
        self,
        session_id: str,
        csrf_token: str
    ) -> bool:
        """
        Validate CSRF token for session.
        
        CRITICAL: This prevents CSRF attacks.
        """
        csrf_protection = CSRFProtection(self.secret_key)
        return csrf_protection.validate_token(session_id, csrf_token)
    
    async def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session.
        
        CRITICAL: Properly clean up session data.
        """
        if not self.redis_client:
            return False
        
        # Get session to find user ID
        session_key = f"session:{session_id}"
        session_json = await self.redis_client.get(session_key)
        
        if session_json:
            try:
                session_data = json.loads(session_json)
                user_id = session_data.get("user_id")
                
                # Remove from user sessions set
                if user_id:
                    user_sessions_key = f"user_sessions:{user_id}"
                    await self.redis_client.srem(user_sessions_key, session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        # Delete session
        await self.redis_client.delete(session_key)
        
        return True
    
    async def destroy_all_user_sessions(self, user_id: str) -> int:
        """
        Destroy all sessions for a user.
        
        Used for security events like password changes.
        """
        if not self.redis_client:
            return 0
        
        user_sessions_key = f"user_sessions:{user_id}"
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        count = 0
        for session_id in session_ids:
            if await self.destroy_session(session_id):
                count += 1
        
        # Clean up user sessions set
        await self.redis_client.delete(user_sessions_key)
        
        return count
    
    async def _enforce_session_limit(self, user_id: str):
        """
        Enforce concurrent session limit.
        
        CRITICAL: Prevents session flooding attacks.
        """
        if not self.redis_client:
            return
        
        user_sessions_key = f"user_sessions:{user_id}"
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        # If at limit, remove oldest session
        if len(session_ids) >= self.max_concurrent_sessions:
            # Get session creation times
            sessions_with_time = []
            for session_id in session_ids:
                session_key = f"session:{session_id}"
                session_json = await self.redis_client.get(session_key)
                if session_json:
                    try:
                        session_data = json.loads(session_json)
                        created_at = datetime.fromisoformat(session_data["created_at"])
                        sessions_with_time.append((session_id, created_at))
                    except Exception:
                        pass
            
            # Sort by creation time and remove oldest
            if sessions_with_time:
                sessions_with_time.sort(key=lambda x: x[1])
                oldest_session_id = sessions_with_time[0][0]
                await self.destroy_session(oldest_session_id)


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CRITICAL: CSRF protection middleware.
    
    This middleware validates CSRF tokens on state-changing requests.
    """
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.session_manager = SecureSessionManager(redis_client)
        self.csrf_protection = CSRFProtection()
        # Methods that require CSRF protection
        self.protected_methods = {"POST", "PUT", "PATCH", "DELETE"}
        # Paths to exclude from CSRF (e.g., login)
        self.excluded_paths = {"/api/v1/auth/login", "/api/v1/auth/register"}
    
    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF validation."""
        # Skip CSRF for safe methods
        if request.method not in self.protected_methods:
            return await call_next(request)
        
        # Skip CSRF for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Get session ID from cookie or header
        session_id = request.cookies.get("session_id") or \
                    request.headers.get("X-Session-ID")
        
        if not session_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "No session found"}
            )
        
        # Get CSRF token from header or form
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token and request.method == "POST":
            # Try to get from form data
            if hasattr(request, "form"):
                form_data = await request.form()
                csrf_token = form_data.get("csrf_token")
        
        if not csrf_token:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "CSRF token missing"}
            )
        
        # Validate CSRF token
        is_valid = await self.session_manager.validate_csrf_token(
            session_id,
            csrf_token
        )
        
        if not is_valid:
            logger.warning(
                f"Invalid CSRF token for session {session_id} from "
                f"IP {request.client.host}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Invalid CSRF token"}
            )
        
        # Process request
        response = await call_next(request)
        return response


# Dependency for FastAPI routes
async def get_session_manager() -> SecureSessionManager:
    """Get session manager instance."""
    # Get Redis client
    redis_client = await get_redis_client()
    return SecureSessionManager(redis_client)


async def get_current_session(
    request: Request,
    session_manager: SecureSessionManager = Depends(get_session_manager)
) -> Dict[str, Any]:
    """
    Get current session data.
    
    CRITICAL: This validates the session and prevents hijacking.
    """
    # Get session ID
    session_id = request.cookies.get("session_id") or \
                request.headers.get("X-Session-ID")
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No session found"
        )
    
    # Get client info
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "")
    
    # Validate session
    session_data = await session_manager.validate_session(
        session_id,
        ip_address,
        user_agent
    )
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    return session_data


async def require_csrf_token(
    request: Request,
    session_data: Dict[str, Any] = Depends(get_current_session),
    session_manager: SecureSessionManager = Depends(get_session_manager)
):
    """
    Require valid CSRF token for request.
    
    CRITICAL: Use this dependency on all state-changing endpoints.
    """
    # Get session ID
    session_id = request.cookies.get("session_id") or \
                request.headers.get("X-Session-ID")
    
    # Get CSRF token
    csrf_token = request.headers.get("X-CSRF-Token")
    
    if not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token required"
        )
    
    # Validate CSRF token
    is_valid = await session_manager.validate_csrf_token(
        session_id,
        csrf_token
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token"
        )
    
    return True


# Example usage in routes
"""
@router.post("/api/v1/portfolios")
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    session: Dict[str, Any] = Depends(get_current_session),
    csrf_valid: bool = Depends(require_csrf_token)
):
    # CRITICAL: Both session and CSRF are validated
    # Safe to process state-changing request
    pass
"""
