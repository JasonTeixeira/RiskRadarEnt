"""
Enterprise Authentication and Authorization System
JWT-based authentication with OAuth2 support, API keys, and fine-grained permissions
"""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from functools import wraps
import hashlib
import hmac

from fastapi import Depends, HTTPException, status, Security, Request
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from pydantic import BaseModel, EmailStr, Field, validator
import redis.asyncio as redis
from prometheus_client import Counter

from app.core.config import settings
from app.core.database import get_db
from app.models.database import User, Organization, APIRequest

logger = logging.getLogger(__name__)

# Metrics
auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['method', 'status'])
token_operations = Counter('token_operations_total', 'Token operations', ['operation', 'status'])

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)

# Password hashing
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4,
)

# Redis client for token blacklisting and session management
redis_client: Optional[redis.Redis] = None


class TokenData(BaseModel):
    """JWT Token payload data"""
    sub: str  # User ID
    email: str
    organization_id: str
    role: str
    permissions: List[str] = []
    token_type: str = "access"
    session_id: Optional[str] = None
    
    
class TokenPair(BaseModel):
    """Access and refresh token pair"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    
    
class AuthenticationService:
    """
    Enterprise authentication service with:
    - JWT token management
    - API key authentication
    - OAuth2 flow support
    - Session management
    - MFA support
    - Rate limiting
    - Token revocation
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using Argon2"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return f"rr_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    async def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        token_operations.labels(operation='create_access', status='success').inc()
        return encoded_jwt
    
    @staticmethod
    async def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        token_operations.labels(operation='create_refresh', status='success').inc()
        return encoded_jwt
    
    @staticmethod
    async def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Check if token is blacklisted
            if redis_client:
                jti = payload.get("jti")
                if jti and await redis_client.exists(f"blacklist:{jti}"):
                    token_operations.labels(operation='decode', status='blacklisted').inc()
                    return None
            
            token_operations.labels(operation='decode', status='success').inc()
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            token_operations.labels(operation='decode', status='invalid').inc()
            return None
    
    @staticmethod
    async def revoke_token(token: str) -> bool:
        """Revoke a token by blacklisting its JTI"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            if jti and redis_client:
                # Calculate TTL based on token expiration
                exp = payload.get("exp")
                if exp:
                    ttl = exp - datetime.utcnow().timestamp()
                    if ttl > 0:
                        await redis_client.setex(
                            f"blacklist:{jti}",
                            int(ttl),
                            "revoked"
                        )
                        token_operations.labels(operation='revoke', status='success').inc()
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            token_operations.labels(operation='revoke', status='error').inc()
            return False
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        username: str,
        password: str
    ) -> Optional[User]:
        """Authenticate user with username/email and password"""
        # Try both username and email
        result = await db.execute(
            select(User).where(
                or_(
                    User.username == username,
                    User.email == username
                )
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            auth_attempts.labels(method='password', status='user_not_found').inc()
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            auth_attempts.labels(method='password', status='account_locked').inc()
            return None
        
        # Verify password
        if not AuthenticationService.verify_password(password, user.hashed_password):
            # Increment failed login count
            user.failed_login_count += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_count >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            await db.commit()
            auth_attempts.labels(method='password', status='invalid_password').inc()
            return None
        
        # Reset failed login count on successful authentication
        user.failed_login_count = 0
        user.last_login = datetime.utcnow()
        user.login_count += 1
        await db.commit()
        
        auth_attempts.labels(method='password', status='success').inc()
        return user
    
    @staticmethod
    async def authenticate_api_key(
        db: AsyncSession,
        api_key: str
    ) -> Optional[User]:
        """Authenticate user with API key"""
        # Hash the API key to compare with stored hash
        key_hash = AuthenticationService.hash_api_key(api_key)
        
        # Look up user by API key hash
        result = await db.execute(
            select(User).where(User.api_key == key_hash)
        )
        user = result.scalar_one_or_none()
        
        if user and user.is_active:
            auth_attempts.labels(method='api_key', status='success').inc()
            
            # Track API request
            api_request = APIRequest(
                user_id=user.id,
                endpoint="api_key_auth",
                method="AUTH",
                status_code=200,
                timestamp=datetime.utcnow()
            )
            db.add(api_request)
            await db.commit()
            
            return user
        
        auth_attempts.labels(method='api_key', status='invalid').inc()
        return None
    
    @staticmethod
    async def create_token_pair(user: User) -> TokenPair:
        """Create access and refresh token pair"""
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "organization_id": str(user.organization_id),
            "role": user.role,
            "permissions": AuthenticationService.get_user_permissions(user),
            "session_id": secrets.token_urlsafe(16)
        }
        
        access_token = await AuthenticationService.create_access_token(token_data)
        refresh_token = await AuthenticationService.create_refresh_token(token_data)
        
        # Store session in Redis
        if redis_client:
            session_key = f"session:{user.id}:{token_data['session_id']}"
            await redis_client.setex(
                session_key,
                86400 * 30,  # 30 days
                "active"
            )
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.JWT_EXPIRATION_HOURS * 3600
        )
    
    @staticmethod
    def get_user_permissions(user: User) -> List[str]:
        """Get user permissions based on role"""
        role_permissions = {
            "admin": [
                "portfolio:*",
                "risk:*",
                "user:*",
                "organization:*",
                "system:*"
            ],
            "risk_manager": [
                "portfolio:read",
                "portfolio:write",
                "portfolio:delete",
                "risk:*",
                "user:read",
                "organization:read"
            ],
            "analyst": [
                "portfolio:read",
                "portfolio:write",
                "risk:read",
                "risk:calculate",
                "user:read"
            ],
            "viewer": [
                "portfolio:read",
                "risk:read"
            ]
        }
        
        return role_permissions.get(user.role, [])
    
    @staticmethod
    async def validate_permissions(
        user: User,
        required_permissions: List[str]
    ) -> bool:
        """Validate user has required permissions"""
        user_permissions = AuthenticationService.get_user_permissions(user)
        
        for required in required_permissions:
            # Check for wildcard permissions
            if f"{required.split(':')[0]}:*" in user_permissions:
                continue
            
            if required not in user_permissions:
                return False
        
        return True


# Dependency functions for FastAPI
async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> User:
    """
    Get current authenticated user from various auth methods
    Priority: Bearer token > OAuth2 token > API key
    """
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try Bearer token first
    if bearer and bearer.credentials:
        token = bearer.credentials
    
    # Try JWT token authentication
    if token:
        payload = await AuthenticationService.decode_token(token)
        if not payload:
            raise credentials_exception
        
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception
        
        user = await db.get(User, user_id)
        if not user or not user.is_active:
            raise credentials_exception
        
        # Store request context
        request.state.user = user
        request.state.auth_method = "jwt"
        
        return user
    
    # Try API key authentication
    if api_key:
        user = await AuthenticationService.authenticate_api_key(db, api_key)
        if not user:
            raise credentials_exception
        
        # Store request context
        request.state.user = user
        request.state.auth_method = "api_key"
        
        return user
    
    raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is active"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def require_permissions(permissions: List[str]):
    """
    Decorator/dependency to require specific permissions
    Usage: @router.get("/", dependencies=[Depends(require_permissions(["portfolio:read"]))])
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ):
        if not await AuthenticationService.validate_permissions(current_user, permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permissions}"
            )
        return current_user
    
    return permission_checker


def require_role(roles: List[str]):
    """
    Decorator/dependency to require specific roles
    Usage: @router.get("/", dependencies=[Depends(require_role(["admin", "risk_manager"]))])
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required one of: {roles}"
            )
        return current_user
    
    return role_checker


class RateLimiter:
    """
    Rate limiting based on user tier and endpoint
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
    
    async def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        """Check rate limits for user"""
        if not redis_client:
            return True
        
        # Get rate limits based on user tier
        tier_limits = {
            "basic": (30, 500),
            "standard": (60, 1000),
            "premium": (120, 5000),
            "enterprise": (600, 50000)
        }
        
        per_minute, per_hour = tier_limits.get(
            current_user.api_tier,
            (self.requests_per_minute, self.requests_per_hour)
        )
        
        # Override with user-specific limits if set
        if current_user.rate_limit_override:
            per_minute = current_user.rate_limit_override
        
        # Check minute limit
        minute_key = f"rate:{current_user.id}:minute:{datetime.utcnow().minute}"
        minute_count = await redis_client.incr(minute_key)
        if minute_count == 1:
            await redis_client.expire(minute_key, 60)
        
        if minute_count > per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {per_minute} requests per minute",
                headers={"Retry-After": "60"}
            )
        
        # Check hour limit
        hour_key = f"rate:{current_user.id}:hour:{datetime.utcnow().hour}"
        hour_count = await redis_client.incr(hour_key)
        if hour_count == 1:
            await redis_client.expire(hour_key, 3600)
        
        if hour_count > per_hour:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {per_hour} requests per hour",
                headers={"Retry-After": "3600"}
            )
        
        # Add rate limit headers to response
        request.state.rate_limit_remaining = per_minute - minute_count
        request.state.rate_limit_reset = 60
        
        return True


# Initialize Redis connection
async def init_redis():
    """Initialize Redis connection for token management"""
    global redis_client
    try:
        redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connected for authentication service")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
