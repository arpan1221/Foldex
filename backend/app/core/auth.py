"""Google OAuth2 authentication and JWT token management."""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import httpx
import jwt
import structlog

from app.config.settings import settings
from app.core.exceptions import AuthenticationError

# Import settings at module level
settings = settings

logger = structlog.get_logger(__name__)

# Google OAuth2 endpoints
GOOGLE_TOKEN_INFO_URL = "https://www.googleapis.com/oauth2/v1/tokeninfo"
GOOGLE_USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token.

    Args:
        data: Data to encode in token (should include 'sub' for user identifier)
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string

    Raises:
        AuthenticationError: If token creation fails
    """
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                hours=settings.JWT_EXPIRATION_HOURS
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        logger.debug("Access token created", user_id=data.get("sub"))
        return encoded_jwt
    except Exception as e:
        logger.error("Failed to create access token", error=str(e))
        raise AuthenticationError("Failed to create access token") from e


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token", error=str(e))
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise AuthenticationError("Token verification failed") from e


async def exchange_authorization_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange Google OAuth2 authorization code for access token.

    Args:
        code: Authorization code from Google OAuth2 callback
        redirect_uri: The redirect URI used in the authorization request

    Returns:
        Dictionary containing:
        - access_token: Google access token
        - refresh_token: Google refresh token (if available)
        - expires_in: Token expiration time in seconds

    Raises:
        AuthenticationError: If code exchange fails
    """
    try:
        # Validate settings are loaded
        if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
            logger.error(
                "Google OAuth credentials not configured",
                has_client_id=bool(settings.GOOGLE_CLIENT_ID),
                has_client_secret=bool(settings.GOOGLE_CLIENT_SECRET),
            )
            raise AuthenticationError("Google OAuth credentials not configured")
        
        # Use granular timeout configuration
        timeout = httpx.Timeout(
            connect=5.0,  # 5 seconds to establish connection
            read=25.0,   # 25 seconds to read response
            write=5.0,   # 5 seconds to write request
            pool=5.0    # 5 seconds to get connection from pool
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_data = {
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
            
            logger.debug(
                "Exchanging authorization code",
                has_client_id=bool(settings.GOOGLE_CLIENT_ID),
                redirect_uri=redirect_uri,
            )
            
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data=request_data,
            )

            if response.status_code != 200:
                logger.error(
                    "Failed to exchange authorization code",
                    status_code=response.status_code,
                    response=response.text,
                )
                raise AuthenticationError("Failed to exchange authorization code")

            token_data = response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise AuthenticationError("No access token in response")

            logger.info("Successfully exchanged authorization code for access token")
            return {
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token"),
                "expires_in": token_data.get("expires_in", 3600),
            }

    except httpx.TimeoutException:
        logger.error("Timeout exchanging authorization code")
        raise AuthenticationError("Timeout exchanging authorization code")
    except httpx.RequestError as e:
        logger.error("Request error exchanging authorization code", error=str(e))
        raise AuthenticationError("Failed to exchange authorization code") from e
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Unexpected error exchanging authorization code", error=str(e))
        raise AuthenticationError("Failed to exchange authorization code") from e


async def validate_google_token(access_token: str) -> Dict[str, Any]:
    """Validate Google OAuth2 access token and fetch user information.

    Optimized to make concurrent HTTP requests instead of sequential,
    reducing total timeout from 60s to 30s in worst case.

    Args:
        access_token: Google OAuth2 access token

    Returns:
        User information from Google including:
        - id: Google user ID
        - email: User email
        - name: User name
        - picture: User profile picture URL
        - verified_email: Email verification status

    Raises:
        AuthenticationError: If token is invalid or validation fails
    """
    try:
        # Make both requests concurrently to reduce total latency
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create both request tasks concurrently
            token_task = client.get(
                GOOGLE_TOKEN_INFO_URL,
                params={"access_token": access_token},
            )
            user_task = client.get(
                GOOGLE_USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            # Execute both requests concurrently
            token_response, user_response = await asyncio.gather(
                token_task, user_task, return_exceptions=True
            )

            # Handle exceptions from concurrent requests
            if isinstance(token_response, Exception):
                if isinstance(token_response, httpx.TimeoutException):
                    logger.error("Timeout validating Google token info")
                    raise AuthenticationError("Timeout validating Google token")
                logger.error(
                    "Request error validating Google token",
                    error=str(token_response)
                )
                raise AuthenticationError("Failed to validate Google token") from token_response

            if isinstance(user_response, Exception):
                if isinstance(user_response, httpx.TimeoutException):
                    logger.error("Timeout fetching Google user info")
                    raise AuthenticationError("Timeout fetching user information")
                logger.error(
                    "Request error fetching Google user info",
                    error=str(user_response)
                )
                raise AuthenticationError("Failed to fetch user information") from user_response

            # Validate token info first
            if token_response.status_code != 200:
                logger.warning(
                    "Google token validation failed",
                    status_code=token_response.status_code,
                )
                raise AuthenticationError("Invalid Google access token")

            token_info = token_response.json()

            # Check if token has required scope
            if "scope" not in token_info:
                raise AuthenticationError("Token missing required scopes")

            scopes = token_info["scope"].split()
            if GOOGLE_DRIVE_SCOPE not in scopes:
                logger.warning("Token missing Drive scope", scopes=scopes)
                raise AuthenticationError(
                    "Token missing required Google Drive scope"
                )

            # Check token expiration
            if "expires_in" in token_info:
                expires_in = token_info["expires_in"]
                if expires_in <= 0:
                    raise AuthenticationError("Google token has expired")

            # Validate user info response
            if user_response.status_code != 200:
                logger.warning(
                    "Failed to fetch user info",
                    status_code=user_response.status_code,
                )
                raise AuthenticationError("Failed to fetch user information")

            user_info = user_response.json()

            # Validate required fields
            if "id" not in user_info or "email" not in user_info:
                raise AuthenticationError("Invalid user information from Google")

            logger.info(
                "Google token validated successfully",
                user_id=user_info.get("id"),
                email=user_info.get("email"),
            )

            return {
                "google_id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "picture": user_info.get("picture"),
                "verified_email": user_info.get("verified_email", False),
            }

    except httpx.TimeoutException:
        logger.error("Timeout validating Google token")
        raise AuthenticationError("Timeout validating Google token")
    except httpx.RequestError as e:
        logger.error("Request error validating Google token", error=str(e))
        raise AuthenticationError("Failed to validate Google token") from e
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Unexpected error validating Google token", error=str(e))
        raise AuthenticationError("Failed to validate Google token") from e


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token with longer expiration.

    Args:
        data: Data to encode in token

    Returns:
        Encoded JWT refresh token string
    """
    try:
        to_encode = data.copy()
        # Refresh tokens expire in 30 days
        expire = datetime.utcnow() + timedelta(days=30)
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        logger.debug("Refresh token created", user_id=data.get("sub"))
        return encoded_jwt
    except Exception as e:
        logger.error("Failed to create refresh token", error=str(e))
        raise AuthenticationError("Failed to create refresh token") from e


def verify_refresh_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT refresh token.

    Args:
        token: JWT refresh token string

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Refresh token expired")
        raise AuthenticationError("Refresh token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid refresh token", error=str(e))
        raise AuthenticationError("Invalid refresh token")
    except Exception as e:
        logger.error("Refresh token verification failed", error=str(e))
        raise AuthenticationError("Refresh token verification failed") from e


async def refresh_google_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh Google OAuth2 access token using refresh token.

    Args:
        refresh_token: Google OAuth2 refresh token

    Returns:
        Dictionary containing:
        - access_token: New Google access token
        - expires_in: Token expiration time in seconds

    Raises:
        AuthenticationError: If token refresh fails
    """
    try:
        # Validate settings are loaded
        if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
            logger.error(
                "Google OAuth credentials not configured",
                has_client_id=bool(settings.GOOGLE_CLIENT_ID),
                has_client_secret=bool(settings.GOOGLE_CLIENT_SECRET),
            )
            raise AuthenticationError("Google OAuth credentials not configured")
        
        # Use granular timeout configuration
        timeout = httpx.Timeout(
            connect=5.0,  # 5 seconds to establish connection
            read=25.0,    # 25 seconds to read response
            write=5.0,    # 5 seconds to write request
            pool=5.0     # 5 seconds to get connection from pool
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_data = {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
            
            logger.debug("Refreshing Google access token")
            
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data=request_data,
            )

            if response.status_code != 200:
                logger.error(
                    "Failed to refresh Google access token",
                    status_code=response.status_code,
                    response=response.text,
                )
                raise AuthenticationError("Failed to refresh Google access token")

            token_data = response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise AuthenticationError("No access token in refresh response")

            expires_in = token_data.get("expires_in", 3600)

            logger.info("Successfully refreshed Google access token")
            return {
                "access_token": access_token,
                "expires_in": expires_in,
            }

    except httpx.TimeoutException:
        logger.error("Timeout refreshing Google access token")
        raise AuthenticationError("Timeout refreshing Google access token")
    except httpx.RequestError as e:
        logger.error("Request error refreshing Google access token", error=str(e))
        raise AuthenticationError("Failed to refresh Google access token") from e
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Unexpected error refreshing Google access token", error=str(e))
        raise AuthenticationError("Failed to refresh Google access token") from e
