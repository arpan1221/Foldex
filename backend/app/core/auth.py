"""Google OAuth2 authentication and JWT token management."""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import httpx
import jwt
import structlog

from app.config.settings import settings
from app.core.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)

# Google OAuth2 endpoints
GOOGLE_TOKEN_INFO_URL = "https://www.googleapis.com/oauth2/v1/tokeninfo"
GOOGLE_USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
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


async def validate_google_token(access_token: str) -> Dict[str, Any]:
    """Validate Google OAuth2 access token and fetch user information.

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
        # First, verify the token with Google
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_response = await client.get(
                GOOGLE_TOKEN_INFO_URL,
                params={"access_token": access_token},
            )

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

            # Fetch user information
            user_response = await client.get(
                GOOGLE_USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )

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
