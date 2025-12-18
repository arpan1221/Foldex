"""Authentication and authorization logic."""

from typing import Optional
from datetime import datetime, timedelta

import jwt
from app.config.settings import settings
from app.core.exceptions import AuthenticationError


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> dict:
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
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")


def validate_google_token(access_token: str) -> dict:
    """Validate Google OAuth2 access token.

    Args:
        access_token: Google OAuth2 access token

    Returns:
        User information from Google

    Raises:
        AuthenticationError: If token is invalid
    """
    # TODO: Implement Google token validation
    # This should call Google's tokeninfo endpoint
    # For now, return placeholder
    raise NotImplementedError("Google token validation not yet implemented")

