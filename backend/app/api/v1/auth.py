"""Authentication API endpoints for Google OAuth2."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import structlog

from app.core.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_refresh_token,
    validate_google_token,
    exchange_authorization_code,
)
from app.core.exceptions import AuthenticationError, DatabaseError
from app.core.security import validate_email
from app.config.settings import settings
from app.models.auth import (
    TokenRequest,
    TokenResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    UserResponse,
)
from app.database.sqlite_manager import SQLiteManager
from app.api.deps import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        User information dictionary

    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Get user from database
        db_manager = SQLiteManager()
        user = await db_manager.get_user_by_id(user_id)

        if not user:
            raise AuthenticationError("User not found")

        return user

    except AuthenticationError as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def exchange_google_token(
    request: TokenRequest, db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """Exchange Google OAuth2 token for JWT tokens.

    Validates the Google token, creates or updates user record,
    and returns JWT access and refresh tokens.

    Args:
        request: Token request with Google access token
        db: Database session

    Returns:
        Token response with JWT tokens

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Check if this is an authorization code or access token
        google_token = request.google_token
        
        # Authorization codes typically start with "4/" and are shorter than access tokens
        # Access tokens are much longer (usually 100-200+ chars) and start with "ya29."
        is_auth_code = google_token.startswith("4/") or (len(google_token) < 200 and not google_token.startswith("ya29."))
        
        google_refresh_token = None
        google_token_expiry = None
        
        if is_auth_code:
            # This is an authorization code - exchange it for an access token
            logger.info("Exchanging authorization code for access token")
            token_response = await exchange_authorization_code(
                code=google_token,
                redirect_uri=settings.GOOGLE_REDIRECT_URI
            )
            google_token = token_response["access_token"]
            google_refresh_token = token_response.get("refresh_token")
            expires_in = token_response.get("expires_in", 3600)
            google_token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            logger.info("Successfully obtained access token from authorization code")
        
        # Validate Google token and get user info
        google_user_info = await validate_google_token(google_token)

        if not validate_email(google_user_info["email"]):
            raise AuthenticationError("Invalid email address")

        # Get or create user in database
        db_manager = SQLiteManager()
        user = await db_manager.get_or_create_user(
            google_id=google_user_info["google_id"],
            email=google_user_info["email"],
            name=google_user_info.get("name"),
            picture=google_user_info.get("picture"),
            verified_email=google_user_info.get("verified_email", False),
            google_access_token=google_token,
            google_refresh_token=google_refresh_token,
            google_token_expiry=google_token_expiry,
            session=db,  # Use the session from dependency injection
        )
        await db.commit()  # Commit the transaction

        # Create JWT tokens
        token_data = {
            "sub": user["user_id"],
            "email": user["email"],
            "google_id": user.get("google_id"),
        }

        access_token = create_access_token(data=token_data)
        refresh_token = create_refresh_token(data=token_data)

        logger.info(
            "Token exchange successful",
            user_id=user["user_id"],
            email=user["email"],
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours in seconds
        )

    except AuthenticationError as e:
        logger.warning("Token exchange failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
    except DatabaseError as e:
        logger.error("Database error in token exchange", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.message}",
        )
    except Exception as e:
        logger.error("Unexpected error in token exchange", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )


@router.post(
    "/refresh", response_model=RefreshTokenResponse, status_code=status.HTTP_200_OK
)
async def refresh_access_token(
    request: RefreshTokenRequest, db: AsyncSession = Depends(get_db_session)
) -> RefreshTokenResponse:
    """Refresh JWT access token using refresh token.

    Args:
        request: Refresh token request
        db: Database session

    Returns:
        New access token response

    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Verify refresh token
        payload = verify_refresh_token(request.refresh_token)
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError("Invalid refresh token payload")

        # Verify user exists
        db_manager = SQLiteManager()
        user = await db_manager.get_user_by_id(user_id)

        if not user:
            raise AuthenticationError("User not found")

        # Create new access token
        token_data = {
            "sub": user["user_id"],
            "email": user["email"],
            "google_id": user.get("google_id"),
        }

        access_token = create_access_token(data=token_data)

        logger.info("Token refresh successful", user_id=user_id)

        return RefreshTokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours in seconds
        )

    except AuthenticationError as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected error in token refresh", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh",
        )


@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
) -> UserResponse:
    """Get current authenticated user information.

    Args:
        current_user: Current user from dependency injection

    Returns:
        User information response

    Raises:
        HTTPException: If user not found
    """
    try:
        return UserResponse(
            user_id=current_user["user_id"],
            email=current_user["email"],
            name=current_user.get("name"),
            picture=current_user.get("picture"),
            verified_email=current_user.get("verified_email", False),
        )
    except Exception as e:
        logger.error("Failed to get user info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information",
        )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Logout endpoint (client should discard tokens).

    Note: Since we use stateless JWT tokens, logout is handled
    client-side by discarding tokens. This endpoint provides
    a way to log the logout event.

    Args:
        current_user: Current user from dependency injection

    Returns:
        Success message
    """
    logger.info("User logged out", user_id=current_user.get("user_id"))
    return {"message": "Logged out successfully"}
