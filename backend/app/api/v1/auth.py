"""Authentication API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.auth import create_access_token, validate_google_token
from app.core.exceptions import AuthenticationError

router = APIRouter()


class TokenRequest(BaseModel):
    """Request model for token exchange."""

    access_token: str


class TokenResponse(BaseModel):
    """Response model for token exchange."""

    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def exchange_google_token(request: TokenRequest) -> TokenResponse:
    """Exchange Google OAuth2 token for JWT.

    Args:
        request: Token request with Google access token

    Returns:
        JWT token response

    Raises:
        HTTPException: If authentication fails
    """
    try:
        user_info = validate_google_token(request.access_token)
        # Create JWT with user info
        jwt_token = create_access_token(data={"sub": user_info.get("email")})
        return TokenResponse(access_token=jwt_token)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=e.message
        )


@router.get("/me")
async def get_current_user_info(
    current_user: dict = Depends(lambda: {}),  # TODO: Implement proper dependency
) -> dict:
    """Get current authenticated user information.

    Args:
        current_user: Current user from dependency

    Returns:
        User information
    """
    return current_user

