"""Authentication data models for API requests and responses."""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """User model for API responses."""

    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    google_id: Optional[str] = Field(None, description="Google user ID")
    picture: Optional[str] = Field(None, description="User profile picture URL")
    verified_email: bool = Field(default=False, description="Email verification status")
    created_at: Optional[datetime] = Field(None, description="User creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="User last update timestamp")


class TokenRequest(BaseModel):
    """Request model for token exchange."""

    google_token: str = Field(
        ..., description="Google OAuth2 access token", alias="google_token"
    )

    class Config:
        populate_by_name = True


class TokenResponse(BaseModel):
    """Response model for token exchange."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""

    refresh_token: str = Field(..., description="JWT refresh token")


class RefreshTokenResponse(BaseModel):
    """Response model for token refresh."""

    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class UserResponse(BaseModel):
    """Response model for user information."""

    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    picture: Optional[str] = Field(None, description="User profile picture URL")
    verified_email: bool = Field(default=False, description="Email verification status")
