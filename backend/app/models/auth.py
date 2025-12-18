"""Authentication data models."""

from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):
    """User model."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email")
    name: Optional[str] = Field(None, description="User name")
    google_id: Optional[str] = Field(None, description="Google user ID")


class Token(BaseModel):
    """Token model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")

