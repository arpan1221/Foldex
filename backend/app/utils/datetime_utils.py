"""Datetime utilities for US Eastern timezone."""

from datetime import datetime
from zoneinfo import ZoneInfo

# US Eastern timezone (handles DST automatically)
US_EASTERN = ZoneInfo("America/New_York")


def get_eastern_time() -> datetime:
    """Get current time in US Eastern timezone.
    
    Returns:
        Current datetime in US Eastern timezone
    """
    return datetime.now(US_EASTERN)


def to_eastern_time(dt: datetime) -> datetime:
    """Convert a datetime to US Eastern timezone.
    
    Args:
        dt: Datetime object (assumed to be UTC if naive)
        
    Returns:
        Datetime in US Eastern timezone
    """
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    
    return dt.astimezone(US_EASTERN)


def eastern_time_to_iso(dt: datetime) -> str:
    """Convert Eastern time datetime to ISO format string.
    
    Args:
        dt: Datetime in US Eastern timezone
        
    Returns:
        ISO format string with timezone
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=US_EASTERN)
    
    return dt.isoformat()

