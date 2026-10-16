"""Optional API-key protection for destructive endpoints."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import get_settings


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require a matching X-API-Key header when ADMIN_API_KEY is set.

    With no key configured the endpoint stays open, which keeps local development simple.
    """
    expected = get_settings().admin_api_key
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
