"""Lightweight, optional token + role auth for the API.

Disabled by default (no token configured) so local/dev and tests are frictionless. When
``NEUROFORGE_API_TOKEN`` is set, endpoints that depend on :func:`require_token` require a matching
``Authorization: Bearer <token>`` header. Roles are advisory and read from ``X-Role`` (``clinician``
or ``researcher``); approval endpoints require the ``clinician`` role.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

VALID_ROLES = {"clinician", "researcher"}


def _expected_token() -> str | None:
    return os.getenv("NEUROFORGE_API_TOKEN")


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = _expected_token()
    if not expected:
        return  # auth disabled
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "invalid or missing bearer token")


def require_clinician(x_role: str | None = Header(default=None)) -> str:
    """Require the clinician role for therapy approval (only enforced if auth/roles configured)."""
    if not _expected_token():
        return x_role or "clinician"  # roles advisory when auth disabled
    if x_role not in VALID_ROLES:
        raise HTTPException(403, "missing or invalid X-Role header")
    if x_role != "clinician":
        raise HTTPException(403, "approval requires the 'clinician' role")
    return x_role
