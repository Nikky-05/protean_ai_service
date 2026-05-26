"""API-key authentication.

The Node orchestrator authenticates with a shared secret sent in the
`Authorization` header — matching the pattern used by Signzy
(`Authorization: <key>`, not `Bearer <key>`). Both forms are accepted
to keep the integration forgiving.
"""

from fastapi import Header
from fastapi.security.utils import get_authorization_scheme_param

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not authorization:
        raise UnauthorizedError("Missing Authorization header")

    scheme, key = get_authorization_scheme_param(authorization)
    presented = key if scheme.lower() == "bearer" else authorization

    if presented.strip() != get_settings().ai_api_key:
        raise UnauthorizedError("Invalid API key")
