"""Domain-level exceptions shared across modules.

Every exception maps to an HTTP status via `app.middleware.error_handler`.
Module-specific errors should subclass `AppError` rather than raising
`HTTPException` directly so logging / observability stays consistent.
"""

from typing import Any


class AppError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AppError):
    status_code = 400
    error_code = "VALIDATION_ERROR"


class UnauthorizedError(AppError):
    status_code = 401
    error_code = "UNAUTHORIZED"


class NotFoundError(AppError):
    status_code = 404
    error_code = "NOT_FOUND"


class UpstreamError(AppError):
    status_code = 502
    error_code = "UPSTREAM_ERROR"


class ModelInferenceError(AppError):
    """Raised by predictor backends when the underlying ML call fails."""

    status_code = 503
    error_code = "MODEL_INFERENCE_ERROR"
