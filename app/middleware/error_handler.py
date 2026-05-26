"""Centralised exception → JSON-response mapping."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.logging import get_logger

log = get_logger(__name__)


def _json_safe_validation_errors(exc: RequestValidationError) -> list[dict]:
    errors = exc.errors()
    for err in errors:
        ctx = err.get("ctx")
        if not isinstance(ctx, dict):
            continue
        for key, value in list(ctx.items()):
            if isinstance(value, Exception):
                ctx[key] = str(value)
    return errors


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                },
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request payload failed validation",
                    "details": _json_safe_validation_errors(exc),
                },
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"},
            },
        )
