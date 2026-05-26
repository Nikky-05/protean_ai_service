"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1.router import api_v1
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.middleware.correlation import CorrelationIdMiddleware
from app.middleware.error_handler import install_error_handlers
from app.modules.image.predictors.factory import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info("startup", env=settings.app_env, image_backend=settings.image_ai_backend)
    # Warm the predictor so the first request doesn't pay model-load time.
    get_predictor()
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Protean AI Services",
        version=__version__,
        description="In-house Python AI service for Protean orchestrated APIs",
        lifespan=lifespan,
    )

    app.add_middleware(CorrelationIdMiddleware)
    install_error_handlers(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(api_v1)
    return app


app = create_app()
