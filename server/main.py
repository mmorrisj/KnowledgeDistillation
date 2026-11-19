"""FastAPI app: a small API plus the built lesson site."""

import sys

import uvicorn
from fastapi import APIRouter, FastAPI

from server.config import Settings
from server.static import SiteNotBuiltError, mount_site


def build_api(settings: Settings) -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/config")
    def config() -> dict[str, bool]:
        return {"tracking_enabled": settings.tracking_enabled}

    return api


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Knowledge Distillation Course", docs_url=None, redoc_url=None)
    app.include_router(build_api(settings))
    mount_site(app, settings)
    return app


def run() -> None:
    """Console entry point: `kd-course-serve`."""
    settings = Settings()
    try:
        app = create_app(settings)
    except SiteNotBuiltError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    uvicorn.run(app, host=settings.host, port=settings.port)
