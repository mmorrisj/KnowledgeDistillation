"""Serve the built Astro site from disk."""

from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles

from server.config import Settings


class SiteNotBuiltError(RuntimeError):
    """Raised at startup when the built site is missing."""


def mount_site(app: FastAPI, settings: Settings) -> None:
    dist = settings.site_dist
    if not (dist / "index.html").is_file():
        raise SiteNotBuiltError(
            f"The lesson site has not been built: '{dist}/index.html' was not found. "
            "Build it with 'make build' (or 'npm ci && npm run build' inside site/), "
            "or point SITE_DIST at an existing build."
        )

    # Unknown API paths must answer as JSON, not fall through to the HTML 404 page.
    @app.api_route("/api/{rest:path}", include_in_schema=False)
    def api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404)

    # html=True serves index.html for folders and 404.html (status 404) for misses.
    app.mount("/", StaticFiles(directory=dist, html=True), name="site")
