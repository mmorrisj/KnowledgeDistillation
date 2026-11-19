from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app


@pytest.fixture
def tmp_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "modules" / "x").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>home</body></html>")
    (dist / "404.html").write_text("<html><body>Page not found</body></html>")
    (dist / "modules" / "x" / "index.html").write_text("<html><body>lesson x</body></html>")
    return dist


@pytest.fixture
def client(tmp_dist: Path) -> TestClient:
    return TestClient(create_app(Settings(site_dist=tmp_dist)))
