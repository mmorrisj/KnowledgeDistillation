import pytest

from server.config import Settings
from server.main import create_app
from server.static import SiteNotBuiltError


def test_serves_index(client):
    assert "home" in client.get("/").text


def test_serves_nested_page(client):
    response = client.get("/modules/x/")
    assert response.status_code == 200
    assert "lesson x" in response.text


def test_unknown_page_gets_styled_404(client):
    response = client.get("/nope/")
    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "Page not found" in response.text


def test_unknown_api_path_is_json_404(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_path_traversal_is_not_served(client, tmp_dist):
    (tmp_dist.parent / "secret.txt").write_text("secret")
    response = client.get("/%2e%2e/secret.txt")
    assert response.status_code == 404
    assert "secret" not in response.text


def test_missing_dist_raises_clear_error(tmp_path):
    with pytest.raises(SiteNotBuiltError, match="npm run build"):
        create_app(Settings(site_dist=tmp_path / "missing"))


def test_run_exits_cleanly_when_site_missing(monkeypatch, tmp_path, capsys):
    from server import main

    monkeypatch.setenv("SITE_DIST", str(tmp_path / "missing"))
    with pytest.raises(SystemExit) as exit_info:
        main.run()
    assert exit_info.value.code == 1
    assert "npm run build" in capsys.readouterr().err
