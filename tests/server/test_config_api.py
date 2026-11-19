from server.config import Settings


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_config_defaults_to_tracking_off(client):
    assert client.get("/api/config").json() == {"tracking_enabled": False}


def test_tracking_flag_read_from_env(monkeypatch):
    monkeypatch.setenv("TRACKING_ENABLED", "true")
    assert Settings().tracking_enabled is True
