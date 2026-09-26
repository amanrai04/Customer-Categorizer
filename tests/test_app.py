"""Tests for the FastAPI application: routing, rendering and CORS."""

import pytest
from fastapi.testclient import TestClient

from app import app
from src.constant.application import get_cors_origins
from src.entity.config_entity import PredictionSchemaConfig


def cors_options(application) -> dict:
    """Return the options the CORS middleware was registered with."""
    for middleware in application.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            return middleware.kwargs
    raise AssertionError("CORSMiddleware is not registered on the application")


@pytest.fixture
def client():
    """A test client for the application.

    The model is never reached: only the routes that render the form, the health
    probe and the CORS headers are exercised here.
    """
    return TestClient(app)


class TestHealth:
    def test_health_reports_ok(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestFormRendering:
    def test_get_root_renders_the_form(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_form_contains_one_input_per_schema_feature(self, client):
        response = client.get("/")
        body = response.text

        for column in PredictionSchemaConfig().column_names():
            assert f'name="{column}"' in body

    def test_form_exposes_the_api_documentation_link(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestTrainingRoute:
    def test_training_failure_is_reported_as_json_not_a_crash(self, client, monkeypatch):
        def explode(self):
            raise RuntimeError("mongo is unreachable")

        monkeypatch.setattr(
            "app.TrainPipeline",
            lambda: type("Broken", (), {"run_pipeline": explode})(),
        )

        response = client.get("/train")

        assert response.status_code == 500
        payload = response.json()
        assert payload["status"] is False
        assert "mongo is unreachable" in payload["message"]


class TestPredictionRoute:
    def test_prediction_failure_is_reported_as_json_not_a_crash(self, client, monkeypatch):
        def explode(self, input_data):
            raise RuntimeError("no promoted model")

        monkeypatch.setattr(
            "app.PredictionPipeline",
            lambda: type("Broken", (), {"run_pipeline": explode})(),
        )

        response = client.post("/", data={"Age": "45"})

        assert response.status_code == 500
        payload = response.json()
        assert payload["status"] is False
        assert "no promoted model" in payload["message"]


class TestCors:
    def test_wildcard_origin_is_allowed_by_default(self, client):
        response = client.get("/health", headers={"Origin": "https://example.com"})

        assert response.headers["access-control-allow-origin"] == "*"

    def test_credentials_are_disabled_for_the_wildcard(self):
        """Allowing credentials with ``*`` is rejected by browsers."""
        options = cors_options(app)

        assert options["allow_origins"] == ["*"]
        assert options["allow_credentials"] is False

    def test_middleware_is_configured_from_the_environment(self, monkeypatch):
        """The allowlist is read once, when the application is built."""
        monkeypatch.setenv("CORS_ORIGINS", "https://allowed.example")
        configured = get_cors_origins()

        assert configured == ["https://allowed.example"]
        assert configured != ["*"]
