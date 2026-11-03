import base64
import json
import time

from authlib.integrations.base_client.errors import OAuthError
from fastapi.testclient import TestClient
import httpx
from itsdangerous import TimestampSigner
import pytest

from webapp.app import create_app

ORIGIN = "https://private.example"
ALLOWED_ID = 123456
SETTINGS = {"SESSION_SECRET": "test-session-secret-that-is-long-enough",
            "PUBLIC_ORIGIN": ORIGIN, "GITHUB_CLIENT_ID": "client-id",
            "ALLOWED_GITHUB_USER_ID": str(ALLOWED_ID), "MODEL_ENDPOINT": "https://model.example",
            "GITHUB_CLIENT_SECRET": "github-secret",
            "MODAL_KEY": "proxy-id", "MODAL_SECRET": "proxy-secret"}


class FakeGitHub:
    def __init__(self, user_id=ALLOWED_ID, login="test-owner", error=False):
        self.user_id, self.login, self.error = user_id, login, error

    async def authorize_access_token(self, request):
        if self.error:
            raise OAuthError("Invalid state")
        return {"access_token": "github-access-token"}

    async def get(self, path, token):
        return httpx.Response(200, json={"id": self.user_id, "login": self.login},
                              request=httpx.Request("GET", "https://api.github.com/user"))


def client(github=None, transport=None, settings=None):
    return TestClient(create_app(settings or SETTINGS, oauth_client=github, transport=transport),
                      base_url=ORIGIN)


def owner(browser):
    response = browser.get("/auth/callback", follow_redirects=False)
    assert response.status_code == 303
    assert "secure" in response.headers["set-cookie"].lower()
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "github-access-token" not in response.headers["set-cookie"]
    return browser.get("/api/me").json()["csrf"]


def test_unconfigured_and_unauthenticated_fail_closed():
    browser = client(settings={"SESSION_SECRET": SETTINGS["SESSION_SECRET"]})
    assert browser.get("/").status_code == 200
    assert browser.get("/login").status_code == 503
    assert browser.get("/api/me").status_code == 401
    assert browser.post("/api/chat", json={}).status_code == 401
    assert browser.get("/.env").status_code == 404
    assert browser.get("/docs").status_code == 404


def test_real_oauth_requires_matching_state():
    browser = client()
    response = browser.get("/login", follow_redirects=False)
    assert response.status_code == 302
    assert "github.com/login/oauth/authorize" in response.headers["location"]
    assert "state=" in response.headers["location"]
    assert browser.get("/auth/callback?code=fake&state=forged").status_code == 401
    assert browser.get("/api/me").status_code == 401


def test_other_account_with_same_display_login_is_denied():
    browser = client(FakeGitHub(user_id=123, login="test-owner"))
    assert browser.get("/auth/callback").status_code == 403
    assert browser.get("/api/me").status_code == 401


def test_verified_id_survives_username_change():
    browser = client(FakeGitHub(login="renamed-owner"))
    owner(browser)
    assert browser.get("/api/me").json()["login"] == "renamed-owner"


def test_forged_cookie_and_expired_session_are_denied():
    browser = client()
    browser.cookies.set("__Host-qwen-session", "forged")
    assert browser.get("/api/me").status_code == 401
    data = base64.b64encode(json.dumps({"user_id": ALLOWED_ID, "expires_at": time.time()-1}).encode())
    signed = TimestampSigner(SETTINGS["SESSION_SECRET"]).sign(data).decode()
    browser.cookies.set("__Host-qwen-session", signed)
    assert browser.get("/api/me").status_code == 401


def test_csrf_origin_and_logout_enforced():
    browser = client(FakeGitHub())
    csrf = owner(browser)
    assert browser.post("/api/chat", headers={"Origin": ORIGIN}, json={}).status_code == 403
    assert browser.post("/api/chat", headers={"Origin": "https://evil.example", "X-CSRF-Token": csrf}, json={}).status_code == 403
    assert browser.post("/logout", headers={"Origin": ORIGIN, "X-CSRF-Token": csrf}).status_code == 200
    assert browser.get("/api/me").status_code == 401


def test_authenticated_stream_only_targets_medical_adapter():
    calls = []

    def upstream(request):
        calls.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == "qwen-medical"
        assert request.headers["Modal-Secret"] == SETTINGS["MODAL_SECRET"]
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Hello!"}}]}\n\ndata: [DONE]\n\n')

    browser = client(FakeGitHub(), httpx.MockTransport(upstream))
    csrf = owner(browser)
    response = browser.post("/api/chat", headers={"Origin": ORIGIN, "X-CSRF-Token": csrf},
                            json={"messages": [{"role": "user", "content": "Hi"}]})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events == [{"type": "token", "text": "Hello!"}, {"type": "done", "usage": None}]
    assert len(calls) == 1
    assert SETTINGS["MODAL_SECRET"] not in response.text


def test_oversize_payload_rejected_before_inference():
    browser = client(FakeGitHub())
    csrf = owner(browser)
    assert browser.post("/api/chat", headers={"Origin": ORIGIN, "X-CSRF-Token": csrf},
                        content=b"x" * 32769).status_code == 413


def test_missing_session_secret_prevents_startup():
    with pytest.raises(RuntimeError):
        create_app({})


def test_missing_owner_allowlist_prevents_login():
    settings = dict(SETTINGS)
    settings.pop("ALLOWED_GITHUB_USER_ID")
    browser = client(FakeGitHub(), settings=settings)
    assert browser.get("/login").status_code == 503
    assert browser.get("/auth/callback").status_code == 503
    assert browser.get("/api/me").status_code == 401


def test_missing_model_endpoint_prevents_proxy_request():
    settings = dict(SETTINGS)
    settings.pop("MODEL_ENDPOINT")
    browser = client(FakeGitHub(), settings=settings)
    csrf = owner(browser)
    assert browser.post("/api/chat", headers={"Origin": ORIGIN, "X-CSRF-Token": csrf},
                        json={"messages": [{"role": "user", "content": "Hi"}]}).status_code == 503
