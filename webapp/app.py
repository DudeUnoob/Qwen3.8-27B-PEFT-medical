"""GitHub allowlisted sessions and a server-side streaming inference proxy."""

import asyncio
import json
import os
from pathlib import Path
import secrets
import time
from typing import Literal

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.sessions import SessionMiddleware

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class Chat(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=16)
    api_version: Literal["v1", "v2"] = "v1"


def create_app(settings=None, *, oauth_client=None, transport=None):
    settings = dict(os.environ if settings is None else settings)
    origin = settings.get("PUBLIC_ORIGIN", "").rstrip("/")
    allowed_id = int(settings.get("ALLOWED_GITHUB_USER_ID", "0"))
    model_endpoints = {"v1": settings.get("MODEL_ENDPOINT", "").rstrip("/"),
                       "v2": settings.get("MODEL_ENDPOINT_V2", "").rstrip("/")}
    session_secret = settings.get("SESSION_SECRET", "")
    if len(session_secret) < 32:
        raise RuntimeError("A persistent SESSION_SECRET of at least 32 characters is required")
    configured = bool(allowed_id > 0 and origin.startswith("https://") and settings.get("GITHUB_CLIENT_ID")
                      and settings.get("GITHUB_CLIENT_SECRET"))
    oauth = OAuth()
    github = oauth_client or oauth.register(
        name="github", client_id=settings.get("GITHUB_CLIENT_ID", ""),
        client_secret=settings.get("GITHUB_CLIENT_SECRET", ""),
        authorize_url="https://github.com/login/oauth/authorize",
        access_token_url="https://github.com/login/oauth/access_token",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": ""},  # Public identity only; no repo/email permissions.
    )
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SessionMiddleware, secret_key=session_secret,
                       session_cookie="__Host-qwen-session", https_only=True,
                       same_site="lax", max_age=8 * 3600)
    slots = asyncio.Semaphore(2)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        })
        return response

    def identity(request):
        if (not configured or request.session.get("user_id") != allowed_id
                or request.session.get("expires_at", 0) <= time.time()):
            raise HTTPException(401, "Sign in to continue")

    def mutation(request):
        identity(request)
        if not origin or request.headers.get("origin") != origin:
            raise HTTPException(403, "Invalid request origin")
        supplied = request.headers.get("x-csrf-token", "")
        expected = request.session.get("csrf", "")
        if not expected or not secrets.compare_digest(supplied, expected):
            raise HTTPException(403, "Invalid request token")

    @app.get("/")
    async def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/me")
    async def me(request: Request):
        identity(request)
        return {"login": request.session["login"], "csrf": request.session["csrf"],
                "api_versions": [v for v, url in model_endpoints.items() if url.startswith("https://")]}

    @app.get("/login")
    async def login(request: Request):
        if not configured:
            raise HTTPException(503, "GitHub login setup is awaiting OAuth app credentials")
        request.session.clear()
        return await github.authorize_redirect(request, origin + "/auth/callback")

    @app.get("/auth/callback")
    async def callback(request: Request):
        if not configured:
            raise HTTPException(503, "Login is not configured")
        try:
            token = await github.authorize_access_token(request)
            response = await github.get("user", token=token)
            response.raise_for_status()
            profile = response.json()
        except (OAuthError, httpx.HTTPError, ValueError):
            request.session.clear()
            raise HTTPException(401, "GitHub sign-in could not be verified; please try again")
        request.session.clear()
        if profile.get("id") != allowed_id:
            raise HTTPException(403, "This application is restricted to its owner")
        # Never put GitHub or Modal access tokens into the browser's session.
        request.session.update(user_id=allowed_id, login=profile["login"],
                               expires_at=int(time.time()) + 8 * 3600,
                               csrf=secrets.token_urlsafe(32))
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    async def logout(request: Request):
        mutation(request)
        request.session.clear()
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        mutation(request)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 32768:
                raise HTTPException(413, "Conversation is too large; start a new chat")
        try:
            data = Chat.model_validate_json(body)
        except ValidationError:
            raise HTTPException(422, "Send between 1 and 16 user/assistant messages and API v1 or v2")
        if data.messages[-1].role != "user":
            raise HTTPException(422, "The last message must be from the user")
        model_endpoint = model_endpoints[data.api_version]
        if not model_endpoint.startswith("https://") or not settings.get("MODAL_KEY") or not settings.get("MODAL_SECRET"):
            raise HTTPException(503, f"Model connection for API {data.api_version} is not configured")

        async def events():
            def event(value):
                return json.dumps(value) + "\n"
            try:
                async with slots, httpx.AsyncClient(
                    timeout=httpx.Timeout(1500, connect=30), transport=transport,
                    follow_redirects=True,
                ) as client:
                    async with client.stream("POST", model_endpoint + "/v1/chat/completions",
                        headers={"Modal-Key": settings["MODAL_KEY"],
                                 "Modal-Secret": settings["MODAL_SECRET"]},
                        json={"model": "qwen-medical", "messages": [m.model_dump() for m in data.messages],
                              "max_tokens": 512, "temperature": 0, "stream": True,
                              "stream_options": {"include_usage": True},
                              "chat_template_kwargs": {"enable_thinking": False}},
                    ) as response:
                        if response.status_code != 200:
                            message = ("The conversation may exceed the model's 2048-token context. Start a new chat."
                                       if response.status_code == 400 else "The model is unavailable. Please try again.")
                            yield event({"type": "error", "message": message})
                            return
                        usage = None
                        async for line in response.aiter_lines():
                            if await request.is_disconnected():
                                return
                            if not line.startswith("data: "):
                                continue
                            if line[6:] == "[DONE]":
                                yield event({"type": "done", "usage": usage})
                                return
                            part = json.loads(line[6:])
                            if "error" in part:
                                yield event({"type": "error", "message": "Generation failed. Please try again."})
                                return
                            usage = part.get("usage") or usage
                            for choice in part.get("choices", []):
                                text = choice.get("delta", {}).get("content")
                                if text:
                                    yield event({"type": "token", "text": text})
                        yield event({"type": "error", "message": "The connection ended early. Please try again."})
            except (httpx.HTTPError, ValueError):
                yield event({"type": "error", "message": "The model connection was interrupted. Please try again."})

        return StreamingResponse(events(), media_type="application/x-ndjson")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
