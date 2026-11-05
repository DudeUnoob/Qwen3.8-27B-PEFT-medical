# Private chat webpage

Configure the deployed website URL in `PUBLIC_ORIGIN` in your local `.env`.

The website runs on CPU in Modal. Its backend forwards authenticated requests
to the existing GPU endpoint using server-side Modal proxy credentials. GitHub
OAuth verifies the account; only the numeric GitHub user ID configured in
`ALLOWED_GITHUB_USER_ID` can create a session. Changing a GitHub username does
not change this allowlist. The app rejects sign-in when that setting is absent.

## GitHub login setup

Copy `.env.example` to `.env`. Fill in `MODAL_KEY`, `MODAL_SECRET`,
`ALLOWED_GITHUB_USER_ID`, and the deployed inference URL as `MODEL_ENDPOINT`.
Set `PUBLIC_ORIGIN` to the website's HTTPS origin, with no trailing slash.
Generate a random session signing secret locally and set `SESSION_SECRET` to it.

1. Open https://github.com/settings/applications/new while signed in to your
   GitHub account.
2. Use application name `Qwen Medical Private`.
3. Set Homepage URL to
   `https://YOUR-WEB-ENDPOINT.modal.run`.
4. Set Authorization callback URL to
   `https://YOUR-WEB-ENDPOINT.modal.run/auth/callback`.
5. Register the application and generate a client secret. Add these two lines
   to the local project's ignored `.env`, preserving the existing values:

   ```dotenv
   GITHUB_CLIENT_ID=your-client-id
   GITHUB_CLIENT_SECRET=your-client-secret
   ```

6. Redeploy from the project directory:

   ```sh
   .venv/bin/python -m modal secret create qwen-medical-web-config --from-dotenv .env --force --env main
   .venv/bin/python -m modal deploy --env main webapp/modal_app.py
   ```

The `PUBLIC_ORIGIN` value must exactly match the webpage URL.
`SESSION_SECRET` must remain stable across restarts. It was generated locally
and is included in the named `qwen-medical-web-config` Modal secret; it is never sent to the browser.
Changing it logs out all browser sessions. Never commit `.env` or paste its
contents into chat. The secret command reads the local `.env` file and stores
its settings in a server-side Modal secret. The website source upload does
not include `.env`. After changing credentials, update the secret and redeploy.

Verification included static assets, rejected anonymous requests, the GitHub
redirect, and a real response through the streaming backend. The owner also
confirmed successful GitHub sign-in and a reply in the browser. Both apps were
then stopped; deploy them again to resume service.

## Access behavior

- The sign-in shell and static assets are public; model access is private.
- Every inference request requires a signed, unexpired owner session, a
  matching Origin, and a CSRF token. The signed session cookie uses Secure,
  HttpOnly, SameSite=Lax, and a `__Host-` cookie name. Sessions expire after
  eight hours. Other verified GitHub users receive 403.
- GitHub login requests public identity only. GitHub access tokens are used
  during callback and discarded, not placed in the session cookie.
- Modal credentials are attached only by the backend. The original GPU API
  remains protected by Modal proxy authentication.
- The header selector picks serving API v1 (`MODEL_ENDPOINT`) or v2
  (`MODEL_ENDPOINT_V2`); a version without a configured endpoint is disabled.
  Both stream tokens to the page as they are generated.
- The webpage streams responses from `qwen-medical`. Conversation history is
  kept in browser memory and is cleared by refresh, new chat, or sign out.
  Requests necessarily reach the inference service; infrastructure logging
  and billing are managed by Modal.
- The model has a 2048-token context limit. Start a new conversation if the
  model rejects a long conversation. Responses are capped at 512 tokens.

## Development and verification

```sh
.venv/bin/python -m pip install -e '.[web,test]'
.venv/bin/python -m pytest -q
```

Tests cover signed-out and unconfigured access, OAuth state mismatch,
non-owner identities, username changes, forged and expired sessions, CSRF,
cross-origin requests, logout, request size, and the streaming proxy with a
mock inference server. There is no production test-login route or bypass.
The local web extra and Modal image both explicitly require Pydantic v2,
matching the request validation APIs used by the application.

The website uses no GPU. The separate inference app retains its limit of
one GPU and scales to zero after idle time. To stop only the website:

```sh
.venv/bin/python -m modal app stop qwen-peft-medical-web --env main
```
