from io import BytesIO
import json

import pytest

from serving import chat, smoke_test


def write_env(root):
    (root / ".env").write_text('''
      # Shared with Modal's dotenv loader.
      export MODEL_ENDPOINT = "https://model.example" # endpoint
      MODAL_KEY = 'test-key'
      MODAL_SECRET = "test-secret"
    ''')


def test_chat_parses_shared_dotenv_syntax(tmp_path, monkeypatch, capsys):
    write_env(tmp_path)
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        return BytesIO(b'data: {"choices":[{"delta":{"content":"Hello!"}}]}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(chat, "urlopen", urlopen)
    assert chat.respond([{"role": "user", "content": "Hi"}], "qwen-medical") == "Hello!"
    assert requests[0].full_url == "https://model.example/v1/chat/completions"
    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert headers["modal-key"] == "test-key"
    assert headers["modal-secret"] == "test-secret"
    assert "total" in capsys.readouterr().out


def test_chat_routes_api_v2_to_its_own_endpoint(tmp_path, monkeypatch, capsys):
    write_env(tmp_path)
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="MODEL_ENDPOINT_V2"):
        chat.respond([{"role": "user", "content": "Hi"}], "qwen-medical", "v2")
    with (tmp_path / ".env").open("a") as env:
        env.write("MODEL_ENDPOINT_V2=https://model-v2.example/\n")
    requests = []
    monkeypatch.setattr(chat, "urlopen", lambda request, **kw: requests.append(request) or BytesIO(
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'))
    assert chat.respond([{"role": "user", "content": "Hi"}], "qwen-medical", "v2") == "Hi"
    assert requests[0].full_url == "https://model-v2.example/v1/chat/completions"


@pytest.mark.parametrize("payload", [b'', b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'])
def test_chat_rejects_stream_without_completion_marker(tmp_path, monkeypatch, capsys, payload):
    write_env(tmp_path)
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat, "urlopen", lambda *a, **kw: BytesIO(payload))
    with pytest.raises(RuntimeError, match="ended before completion"):
        chat.respond([{"role": "user", "content": "Hi"}], "qwen-medical")
    assert "total" not in capsys.readouterr().out


def test_one_shot_reports_interrupted_stream_as_failure(monkeypatch):
    monkeypatch.setattr("sys.argv", ["chat.py", "Hi"])
    monkeypatch.setattr(chat, "respond", lambda *a: (_ for _ in ()).throw(RuntimeError("Stream ended before completion")))
    with pytest.raises(SystemExit) as error:
        chat.main()
    assert error.value.code == 1


def test_interactive_retry_does_not_keep_partial_turn(tmp_path, monkeypatch):
    write_env(tmp_path)
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["chat.py"])
    inputs = iter(["first question", "retry question", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    calls = []

    def urlopen(request, **kwargs):
        calls.append(json.loads(request.data))
        payload = b'data: {"choices":[{"delta":{"content":"answer"}}]}\n'
        if len(calls) == 2:
            payload += b'data: [DONE]\n'
        return BytesIO(payload)

    monkeypatch.setattr(chat, "urlopen", urlopen)
    chat.main()
    assert len(calls) == 2
    assert calls[1]["messages"] == [{"role": "user", "content": "retry question"}]


def test_smoke_test_uses_same_dotenv_parser(tmp_path, monkeypatch):
    write_env(tmp_path)
    (tmp_path / "serving").mkdir()
    monkeypatch.setattr(smoke_test, "ROOT", tmp_path)
    class Remote:
        def get_web_url(self):
            return "https://model.example"
    monkeypatch.setattr(smoke_test.modal.Function, "from_name", lambda *a, **kw: Remote())
    requests = []

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            from urllib.error import HTTPError
            raise HTTPError(request, 401, "Unauthorized", {}, None)
        requests.append(request)
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["modal-key"] == "test-key"
        assert headers["modal-secret"] == "test-secret"
        result = ({"data": [{"id": "qwen-medical"}]} if request.data is None else
                  {"model": "qwen-medical", "choices": [{"message": {"content": "Hello!"}}]})
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(smoke_test, "urlopen", urlopen)
    smoke_test.main()
    assert len(requests) == 2
    assert (tmp_path / "serving/deployment.json").exists()
