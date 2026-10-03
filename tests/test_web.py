"""Checks for the web demo (Lesson 18). Run: uv run pytest"""
import pytest
from fastapi.testclient import TestClient

from tinygpt import web

client = TestClient(web.app)


@pytest.fixture(autouse=True)
def fresh_limits():
    web.recent_requests.clear()
    web.writer.release()
    yield
    web.recent_requests.clear()
    web.writer.release()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_page_is_served_with_security_headers():
    r = client.get("/")
    assert r.status_code == 200 and "TinyGPT" in r.text
    assert r.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_info_describes_the_model():
    data = client.get("/api/info").json()
    assert data["parameters"] == web.model.num_params()
    assert data["vocab_size"] == web.model.config.vocab_size


def test_generate_streams_text():
    r = client.post("/api/generate", json={"prompt": "The coil ", "max_tokens": 10})
    assert r.status_code == 200
    assert len(r.text) > 0
    assert web.writer.try_acquire(), "the writing slot must be free again afterwards"


def test_generate_handles_any_language():
    r = client.post("/api/generate", json={"prompt": "Никола Тесла, ćao ⚡", "max_tokens": 3})
    assert r.status_code == 200


@pytest.mark.parametrize("body", [
    {"prompt": "x" * 201},                     # prompt too long
    {"prompt": "hi", "max_tokens": 10_000},    # asks for too much
    {"prompt": "hi", "temperature": 5},        # out of range
    {"prompt": "hi", "unexpected": True},      # unknown field
])
def test_generate_rejects_bad_requests(body):
    assert client.post("/api/generate", json=body).status_code == 422


def test_busy_server_says_try_again():
    assert web.writer.try_acquire()            # someone else is writing
    r = client.post("/api/generate", json={"prompt": "The", "max_tokens": 2})
    assert r.status_code == 429 and "busy" in r.json()["detail"]


def test_rate_limit_per_visitor():
    for _ in range(web.RATE_LIMIT):
        assert client.post("/api/generate", json={"prompt": "a", "max_tokens": 1}).status_code == 200
    r = client.post("/api/generate", json={"prompt": "a", "max_tokens": 1})
    assert r.status_code == 429 and "Too many" in r.json()["detail"]


def test_stuck_writing_slot_frees_itself():
    slot = web.OneAtATime(timeout=0)
    assert slot.try_acquire()
    assert slot.try_acquire(), "a slot older than the timeout counts as free"
