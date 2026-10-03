"""Web demo: a page where anyone can type a prompt and watch the Tesla GPT write (Lesson 18).

Run locally:  uv run uvicorn tinygpt.web:app --port 8000     then open http://localhost:8000
"""
import codecs
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from tinygpt.export import DEFAULT_MODEL
from tinygpt.train import load_model

MODEL_PATH = Path(os.environ.get("TINYGPT_MODEL", DEFAULT_MODEL))
PAGE = Path(__file__).parent / "static" / "index.html"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10"))   # requests per visitor per minute
if "TORCH_THREADS" in os.environ:                                  # 1 on the small Render server
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

model, tokenizer, checkpoint = load_model(MODEL_PATH)
app = FastAPI(title="TinyGPT Tesla", docs_url=None, redoc_url=None, openapi_url=None)


class OneAtATime:
    """The free server has a fraction of one CPU: write one answer at a time.
    A slot that was never released (e.g. a dropped connection) frees itself after `timeout` seconds."""

    def __init__(self, timeout=120):
        self.timeout = timeout
        self._lock = threading.Lock()
        self._since = None

    def try_acquire(self):
        with self._lock:
            if self._since is not None and time.monotonic() - self._since < self.timeout:
                return False
            self._since = time.monotonic()
            return True

    def release(self):
        with self._lock:
            self._since = None


writer = OneAtATime()
recent_requests = defaultdict(deque)   # visitor -> times of their recent requests


def visitor_id(request):
    forwarded = request.headers.get("x-forwarded-for")   # Render's proxy puts the real address here
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


def check_rate_limit(visitor):
    now = time.monotonic()
    if len(recent_requests) > 5000:                       # forget visitors from long ago
        for key in [k for k, q in recent_requests.items() if not q or now - q[-1] > 60]:
            del recent_requests[key]
    times = recent_requests[visitor]
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    times.append(now)


def token_bytes(token_id):
    if hasattr(tokenizer, "vocab"):                       # BPE: each token is a few bytes
        return tokenizer.vocab[token_id]
    return tokenizer.itos[token_id].encode("utf-8")       # character tokenizer


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field("", max_length=200)
    max_tokens: int = Field(120, ge=1, le=250)
    temperature: float = Field(0.5, ge=0.1, le=1.5)    # safer defaults read best for a model this small
    top_k: int | None = Field(10, ge=1, le=512)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; frame-ancestors 'none'"
    )
    return response


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/info")
def info():
    c = model.config
    return {
        "parameters": model.num_params(),
        "layers": c.n_layer, "heads": c.n_head, "embedding_size": c.n_embd,
        "context_tokens": c.block_size, "vocab_size": c.vocab_size,
        "training_steps": checkpoint["step"], "training_minutes": round(checkpoint["minutes"]),
        "validation_loss": round(checkpoint["best_val"], 3),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest, request: Request):
    check_rate_limit(visitor_id(request))
    try:
        ids = tokenizer.encode(req.prompt or "\n")
    except KeyError:
        raise HTTPException(400, "The prompt contains characters this model doesn't know.")
    if not writer.try_acquire():
        raise HTTPException(429, "The model is busy writing for someone else. Try again in a few seconds.")

    def stream():
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")   # BPE tokens can split a letter
            idx = torch.tensor([ids[-model.config.block_size:]])
            for token in model.generate_stream(idx, req.max_tokens, req.temperature, req.top_k):
                piece = decoder.decode(token_bytes(token.item()))
                if piece:
                    yield piece
            yield decoder.decode(b"", final=True)
        finally:
            writer.release()

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store"})
