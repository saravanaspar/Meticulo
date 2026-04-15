"""Official Python SDK client for Meticulo APIs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .types import ChatMessage, ModelInfo


class MeticuloAPIError(RuntimeError):
    """Raised when the Meticulo API returns an error."""

    def __init__(self, message: str, status: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


class MeticuloClient:
    """Synchronous Python client for Meticulo's OpenAI and Ollama-compatible APIs."""

    def __init__(self, base_url: str = "http://127.0.0.1:22434", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---------- Core HTTP helpers ----------
    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ):
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, method=method, headers=headers)

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            if stream:
                return resp
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            msg = f"HTTP {e.code} for {path}"
            try:
                parsed = json.loads(body)
                err = parsed.get("error", {})
                if isinstance(err, dict) and err.get("message"):
                    msg = err["message"]
            except Exception:
                pass
            raise MeticuloAPIError(msg, status=e.code, body=body) from e

        except urllib.error.URLError as e:
            raise MeticuloAPIError(f"Connection failed to {url}: {e}") from e

    @staticmethod
    def _normalize_messages(messages: Iterable[Union[ChatMessage, Dict[str, str]]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                out.append(asdict(msg))
            elif isinstance(msg, dict):
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", ""))
                out.append({"role": role, "content": content})
            else:
                raise TypeError("messages must contain ChatMessage or dict items")
        return out

    # ---------- Health / model management ----------
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/health")

    def list_models(self) -> List[ModelInfo]:
        data = self._request("GET", "/v1/models")
        models = []
        for m in data.get("data", []):
            models.append(
                ModelInfo(
                    id=m.get("id", ""),
                    object=m.get("object", "model"),
                    owned_by=m.get("owned_by", "local"),
                    size=m.get("size", 0),
                    size_human=m.get("size_human", ""),
                )
            )
        return models

    def ps(self) -> Dict[str, Any]:
        return self._request("GET", "/api/ps")

    def pull_model(self, model: str) -> Dict[str, Any]:
        return self._request("POST", "/api/pull", {"model": model})

    def load_model(
        self,
        model: str,
        *,
        n_gpu_layers: Optional[int] = None,
        ctx_size: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_parallel: Optional[int] = None,
        flash_attn: Optional[bool] = None,
        mlock: Optional[bool] = None,
        threads: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model}
        if n_gpu_layers is not None:
            payload["n_gpu_layers"] = n_gpu_layers
        if ctx_size is not None:
            payload["ctx_size"] = ctx_size
        if n_batch is not None:
            payload["n_batch"] = n_batch
        if n_ubatch is not None:
            payload["n_ubatch"] = n_ubatch
        if n_parallel is not None:
            payload["n_parallel"] = n_parallel
        if flash_attn is not None:
            payload["flash_attn"] = flash_attn
        if mlock is not None:
            payload["mlock"] = mlock
        if threads is not None:
            payload["threads"] = threads
        if extra_args is not None:
            payload["extra_args"] = extra_args
        return self._request("POST", "/api/load", payload)

    def unload_model(self) -> Dict[str, Any]:
        return self._request("POST", "/api/unload", {})

    def delete_model(self, name: str) -> Dict[str, Any]:
        return self._request("DELETE", "/api/delete", {"name": name})

    # ---------- Inference ----------
    def chat(
        self,
        model: str,
        messages: Iterable[Union[ChatMessage, Dict[str, str]]],
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return self._request("POST", "/v1/chat/completions", payload)

    def chat_stream(
        self,
        model: str,
        messages: Iterable[Union[ChatMessage, Dict[str, str]]],
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = self._request("POST", "/v1/chat/completions", payload, stream=True)
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # SSE format: "data: {...}" or "data: [DONE]"
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Some backends may split events; skip malformed partial lines.
                    continue
        finally:
            resp.close()

    def completion(
        self,
        model: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return self._request("POST", "/v1/completions", payload)

    def embeddings(self, model: str, input_data: Union[str, List[str]]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": input_data,
        }
        return self._request("POST", "/v1/embeddings", payload)
