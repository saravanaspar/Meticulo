"""
Meticulo API Server - OpenAI-compatible API that proxies to meticulo-server.

This is a thin proxy that forwards requests to the running meticulo-server,
adding NO overhead, NO rate limiting, NO token capping.

Why a proxy instead of directly exposing meticulo-server?
- Adds model management (hot-swap models)
- Adds model listing endpoint
- Provides /api/generate (Ollama-compatible) alongside OpenAI format
- Zero processing overhead on the inference path
"""

import json
import re
import signal
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from .config import DEFAULTS
from .model_manager import get_model_path, list_models, pull_model
from .runner import get_runner


def _extract_thinking_response(text: str) -> dict:
    """Extract <thinking> and <response> tags from model output."""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    response_match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
    
    result = {}
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()
    else:
        result["thinking"] = ""
    
    if response_match:
        # Try to parse as JSON
        try:
            result["response"] = json.loads(response_match.group(1).strip())
        except json.JSONDecodeError:
            result["response"] = response_match.group(1).strip()
    else:
        # No tags - treat whole as response
        try:
            result["response"] = json.loads(text.strip())
        except json.JSONDecodeError:
            result["response"] = text.strip()
    
    return result


class MeticuloHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies to meticulo-server with zero overhead."""

    # Reference to the shared state
    server_instance = None

    def log_message(self, format, *args):
        """Suppress default logging for clean output."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": {"message": message, "type": "error"}}, status)

    def _proxy_to_engine(self, path: str, body: Optional[bytes] = None, method: str = "POST"):
        """Forward request directly to meticulo-server - zero processing overhead."""
        runner = get_runner()
        if not runner.is_running():
            self._send_error("No model loaded. Use POST /api/load to load a model.", 503)
            return

        target_url = f"http://{runner.host}:{runner.port}{path}"

        try:
            req = urllib.request.Request(target_url, data=body, method=method)
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req) as resp:
                # Check if streaming
                content_type = resp.headers.get("Content-Type", "")

                if "text/event-stream" in content_type:
                    # Stream directly through
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    # Forward response as-is
                    response_data = resp.read()
                    self.send_response(resp.status)
                    for key in ("Content-Type", "Content-Length"):
                        val = resp.headers.get(key)
                        if val:
                            self.send_header(key, val)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(response_data)

        except urllib.error.URLError as e:
            self._send_error(f"meticulo-server error: {e}", 502)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "" or path == "/":
            self._send_json({
                "name": "Meticulo",
                "version": "0.1.0",
                "description": "Zero-overhead local LLM runner",
            })

        elif path == "/health" or path == "/api/health":
            runner = get_runner()
            status = "running" if runner.is_running() else "no model loaded"
            self._send_json({
                "status": status,
                "model": str(runner.model_path) if runner.model_path else None,
            })

        elif path == "/v1/models" or path == "/api/tags":
            models = list_models()
            # OpenAI format
            model_list = []
            for m in models:
                model_list.append({
                    "id": m["name"],
                    "object": "model",
                    "owned_by": "local",
                    "size": m.get("size", 0),
                    "size_human": m.get("size_human", ""),
                })
            self._send_json({"object": "list", "data": model_list})

        elif path == "/api/ps":
            runner = get_runner()
            if runner.is_running():
                self._send_json({
                    "models": [{
                        "name": runner.model_path.name if runner.model_path else "",
                        "model": str(runner.model_path) if runner.model_path else "",
                        "size": runner.model_path.stat().st_size if runner.model_path and runner.model_path.exists() else 0,
                    }]
                })
            else:
                self._send_json({"models": []})

        elif path == "/api/catalog":
            from .config import get_model_registry
            registry = get_model_registry()
            # Filter out comment/separator keys
            names = sorted([
                k for k in registry
                if not k.startswith("_") and not k.startswith("__")
            ])
            self._send_json({"models": names})

        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        body = self._read_body()

        # === OpenAI-compatible endpoints (proxy to meticulo-server) ===
        if path == "/v1/chat/completions":
            if body:
                data = json.loads(body)
                body = self._handle_structured_request(data)
                if body is None:
                    return  # Error already sent
            self._proxy_to_engine("/v1/chat/completions", body)

        elif path == "/v1/completions":
            self._proxy_to_engine("/completion", body)

        elif path == "/v1/embeddings":
            self._proxy_to_engine("/embedding", body)

        # === Ollama-compatible endpoints ===

        elif path == "/api/generate":
            self._proxy_to_engine("/completion", body)

        elif path == "/api/chat":
            self._proxy_to_engine("/v1/chat/completions", body)

        # === Meticulo management endpoints ===
        elif path == "/api/load":
            self._handle_load(body)

        elif path == "/api/unload":
            self._handle_unload()

        elif path == "/api/pull":
            self._handle_pull(body)

        else:
            self._send_error("Not found", 404)

    def do_DELETE(self):
        path = self.path.rstrip("/")
        if path == "/api/delete":
            body = self._read_body()
            data = json.loads(body) if body else {}
            name = data.get("name", "")
            if not name:
                self._send_error("Missing 'name' field")
                return
            from .model_manager import delete_model
            if delete_model(name):
                self._send_json({"status": "deleted"})
            else:
                self._send_error("Model not found", 404)
        else:
            self._send_error("Not found", 404)

    def _handle_load(self, body: bytes):
        """Load/swap a model."""
        data = json.loads(body) if body else {}
        model_name = data.get("model", "")
        if not model_name:
            self._send_error("Missing 'model' field")
            return

        model_path = get_model_path(model_name)
        if not model_path:
            self._send_error(f"Model '{model_name}' not found. Pull it first.", 404)
            return

        runner = get_runner()

        # Stop current model if running
        if runner.is_running():
            runner.stop()
            time.sleep(1)

        # Start with full performance settings
        server_kwargs = {}

    def _handle_structured_request(self, data: dict) -> bytes:
        """Handle structured output and thinking mode."""
        
        # Handle thinking mode
        thinking = data.get("thinking", False)
        if thinking:
            # Inject thinking instruction
            messages = data.get("messages", [])
            if messages:
                last = messages[-1]
                if last.get("role") == "user":
                    last["content"] = (
                        "Before responding, work through your reasoning step by step in <thinking> tags. "
                        "Then provide your final response in <response> tags as valid JSON.\n\n" 
                        + last["content"]
                    )
            data["thinking"] = True
        
        # Handle response_format
        response_format = data.get("response_format", {})
        if response_format:
            format_type = response_format.get("type")
            
            if format_type in ("json_schema", "json_object"):
                # Build JSON instruction
                if format_type == "json_schema":
                    json_schema = response_format.get("json_schema", {})
                    schema_json = json.dumps(json_schema) if json_schema else None
                else:
                    schema_json = None

                messages = data.get("messages", [])
                
                if schema_json:
                    instruction = f"Provide response in <response> tags as valid JSON matching:\n{schema_json}"
                else:
                    instruction = "Provide response in <response> tags as valid JSON."
                
                instruction += " No markdown, no explanation outside tags."
                
                if messages:
                    last = messages[-1]
                    if last.get("role") == "user":
                        last["content"] = instruction + "\n\n" + last["content"]
                    else:
                        messages.append({"role": "user", "content": instruction})
                else:
                    messages = [{"role": "user", "content": instruction}]
                
                data["messages"] = messages
                data.pop("response_format", None)
        
        return json.dumps(data).encode()
        for key in ("n_gpu_layers", "ctx_size", "n_batch", "n_ubatch",
                     "n_parallel", "flash_attn", "mlock", "threads"):
            if key in data:
                server_kwargs[key] = data[key]

        # Pass extra_args if provided
        if "extra_args" in data:
            server_kwargs["extra_args"] = data["extra_args"]

        success = runner.start_server(model_path, **server_kwargs)
        if success:
            self._send_json({"status": "loaded", "model": model_name})
        else:
            self._send_error("Failed to load model", 500)

    def _handle_unload(self):
        """Unload current model."""
        runner = get_runner()
        runner.stop()
        self._send_json({"status": "unloaded"})

    def _handle_pull(self, body: bytes):
        """Pull a model (blocking)."""
        data = json.loads(body) if body else {}
        model_name = data.get("model", "")
        if not model_name:
            self._send_error("Missing 'model' field")
            return

        try:
            path = pull_model(model_name)
            self._send_json({"status": "pulled", "model": model_name, "path": str(path)})
        except Exception as e:
            self._send_error(str(e), 500)


def start_api_server(
    host: str = "127.0.0.1",
    port: int = 22434,
    model: str = None,
    server_kwargs: Optional[dict] = None,
):
    """
    Start the Meticulo API server.

    Port 22434 by default (avoids conflict with Ollama on 11434).
    Set METICULO_PORT env var or --port flag to override.
    """
    # If a model is specified, load it immediately
    if model:
        model_path = get_model_path(model)
        if not model_path:
            print(f"Model '{model}' not found locally. Pulling...")
            model_path = pull_model(model)

        runner = get_runner()
        # Use a different port for meticulo-server backend
        backend_port = port + 1
        kwargs = dict(server_kwargs or {})
        success = runner.start_server(model_path, port=backend_port, **kwargs)
        if not success:
            print("Failed to start meticulo-server backend.")
            sys.exit(1)

    server = HTTPServer((host, port), MeticuloHandler)
    MeticuloHandler.server_instance = server

    def shutdown_handler(sig, frame):
        print("\nShutting down...")
        runner = get_runner()
        runner.stop()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"\n{'═' * 60}")
    print(f"  Meticulo API Server")
    print(f"  Listening on http://{host}:{port}")
    print(f"  OpenAI-compatible: http://{host}:{port}/v1/")
    print(f"  Ollama-compatible: http://{host}:{port}/api/")
    if model:
        print(f"  Model loaded: {model}")
    print(f"{'═' * 60}\n")

    server.serve_forever()


def start_embedding_server(
    host: str = "127.0.0.1",
    port: int = 8081,
    model: str = None,
    server_kwargs: Optional[dict] = None,
):
    """
    Start the Meticulo embedding server on a separate port.
    
    Default port 8081 (different from chat server at 22434).
    """
    from .model_manager import get_model_path, pull_model
    from .runner import get_runner
    
    if not model:
        print("Error: --model is required for embedding server")
        sys.exit(1)
    
    model_path = get_model_path(model)
    if not model_path:
        print(f"Model '{model}' not found locally. Pulling...")
        model_path = pull_model(model)

    runner = get_runner()
    # Embedding server uses llama-server with embedding mode
    kwargs = dict(server_kwargs or {})
    kwargs["embedding"] = True
    
    success = runner.start_server(model_path, port=port, **kwargs)
    if not success:
        print("Failed to start embedding server.")
        sys.exit(1)
    
    print(f"\n{'═' * 60}")
    print(f"  Meticulo Embedding Server")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Embeddings endpoint: http://{host}:{port}/v1/embeddings")
    print(f"  Model loaded: {model}")
    print(f"{'═' * 60}\n")
    
    # Keep running
    while True:
        time.sleep(1)
