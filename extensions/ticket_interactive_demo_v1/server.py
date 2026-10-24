"""Loopback-only demo HTTP transport, requiring a per-process browser token."""
from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import core


def create_server(service, *, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def respond(self, code, value, content_type="application/json; charset=utf-8"):
            encoded = (core.wire.canonical(value) if content_type.startswith("application/json") else value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; "
                             f"script-src 'nonce-{token}'; style-src 'nonce-{token}'; "
                             "frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(encoded)

        def valid_host(self):
            number = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{number}", f"localhost:{number}"}

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {"error": "host_not_allowed"})
            if self.path == "/":
                html = (core.HERE / "web.html").read_text(encoding="utf-8").replace("__DEMO_TOKEN__", token)
                return self.respond(200, html, "text/html; charset=utf-8")
            if self.path == "/api/health":
                return self.respond(200, {"provider": service.provider_kind, "runtime": "trusted_completion_v1.7",
                                          "backend": "in_memory", "rolled_back": service.rolled_back,
                                          "model_loaded": service.provider_kind == "local-qwen"})
            self.respond(404, {"error": "not_found"})

        def do_POST(self):
            number = self.server.server_address[1]
            allowed_origins = {f"http://127.0.0.1:{number}", f"http://localhost:{number}"}
            if (not self.valid_host() or self.headers.get("X-Demo-Token") != token or
                    self.headers.get("Origin") not in allowed_origins):
                self.close_connection = True
                return self.respond(403, {"error": "browser_token_or_origin_rejected"})
            try:
                self.connection.settimeout(20)
                core.require(self.headers.get("Content-Type", "").split(";")[0] == "application/json", "JSON required")
                core.require(not self.headers.get("Transfer-Encoding"), "Chunked bodies unsupported")
                size = int(self.headers.get("Content-Length", "0"))
                core.require(0 < size <= 16000, "Request body size invalid")
                raw = self.rfile.read(size)
                core.require(len(raw) == size, "Incomplete request")
                body = json.loads(raw.decode("utf-8"))
                core.require(isinstance(body, dict), "Body must be an object")
                if self.path == "/api/session":
                    result = service.new_session()
                elif self.path == "/api/state":
                    result = {"state": service.get_state(body.get("session_id"))}
                elif self.path == "/api/run":
                    result = service.handle(body.get("session_id"), body.get("request_key"), body.get("request"))
                elif self.path == "/api/rollback":
                    result = service.rollback()
                else:
                    return self.respond(404, {"error": "not_found"})
                self.respond(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)[:500]})
            except Exception as exc:
                self.respond(500, {"error": "服务异常：" + type(exc).__name__ + "；请检查本地日志"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
