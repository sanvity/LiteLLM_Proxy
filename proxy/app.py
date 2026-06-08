import os
import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ProxyConfig
from .router import LiteLLMProxyRouter

# Internal Streamlit URL — never exposed publicly. FastAPI proxies to it.
# Override via STREAMLIT_INTERNAL_URL env var in cloud environments.
STREAMLIT_INTERNAL_URL = os.environ.get("STREAMLIT_INTERNAL_URL", "http://127.0.0.1:8501")

# Headers that must NOT be forwarded (hop-by-hop per RFC 7230)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
    "proxy-connection", "content-length",
})

LOADING_HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>LiteLLM Gateway — Starting</title>"
    "<style>*{margin:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "background:#0e1117;color:#fafafa;display:flex;align-items:center;justify-content:center;height:100vh}"
    ".wrap{text-align:center}.logo{font-size:3rem;margin-bottom:1rem}.title{font-size:1.5rem;font-weight:600;"
    "color:#ff4b4b;margin-bottom:.5rem}.sub{color:#8b9dc3;font-size:.95rem;margin-bottom:2rem}"
    ".spinner{width:44px;height:44px;border:4px solid #2a2d3e;border-top-color:#ff4b4b;"
    "border-radius:50%;animation:spin .9s linear infinite;margin:0 auto}"
    "@keyframes spin{to{transform:rotate(360deg)}}</style></head>"
    "<body><div class='wrap'><div class='logo'>🛡️</div>"
    "<div class='title'>LiteLLM Gateway Console</div>"
    "<div class='sub'>Dashboard is warming up — auto-refreshing in 3s…</div>"
    "<div class='spinner'></div></div>"
    "<script>setTimeout(()=>location.reload(),3000)</script></body></html>"
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model ID to use for this request (e.g., primary-cluster)")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the chat history so far")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature to use")
    max_tokens: Optional[int] = Field(default=1000, description="The maximum number of tokens to generate")
    stream: Optional[bool] = Field(default=False, description="If true, stream tokens")
    mock_sandbox: Optional[bool] = Field(default=False, description="Force request to run in mock sandbox mode")


class UIPiiConfig(BaseModel):
    pii_enabled: bool = Field(default=False)
    pii_action: str = Field(default="MASK", description="BLOCK, MASK, or REWRITE")
    pii_policy: Optional[Dict[str, str]] = Field(default=None)


class LiteLLMProxyApp:
    """
    Class-based FastAPI Application container for the LiteLLM Proxy microservice.
    Serves the Streamlit UI transparently through a single HTTP port via reverse proxy,
    enabling single-URL cloud deployment on Railway, Render, Fly.io, etc.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ProxyConfig(config_path=config_path)
        self.router = LiteLLMProxyRouter(config=self.config)

        self.app = FastAPI(
            title="LiteLLM Load-Balancing Routing Proxy",
            description="OOP-based Microservice for intelligent LLM routing, fallbacks, TPM/RPM limits, and PII shielding.",
            version="2.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Persistent async HTTP client — shared for all proxy requests
        self._http_client = httpx.AsyncClient(
            base_url=STREAMLIT_INTERNAL_URL,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
        )

        self._register_routes()

    # ------------------------------------------------------------------
    # STREAMLIT REVERSE PROXY HELPERS
    # ------------------------------------------------------------------

    async def _proxy_http(self, request: Request, path: str) -> Response:
        """
        Transparently forwards an HTTP request to the internal Streamlit server.
        Preserves query string, strips hop-by-hop headers, and injects
        X-Forwarded-* headers so Streamlit knows its real public host.
        """
        qs = request.url.query
        upstream_path = path + (f"?{qs}" if qs else "")

        # Forward real client IP and protocol (needed when behind Railway TLS proxy)
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        fwd_headers["x-forwarded-host"] = request.headers.get("host", "")
        fwd_headers["x-forwarded-proto"] = request.headers.get("x-forwarded-proto", "https")

        body = await request.body()
        try:
            upstream_resp = await self._http_client.request(
                method=request.method,
                url=upstream_path,
                headers=fwd_headers,
                content=body,
            )
            resp_headers = {
                k: v for k, v in upstream_resp.headers.items()
                if k.lower() not in _HOP_BY_HOP
            }
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type=upstream_resp.headers.get("content-type"),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return HTMLResponse(content=LOADING_HTML, status_code=503)

    async def _proxy_ws(self, websocket: WebSocket, path: str):
        """
        Bidirectional WebSocket proxy to Streamlit's internal WS endpoint.
        Handles both binary and text frames (Streamlit uses both).
        """
        await websocket.accept()

        ws_base = STREAMLIT_INTERNAL_URL.replace("http://", "ws://").replace("https://", "wss://")
        # Preserve any query params from the original WS request
        qs = websocket.url.query
        ws_url = f"{ws_base}{path}" + (f"?{qs}" if qs else "")

        # Pass subprotocols from client to upstream
        subprotocols = list(websocket.headers.get("sec-websocket-protocol", "").split(", "))
        subprotocols = [s for s in subprotocols if s]

        try:
            import websockets as ws_lib
            extra_headers = {
                "X-Forwarded-Host": websocket.headers.get("host", ""),
                "X-Forwarded-Proto": "https",
            }
            conn_kwargs = dict(extra_headers=extra_headers)
            if subprotocols:
                conn_kwargs["subprotocols"] = subprotocols

            async with ws_lib.connect(ws_url, **conn_kwargs) as upstream:

                async def client_to_upstream():
                    try:
                        while True:
                            msg = await websocket.receive()
                            if "bytes" in msg and msg["bytes"] is not None:
                                await upstream.send(msg["bytes"])
                            elif "text" in msg and msg["text"] is not None:
                                await upstream.send(msg["text"])
                    except (WebSocketDisconnect, Exception):
                        pass

                async def upstream_to_client():
                    try:
                        async for message in upstream:
                            if isinstance(message, bytes):
                                await websocket.send_bytes(message)
                            else:
                                await websocket.send_text(message)
                    except Exception:
                        pass

                await asyncio.gather(client_to_upstream(), upstream_to_client())

        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # ROUTE REGISTRATION
    # ------------------------------------------------------------------

    def _register_routes(self):
        """Registers all FastAPI API endpoints and the Streamlit reverse proxy routes."""

        # -----------------------------------------------------------
        # STREAMLIT REVERSE PROXY
        # Catches all Streamlit paths and proxies them to port 8501.
        # Specific API routes below take precedence over the catch-all.
        # -----------------------------------------------------------

        @self.app.get("/")
        @self.app.head("/")
        async def proxy_root(request: Request):
            """Serves the Streamlit dashboard (reverse-proxied from internal port 8501)."""
            return await self._proxy_http(request, "/")

        # Streamlit's primary WebSocket for script runner + component updates
        @self.app.websocket("/_stcore/stream")
        async def proxy_ws_stream(websocket: WebSocket):
            await self._proxy_ws(websocket, "/_stcore/stream")

        # All other _stcore paths (health, script_run_id, upload_file, etc.)
        @self.app.api_route(
            "/_stcore/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]
        )
        async def proxy_stcore(request: Request, path: str):
            return await self._proxy_http(request, f"/_stcore/{path}")

        # Streamlit static assets (JS, CSS, fonts, images)
        @self.app.api_route(
            "/static/{path:path}",
            methods=["GET", "HEAD"]
        )
        async def proxy_static(request: Request, path: str):
            return await self._proxy_http(request, f"/static/{path}")

        # Streamlit component assets
        @self.app.api_route(
            "/component/{path:path}",
            methods=["GET", "HEAD"]
        )
        async def proxy_component(request: Request, path: str):
            return await self._proxy_http(request, f"/component/{path}")

        # Streamlit media/file upload endpoint
        @self.app.api_route(
            "/media/{path:path}",
            methods=["GET", "HEAD", "POST"]
        )
        async def proxy_media(request: Request, path: str):
            return await self._proxy_http(request, f"/media/{path}")

        # Streamlit /stream endpoint (used in some versions for SSE)
        @self.app.api_route("/stream", methods=["GET", "POST"])
        async def proxy_stream(request: Request):
            return await self._proxy_http(request, "/stream")

        # Legacy HTML dashboard
        @self.app.get("/old-ui", response_class=HTMLResponse)
        async def serve_old_ui():
            """Serves the legacy HTML dashboard template."""
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                template_path = os.path.join(current_dir, "templates", "index.html")
                with open(template_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content, status_code=200, headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                })
            except Exception as e:
                return HTMLResponse(
                    content=f"<html><body><h1>LiteLLM Proxy Online</h1><p>UI error: {e}</p></body></html>",
                    status_code=500
                )

        # -----------------------------------------------------------
        # INFRASTRUCTURE ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/health")
        async def health():
            """Health check for Railway / Kubernetes / Fly.io."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "routing_strategy": self.config.routing_strategy,
                "endpoints_loaded": len(self.config.endpoints)
            }

        @self.app.get("/metrics")
        async def metrics():
            """Real-time load-balancing and token metrics."""
            return {
                "timestamp": time.time(),
                "metrics": self.router.get_metrics(),
                "active_routing_rules": {
                    "strategy": self.config.routing_strategy,
                    "retries": self.config.num_retries,
                    "timeout": self.config.timeout
                },
                "registered_virtual_models": list(set(e.model_name for e in self.config.endpoints)),
                "registered_physical_providers": list(set(e.model.split("/")[0] for e in self.config.endpoints))
            }

        # -----------------------------------------------------------
        # LLM API ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/v1/models")
        async def list_models():
            """Lists virtual and physical backend models."""
            data = []
            for vm in set(e.model_name for e in self.config.endpoints):
                data.append({"id": vm, "object": "model", "created": 1686935002, "owned_by": "proxy-system", "type": "virtual"})
            for ep in self.config.endpoints:
                data.append({
                    "id": ep.model, "object": "model", "created": 1686935002,
                    "owned_by": ep.model.split("/")[0], "type": "physical",
                    "tpm_limit": ep.tpm, "rpm_limit": ep.rpm, "tpr_limit": ep.tpr
                })
            return {"object": "list", "data": data}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completion with load balancing and PII shielding."""
            messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
            virtual_models = set(e.model_name for e in self.config.endpoints)
            if request.model not in virtual_models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model}' not found. Available: {list(virtual_models)}"
                )
            try:
                return await self.router.execute_chat_completion(
                    model=request.model,
                    messages=messages_dict,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    mock_sandbox=request.mock_sandbox
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # -----------------------------------------------------------
        # PREFERENCE / CREDIT ROUTING ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/preference-config")
        async def get_preference_config():
            return {
                "preference_enabled": self.router.preference_enabled,
                "preference_list": self.router.preference_list,
                "credit_limits": self.router.credit_limits,
                "accumulated_spend": self.router.accumulated_spend,
                "available_physical_models": list(set(e.model for e in self.config.endpoints))
            }

        @self.app.post("/preference-config")
        async def update_preference_config(config: dict = Body(...)):
            self.router.preference_enabled = config.get("preference_enabled", self.router.preference_enabled)
            self.router.preference_list = config.get("preference_list", self.router.preference_list)
            for m, limit in config.get("credit_limits", {}).items():
                self.router.credit_limits[m] = float(limit)
            return {"status": "success", "message": "Preference routing configuration synchronized."}

        @self.app.post("/preference-config/reset")
        async def reset_preference_spend():
            for m in list(self.router.accumulated_spend.keys()):
                self.router.accumulated_spend[m] = 0.0
            return {"status": "success", "message": "Spend counters reset to zero."}

        # -----------------------------------------------------------
        # PII GUARDRAIL CONFIGURATION ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/ui/pii-config")
        async def get_pii_config():
            return {
                "pii_enabled": getattr(self.router, "pii_enabled", False),
                "pii_action": getattr(self.router, "pii_action", "MASK"),
                "pii_policy": getattr(self.router, "pii_policy", None)
            }

        @self.app.post("/ui/pii-config")
        async def update_pii_config(config: UIPiiConfig):
            self.router.pii_enabled = config.pii_enabled
            self.router.pii_action = config.pii_action
            self.router.pii_policy = config.pii_policy
            return {
                "status": "success",
                "message": f"PII Guardrail updated: enabled={config.pii_enabled}, action={config.pii_action}."
            }

    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance for use with ASGI servers."""
        return self.app