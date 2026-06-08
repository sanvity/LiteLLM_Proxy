import os
import time
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ProxyConfig
from .router import LiteLLMProxyRouter

# Internal Streamlit URL — never exposed publicly. FastAPI proxies to it.
# Override via STREAMLIT_INTERNAL_URL env var in cloud environments.
STREAMLIT_INTERNAL_URL = os.environ.get("STREAMLIT_INTERNAL_URL", "http://127.0.0.1:8501")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model ID to use for this request (e.g., oss-chat-fast)")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the chat history so far")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature to use")
    max_tokens: Optional[int] = Field(default=1000, description="The maximum number of tokens to generate")
    stream: Optional[bool] = Field(default=False, description="If true, stream tokens (Note: non-streaming fully optimized in this proxy version)")
    mock_sandbox: Optional[bool] = Field(default=False, description="Force request to run in mock sandbox mode for test verification")


class UIPiiConfig(BaseModel):
    pii_enabled: bool = Field(default=False)
    pii_action: str = Field(default="MASK", description="BLOCK, MASK, or REWRITE")
    pii_policy: Optional[Dict[str, str]] = Field(default=None)


class LiteLLMProxyApp:
    """
    Class-based FastAPI Application container for the LiteLLM Proxy microservice.
    Serves the Streamlit UI transparently through a single HTTP port via reverse proxy,
    enabling cloud deployment where only one port is exposed.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Initialize configuration and the routing engine
        self.config = ProxyConfig(config_path=config_path)
        self.router = LiteLLMProxyRouter(config=self.config)

        # Initialize FastAPI app
        self.app = FastAPI(
            title="LiteLLM Load-Balancing Routing Proxy",
            description="OOP-based Microservice for intelligent LLM routing, fallbacks, TPM/RPM limits, and PII shielding.",
            version="2.0.0"
        )

        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Async HTTP client for proxying Streamlit requests (single shared instance)
        self._http_client = httpx.AsyncClient(
            base_url=STREAMLIT_INTERNAL_URL,
            follow_redirects=True,
            timeout=30.0
        )

        # Register all routes
        self._register_routes()

    # ------------------------------------------------------------------
    # STREAMLIT HTTP REVERSE PROXY HELPER
    # ------------------------------------------------------------------

    async def _proxy_http(self, request: Request, path: str) -> Response:
        """
        Transparently forwards an HTTP request to the internal Streamlit server.
        Strips hop-by-hop headers and returns the upstream response verbatim.
        Returns a friendly 503 with auto-refresh if Streamlit is still starting.
        """
        # Preserve query string
        qs = request.url.query
        upstream_url = path + (f"?{qs}" if qs else "")

        # Headers that must NOT be forwarded to the upstream
        HOP_BY_HOP = {
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "transfer-encoding", "upgrade", "host"
        }
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP
        }

        body = await request.body()
        try:
            upstream_resp = await self._http_client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body
            )
            # Strip hop-by-hop from response headers
            resp_headers = {
                k: v for k, v in upstream_resp.headers.items()
                if k.lower() not in HOP_BY_HOP
            }
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type=upstream_resp.headers.get("content-type")
            )
        except httpx.ConnectError:
            # Streamlit hasn't fully started yet — return a friendly loading page
            return HTMLResponse(
                content=(
                    "<html><head><title>Starting...</title></head><body style='font-family:sans-serif;"
                    "display:flex;align-items:center;justify-content:center;height:100vh;margin:0;"
                    "background:#f0f2f6'>"
                    "<div style='text-align:center'>"
                    "<h2 style='color:#262730'>🛡️ LiteLLM Gateway Console</h2>"
                    "<p style='color:#6c757d'>Dashboard is warming up, please wait...</p>"
                    "<div style='width:40px;height:40px;border:4px solid #e0e0e0;"
                    "border-top-color:#ff4b4b;border-radius:50%;animation:spin 1s linear infinite;"
                    "margin:20px auto'></div>"
                    "<style>@keyframes spin{to{transform:rotate(360deg)}}</style>"
                    "<script>setTimeout(()=>location.reload(),3000)</script>"
                    "</div></body></html>"
                ),
                status_code=503
            )

    # ------------------------------------------------------------------
    # ROUTE REGISTRATION
    # ------------------------------------------------------------------

    def _register_routes(self):
        """Registers all FastAPI endpoints to the underlying app instance."""

        # -----------------------------------------------------------
        # STREAMLIT REVERSE PROXY (HTTP + WebSocket)
        # Routes / and Streamlit internals to the background process.
        # This makes the entire app work on a single public port.
        # -----------------------------------------------------------

        @self.app.get("/")
        @self.app.head("/")
        async def proxy_streamlit_root(request: Request):
            """Transparently proxies the Streamlit UI root."""
            return await self._proxy_http(request, "/")

        @self.app.websocket("/_stcore/stream")
        async def proxy_streamlit_ws(websocket: WebSocket):
            """Proxies Streamlit WebSocket connections for live component updates."""
            await websocket.accept()
            import websockets as ws_lib
            ws_url = STREAMLIT_INTERNAL_URL.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = f"{ws_url}/_stcore/stream"
            try:
                async with ws_lib.connect(ws_url) as upstream:
                    import asyncio

                    async def client_to_upstream():
                        try:
                            while True:
                                data = await websocket.receive_bytes()
                                await upstream.send(data)
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

        @self.app.api_route("/_stcore/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
        async def proxy_stcore(request: Request, path: str):
            """Proxies Streamlit internal _stcore paths (health check, script runner, file uploads)."""
            return await self._proxy_http(request, f"/_stcore/{path}")

        @self.app.api_route("/stream", methods=["GET", "POST"])
        async def proxy_streamlit_stream(request: Request):
            """Proxies Streamlit's /stream endpoint."""
            return await self._proxy_http(request, "/stream")

        # Legacy HTML dashboard (for debugging)
        @self.app.get("/old-ui", response_class=HTMLResponse)
        async def serve_old_ui():
            """Serves the legacy HTML dashboard template."""
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                template_path = os.path.join(current_dir, "templates", "index.html")
                with open(template_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                headers = {
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
                return HTMLResponse(content=html_content, status_code=200, headers=headers)
            except Exception as e:
                return HTMLResponse(
                    content=f"<html><body><h1>LiteLLM Proxy API Online</h1><p>UI loading error: {e}</p></body></html>",
                    status_code=500
                )

        # -----------------------------------------------------------
        # INFRASTRUCTURE ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/health")
        async def health():
            """Health check endpoint for container orchestrators (Kubernetes, Railway, etc.)."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "routing_strategy": self.config.routing_strategy,
                "endpoints_loaded": len(self.config.endpoints)
            }

        @self.app.get("/metrics")
        async def metrics():
            """Returns real-time load-balancing, token rates, and fallback statistics."""
            router_metrics = self.router.get_metrics()
            return {
                "timestamp": time.time(),
                "metrics": router_metrics,
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
            """Lists both logical virtual models and physical backend models."""
            data = []

            # Virtual Models (Logical endpoints exposed to microservices)
            virtual_models = list(set(e.model_name for e in self.config.endpoints))
            for vm in virtual_models:
                data.append({
                    "id": vm,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "proxy-system",
                    "type": "virtual"
                })

            # Physical Models (Concrete implementations supporting them)
            for ep in self.config.endpoints:
                data.append({
                    "id": ep.model,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": ep.model.split("/")[0],
                    "type": "physical",
                    "tpm_limit": ep.tpm,
                    "rpm_limit": ep.rpm,
                    "tpr_limit": ep.tpr
                })

            return {"object": "list", "data": data}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """
            OpenAI-compatible chat completion endpoint.
            Estimates prompt tokens, validates against TPR, and routes with fallbacks.
            """
            messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]

            virtual_models = set(e.model_name for e in self.config.endpoints)
            if request.model not in virtual_models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Requested model '{request.model}' not found in proxy configuration. "
                           f"Available virtual models: {list(virtual_models)}"
                )

            try:
                response = await self.router.execute_chat_completion(
                    model=request.model,
                    messages=messages_dict,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    mock_sandbox=request.mock_sandbox
                )
                return response
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # -----------------------------------------------------------
        # PREFERENCE / CREDIT ROUTING ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/preference-config")
        async def get_preference_config():
            """Returns current user priority preferences, credit limits, and accumulated spending."""
            return {
                "preference_enabled": self.router.preference_enabled,
                "preference_list": self.router.preference_list,
                "credit_limits": self.router.credit_limits,
                "accumulated_spend": self.router.accumulated_spend,
                "available_physical_models": list(set(e.model for e in self.config.endpoints))
            }

        @self.app.post("/preference-config")
        async def update_preference_config(config: dict = Body(...)):
            """Updates the priority preference list, credit limits, and enablement flag."""
            self.router.preference_enabled = config.get("preference_enabled", self.router.preference_enabled)
            self.router.preference_list = config.get("preference_list", self.router.preference_list)
            limits = config.get("credit_limits", {})
            for m, limit in limits.items():
                self.router.credit_limits[m] = float(limit)
            return {"status": "success", "message": "Preference routing configuration synchronized in memory."}

        @self.app.post("/preference-config/reset")
        async def reset_preference_spend():
            """Resets all accumulated spend counters to $0.00."""
            for m in list(self.router.accumulated_spend.keys()):
                self.router.accumulated_spend[m] = 0.0
            return {"status": "success", "message": "All accumulated model credit spending counters reset to zero."}

        # -----------------------------------------------------------
        # PII GUARDRAIL CONFIGURATION ENDPOINTS
        # -----------------------------------------------------------

        @self.app.get("/ui/pii-config")
        async def get_pii_config():
            """Returns the current PII guardrail configuration."""
            return {
                "pii_enabled": getattr(self.router, "pii_enabled", False),
                "pii_action": getattr(self.router, "pii_action", "MASK"),
                "pii_policy": getattr(self.router, "pii_policy", None)
            }

        @self.app.post("/ui/pii-config")
        async def update_pii_config(config: UIPiiConfig):
            """Updates the PII guardrail configuration."""
            self.router.pii_enabled = config.pii_enabled
            self.router.pii_action = config.pii_action
            self.router.pii_policy = config.pii_policy
            return {
                "status": "success",
                "message": f"PII Guardrail configuration synchronized in memory: "
                           f"enabled={config.pii_enabled}, action={config.pii_action}."
            }

    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance (useful for running with ASGI servers)."""
        return self.app