import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ProxyConfig
from .router import LiteLLMProxyRouter


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
    Exposes an OpenAI-compatible API with intelligent routing, PII shielding,
    and load balancing across multiple LLM providers.

    The Streamlit dashboard (app.py) runs as a separate service and connects
    to this API via the FASTAPI_URL environment variable.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ProxyConfig(config_path=config_path)
        self.router = LiteLLMProxyRouter(config=self.config)

        self.app = FastAPI(
            title="LiteLLM Load-Balancing Routing Proxy",
            description=(
                "OOP-based Microservice for intelligent LLM routing, "
                "automatic fallbacks, TPM/RPM rate limiting, and DeBERTa-v3 PII shielding."
            ),
            version="2.0.0"
        )

        # Wide-open CORS — the Streamlit service calls this API cross-origin
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes()

    def _register_routes(self):
        """Registers all FastAPI endpoints."""

        # -----------------------------------------------------------
        # ROOT — API info page (no Streamlit proxy needed)
        # -----------------------------------------------------------

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """API landing page — links to the Streamlit dashboard service."""
            streamlit_url = os.environ.get("STREAMLIT_URL", "#")
            return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LiteLLM Gateway API</title>
  <style>
    * {{ margin: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0e1117; color: #fafafa;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .card {{ background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 16px;
             padding: 3rem; max-width: 540px; width: 90%; text-align: center; }}
    .logo {{ font-size: 3.5rem; margin-bottom: 1rem; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; color: #ff4b4b; margin-bottom: .5rem; }}
    p {{ color: #8b9dc3; line-height: 1.6; margin-bottom: 2rem; font-size: .95rem; }}
    .btn {{ display: inline-block; padding: .75rem 2rem; border-radius: 8px;
            text-decoration: none; font-weight: 600; font-size: .9rem; transition: .2s; }}
    .btn-primary {{ background: #ff4b4b; color: #fff; margin-right: .75rem; }}
    .btn-primary:hover {{ background: #e03e3e; }}
    .btn-secondary {{ background: #2a2d3e; color: #d1d5db; border: 1px solid #3a3d4e; }}
    .btn-secondary:hover {{ background: #3a3d4e; }}
    .pill {{ display: inline-block; background: #22c55e22; color: #22c55e;
             border: 1px solid #22c55e44; border-radius: 20px; padding: .2rem .8rem;
             font-size: .75rem; font-weight: 600; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🛡️</div>
    <div class="pill">● API Online</div>
    <h1>LiteLLM Gateway API</h1>
    <p>Intelligent LLM routing with DeBERTa PII shielding.<br>
       Connect your Streamlit dashboard to start routing queries.</p>
    <a class="btn btn-primary" href="{streamlit_url}" target="_blank">Open Dashboard →</a>
    <a class="btn btn-secondary" href="/docs">API Docs</a>
  </div>
</body>
</html>""", status_code=200)

        # -----------------------------------------------------------
        # INFRASTRUCTURE
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
                data.append({
                    "id": vm, "object": "model", "created": 1686935002,
                    "owned_by": "proxy-system", "type": "virtual"
                })
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