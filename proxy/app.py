import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ProxyConfig
from .router import LiteLLMProxyRouter

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

class LiteLLMProxyApp:
    """
    Class-based FastAPI Application container for the LiteLLM Proxy microservice.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Initialize configuration and the routing engine
        self.config = ProxyConfig(config_path=config_path)
        self.router = LiteLLMProxyRouter(config=self.config)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="LiteLLM Load-Balancing Routing Proxy",
            description="OOP-based Microservice for intelligent LLM routing, fallbacks, TPM/RPM limits, and TPR filtering.",
            version="1.0.0"
        )
        
        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Registers FastAPI endpoints to the underlying app instance."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            """Serves the interactive single-page application dashboard for model queries and metric evaluations."""
            try:
                # Resolve index.html path relative to app.py
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
                # Fallback raw HTML if template is missing or fails to load
                return HTMLResponse(
                    content=f"<html><body><h1>LiteLLM Proxy API Online</h1><p>UI loading error: {e}</p></body></html>", 
                    status_code=500
                )
        
        @self.app.get("/health")
        async def health():
            """Simple health check endpoint for container orchestrators like Kubernetes."""
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
            
            # Enrich with active configuration details
            pii_settings = getattr(self.config, "pii_shield_settings", None)
            return {
                "timestamp": time.time(),
                "metrics": router_metrics,
                "active_routing_rules": {
                    "strategy": self.config.routing_strategy,
                    "retries": self.config.num_retries,
                    "timeout": self.config.timeout
                },
                "pii_shield_settings": {
                    "enabled": pii_settings.enabled if pii_settings else False,
                    "entities": pii_settings.entities if pii_settings else [],
                    "custom_regex_rules": [r.get("name") for r in pii_settings.custom_regex_rules] if pii_settings else []
                },
                "registered_virtual_models": list(set(e.model_name for e in self.config.endpoints)),
                "registered_physical_providers": list(set(e.model.split("/")[0] for e in self.config.endpoints))
            }

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
            # Convert ChatMessage list to standard list of dicts
            messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
            
            # Validate requested model exists
            virtual_models = set(e.model_name for e in self.config.endpoints)
            if request.model not in virtual_models:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Requested model '{request.model}' not found in proxy configuration. "
                           f"Available virtual models: {list(virtual_models)}"
                )

            try:
                # Execute complete load-balanced query
                response = self.router.execute_chat_completion(
                    model=request.model,
                    messages=messages_dict,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    mock_sandbox=request.mock_sandbox
                )
                return response
                
            except ValueError as ve:
                # Custom error if prompt exceeds model capacities or constraints (TPR check failures)
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                # Global failure (e.g. rate limit, backend down, misconfiguration)
                raise HTTPException(status_code=500, detail=str(e))
                
    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance (useful for running with ASGI servers)."""
        return self.app
