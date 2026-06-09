import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ProxyConfig
from .router import LiteLLMProxyRouter

class EntityLabel(BaseModel):
    start: int = Field(..., description="Start character offset of the entity")
    end: int = Field(..., description="End character offset of the entity")
    label: str = Field(..., description="Label of the entity (e.g. person, email address)")

class TrainingSample(BaseModel):
    text: str = Field(..., description="The input text sample")
    entities: List[EntityLabel] = Field(..., description="List of labeled entities in the text")

class TrainDebertaRequest(BaseModel):
    dataset: List[TrainingSample] = Field(..., description="List of training text samples and labels")
    epochs: Optional[int] = Field(default=3, description="Number of training epochs")
    learning_rate: Optional[float] = Field(default=5e-5, description="Learning rate")
    batch_size: Optional[int] = Field(default=8, description="Batch size for training")

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
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Initialize configuration and the routing engine
        self.config = ProxyConfig(config_path=config_path)
        self.router = LiteLLMProxyRouter(config=self.config)
        
        # Training state parameters
        self.training_status = "idle"
        self.training_progress = ""
        self.training_error = None
        
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
                response = await self.router.execute_chat_completion(
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
            return {"status": "success", "message": f"PII Guardrail configuration synchronized in memory: enabled={config.pii_enabled}, action={config.pii_action}."}

        def run_training_in_background(req: TrainDebertaRequest):
            try:
                self.training_status = "training"
                self.training_progress = "Tokenizing and prepping dataset..."
                self.training_error = None
                
                # Format dataset to dict for the training function
                dataset_dicts = []
                for sample in req.dataset:
                    ents = [{"start": e.start, "end": e.end, "label": e.label} for e in sample.entities]
                    dataset_dicts.append({
                        "text": sample.text,
                        "entities": ents
                    })
                
                # Resolve output directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.abspath(os.path.join(current_dir, "..", "models", "finetuned-deberta"))
                
                self.training_progress = "Executing Hugging Face Trainer fine-tuning loop..."
                
                from guardrails.deberta_pii_guardrail import train_deberta_model, active_guardrails
                
                train_deberta_model(
                    dataset=dataset_dicts,
                    output_dir=output_dir,
                    epochs=req.epochs if req.epochs is not None else 3,
                    learning_rate=req.learning_rate if req.learning_rate is not None else 5e-5,
                    batch_size=req.batch_size if req.batch_size is not None else 8
                )
                
                self.training_progress = "Reloading active model pipelines..."
                # Reload model in all running instances
                for cb in active_guardrails:
                    cb.reload_model()
                    
                self.training_status = "completed"
                self.training_progress = "Training successfully completed. Local model active."
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                self.training_status = "failed"
                self.training_error = f"{str(e)}\n\n{error_trace}"
                self.training_progress = "Training failed."

        @self.app.post("/ui/train-deberta")
        async def train_deberta(request: TrainDebertaRequest, background_tasks: BackgroundTasks):
            """Triggers asynchronous fine-tuning of the DeBERTa PII guardrail model."""
            if self.training_status == "training":
                raise HTTPException(status_code=400, detail="Training is already in progress.")
                
            self.training_status = "training"
            self.training_progress = "Starting training background task..."
            self.training_error = None
            
            background_tasks.add_task(run_training_in_background, request)
            return {"status": "training", "message": "DeBERTa fine-tuning background thread spawned successfully."}

        @self.app.get("/ui/train-deberta/status")
        async def train_deberta_status():
            """Returns the current status of the model training process."""
            return {
                "status": self.training_status,
                "progress": self.training_progress,
                "error": self.training_error
            }

        @self.app.post("/ui/train-deberta/reset")
        async def train_deberta_reset():
            """Resets the training status to idle (allowed only if not actively training)."""
            if self.training_status == "training":
                raise HTTPException(status_code=400, detail="Cannot reset while training is actively running.")
            self.training_status = "idle"
            self.training_progress = ""
            self.training_error = None
            return {"status": "success", "message": "Training status reset to idle."}



                
    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance (useful for running with ASGI servers)."""
        return self.app