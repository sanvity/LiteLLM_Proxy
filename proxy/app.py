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

class TeamGuardrailRegisterRequest(BaseModel):
    guardrail_name: str = Field(..., description="Unique name for the guardrail")
    litellm_params: Dict[str, Any] = Field(..., description="YAML parameters matching Generic Guardrail API")
    guardrail_info: Optional[Dict[str, Any]] = Field(None, description="Optional metadata description")

class GuardrailTestRequest(BaseModel):
    text: str = Field(..., description="The prompt or text to test against selected guardrails")
    guardrails: Optional[List[str]] = Field(None, description="List of guardrail names to compare. Runs all if not specified.")

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


        @self.app.post("/guardrails/register")
        async def register_guardrail(request: TeamGuardrailRegisterRequest):
            """
            Developer dynamic team-scoped guardrail registration.
            """
            params = request.litellm_params
            provider = params.get("guardrail")
            if not provider or provider not in ["aporia", "litellm_content_filter"]:
                raise HTTPException(status_code=400, detail="Registration requires 'aporia' or 'litellm_content_filter' provider.")
            
            # Simple metadata validations
            if provider == "aporia":
                if not params.get("api_base") or not params.get("api_key"):
                    raise HTTPException(status_code=400, detail="Missing required 'api_base' or 'api_key' parameter for Aporia.")
                
            import uuid
            guardrail_id = str(uuid.uuid4())
            
            # Store in router team-based registry
            self.router.team_guardrails[guardrail_id] = {
                "guardrail_id": guardrail_id,
                "guardrail_name": request.guardrail_name,
                "litellm_params": params,
                "guardrail_info": request.guardrail_info or {},
                "status": "pending_review",
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            return {
                "guardrail_id": guardrail_id,
                "guardrail_name": request.guardrail_name,
                "status": "pending_review",
                "submitted_at": self.router.team_guardrails[guardrail_id]["submitted_at"]
            }

        @self.app.get("/guardrails/submissions")
        async def get_submissions(status: Optional[str] = None):
            """
            Lists all team-submitted guardrails for comparison and auditing.
            """
            submissions = list(self.router.team_guardrails.values())
            if status:
                submissions = [s for s in submissions if s.get("status") == status]
            return {"submissions": submissions}

        @self.app.get("/aporia/control-plane")
        async def get_aporia_control_plane():
            from .router import AporiaControlPlaneState
            state = AporiaControlPlaneState()
            return {
                "master_switch": state.master_switch,
                "evaluators": state.evaluators,
                "sensitivity": state.sensitivity,
                "remediation_actions": state.remediation_actions,
                "custom_shadow_keywords": state.custom_shadow_keywords,
                "session_logs": state.session_logs[-100:]
            }

        @self.app.post("/aporia/control-plane")
        async def update_aporia_control_plane(config: dict):
            from .router import AporiaControlPlaneState
            state = AporiaControlPlaneState()
            state.master_switch = config.get("master_switch", state.master_switch)
            state.evaluators = config.get("evaluators", state.evaluators)
            state.sensitivity = config.get("sensitivity", state.sensitivity)
            state.remediation_actions = config.get("remediation_actions", state.remediation_actions)
            state.custom_shadow_keywords = config.get("custom_shadow_keywords", state.custom_shadow_keywords)
            return {"status": "success", "message": "Aporia control plane configuration updated successfully."}

        @self.app.post("/guardrails/submissions/{guardrail_id}/approve")
        async def approve_submission(guardrail_id: str):
            """
            Admin-scoped endpoint to approve and dynamically activate a team guardrail.
            """
            if guardrail_id not in self.router.team_guardrails:
                raise HTTPException(status_code=404, detail="Guardrail submission not found.")
                
            submission = self.router.team_guardrails[guardrail_id]
            submission["status"] = "active"
            
            # Dynamically compile and mount in memory
            name = submission.get("guardrail_name")
            params = submission.get("litellm_params", {})
            provider = params.get("guardrail")
            
            from .router import LiteLLMContentFilter, GenericGuardrailSimulator
            if provider == "litellm_content_filter":
                inst = LiteLLMContentFilter(router=self.router, config_dict=submission)
            else:
                inst = GenericGuardrailSimulator(router=self.router, config_dict=submission)
                
            if name not in self.router.guardrail_instances:
                self.router.guardrail_instances[name] = []
            self.router.guardrail_instances[name].append(inst)
            
            self.router.log_event(f"[Admin] Approved and dynamically activated team guardrail '{name}'.", "success")
            return {"message": "Guardrail successfully approved and activated in memory.", "status": "active"}

        @self.app.post("/guardrails/submissions/{guardrail_id}/reject")
        async def reject_submission(guardrail_id: str):
            """
            Decline a team guardrail submission.
            """
            if guardrail_id not in self.router.team_guardrails:
                raise HTTPException(status_code=404, detail="Guardrail submission not found.")
                
            submission = self.router.team_guardrails[guardrail_id]
            submission["status"] = "rejected"
            self.router.log_event(f"[Admin] Rejected team guardrail submission '{submission.get('guardrail_name')}'", "error")
            return {"message": "Guardrail submission rejected.", "status": "rejected"}

        @self.app.post("/guardrails/test")
        async def test_guardrails(request: GuardrailTestRequest):
            """
            Interactive testing playground to compare and evaluate multiple guardrails on a sample input.
            """
            text = request.text
            guardrail_names = request.guardrails
            
            from .router import LocalPresidioPIIMasking, LiteLLMContentFilter, GenericGuardrailSimulator
            if not guardrail_names:
                guardrail_names = list(self.router.guardrail_instances.keys())
                
            results = []
            
            for name in guardrail_names:
                # Guardrail Load Balancing: Retrieve a load-balanced instance of the guardrail by name
                inst = self.router.get_guardrail_instance(name)
                if not inst:
                    # Check if there is an active team guardrail with this name
                    matching_team = next((g for g in self.router.team_guardrails.values() if g.get("guardrail_name") == name and g.get("status") == "active"), None)
                    if matching_team:
                        inst = GenericGuardrailSimulator(router=self.router, config_dict=matching_team)
                        
                if not inst:
                    results.append({
                        "guardrail_name": name,
                        "status": "not_found",
                        "passed": True,
                        "action": "ALLOW",
                        "output": text,
                        "reason": f"Guardrail '{name}' not found or inactive."
                    })
                    continue
                    
                # Run the selected guardrail instance
                try:
                    if isinstance(inst, LocalPresidioPIIMasking):
                        sandbox_data = {"metadata": {}}
                        output_text = inst.shield_text(text, sandbox_data)
                        pii_tokens = sandbox_data["metadata"].get("pii_tokens", {})
                        passed = (output_text == text)
                        results.append({
                            "guardrail_name": name,
                            "status": "success",
                            "passed": passed,
                            "action": "MASK" if not passed else "ALLOW",
                            "output": output_text,
                            "reason": f"Detected PII tokens: {list(pii_tokens.values())}" if pii_tokens else "No PII detected."
                        })
                    elif isinstance(inst, LiteLLMContentFilter):
                        is_blocked, action, reason = await inst.check_text(text)
                        output_text = inst.mask_text(text) if action == "MASK" else text
                        results.append({
                            "guardrail_name": name,
                            "status": "success",
                            "passed": not is_blocked,
                            "action": action if is_blocked else "ALLOW",
                            "output": output_text if action == "MASK" else text,
                            "reason": reason if is_blocked else "Passes content filter checks."
                        })
                    elif isinstance(inst, GenericGuardrailSimulator):
                        is_blocked, action, reason = await inst.check_text(text)
                        output_text = inst.mask_text(text) if action == "MASK" else text
                        results.append({
                            "guardrail_name": name,
                            "status": "success",
                            "passed": not is_blocked,
                            "action": action if is_blocked else "ALLOW",
                            "output": output_text,
                            "reason": reason if is_blocked else "Passes security API checks."
                        })
                except Exception as e:
                    results.append({
                        "guardrail_name": name,
                        "status": "error",
                        "passed": True,
                        "action": "ALLOW",
                        "output": text,
                        "reason": f"Error running guardrail: {e}"
                    })
                    
            return {"results": results}
                
    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance (useful for running with ASGI servers)."""
        return self.app