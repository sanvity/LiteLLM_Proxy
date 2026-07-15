import os
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
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
    epochs: Optional[int] = Field(default=15, description="Number of training epochs")
    learning_rate: Optional[float] = Field(default=1e-4, description="Learning rate")
    batch_size: Optional[int] = Field(default=8, description="Batch size for training")
    use_optuna: Optional[bool] = Field(default=False, description="Whether to run hyperparameter tuning with Optuna first")
    optuna_trials: Optional[int] = Field(default=3, description="Number of trials for Optuna tuning")
    baseline_version_id: Optional[str] = Field(default=None, description="Model version ID to start fine-tuning from (or None/base for default)")


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
    bypass_guardrails: Optional[bool] = Field(default=False, description="If true, bypass all PII guardrails for this request (e.g. for synthesis)")

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
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        finetuned_config = os.path.abspath(os.path.join(current_dir, "..", "models", "finetuned-deberta", "config.json"))
        if os.path.exists(finetuned_config):
            self.training_status = "completed"
            self.training_progress = "Training successfully completed. Local model active."
        else:
            self.training_status = "idle"
            self.training_progress = ""
        self.training_error = None
        
        # MLOps Model Registry and Evaluation state parameters
        from .mlops import ModelRegistry
        self.registry = ModelRegistry(os.path.abspath(os.path.join(current_dir, "..", "models")))
        self.evaluation_status = "idle"
        self.evaluation_progress = ""
        self.evaluation_error = None
        
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
        
        @self.app.get("/", response_class=RedirectResponse)
        async def serve_ui():
            """Redirects to the active Streamlit frontend app."""
            return RedirectResponse(url="https://sanvity-litellm-proxy-app-surcgq.streamlit.app")
        
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

        @self.app.get("/api/evaluation")
        async def get_evaluation_data():
            """Returns the latest PII guardrail evaluation data for the frontend."""
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eval_json_path = os.path.abspath(os.path.join(current_dir, "..", "data", "evaluation_data.json"))
            
            if not os.path.exists(eval_json_path):
                raise HTTPException(status_code=404, detail="Evaluation data not found. Run evaluate.py to generate evaluation report.")
            
            try:
                with open(eval_json_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load evaluation data: {str(e)}")

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
                    mock_sandbox=request.mock_sandbox,
                    bypass_guardrails=request.bypass_guardrails
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
            from guardrails.deberta_pii_guardrail import get_active_model_labels, get_custom_model_labels
            labels = get_active_model_labels()
            model_custom = get_custom_model_labels()
            saved_custom = getattr(self.router, "custom_labels", [])
            custom_labels = sorted(list(set(model_custom + saved_custom)))
            combined_labels = sorted(list(set(labels + custom_labels)))
            return {
                "pii_enabled": getattr(self.router, "pii_enabled", False),
                "pii_action": getattr(self.router, "pii_action", "MASK"),
                "pii_policy": getattr(self.router, "pii_policy", None),
                "active_labels": combined_labels,
                "custom_labels": custom_labels
            }

        @self.app.post("/ui/pii-config")
        async def update_pii_config(config: UIPiiConfig):
            """Updates the PII guardrail configuration."""
            self.router.pii_enabled = config.pii_enabled
            self.router.pii_action = config.pii_action
            self.router.pii_policy = config.pii_policy
            self.router.save_pii_guardrail_config()
            if config.pii_enabled:
                import threading
                threading.Thread(target=lambda: self.router._get_pii_guardrail().model, daemon=True).start()
            return {"status": "success", "message": f"PII Guardrail configuration synchronized in memory and saved to disk: enabled={config.pii_enabled}, action={config.pii_action}."}

        @self.app.get("/ui/safety-config")
        async def get_safety_config():
            """Returns the current content safety guardrail configuration."""
            return {
                "enabled": getattr(self.router, "safety_enabled", {
                    "jailbreak": False, "toxicity": False, "prompt_injection": False
                }),
                "action": getattr(self.router, "safety_action", {
                    "jailbreak": "BLOCK", "toxicity": "BLOCK", "prompt_injection": "BLOCK"
                }),
            }

        @self.app.post("/ui/safety-config")
        async def update_safety_config(payload: dict = Body(...)):
            """Updates and persists the content safety guardrail configuration."""
            enabled = payload.get("enabled", {})
            action  = payload.get("action", {})
            for key in ("jailbreak", "toxicity", "prompt_injection"):
                if key in enabled:
                    self.router.safety_enabled[key] = bool(enabled[key])
                if key in action:
                    self.router.safety_action[key] = action[key]
            self.router.save_safety_guardrail_config()
            return {"status": "success", "message": "Safety guardrail configuration saved."}



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
                
                # Prep staging directory based on selected baseline
                import shutil
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)
                os.makedirs(output_dir, exist_ok=True)
                
                if req.baseline_version_id and req.baseline_version_id != "base":
                    baseline_path = os.path.abspath(os.path.join(current_dir, "..", "models", "versions", req.baseline_version_id))
                    if os.path.exists(baseline_path):
                        from litellm._logging import verbose_proxy_logger
                        verbose_proxy_logger.info(f"[DeBERTa Training] Copying baseline model {req.baseline_version_id} to staging...")
                        self.training_progress = f"Copying baseline model {req.baseline_version_id} to staging..."
                        for fname in os.listdir(baseline_path):
                            src_f = os.path.join(baseline_path, fname)
                            if os.path.isfile(src_f):
                                shutil.copy(src_f, os.path.join(output_dir, fname))
                
                # ── Free the inference pipeline BEFORE spawning training ─────────
                from guardrails.deberta_pii_guardrail import active_guardrails
                from litellm._logging import verbose_proxy_logger
                verbose_proxy_logger.info(
                    "[DeBERTa Training] Unloading inference pipeline to free memory for training subprocess..."
                )
                for cb in active_guardrails:
                    cb._model = None
                import gc
                import sys
                import json
                import subprocess
                gc.collect()

                if req.use_optuna:
                    self.training_progress = "Running hyperparameter optimization with Optuna..."
                else:
                    self.training_progress = "Executing Hugging Face Trainer fine-tuning loop..."
                
                # Save configuration options to a temporary JSON file
                config_data = {
                    "dataset": dataset_dicts,
                    "output_dir": output_dir,
                    "epochs": req.epochs if req.epochs is not None else 3,
                    "learning_rate": req.learning_rate if req.learning_rate is not None else 5e-5,
                    "batch_size": req.batch_size if req.batch_size is not None else 8,
                    "use_optuna": req.use_optuna if req.use_optuna is not None else False,
                    "optuna_trials": req.optuna_trials if req.optuna_trials is not None else 3
                }

                import tempfile
                fd, temp_config_path = tempfile.mkstemp(suffix=".json", prefix="deberta_train_cfg_")
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
                        json.dump(config_data, tmp_f)
                    
                    # Spawn subprocess with LITELLM_TRAINING_ACTIVE=1 environment variable
                    env = os.environ.copy()
                    env["LITELLM_TRAINING_ACTIVE"] = "1"
                    
                    # Resolve script path
                    script_path = os.path.abspath(os.path.join(current_dir, "..", "guardrails", "deberta_pii_guardrail.py"))
                    
                    # Run subprocess
                    process_args = [sys.executable, script_path, "--train", temp_config_path]
                    verbose_proxy_logger.info(f"[DeBERTa Training] Spawning subprocess: {' '.join(process_args)}")
                    
                    res = subprocess.run(
                        process_args,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if res.returncode != 0:
                        raise RuntimeError(
                            f"Training subprocess failed with exit code {res.returncode}.\n"
                            f"Stderr: {res.stderr}\nStdout: {res.stdout}"
                        )
                finally:
                    if os.path.exists(temp_config_path):
                        try:
                            os.remove(temp_config_path)
                        except Exception:
                            pass
                
                self.training_progress = "Reloading active model pipelines..."
                # Reload model in all running instances
                for cb in active_guardrails:
                    cb.reload_model()
                
                # Permanently save new custom labels to configuration file
                from guardrails.deberta_pii_guardrail import get_custom_model_labels
                new_custom = get_custom_model_labels()
                if new_custom:
                    existing_custom = getattr(self.router, "custom_labels", [])
                    updated_custom = sorted(list(set(existing_custom + new_custom)))
                    self.router.custom_labels = updated_custom
                    self.router.save_pii_guardrail_config()
                    
                self.training_status = "completed"
                self.training_progress = "Training successfully completed. Local model active."
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                self.training_status = "failed"
                self.training_error = f"{str(e)}\n\n{error_trace}"
                self.training_progress = "Training failed."

        async def execute_train(request: TrainDebertaRequest, background_tasks: BackgroundTasks):
            if self.training_status == "training":
                raise HTTPException(status_code=400, detail="Training is already in progress.")
                
            self.training_status = "training"
            self.training_progress = "Starting training background task..."
            self.training_error = None
            
            background_tasks.add_task(run_training_in_background, request)
            return {"status": "training", "message": "DeBERTa fine-tuning background thread spawned successfully."}

        async def execute_status():
            history = []
            current_dir = os.path.dirname(os.path.abspath(__file__))
            history_filepath = os.path.abspath(os.path.join(current_dir, "..", "models", "finetuned-deberta", "training_history.json"))
            try:
                if os.path.exists(history_filepath):
                    with open(history_filepath, "r", encoding="utf-8") as f:
                        history = json.load(f)
                else:
                    from litellm._logging import verbose_proxy_logger
                    verbose_proxy_logger.warning(f"[DeBERTa Status] File does not exist: {history_filepath}")
            except Exception as e:
                import traceback
                from litellm._logging import verbose_proxy_logger
                verbose_proxy_logger.error(f"[DeBERTa Status] Failed to read training history file: {e}\n{traceback.format_exc()}")
            return {
                "status": self.training_status,
                "progress": self.training_progress,
                "error": self.training_error,
                "history": history
            }

        async def execute_reset():
            if self.training_status == "training":
                raise HTTPException(status_code=400, detail="Cannot reset while training is actively running.")
            self.training_status = "idle"
            self.training_progress = ""
            self.training_error = None
            return {"status": "success", "message": "Training status reset to idle."}

        @self.app.post("/ui/train-deberta")
        async def train_deberta(request: TrainDebertaRequest, background_tasks: BackgroundTasks):
            """Triggers asynchronous fine-tuning of the DeBERTa PII guardrail model (UI endpoint)."""
            return await execute_train(request, background_tasks)

        @self.app.post("/v1/deberta/train")
        @self.app.post("/deberta/train")
        async def train_deberta_api(request: TrainDebertaRequest, background_tasks: BackgroundTasks):
            """Triggers asynchronous fine-tuning of the DeBERTa PII guardrail model (API endpoint)."""
            return await execute_train(request, background_tasks)

        @self.app.get("/ui/train-deberta/status")
        async def train_deberta_status():
            """Returns the current status of the model training process (UI endpoint)."""
            return await execute_status()

        @self.app.get("/v1/deberta/train/status")
        @self.app.get("/deberta/train/status")
        async def train_deberta_status_api():
            """Returns the current status of the model training process (API endpoint)."""
            return await execute_status()

        @self.app.post("/ui/train-deberta/reset")
        async def train_deberta_reset():
            """Resets the training status to idle (allowed only if not actively training) (UI endpoint)."""
            return await execute_reset()

        @self.app.post("/v1/deberta/train/reset")
        @self.app.post("/deberta/train/reset")
        async def train_deberta_reset_api():
            """Resets the training status to idle (allowed only if not actively training) (API endpoint)."""
            return await execute_reset()

        @self.app.post("/ui/pii-detect")
        async def ui_pii_detect(request: dict):
            """Endpoint to run raw PII detection using the active model for UI validation."""
            text = request.get("text", "")
            from guardrails.deberta_pii_guardrail import DeBERTaPIIGuardrail, active_guardrails
            if active_guardrails:
                guardrail = active_guardrails[0]
            else:
                guardrail = DeBERTaPIIGuardrail()
            cfg = guardrail._get_config({})
            entities = guardrail.detect(text, cfg)
            # sort by start ascending
            entities.sort(key=lambda e: e["start"])
            return {"entities": entities}

        @self.app.get("/ui/mlops/registry")
        async def get_mlops_registry():
            """Returns the list of all saved versions and active version ID."""
            try:
                versions = self.registry.get_versions()
                active_version = self.registry.get_active_version()
                return {"active_version": active_version, "versions": versions}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ui/mlops/save")
        async def save_model_version(request: dict):
            """Saves current staging model as a new version."""
            name = request.get("name", "Unnamed Version")
            description = request.get("description", "")
            config = request.get("config", {})
            try:
                vinfo = self.registry.save_version(name, description, config)
                return {"status": "success", "version": vinfo}
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ui/mlops/deploy")
        async def deploy_model_version(request: dict):
            """Deploys a specific model version."""
            version_id = request.get("version_id") # version ID or None (to use default/staging)
            try:
                deployed = self.registry.deploy_version(version_id)
                # Trigger reload of model across active pipelines
                from guardrails.deberta_pii_guardrail import active_guardrails
                for cb in active_guardrails:
                    cb.reload_model()
                return {"status": "success", "message": f"Successfully deployed {deployed}."}
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ui/mlops/delete")
        async def delete_model_version(request: dict):
            """Deletes a specific model version."""
            version_id = request.get("version_id")
            try:
                self.registry.delete_version(version_id)
                return {"status": "success", "message": f"Version {version_id} deleted."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ui/mlops/evaluate")
        async def evaluate_model_version(request: dict, background_tasks: BackgroundTasks):
            """Triggers evaluation on staging or a specific version in background."""
            version_id = request.get("version_id", "staging")
            if self.evaluation_status == "running":
                raise HTTPException(status_code=400, detail="Evaluation is already in progress.")
            
            self.evaluation_status = "running"
            self.evaluation_progress = f"Starting evaluation for {version_id}..."
            self.evaluation_error = None
            
            background_tasks.add_task(self.run_evaluation_in_background, version_id)
            return {"status": "running", "message": f"Evaluation for {version_id} started."}

        @self.app.get("/ui/mlops/evaluate/status")
        async def get_evaluation_status():
            """Returns the current background evaluation status."""
            return {
                "status": self.evaluation_status,
                "progress": self.evaluation_progress,
                "error": self.evaluation_error
            }

        @self.app.get("/ui/mlops/report/{version_id}")
        async def serve_version_report(version_id: str):
            """Serves the HTML evaluation report for a specific version or staging."""
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if version_id == "staging":
                report_path = os.path.abspath(os.path.join(current_dir, "..", "models", "finetuned-deberta", "evaluation_report.html"))
            else:
                report_path = os.path.abspath(os.path.join(current_dir, "..", "models", "versions", version_id, "evaluation_report.html"))
                
            if not os.path.exists(report_path):
                raise HTTPException(status_code=404, detail="Evaluation report not found. Run evaluation first.")
                
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")

    def run_evaluation_in_background(self, version_id: str):
        try:
            import sys
            import subprocess
            import gc
            import torch
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Resolve paths
            if version_id == "staging":
                model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "finetuned-deberta"))
                output_dir = model_path
            else:
                model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "versions", version_id))
                output_dir = model_path
                
            if not os.path.exists(os.path.join(model_path, "config.json")) and not os.path.exists(os.path.join(model_path, "adapter_config.json")):
                raise ValueError(f"Model path {model_path} does not exist or is missing configuration files.")
                
            self.evaluation_progress = "Unloading inference pipeline to free memory..."
            # Unload inference pipeline
            from guardrails.deberta_pii_guardrail import active_guardrails
            for cb in active_guardrails:
                cb._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.evaluation_progress = "Running model evaluation script..."
            script_path = os.path.abspath(os.path.join(current_dir, "..", "scripts", "evaluate.py"))
            
            process_args = [
                sys.executable,
                script_path,
                "--model_path", model_path,
                "--output_dir", output_dir
            ]
            
            res = subprocess.run(
                process_args,
                capture_output=True,
                text=True,
                check=False
            )
            
            if res.returncode != 0:
                raise RuntimeError(f"Evaluation subprocess failed with exit code {res.returncode}.\nStderr: {res.stderr}")
                
            self.evaluation_progress = "Parsing evaluation results..."
            # Read evaluation_data.json from output_dir
            eval_data_path = os.path.join(output_dir, "evaluation_data.json")
            if not os.path.exists(eval_data_path):
                raise RuntimeError("Evaluation script finished but evaluation_data.json was not found.")
                
            with open(eval_data_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                
            ft_metrics = eval_data.get("finetuned_metrics", {})
            macro_metrics = ft_metrics.get("macro", {})
            
            # If not staging, update version registry
            if version_id != "staging":
                self.registry.update_version_metrics(version_id, macro_metrics)
                
            # Reload model across active pipelines
            for cb in active_guardrails:
                cb.reload_model()
                
            self.evaluation_status = "completed"
            self.evaluation_progress = "Evaluation successfully completed."
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.evaluation_status = "failed"
            self.evaluation_error = f"{str(e)}\n\n{error_trace}"
            self.evaluation_progress = "Evaluation failed."

    def get_app(self) -> FastAPI:
        """Returns the FastAPI instance (useful for running with ASGI servers)."""
        return self.app