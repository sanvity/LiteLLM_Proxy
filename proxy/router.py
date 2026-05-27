import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

from .config import ProxyConfig, ModelEndpointConfig

logger = logging.getLogger("proxy.router")

from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking

class LocalPresidioPIIMasking(_OPTIONAL_PresidioPIIMasking):
    """
    Local, stateless implementation of LiteLLM's standard Presidio PII Masking guardrail.
    Inherits natively from _OPTIONAL_PresidioPIIMasking and overrides HTTP calls
    to execute locally using presidio-analyzer and presidio-anonymizer.
    """
    def __init__(self, router, **kwargs):
        # Prevent base class validation from throwing missing URL exceptions
        kwargs.setdefault("presidio_analyzer_api_base", "http://localhost:5002/")
        kwargs.setdefault("presidio_anonymizer_api_base", "http://localhost:5001/")
        kwargs.setdefault("output_parse_pii", True)
        kwargs.setdefault("guardrail_name", "presidio-pii")
        
        super().__init__(**kwargs)
        self.router = router
        
        # Initialize local Presidio engines
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Dynamically register custom regex patterns from configuration
        self._register_custom_rules()

    def validate_environment(self, **kwargs):
        # Overridden to prevent requiring external base URLs in environment
        self.presidio_analyzer_api_base = "http://localhost:5002/"
        self.presidio_anonymizer_api_base = "http://localhost:5001/"

    def _register_custom_rules(self):
        """Registers user-defined config regex rules dynamically into Presidio's native recognizer registry."""
        from presidio_analyzer import PatternRecognizer, Pattern
        pii_settings = getattr(self.router.config, "pii_shield_settings", None)
        if not pii_settings:
            return
            
        custom_rules = pii_settings.custom_regex_rules
        for idx, rule in enumerate(custom_rules, 1):
            rule_name = rule.get("name", f"CUSTOM_RULE_{idx}")
            pattern_str = rule.get("pattern", "")
            if pattern_str:
                try:
                    # Map common name rules to PERSON, others to custom uppercase entity types
                    entity_type = "PERSON" if rule_name in ["Sanvi", "Jain"] else rule_name.upper()
                    
                    custom_pattern = Pattern(
                        name=f"{rule_name.lower()}_pattern",
                        regex=pattern_str,
                        score=1.0
                    )
                    custom_recognizer = PatternRecognizer(
                        supported_entity=entity_type,
                        patterns=[custom_pattern]
                    )
                    self.analyzer.registry.add_recognizer(custom_recognizer)
                    logger.info(f"Registered local Presidio PatternRecognizer: '{rule_name}' with pattern '{pattern_str}' -> entity '{entity_type}'")
                except Exception as e:
                    logger.error(f"Failed to register local Presidio PatternRecognizer for '{rule_name}': {e}")

    async def analyze_text(
        self,
        text: str,
        presidio_config: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Scans prompt text using local Presidio Analyzer, resolves overlaps, and returns list of matches."""
        if not text or not text.strip():
            return []
            
        pii_settings = getattr(self.router.config, "pii_shield_settings", None)
        active_entities = pii_settings.entities if pii_settings else ["PERSON", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        
        # Analyze locally
        results = self.analyzer.analyze(
            text=text,
            language=self.presidio_language,
            entities=active_entities
        )
        
        # Format results as expected by _OPTIONAL_PresidioPIIMasking
        matches = [
            {
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score,
            }
            for res in results
        ]
        
        # Resolve overlaps (keep only non-overlapping matches with highest score/length)
        matches.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        non_overlapping = []
        last_end = 0
        for match in matches:
            if match["start"] >= last_end:
                non_overlapping.append(match)
                last_end = match["end"]
                
        return non_overlapping

    async def _post_presidio_anonymize(self, text: str, analyze_results: Any) -> Dict[str, Any]:
        """Anonymizes text using local Presidio AnonymizerEngine."""
        from presidio_analyzer import RecognizerResult
        
        results = []
        for d in analyze_results:
            r = RecognizerResult(
                entity_type=d["entity_type"],
                start=d["start"],
                end=d["end"],
                score=d.get("score", 1.0)
            )
            results.append(r)
            
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
        
        items = []
        for x in anonymized_result.items:
            items.append({
                "entity_type": x.entity_type,
                "start": x.start,
                "end": x.end
            })
            
        return {
            "text": anonymized_result.text,
            "items": items
        }

    def shield_text(self, text: str, request_data: dict) -> str:
        """Synchronously redacts and numbers PII tokens. Used primarily for sandbox mock completions."""
        if not text or not text.strip():
            return text
            
        pii_settings = getattr(self.router.config, "pii_shield_settings", None)
        if pii_settings and not pii_settings.enabled:
            return text
            
        active_entities = pii_settings.entities if pii_settings else ["PERSON", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        
        results = self.analyzer.analyze(
            text=text,
            language=self.presidio_language,
            entities=active_entities
        )
        
        dict_results = [
            {
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score,
            }
            for res in results
        ]
        
        # Resolve overlaps
        dict_results.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        non_overlapping = []
        last_end = 0
        for match in dict_results:
            if match["start"] >= last_end:
                non_overlapping.append(match)
                last_end = match["end"]
        
        masked_entity_count = {}
        # Apply standard LiteLLM numbered replacement and mapping
        return self._finalize_presidio_anonymize_numbered_tokens(
            text=text,
            analyze_results=non_overlapping,
            request_data=request_data,
            masked_entity_count=masked_entity_count
        )

    def unmask_text(self, text: str, request_data: dict) -> str:
        """Synchronously restores raw PII values into redacted text."""
        metadata = request_data.get("metadata", {}) if request_data else {}
        pii_tokens = metadata.get("pii_tokens", {})
        if not text or not pii_tokens:
            return text
            
        return self._unmask_pii_text(text, pii_tokens)

    def _finalize_presidio_anonymize_numbered_tokens(
        self,
        text: str,
        analyze_results: Any,
        request_data: Optional[Dict],
        masked_entity_count: Dict[str, int],
    ) -> str:
        """Overrides base class method to implement per-type counter token numbering (e.g. <PERSON_1>)."""
        new_text = text
        if request_data is None:
            request_data = {}
        if not request_data.get("metadata"):
            request_data["metadata"] = {}
        if "pii_tokens" not in request_data["metadata"]:
            request_data["metadata"]["pii_tokens"] = {}
        pii_tokens = request_data["metadata"]["pii_tokens"]

        # Sort detections by start position forward
        sorted_forward = sorted(analyze_results, key=lambda x: x["start"])
        
        # Keep per-type counters
        type_counters = {}
        seq_map = {}
        
        for ar in sorted_forward:
            etype = ar["entity_type"]
            if etype not in type_counters:
                type_counters[etype] = 1
            seq_map[(ar["start"], ar["end"])] = type_counters[etype]
            type_counters[etype] += 1

        # Replace in reverse order to keep positions intact
        for ar in reversed(sorted_forward):
            start = ar["start"]
            end = ar["end"]
            entity_type = ar["entity_type"]
            seq = seq_map[(start, end)]
            replacement = f"<{entity_type}_{seq}>"
            
            pii_tokens[replacement] = text[start:end]
            new_text = new_text[:start] + replacement + new_text[end:]
            masked_entity_count[entity_type] = masked_entity_count.get(entity_type, 0) + 1
            
        return new_text

    def log_pre_api_call(self, model: str, messages: List[Dict[str, str]], kwargs: Dict[str, Any]):
        """
        Interceptive pre-call hook invoked natively by LiteLLM completion/router.
        Masks user prompt in-place and saves transient PII tokens in the request metadata.
        """
        pii_settings = getattr(self.router.config, "pii_shield_settings", None)
        if not pii_settings or not pii_settings.enabled:
            return

        # Prepare request-scoped metadata
        if "metadata" not in kwargs or not isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = {}
        if "pii_tokens" not in kwargs["metadata"]:
            kwargs["metadata"]["pii_tokens"] = {}
        
        pii_redacted = False
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                # Anonymize user messages using local helper
                sanitized_content = self.shield_text(content, kwargs)
                if sanitized_content != content:
                    pii_redacted = True
                    msg["content"] = sanitized_content

        if pii_redacted:
            placeholders = list(kwargs["metadata"]["pii_tokens"].keys())
            self.router.log_event(
                f"[PII Guardrail Callback] Natively redacted PII pre-call. Placeholders: {placeholders}", 
                "warning"
            )

    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any):
        """Synchronous success hook. Restores raw PII values back into the response."""
        self._unmask_response(kwargs, response_obj)

    async def async_log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any):
        """Asynchronous success hook. Restores raw PII values back into the response."""
        self._unmask_response(kwargs, response_obj)

    def _unmask_response(self, kwargs: Dict[str, Any], response_obj: Any):
        """Parses completion response and restores raw user values using in-memory metadata."""
        metadata = kwargs.get("metadata", {}) if kwargs else {}
        if not metadata or not isinstance(metadata, dict):
            return
            
        pii_tokens = metadata.get("pii_tokens")
        if not pii_tokens:
            return

        try:
            choices = getattr(response_obj, "choices", [])
            replaced_count = 0
            for choice in choices:
                msg = getattr(choice, "message", None)
                if msg:
                    content = getattr(msg, "content", "")
                    if content:
                        restored_content = self.unmask_text(content, kwargs)
                        if restored_content != content:
                            msg.content = restored_content
                            for placeholder in pii_tokens.keys():
                                if placeholder in content:
                                    replaced_count += 1
                                    
            if replaced_count > 0:
                self.router.log_event(
                    f"[PII De-anonymizer Callback] Natively restored {replaced_count} placeholders in response choice.", 
                    "success"
                )
        except Exception as e:
            logger.error(f"Error during native callback de-anonymization: {e}")

class LiteLLMProxyRouter:
    """
    Core OOP Routing Engine wrapping LiteLLM's Router.
    Implements Token Per Request (TPR) checks, load balancing, fallback routing, and token usage optimization.
    """
    def __init__(self, config: ProxyConfig):
        self.config = config
        
        # Configure litellm global settings
        litellm.telemetry = False
        litellm.drop_params = True # Safely drop unsupported params per provider
        
        # Convert our configurations to the shape LiteLLM expects
        model_list = self.config.to_litellm_model_list()
        
        # Initialize LiteLLM's core Router
        logger.info(f"Initializing LiteLLM Router with strategy: {self.config.routing_strategy}")
        self.router = Router(
            model_list=model_list,
            routing_strategy=self.config.routing_strategy,
            num_retries=self.config.num_retries,
            timeout=self.config.timeout,
            fallbacks=self.config.general_fallbacks,
            context_window_fallbacks=self.config.context_window_fallbacks
        )
        
        # Local metrics tracking for the microservice
        self.metrics_lock = threading.Lock()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "provider_calls": {}, # tracks calls per physical model
            "fallback_events": 0,
        }
        
        self.usage_history = []
        self.routing_logs = []
        
        # Initialize Tiktoken for TPR (Tokens Per Request) estimation
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Tiktoken tokenizer initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken, falling back to character approximation: {e}")
            self.tokenizer = None

        # Register local Presidio PII Guardrail natively in LiteLLM callbacks
        self.pii_guardrail_callback = LocalPresidioPIIMasking(router=self)
        if self.pii_guardrail_callback not in litellm.callbacks:
            litellm.callbacks.append(self.pii_guardrail_callback)

        self.log_event("Class-based LiteLLM Proxy online. Load balancer initialized.", "routing")

    def log_event(self, message: str, type: str = "routing"):
        """Logs an event and pushes it to a thread-safe rolling buffer for the telemetry UI."""
        logger.info(message)
        with self.metrics_lock:
            self.routing_logs.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "message": message,
                "type": type
            })
            if len(self.routing_logs) > 50:
                self.routing_logs.pop(0)

    def _prune_usage_history(self, now: float):
        """Removes usage records older than 60 seconds."""
        cutoff = now - 60.0
        self.usage_history = [r for r in self.usage_history if r["timestamp"] > cutoff]

    def get_endpoint_usage(self, physical_model: str) -> Tuple[int, int]:
        """Returns the current (TPM, RPM) usage in the last 60 seconds for the given physical model."""
        now = time.time()
        with self.metrics_lock:
            self._prune_usage_history(now)
            tpm = 0
            rpm = 0
            for r in self.usage_history:
                if r["model"] == physical_model:
                    tpm += r["tokens"]
                    rpm += 1
            return tpm, rpm

    def get_fallbacks_for_model(self, virtual_model: str) -> List[str]:
        """Resolves target fallback clusters from the configuration."""
        fallbacks = []
        for fb_dict in self.config.general_fallbacks:
            if isinstance(fb_dict, dict) and virtual_model in fb_dict:
                fallbacks.extend(fb_dict[virtual_model])
        return fallbacks

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count of a given string using tiktoken or robust approximation."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: Llama/Mistral models average ~4 characters per token
        return max(1, len(text) // 4)

    def estimate_request_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimates the total token count of incoming chat messages."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            role = m.get("role", "")
            total += self.estimate_tokens(content) + self.estimate_tokens(role) + 4
        return total + 2 # overhead

    def shield_prompt_payload_reversible(self, text_content: str) -> Tuple[str, Dict[str, str]]:
        """
        Wrapper to route prompt shielding to the native LocalPresidioPIIMasking callback.
        """
        request_data = {"metadata": {}}
        sanitized = self.pii_guardrail_callback.shield_text(text_content, request_data)
        pii_tokens = request_data["metadata"].get("pii_tokens", {})
        return sanitized, pii_tokens

    def shield_prompt_payload(self, text_content: str) -> str:
        """
        Legacy wrapper for shield_prompt_payload_reversible.
        """
        sanitized, _ = self.shield_prompt_payload_reversible(text_content)
        return sanitized

    def restore_pii_content(self, text: str, pii_map: Dict[str, str]) -> str:
        """
        Restores raw PII values back into the assistant response content.
        Uses exact placeholder replacement.
        """
        request_data = {"metadata": {"pii_tokens": pii_map}}
        return self.pii_guardrail_callback.unmask_text(text, request_data)
    def classify_prompt_complexity(self, messages: List[Dict[str, str]], required_context: int) -> str:
        """
        Classifies prompt complexity into 'low', 'medium', or 'high' based on:
        1. Context window requirement (required_context)
        2. Semantic features (presence of reasoning/coding keywords, structural code, etc.)
        """
        # Feature 1: Context requirement
        if required_context > 8192:
            return "high"
        
        # Feature 2: Semantic check on the latest user message
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        latest_user_msg = user_msgs[-1] if user_msgs else ""
        
        # Keywords suggesting high complexity (reasoning, coding, architecture, deep math)
        high_complexity_keywords = [
            "code", "python", "javascript", "c++", "rust", "java", "html", "css", "sql", "git",
            "algorithm", "function", "refactor", "debug", "optimize", "regex", "database",
            "proof", "theorem", "math", "calculus", "derive", "solve", "equation",
            "analyze", "evaluate", "architecture", "design pattern", "system design",
            "compare and contrast", "step by step", "reasoning", "logical deduction"
        ]
        
        # Keywords suggesting medium complexity (formatting, summarizing, translations, drafting)
        medium_complexity_keywords = [
            "summar", "summary", "report", "translation", "translate", "synopsis", "outline", "draft", 
            "rewrite", "rephrase", "format", "extract", "list", "bullet points", "email",
            "explain", "what is", "how does"
        ]
        
        msg_lower = latest_user_msg.lower()
        
        # Count high-complexity indicators (keywords or code-like patterns)
        high_count = sum(1 for kw in high_complexity_keywords if kw in msg_lower)
        # Check for code blocks (```) or braces/indentation suggesting code
        if "```" in msg_lower or (msg_lower.count("{") > 2 and msg_lower.count("}") > 2) or "def " in msg_lower or "import " in msg_lower:
            high_count += 3
            
        medium_count = sum(1 for kw in medium_complexity_keywords if kw in msg_lower)
        
        if high_count >= 2 or (high_count >= 1 and required_context > 2048):
            return "high"
        elif medium_count >= 1 or required_context > 1024 or len(latest_user_msg) > 500:
            return "medium"
        else:
            return "low"

    def execute_chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes a chat completion request, applying local PII Guardrail shielding,
        TPM/TPR evaluation limits, and automatic multi-cluster routing.
        """
        with self.metrics_lock:
            self.metrics["total_requests"] += 1

        # 1. Local PII Shielding (Reversible / Native Guardrail Callbacks)
        sanitized_messages = []
        pii_redacted = False
        sandbox_request_data = {"metadata": {}}
        
        # Check if the guardrail is enabled in config
        pii_settings = getattr(self.config, "pii_shield_settings", None)
        pii_enabled = pii_settings.enabled if pii_settings else False
        
        # Determine early if the request is running in mock sandbox mode
        endpoints = self.config.get_endpoints_for_model(model)
        is_mock_early = (
            kwargs.get("mock_sandbox", False) or 
            (endpoints and endpoints[0].api_key and "mock" in endpoints[0].api_key.lower())
        )
        
        if is_mock_early:
            if pii_enabled:
                for msg in messages:
                    content = msg.get("content", "")
                    role = msg.get("role", "")
                    if role == "user":
                        sanitized_content = self.pii_guardrail_callback.shield_text(content, sandbox_request_data)
                        if sanitized_content != content:
                            pii_redacted = True
                        sanitized_messages.append({"role": role, "content": sanitized_content})
                    else:
                        sanitized_messages.append(msg)
                if pii_redacted:
                    placeholders = list(sandbox_request_data["metadata"]["pii_tokens"].keys())
                    self.log_event(f"[PII Shield] Sensitive information detected and redacted locally (Sandbox Mock). Placeholders: {placeholders}", "warning")
            else:
                sanitized_messages = [m.copy() for m in messages]
        else:
            # Real LLM calls let LiteLLM's native callback handle PII masking in-place during pre-call log hooks
            sanitized_messages = [m.copy() for m in messages]

        estimated_prompt_tokens = self.estimate_request_tokens(sanitized_messages)
        max_tokens = kwargs.get("max_tokens", 1000)
        required_context = estimated_prompt_tokens + max_tokens
        
        # Classify complexity
        complexity = self.classify_prompt_complexity(sanitized_messages, required_context)
        
        self.log_event(
            f"[Analysis] New request on '{model}'. Size: {estimated_prompt_tokens} prompt + {max_tokens} response = {required_context} required TPR. "
            f"Prompt Complexity classified as: '{complexity.upper()}'.", 
            "routing"
        )

        # 2. Select target endpoints and apply proactive TPR & TPM/RPM load-balancing/routing
        search_clusters = [model]
        search_clusters.extend(self.get_fallbacks_for_model(model))
        
        selected_cluster = model
        selected_endpoint = None
        is_fallback_triggered = False
        
        for cluster in search_clusters:
            endpoints = self.config.get_endpoints_for_model(cluster)
            if not endpoints:
                continue
                
            suitable_endpoints = []
            for ep in endpoints:
                # Check TPR
                if required_context > ep.tpr:
                    self.log_event(f"[TPR Limit] Node '{ep.model}' rejected: required context {required_context} exceeds limit {ep.tpr}.", "warning")
                    continue
                    
                # Check TPM & RPM
                current_tpm, current_rpm = self.get_endpoint_usage(ep.model)
                if current_tpm + required_context > ep.tpm:
                    self.log_event(f"[TPM Limit] Node '{ep.model}' rejected: current TPM {current_tpm} + required {required_context} exceeds limit {ep.tpm}.", "warning")
                    continue
                if current_rpm + 1 > ep.rpm:
                    self.log_event(f"[RPM Limit] Node '{ep.model}' rejected: current RPM {current_rpm} + 1 exceeds limit {ep.rpm}.", "warning")
                    continue
                    
                suitable_endpoints.append((ep, current_tpm, current_rpm))
                
            if suitable_endpoints:
                # Complexity-Aware and Cost-Aware Multi-Objective Load Balancing:
                # 1. Primary Objective: Minimize tier mismatch penalty (align with classified complexity)
                # 2. Secondary Objective: Minimize cost_per_million (least credit usage)
                # 3. Tertiary Objective: Minimize utilization (balanced resource usage)
                def get_suitability_score(item):
                    ep, t_used, r_used = item
                    
                    tier_map = {"low": 1, "medium": 2, "high": 3}
                    p_tier = tier_map.get(complexity, 2)
                    ep_tier = tier_map.get(ep.complexity_tier, 2)
                    
                    tier_mismatch = abs(p_tier - ep_tier)
                    
                    # Strong penalty if high complexity prompt is sent to a low reasoning node
                    if complexity == "high" and ep.complexity_tier == "low":
                        tier_mismatch += 5.0
                    # Strong penalty if low complexity prompt is sent to an expensive high-tier node
                    if complexity == "low" and ep.complexity_tier == "high":
                        tier_mismatch += 5.0
                        
                    util = max(t_used / ep.tpm, r_used / ep.rpm)
                    return (tier_mismatch, ep.cost_per_million, util)
                    
                selected_endpoint, t_used, r_used = min(suitable_endpoints, key=get_suitability_score)
                selected_cluster = cluster
                if cluster != model:
                    is_fallback_triggered = True
                    with self.metrics_lock:
                        self.metrics["fallback_events"] += 1
                    self.log_event(f"[Fallback Route] Overload/limit on '{model}'. Cascading to cluster '{cluster}'.", "warning")
                
                self.log_event(
                    f"[Complexity-Aware Selection] Node '{selected_endpoint.model}' selected in cluster '{cluster}' "
                    f"(Complexity Tier: {selected_endpoint.complexity_tier.upper()}, Cost: ${selected_endpoint.cost_per_million}/M tokens, TPM usage: {t_used}/{selected_endpoint.tpm}).", 
                    "success"
                )
                break
                
        # Relax TPR constraint if prompt is extremely large and exceeds all limits
        if not selected_endpoint:
            all_eps = []
            for cluster in search_clusters:
                all_eps.extend(self.config.get_endpoints_for_model(cluster))
            if all_eps:
                max_tpr = max(e.tpr for e in all_eps)
                best_eps = [e for e in all_eps if e.tpr == max_tpr]
                backup_eps = [e for e in best_eps if e.model_name == "backup-cluster"]
                selected_endpoint = backup_eps[0] if backup_eps else best_eps[0]
                selected_cluster = selected_endpoint.model_name
                if selected_cluster != model:
                    is_fallback_triggered = True
                    with self.metrics_lock:
                        self.metrics["fallback_events"] += 1
                self.log_event(f"[TPR Overlimit] Required {required_context} exceeds all limits. Escalating to highest capacity node: '{selected_endpoint.model}'.", "warning")
            else:
                raise ValueError(f"No active endpoints configured for model routing group '{model}'.")

        # Determine if we execute in Mock Sandbox Mode
        is_mock = (
            kwargs.get("mock_sandbox", False) or
            (selected_endpoint and selected_endpoint.api_key and "mock" in selected_endpoint.api_key.lower())
        )
        
        if is_mock:
            response = self._execute_mock_sandbox_completion(
                virtual_model=selected_cluster,
                endpoint=selected_endpoint,
                estimated_prompt_tokens=estimated_prompt_tokens,
                max_tokens=max_tokens,
                messages=sanitized_messages
            )
            response["prompt_complexity"] = complexity
            if pii_enabled:
                choices = response.get("choices", [])
                for choice in choices:
                    msg = choice.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        restored = self.pii_guardrail_callback.unmask_text(content, sandbox_request_data)
                        if restored != content:
                            msg["content"] = restored
                            self.log_event(f"[PII De-anonymizer] Mapped placeholders back in sandbox response choice.", "success")
            return response

        # Real API Execution via LiteLLM Router - directly let LITELLM route the model!
        try:
            self.log_event(f"[API Dispatch] Sending request to backend '{selected_endpoint.model}' in cluster '{selected_cluster}'...", "routing")
            start_time = time.time()
            
            # Clean kwargs
            litellm_kwargs = kwargs.copy()
            litmm_kwargs_to_pop = ["mock_sandbox", "fallbacks"]
            for k in litmm_kwargs_to_pop:
                litellm_kwargs.pop(k, None)

            # We pass the selected target cluster to the LiteLLM Router so it uses its config
            # and automatically selects the active backend under that cluster!
            response = self.router.completion(
                model=selected_cluster,
                messages=sanitized_messages,
                **litellm_kwargs
            )
            
            latency = time.time() - start_time
            actual_routed_model = response.get("model", selected_endpoint.model)
            self.log_event(f"[Success] API call completed by '{actual_routed_model}' in {latency:.2f}s.", "success")
            
            # Update metrics and usage history
            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", estimated_prompt_tokens)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
            with self.metrics_lock:
                self.usage_history.append({
                    "timestamp": time.time(),
                    "model": actual_routed_model,
                    "tokens": total_tokens
                })
            
            self._update_success_metrics(actual_routed_model, input_tokens, output_tokens)
            try:
                response["prompt_complexity"] = complexity
            except Exception:
                pass
            
            return response
            
        except Exception as e:
            self.log_event(f"[API Error] Backend '{selected_endpoint.model}' failed: {e}. Trying fallback execution chain...", "error")
            
            # Try immediate physical model failover!
            for fallback_cluster in search_clusters:
                if fallback_cluster == selected_cluster:
                    continue
                fallback_eps = self.config.get_endpoints_for_model(fallback_cluster)
                if not fallback_eps:
                    continue
                alt_ep = fallback_eps[0]
                
                try:
                    self.log_event(f"[Failover API Dispatch] Cascading to alternate backend '{alt_ep.model}' in '{fallback_cluster}'...", "warning")
                    start_time = time.time()
                    
                    response = self.router.completion(
                        model=fallback_cluster,
                        messages=sanitized_messages,
                        **litellm_kwargs
                    )
                    latency = time.time() - start_time
                    actual_routed_model = response.get("model", alt_ep.model)
                    
                    self.log_event(f"[Failover Success] Alternate backend '{actual_routed_model}' succeeded in {latency:.2f}s.", "success")
                    
                    usage = response.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", estimated_prompt_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    total_tokens = input_tokens + output_tokens
                    
                    with self.metrics_lock:
                        self.usage_history.append({
                            "timestamp": time.time(),
                            "model": actual_routed_model,
                            "tokens": total_tokens
                        })
                        self.metrics["fallback_events"] += 1
                        
                    self._update_success_metrics(actual_routed_model, input_tokens, output_tokens)
                    try:
                        response["prompt_complexity"] = complexity
                    except Exception:
                        pass
                    
                    return response
                    
                except Exception as ex:
                    self.log_event(f"[Failover API Error] Alternate backend '{alt_ep.model}' also failed: {ex}.", "error")
            
            with self.metrics_lock:
                self.metrics["failed_requests"] += 1
            raise RuntimeError(f"All backends in the routing chain failed. Last error: {e}")

    def _update_success_metrics(self, model: str, input_tokens: int, output_tokens: int):
        """Updates metrics safely across threads."""
        with self.metrics_lock:
            self.metrics["successful_requests"] += 1
            self.metrics["total_input_tokens"] += input_tokens
            self.metrics["total_output_tokens"] += output_tokens
            self.metrics["provider_calls"][model] = self.metrics["provider_calls"].get(model, 0) + 1

    def _execute_mock_sandbox_completion(
        self,
        virtual_model: str,
        endpoint: ModelEndpointConfig,
        estimated_prompt_tokens: int,
        max_tokens: int,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Simulates a model completion response instantly for local development and validation."""
        logger.info(f"[MOCK SANDBOX] Simulating completion for '{endpoint.model}'")
        time.sleep(0.1) # Simulate minimal network latency
        
        last_message = messages[-1].get("content", "") if messages else "Hello"
        mock_reply = (
            f"[LiteLLM Proxy Mock - {endpoint.model}]\n"
            f"Routing Group: {virtual_model}\n"
            f"Optimized TPR Context: {endpoint.tpr} max tokens.\n"
            f"Acknowledged request: '{last_message[:200]}...'"
        )
        
        output_tokens = self.estimate_tokens(mock_reply)
        total_tokens = estimated_prompt_tokens + output_tokens
        
        with self.metrics_lock:
            self.usage_history.append({
                "timestamp": time.time(),
                "model": endpoint.model,
                "tokens": total_tokens
            })
            
        self._update_success_metrics(endpoint.model, estimated_prompt_tokens, output_tokens)
        self.log_event(f"[Mock Success] Simulated reply from '{endpoint.model}' in cluster '{virtual_model}' (Tokens: {total_tokens}).", "success")
        
        # Structure the dict like an OpenAI / LiteLLM chat.completion response
        return {
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": endpoint.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": mock_reply
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": estimated_prompt_tokens + output_tokens
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Thread-safe getter for router metrics."""
        with self.metrics_lock:
            m = self.metrics.copy()
            m["logs"] = list(self.routing_logs)
            return m
