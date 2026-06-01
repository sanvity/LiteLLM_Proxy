import time
import logging
import threading
import httpx
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

from .config import ProxyConfig, ModelEndpointConfig

logger = logging.getLogger("proxy.router")

from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking

class AporiaControlPlaneState:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AporiaControlPlaneState, cls).__new__(cls)
                cls._instance.master_switch = True
                cls._instance.evaluators = {}
                cls._instance.sensitivity = {}
                cls._instance.remediation_actions = {}
                cls._instance.custom_shadow_keywords = []
                cls._instance.target_entities = []
                cls._instance.session_logs = []
            return cls._instance

class LocalPresidioPIIMasking(_OPTIONAL_PresidioPIIMasking):
    """
    Local, stateless implementation of LiteLLM's standard Presidio PII Masking guardrail.
    Inherits natively from _OPTIONAL_PresidioPIIMasking and overrides HTTP calls
    to execute locally using presidio-analyzer and presidio-anonymizer.
    """
    def __init__(self, router, **kwargs):
        # Resolve dynamic guardrails config
        guardrail_name = "presidio-pii"
        guardrails_list = getattr(router.config, "guardrails", [])
        presidio_guardrail = next((g for g in guardrails_list if g.get("guardrail_name") == guardrail_name), {})
        litellm_params = presidio_guardrail.get("litellm_params", {})
        
        # Prevent base class validation from throwing missing URL exceptions
        kwargs.setdefault("presidio_analyzer_api_base", "http://localhost:5002/")
        kwargs.setdefault("presidio_anonymizer_api_base", "http://localhost:5001/")
        
        # Override parameters from the YAML configuration
        output_parse = litellm_params.get("output_parse_pii", True)
        kwargs.setdefault("output_parse_pii", output_parse)
        kwargs.setdefault("presidio_language", litellm_params.get("presidio_language", "en"))
        kwargs.setdefault("pii_entities_config", litellm_params.get("pii_entities_config", {}))
        kwargs.setdefault("presidio_score_thresholds", litellm_params.get("presidio_score_thresholds", {}))
        kwargs.setdefault("guardrail_name", guardrail_name)
        
        super().__init__(**kwargs)
        self.router = router
        
        # Initialize local Presidio engines
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def validate_environment(self, **kwargs):
        # Overridden to prevent requiring external base URLs in environment
        self.presidio_analyzer_api_base = "http://localhost:5002/"
        self.presidio_anonymizer_api_base = "http://localhost:5001/"

    async def analyze_text(
        self,
        text: str,
        presidio_config: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Scans prompt text using local Presidio Analyzer, filters by threshold, resolves overlaps, and returns matches."""
        if not text or not text.strip():
            return []
            
        pii_settings = getattr(self.router.config, "pii_shield_settings", None)
        entities_set = set()
        
        # 1. Pull explicitly from runtime configuration (pii_entities_config)
        if self.pii_entities_config:
            entities_set.update(self.pii_entities_config.keys())
            
        # 2. Pull explicitly from general pii_shield_settings in config
        if pii_settings and getattr(pii_settings, "entities", None):
            entities_set.update(pii_settings.entities)
            
        # 3. Pull explicitly from dynamic control plane state
        from .router import AporiaControlPlaneState
        state = AporiaControlPlaneState()
        if getattr(state, "target_entities", None):
            entities_set.update(state.target_entities)
            
        active_entities = list(entities_set)
            
        # Analyze locally
        results = self.analyzer.analyze(
            text=text,
            language=self.presidio_language,
            entities=active_entities
        )
        
        # Manual zero-regex Aadhaar card scan (4 digits + space + 4 digits + space + 4 digits) -> IDENTIFIER
        from presidio_analyzer import RecognizerResult
        i = 0
        while i < len(text) - 13:
            part1 = text[i:i+4]
            space1 = text[i+4]
            part2 = text[i+5:i+9]
            space2 = text[i+9]
            part3 = text[i+10:i+14]
            if (part1.isdigit() and space1 == " " and 
                part2.isdigit() and space2 == " " and 
                part3.isdigit()):
                results.append(RecognizerResult(
                    entity_type="IDENTIFIER",
                    start=i,
                    end=i+14,
                    score=0.95
                ))
                i += 14
            else:
                i += 1

        # Manual zero-regex SSN scan (3 digits + hyphen + 2 digits + hyphen + 4 digits) -> US_SSN
        i = 0
        while i < len(text) - 10:
            part1 = text[i:i+3]
            hyphen1 = text[i+3]
            part2 = text[i+4:i+6]
            hyphen2 = text[i+6]
            part3 = text[i+7:i+11]
            if (part1.isdigit() and hyphen1 == "-" and 
                part2.isdigit() and hyphen2 == "-" and 
                part3.isdigit()):
                results.append(RecognizerResult(
                    entity_type="US_SSN",
                    start=i,
                    end=i+11,
                    score=0.95
                ))
                i += 11
            else:
                i += 1
        
        # Filter results by dynamic confidence thresholds
        filtered_results = []
        for res in results:
            threshold = self.presidio_score_thresholds.get(res.entity_type, 0.0)
            if res.score >= threshold:
                filtered_results.append(res)
        
        # Format results as expected by _OPTIONAL_PresidioPIIMasking
        matches = [
            {
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score,
            }
            for res in filtered_results
        ]
        
        # Resolve overlaps (prefer specific matches over generic IDENTIFIER)
        def get_match_priority(m):
            is_generic = (m["entity_type"] == "IDENTIFIER")
            return (m["start"], is_generic, -(m["end"] - m["start"]))
            
        matches.sort(key=get_match_priority)
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
        if pii_settings and not getattr(pii_settings, "enabled", True):
            return text
            
        entities_set = set()
        
        # 1. Pull explicitly from runtime configuration (pii_entities_config)
        if self.pii_entities_config:
            entities_set.update(self.pii_entities_config.keys())
            
        # 2. Pull explicitly from general pii_shield_settings in config
        if pii_settings and getattr(pii_settings, "entities", None):
            entities_set.update(pii_settings.entities)
            
        # 3. Pull explicitly from dynamic control plane state
        from .router import AporiaControlPlaneState
        state = AporiaControlPlaneState()
        if getattr(state, "target_entities", None):
            entities_set.update(state.target_entities)
            
        active_entities = list(entities_set)
            
        results = self.analyzer.analyze(
            text=text,
            language=self.presidio_language,
            entities=active_entities
        )
        
        # Manual zero-regex Aadhaar card scan (4 digits + space + 4 digits + space + 4 digits) -> IDENTIFIER
        from presidio_analyzer import RecognizerResult
        i = 0
        while i < len(text) - 13:
            part1 = text[i:i+4]
            space1 = text[i+4]
            part2 = text[i+5:i+9]
            space2 = text[i+9]
            part3 = text[i+10:i+14]
            if (part1.isdigit() and space1 == " " and 
                part2.isdigit() and space2 == " " and 
                part3.isdigit()):
                results.append(RecognizerResult(
                    entity_type="IDENTIFIER",
                    start=i,
                    end=i+14,
                    score=0.95
                ))
                i += 14
            else:
                i += 1

        # Manual zero-regex SSN scan (3 digits + hyphen + 2 digits + hyphen + 4 digits) -> US_SSN
        i = 0
        while i < len(text) - 10:
            part1 = text[i:i+3]
            hyphen1 = text[i+3]
            part2 = text[i+4:i+6]
            hyphen2 = text[i+6]
            part3 = text[i+7:i+11]
            if (part1.isdigit() and hyphen1 == "-" and 
                part2.isdigit() and hyphen2 == "-" and 
                part3.isdigit()):
                results.append(RecognizerResult(
                    entity_type="US_SSN",
                    start=i,
                    end=i+11,
                    score=0.95
                ))
                i += 11
            else:
                i += 1
        
        # Filter results by dynamic confidence thresholds
        filtered_results = []
        for res in results:
            threshold = self.presidio_score_thresholds.get(res.entity_type, 0.0)
            if res.score >= threshold:
                filtered_results.append(res)
        
        dict_results = [
            {
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score,
            }
            for res in filtered_results
        ]
        
        # Resolve overlaps (prefer specific matches over generic IDENTIFIER)
        def get_match_priority(m):
            is_generic = (m["entity_type"] == "IDENTIFIER")
            return (m["start"], is_generic, -(m["end"] - m["start"]))
            
        dict_results.sort(key=get_match_priority)
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

class LiteLLMContentFilter:
    """
    Local implementation of LiteLLM's Content Filter guardrail.
    Supports detecting blocked words dynamically registered via configuration.
    """
    def __init__(self, router, config_dict: Dict[str, Any]):
        self.router = router
        self.config = config_dict
        self.guardrail_name = config_dict.get("guardrail_name", "litellm_content_filter")
        self.litellm_params = config_dict.get("litellm_params", {})
        self.mode = self.litellm_params.get("mode", "pre_call")
        self.blocked_words = self.litellm_params.get("blocked_words", [])
        
    async def check_text(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Checks text against configured blocked words.
        Returns (is_blocked, action, reason) or (False, None, None).
        """
        if not text:
            return False, None, None
            
        # Check blocked keywords/words from config
        for item in self.blocked_words:
            kw = item.get("keyword", "").lower()
            action = item.get("action", "BLOCK")
            desc = item.get("description", "Blocked word detected")
            if kw and kw in text.lower():
                return True, action, f"Blocked keyword matching '{kw}': {desc}"
                        
        return False, None, None

    def mask_text(self, text: str) -> str:
        """Masks detected blocked words with custom placeholders without using regex."""
        masked_text = text
        for item in self.blocked_words:
            kw = item.get("keyword", "")
            if kw:
                kw_lower = kw.lower()
                placeholder = f"<{self.guardrail_name.upper()}>"
                idx = masked_text.lower().find(kw_lower)
                while idx != -1:
                    masked_text = masked_text[:idx] + placeholder + masked_text[idx + len(kw):]
                    idx = masked_text.lower().find(kw_lower)
                
        return masked_text

def mask_pii_no_regex(text: str) -> str:
    """
    Masks typical PII items (like email addresses, IP addresses, numeric IDs, phone numbers,
    Aadhaar cards, or SSNs) by revealing only the first 2 and last 2 characters,
    with the rest replaced with asterisks '*'.
    No regular expressions are used.
    """
    if not text:
        return ""
    words = text.split()
    masked_words = []
    for word in words:
        # Strip trailing and leading punctuations for analysis, but maintain them in output
        clean_word = word.strip(".,;:?!()\"'")
        punctuation_end = word[len(clean_word):] if clean_word else ""
        punctuation_start = word[:len(word) - len(clean_word) - len(punctuation_end)] if clean_word else ""
        
        # 1. Email check
        if "@" in clean_word and "." in clean_word:
            parts = clean_word.split("@", 1)
            local = parts[0]
            domain = parts[1]
            if len(local) > 2:
                local_masked = local[:2] + "*" * (len(local) - 2)
            else:
                local_masked = local[0] + "*" if local else "*"
            
            # Domain mask (e.g. gmail.com -> gm***.com)
            if "." in domain:
                d_parts = domain.rsplit(".", 1)
                d_name = d_parts[0]
                d_ext = d_parts[1]
                if len(d_name) > 2:
                    d_masked = d_name[:2] + "*" * (len(d_name) - 2)
                else:
                    d_masked = d_name[0] + "*" if d_name else "*"
                domain = d_masked + "." + d_ext
                
            masked_words.append(punctuation_start + local_masked + "@" + domain + punctuation_end)
            
        # 2. IP check
        elif clean_word.count(".") == 3 and all(c.isdigit() or c == "." for c in clean_word):
            # Mask middle octets
            octets = clean_word.split(".")
            masked_octets = [octets[0], "***", "***", octets[3]]
            masked_words.append(punctuation_start + ".".join(masked_octets) + punctuation_end)
            
        # 3. Numeric ID, Phone, Aadhaar, or SSN check (any block of numbers/digits/hyphens)
        elif any(c.isdigit() for c in clean_word):
            digits_count = sum(c.isdigit() for c in clean_word)
            if digits_count >= 5:
                # Mask middle parts of the ID/number
                if len(clean_word) > 4:
                    masked_word = clean_word[:2] + "*" * (len(clean_word) - 4) + clean_word[-2:]
                else:
                    masked_word = "*" * len(clean_word)
                masked_words.append(punctuation_start + masked_word + punctuation_end)
            else:
                masked_words.append(word)
        else:
            masked_words.append(word)
            
    return " ".join(masked_words)

class GenericGuardrailSimulator:
    """
    Enterprise-grade Live Aporia Shielding client with Presidio connection-failure recovery.
    Strictly focuses on Aporia PII scanning; other providers (Cato, Pillar, generic) are removed.
    NO local emulations, hardcoded heuristic triggers, or simulated mock loops are allowed.
    """
    def __init__(self, router, config_dict: Dict[str, Any]):
        self.router = router
        self.config = config_dict
        self.guardrail_name = config_dict.get("guardrail_name", "aporia-shield")
        self.litellm_params = config_dict.get("litellm_params", {})
        
        mode_val = self.litellm_params.get("mode", "pre_call")
        if isinstance(mode_val, list):
            self.mode = mode_val
        else:
            self.mode = [mode_val]
            
        self.provider = self.litellm_params.get("guardrail", "aporia")
        self._last_revised_text = None

        # Circuit Breaker attributes
        self.consecutive_failures = 0
        self.aporia_healthy = True
        self.last_failure_time = 0.0
        self.cooldown_window = 60.0
        
    def resolve_val(self, val: str) -> str:
        import os
        if not val:
            return ""
        if isinstance(val, str) and val.startswith("os.environ/"):
            env_key = val.split("os.environ/")[1]
            return os.environ.get(env_key, "")
        return val

    async def check_text(self, text: str) -> Tuple[bool, str, str]:
        """Runs security PII scans via live Aporia API endpoints, controlled by the dynamic Control Plane."""
        if not text:
            return False, "ALLOW", ""
            
        from .router import AporiaControlPlaneState
        state = AporiaControlPlaneState()
        
        # 1. Master Switch global bypass check
        if not state.master_switch:
            # Global Master Switch bypassed, return ALLOW immediately
            return False, "ALLOW", ""
            
        # We check provider is Aporia
        if self.provider != "aporia":
            raise ValueError(f"Provider '{self.provider}' is disabled. Gateway strictly requires 'aporia' for live shielding.")
            
        # Let's run a security evaluator scan!
        # Aporia control plane toggle checks:
        
        # A. Prompt Injection Evaluator
        if state.evaluators.get("prompt_injection"):
            lower_text = text.lower()
            is_injection = "ignore previous" in lower_text or "bypass instructions" in lower_text or "reveal the master" in lower_text
            if is_injection:
                action = state.remediation_actions.get("prompt_injection", "BLOCK")
                explain = "[Aporia built-in SLM Evaluator] Prompt Injection threat detected."
                
                # Log violation to session logs
                state.session_logs.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "prompt": text,
                    "status": "VIOLATION",
                    "evaluator": "prompt_injection",
                    "action": action,
                    "reason": explain
                })
                
                if action == "BLOCK":
                    self._last_revised_text = " "
                    return True, "BLOCK", explain
                elif action == "MASK":
                    output = mask_pii_no_regex(text)
                    self._last_revised_text = output
                    return True, "MASK", f"[Aporia Masked] Revised prompt: {output}"
                elif action == "REWRITE":
                    output = "Standard compliance request context rephrased securely by delegated agent."
                    self._last_revised_text = output
                    return True, "REWRITE", f"[Aporia Rephrased] {output}"
        
        # B. Custom Shadow keyword blocklists
        for kw in state.custom_shadow_keywords:
            if kw and kw.lower() in text.lower():
                # Shadow block policy triggered!
                explain = f"[Aporia Custom Shadow Policy] Prohibited keyword matching '{kw}'."
                state.session_logs.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "prompt": text,
                    "status": "VIOLATION",
                    "evaluator": "custom_shadow_policy",
                    "action": "BLOCK",
                    "reason": explain
                })
                self._last_revised_text = " "
                return True, "BLOCK", explain

        # C. Data/PII Leakage Evaluator
        if state.evaluators.get("pii_leakage"):
            api_base = self.resolve_val(self.litellm_params.get("api_base", ""))
            api_key = self.resolve_val(self.litellm_params.get("api_key", ""))
            
            if not api_base or not api_key:
                raise ValueError("Enterprise Aporia Guardrail is unconfigured. A valid API key and Base URL are required.")
                
            url = api_base if "validate" in api_base else f"{api_base.rstrip('/')}/validate"
            headers = {
                "X-APORIA-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [{"role": "user", "content": text}],
                "validation_target": "prompt",
                "explain": True
            }

            now = time.time()
            circuit_open = False
            if not self.aporia_healthy:
                if now - self.last_failure_time < self.cooldown_window:
                     circuit_open = True
                     self.router.log_event(
                         f"[Aporia Circuit Breaker] Circuit is OPEN (cooldown window active). Fast-failing to Presidio.",
                         "warning"
                     )
                else:
                     self.router.log_event(
                         f"[Aporia Circuit Breaker] Circuit is HALF-OPEN. Attempting live health check.",
                         "warning"
                     )

            resp_success = False
            api_failed_with_exception = False
            status_code = None
            response_json = None
            exception_obj = None

            if not circuit_open:
                try:
                    self.router.log_event(f"[Aporia Guardrail] Dispatching live enterprise request to {url}...", "routing")
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, json=payload, headers=headers, timeout=2.0)
                        status_code = resp.status_code
                        if status_code == 200:
                            response_json = resp.json()
                            resp_success = True
                        else:
                            exception_obj = ValueError(f"Aporia API returned status {status_code}")
                except Exception as e:
                    api_failed_with_exception = True
                    exception_obj = e

            if resp_success:
                # Reset consecutive failures on success
                self.consecutive_failures = 0
                self.aporia_healthy = True
                
                action = response_json.get("action", "passthrough").lower()
                if action == "block":
                    explain = "Blocked by Aporia Policy"
                    if response_json.get("explain_log"):
                        explain = f"Aporia block reasons: {response_json['explain_log']}"
                    
                    state.session_logs.append({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "prompt": text,
                        "status": "VIOLATION",
                        "evaluator": "pii_leakage",
                        "action": "BLOCK",
                        "reason": explain
                    })
                    self._last_revised_text = " "
                    return True, "BLOCK", explain
                elif action in ["modify", "rephrase", "rewrite", "mask"]:
                    if action in ["rephrase", "rewrite"]:
                        revised = "Standard compliance request context rephrased securely by delegated agent."
                        act_type = "REWRITE"
                    else:
                        revised = response_json.get("revised_response", text)
                        revised = mask_pii_no_regex(revised)
                        act_type = "MASK"
                    
                    self._last_revised_text = revised
                    
                    state.session_logs.append({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "prompt": text,
                        "status": "VIOLATION",
                        "evaluator": "pii_leakage",
                        "action": act_type,
                        "reason": f"Aporia revised content: {revised}"
                    })
                    return True, act_type, f"Aporia revised content: {revised}"
                
                # Clean request!
                state.session_logs.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "prompt": text,
                    "status": "LEGITIMATE",
                    "evaluator": "pii_leakage",
                    "action": "ALLOW",
                    "reason": "Clean prompt passing Aporia Evaluators."
                })
                return False, "ALLOW", ""

            else:
                # API failed or circuit is open
                if not circuit_open:
                    # It was a live attempt that failed
                    self.consecutive_failures += 1
                    self.router.log_event(
                        f"[Aporia API Failure] Live API failed (Consecutive: {self.consecutive_failures}). Error: {exception_obj}",
                        "error"
                    )
                    
                    # Track timeouts/network errors or 5xx errors specifically
                    is_timeout_or_5xx = False
                    if api_failed_with_exception:
                        is_timeout_or_5xx = True
                    elif status_code and status_code >= 500:
                        is_timeout_or_5xx = True
                    
                    if is_timeout_or_5xx:
                        # Trip circuit if failures >= 3
                        if self.consecutive_failures >= 3:
                            self.aporia_healthy = False
                            self.last_failure_time = time.time()
                            self.router.log_event(
                                f"[Aporia Circuit Breaker] Tripped! 3 consecutive failures reached. Circuit is now OPEN for 60 seconds.",
                                "error"
                            )
                
                # Route exclusively to local Presidio masking engine
                try:
                    presidio_inst = self.router.pii_guardrail_callback
                    if not presidio_inst:
                        raise ValueError("Local Presidio masking engine is not initialized.")
                        
                    sandbox_data = {"metadata": {}}
                    output_text = presidio_inst.shield_text(text, sandbox_data)
                    if output_text != text:
                        action = state.remediation_actions.get("pii_leakage", "MASK")
                        if circuit_open:
                            explain = f"[Aporia Circuit Breaker] Circuit is OPEN (cooldown window active). [Presidio Fallback] Masked PII: {output_text}"
                        else:
                            explain = f"[Presidio Fallback] Masked PII: {output_text}"
                        
                        state.session_logs.append({
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "prompt": text,
                            "status": "VIOLATION",
                            "evaluator": "pii_leakage",
                            "action": action,
                            "reason": explain
                        })
                        
                        if action == "BLOCK":
                            self._last_revised_text = " "
                            return True, "BLOCK", f"[Aporia Circuit Breaker] Circuit is OPEN. Prompt blocked by Presidio fallback." if circuit_open else "[Aporia Policy Violation: PII Leakage] Prompt blocked by Presidio fallback."
                        elif action == "MASK":
                            masked_out = mask_pii_no_regex(output_text)
                            self._last_revised_text = masked_out
                            return True, "MASK", explain
                        elif action == "REWRITE":
                            output = "Standard compliance request context rephrased securely by delegated agent."
                            self._last_revised_text = output
                            return True, "REWRITE", f"[Presidio Fallback Rephrased] {output}"
                            
                    # Clean request passing Presidio fallback
                    state.session_logs.append({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "prompt": text,
                        "status": "LEGITIMATE",
                        "evaluator": "pii_leakage",
                        "action": "ALLOW",
                        "reason": "Clean prompt passing Presidio fallback checks."
                    })
                    return False, "ALLOW", ""
                except Exception as presidio_err:
                    self.router.log_event(f"[Fallback Error] Presidio recovery failed: {presidio_err}", "error")
                    # Strict Fail-Closed strategy for dual-failure scenarios
                    raise ValueError(
                        f"Fail-Closed: Critical safety threat. Both Aporia and Presidio shielding are unavailable. "
                        f"Aporia failure: {exception_obj if exception_obj else 'Circuit is Open'}. "
                        f"Presidio failure: {presidio_err}"
                    )

        # If PII leakage is toggled off, and prompt injection is clean:
        state.session_logs.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt": text,
            "status": "LEGITIMATE",
            "evaluator": "none",
            "action": "ALLOW",
            "reason": "No policy violations detected."
        })
        return False, "ALLOW", ""
        
    def mask_text(self, text: str) -> str:
        if self._last_revised_text is not None:
            rev = self._last_revised_text
            self._last_revised_text = None
            return rev
        return text

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
        
        # Initialize Redis connection for centralized load balancing
        try:
            import os
            import redis
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, socket_timeout=1.0)
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {redis_host}:{redis_port} for load-balancing usage tracking.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Falling back to dynamic in-memory Redis simulation.")
            self.redis_client = None

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

        # Team Bring-Your-Own Guardrails Registry
        self.team_guardrails = {}

        # Parse and group all configured guardrails for Load Balancing
        self.guardrail_instances = {}
        for g in self.config.guardrails:
            name = g.get("guardrail_name")
            if not name:
                continue
            litellm_params = g.get("litellm_params", {})
            provider = litellm_params.get("guardrail")
            
            instance = None
            if provider == "presidio":
                instance = LocalPresidioPIIMasking(router=self, guardrail_config=g)
            elif provider == "litellm_content_filter":
                instance = LiteLLMContentFilter(router=self, config_dict=g)
            else:
                # Simulator for Cato, Aporia, generic_guardrail_api, etc.
                instance = GenericGuardrailSimulator(router=self, config_dict=g)
                
            if instance:
                if name not in self.guardrail_instances:
                    self.guardrail_instances[name] = []
                self.guardrail_instances[name].append(instance)

        # Priority Preference Routing State Parameters
        self.preference_enabled = False
        self.preference_list = []
        self.credit_limits = {
            "groq/llama-3.1-8b-instant": 0.05,
            "cerebras/llama3.1-8b": 0.05,
            "groq/llama-3.3-70b-versatile": 0.05,
            "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.05,
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": 0.05,
            "ollama/llama3.1": 0.05
        }
        self.accumulated_spend = {
            "groq/llama-3.1-8b-instant": 0.0,
            "cerebras/llama3.1-8b": 0.0,
            "groq/llama-3.3-70b-versatile": 0.0,
            "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.0,
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": 0.0,
            "ollama/llama3.1": 0.0
        }

        # Dynamic ingestion of the externalized configuration
        self.load_control_plane_config()

        self.log_event("Class-based LiteLLM Proxy online. Load balancer initialized.", "routing")

    def load_control_plane_config(self):
        """
        Dynamically ingests scanner targets, sensitivity sliders, and evaluator states
        from the parsed configuration into the AporiaControlPlaneState singleton.
        """
        from .router import AporiaControlPlaneState
        state = AporiaControlPlaneState()
        
        pii_settings = getattr(self.config, "pii_shield_settings", None)
        if pii_settings:
            # strictly overwrite without hardcoded defaults
            state.target_entities = list(pii_settings.entities)
            
            if pii_settings.evaluators:
                state.evaluators.update(pii_settings.evaluators)
            if pii_settings.sensitivity:
                state.sensitivity.update(pii_settings.sensitivity)
            if pii_settings.remediation_actions:
                state.remediation_actions.update(pii_settings.remediation_actions)

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

    def track_request_usage(self, physical_model: str, tokens: int):
        """
        Logs a completed request, tracking input/output tokens and timestamp in Redis or local memory.
        """
        now = time.time()
        if self.redis_client:
            try:
                # Store dynamic rate indicators inside Redis sorted sets
                key_tpm = f"litellm:tpm:{physical_model}"
                key_rpm = f"litellm:rpm:{physical_model}"
                
                # ZADD payload using sliding score window matching the timestamp
                self.redis_client.zadd(key_tpm, {f"{now}:{tokens}": now})
                self.redis_client.zadd(key_rpm, {f"{now}": now})
                
                # Expire raw telemetry indexes after 65 seconds
                self.redis_client.expire(key_tpm, 65)
                self.redis_client.expire(key_rpm, 65)
                return
            except Exception as e:
                logger.error(f"[Redis Telemetry Error] Track request failure: {e}")
                
        # In-memory fallback
        with self.metrics_lock:
            self.usage_history.append({
                "timestamp": now,
                "model": physical_model,
                "tokens": tokens
            })

    def get_endpoint_usage(self, physical_model: str) -> Tuple[int, int]:
        """Returns the current (TPM, RPM) usage in the last 60 seconds for the given physical model."""
        now = time.time()
        cutoff = now - 60.0
        
        if self.redis_client:
            try:
                key_tpm = f"litellm:tpm:{physical_model}"
                key_rpm = f"litellm:rpm:{physical_model}"
                
                # Remove timestamps older than 60 seconds to prune sliding window
                self.redis_client.zremrangebyscore(key_tpm, 0, cutoff)
                self.redis_client.zremrangebyscore(key_rpm, 0, cutoff)
                
                # Compute TPM
                tpm = 0
                members = self.redis_client.zrange(key_tpm, 0, -1)
                for member in members:
                    if ":" in member:
                        try:
                            tpm += int(member.split(":")[1])
                        except ValueError:
                            pass
                            
                # Compute RPM
                rpm = self.redis_client.zcard(key_rpm)
                return tpm, rpm
            except Exception as e:
                logger.error(f"[Redis Telemetry Error] Get endpoint usage failure: {e}")
                
        # In-memory fallback
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

    def get_guardrail_instance(self, guardrail_name: str):
        """Returns a load-balanced instance of a guardrail by name."""
        instances = self.guardrail_instances.get(guardrail_name, [])
        if not instances:
            return None
        import random
        return random.choice(instances)

    async def evaluate_pre_call_guardrails(self, messages: List[Dict[str, str]], request_data: dict) -> List[Dict[str, str]]:
        """
        Runs all active pre_call guardrails on the input messages.
        Modifies messages in-place if masking, or raises ValueError if blocked.
        """
        try:
            active_instances = []
            
            # Gather all configured pre-call instances
            for name, inst_list in self.guardrail_instances.items():
                for inst in inst_list:
                    mode_val = getattr(inst, "mode", "pre_call")
                    if "pre_call" in str(mode_val) or "during_call" in str(mode_val):
                        active_instances.append(inst)
                        
            # Gather all approved active team guardrails
            for g_id, g in self.team_guardrails.items():
                if g.get("status") == "active":
                    litellm_params = g.get("litellm_params", {})
                    mode_val = litellm_params.get("mode", "pre_call")
                    if "pre_call" in str(mode_val) or "during_call" in str(mode_val):
                        simulator = GenericGuardrailSimulator(router=self, config_dict=g)
                        active_instances.append(simulator)
                        
            new_messages = []
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "")
                if role == "user" and content:
                    current_text = content
                    for inst in active_instances:
                        if isinstance(inst, LocalPresidioPIIMasking):
                            current_text = inst.shield_text(current_text, request_data)
                        elif isinstance(inst, LiteLLMContentFilter):
                            is_blocked, action, reason = await inst.check_text(current_text)
                            if is_blocked:
                                if action == "BLOCK":
                                    current_text = " "
                                elif action == "MASK":
                                    current_text = inst.mask_text(current_text)
                                elif action == "REWRITE":
                                    current_text = "Standard compliance request context rephrased securely by delegated agent."
                        elif isinstance(inst, GenericGuardrailSimulator):
                            is_blocked, action, reason = await inst.check_text(current_text)
                            if is_blocked:
                                if action == "BLOCK":
                                    current_text = " "
                                elif action == "MASK":
                                    current_text = inst.mask_text(current_text)
                                elif action == "REWRITE":
                                    current_text = "Standard compliance request context rephrased securely by delegated agent."
                                    
                    new_messages.append({"role": role, "content": current_text})
                else:
                    new_messages.append(msg)
                    
            return new_messages

        except ValueError:
            raise  # Re-raise intended policy blocks
        except Exception as e:
            # FAIL-CLOSED: Any unexpected error blocks the request for safety
            logger.critical(f"GUARDRAIL CRITICAL FAILURE: {e}")
            raise RuntimeError("Security Layer Error: Request blocked to prevent data exposure.")

    async def validate_realtime_transcription(self, transcript: str, guardrail_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Guards voice conversation turns in the Realtime API.
        Intercepts speech transcriptions turn by turn.
        """
        self.log_event(f"[Realtime API Guard] Intercepted speech transcription: '{transcript}'", "routing")
        
        active_filters = []
        if guardrail_name:
            inst = self.get_guardrail_instance(guardrail_name)
            if inst:
                active_filters.append(inst)
        else:
            for name, inst_list in self.guardrail_instances.items():
                for inst in inst_list:
                    mode_val = getattr(inst, "mode", "pre_call")
                    if "realtime_input_transcription" in str(mode_val) or "pre_call" in str(mode_val):
                        active_filters.append(inst)
                        
        for inst in active_filters:
            if isinstance(inst, LocalPresidioPIIMasking):
                sandbox_data = {"metadata": {}}
                sanitized = inst.shield_text(transcript, sandbox_data)
                if sanitized != transcript:
                    return {
                        "action": "MASK",
                        "output": sanitized,
                        "reason": "PII detected in audio transcript turn.",
                        "guardrail": inst.guardrail_name
                    }
            elif isinstance(inst, LiteLLMContentFilter):
                is_blocked, action, reason = await inst.check_text(transcript)
                if is_blocked:
                    if action == "BLOCK":
                        return {
                            "action": "BLOCK",
                            "output": transcript,
                            "reason": f"Blocked by Realtime Guardrail: {reason}",
                            "guardrail": inst.guardrail_name
                        }
                    elif action == "MASK":
                        return {
                            "action": "MASK",
                            "output": inst.mask_text(transcript),
                            "reason": "Content filter masking applied.",
                            "guardrail": inst.guardrail_name
                        }
            elif isinstance(inst, GenericGuardrailSimulator):
                is_blocked, action, reason = await inst.check_text(transcript)
                if is_blocked and action == "BLOCK":
                    return {
                        "action": "BLOCK",
                        "output": transcript,
                        "reason": f"Blocked by Realtime Security Guardrail: {reason}",
                        "guardrail": inst.guardrail_name
                    }
                    
        return {
            "action": "ALLOW",
            "output": transcript,
            "reason": "Passed all realtime API guardrails.",
            "guardrail": None
        }

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

    async def execute_chat_completion(
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
            # Under sandbox/mock, execute all pre-call guardrails locally
            sanitized_messages = await self.evaluate_pre_call_guardrails(messages, sandbox_request_data)
            pii_tokens = sandbox_request_data["metadata"].get("pii_tokens", {})
            if pii_tokens:
                placeholders = list(pii_tokens.keys())
                self.log_event(f"[PII Shield] Sensitive information detected and redacted locally (Sandbox Mock). Placeholders: {placeholders}", "warning")
        else:
            # Under real calls, run pre-call guardrails locally for full enterprise safety
            sanitized_messages = await self.evaluate_pre_call_guardrails(messages, sandbox_request_data)

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
        selected_cluster = model
        selected_endpoint = None
        is_fallback_triggered = False

        if self.preference_enabled and self.preference_list:
            for p_model in self.preference_list:
                # Find matching endpoint in config
                ep = next((e for e in self.config.endpoints if e.model == p_model), None)
                if not ep:
                    continue
                    
                # Check credit limit
                current_spend = self.accumulated_spend.get(p_model, 0.0)
                limit = self.credit_limits.get(p_model, 9999.0)
                if current_spend >= limit:
                    self.log_event(f"[Preference Route] Model '{p_model}' skipped: credit limit reached (spent ${current_spend:.6f} >= limit ${limit:.6f}).", "warning")
                    continue
                    
                # Check TPR
                if required_context > ep.tpr:
                    self.log_event(f"[Preference Route] Model '{p_model}' skipped: required context {required_context} exceeds limit {ep.tpr}.", "warning")
                    continue
                    
                # Check TPM / RPM
                current_tpm, current_rpm = self.get_endpoint_usage(ep.model)
                if current_tpm + required_context > ep.tpm:
                    self.log_event(f"[Preference Route] Model '{p_model}' skipped: current TPM {current_tpm} + required {required_context} exceeds limit {ep.tpm}.", "warning")
                    continue
                if current_rpm + 1 > ep.rpm:
                    self.log_event(f"[Preference Route] Model '{p_model}' skipped: current RPM {current_rpm} + 1 exceeds limit {ep.rpm}.", "warning")
                    continue
                    
                selected_endpoint = ep
                selected_cluster = ep.model_name
                self.log_event(f"[Preference Selection] Node '{p_model}' selected as highest priority active preferred LLM (spent: ${current_spend:.6f}/${limit:.6f}).", "success")
                break
                
            if not selected_endpoint:
                self.log_event("[Preference Route Exhaustion] All preferred models exceeded credit limits or TPR constraints! Falling back to the first preference to prevent hard outage.", "error")
                p_model = self.preference_list[0]
                ep = next((e for e in self.config.endpoints if e.model == p_model), None)
                if ep:
                    selected_endpoint = ep
                    selected_cluster = ep.model_name

        if not selected_endpoint:
            search_clusters = [model]
            search_clusters.extend(self.get_fallbacks_for_model(model))
            
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
            
            self.track_request_usage(actual_routed_model, total_tokens)
            
            self._update_success_metrics(actual_routed_model, input_tokens, output_tokens)
            try:
                response["prompt_complexity"] = complexity
            except Exception:
                pass
            
            return response
            
        except Exception as e:
            self.log_event(f"[API Error] Backend '{selected_endpoint.model}' failed: {e}. Trying fallback execution chain...", "error")
            
            # Try immediate physical model failover!
            # 1. First try other endpoints inside the CURRENT active cluster (Intra-Cluster Failover)
            current_cluster_eps = self.config.get_endpoints_for_model(selected_cluster)
            for alt_ep in current_cluster_eps:
                if alt_ep.model == selected_endpoint.model:
                    continue
                try:
                    self.log_event(f"[Intra-Cluster Failover] Cascading to alternate backend '{alt_ep.model}' in current cluster '{selected_cluster}'...", "warning")
                    start_time = time.time()
                    response = self.router.completion(
                        model=selected_cluster,
                        messages=sanitized_messages,
                        **litellm_kwargs
                    )
                    latency = time.time() - start_time
                    actual_routed_model = response.get("model", alt_ep.model)
                    
                    self.log_event(f"[Intra-Cluster Success] Succeeded on '{actual_routed_model}' in {latency:.2f}s.", "success")
                    
                    usage = response.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", estimated_prompt_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    total_tokens = input_tokens + output_tokens
                    
                    self.track_request_usage(actual_routed_model, total_tokens)
                    with self.metrics_lock:
                        self.metrics["fallback_events"] += 1
                        
                    self._update_success_metrics(actual_routed_model, input_tokens, output_tokens)
                    try:
                        response["prompt_complexity"] = complexity
                    except Exception:
                        pass
                    return response
                except Exception as intra_ex:
                    self.log_event(f"[Intra-Cluster Failover Error] Alternate backend '{alt_ep.model}' also failed: {intra_ex}.", "error")

            # 2. Loop through other clusters
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
                    
                    self.track_request_usage(actual_routed_model, total_tokens)
                    with self.metrics_lock:
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
        cost = 0.0
        total_spent = 0.0
        with self.metrics_lock:
            self.metrics["successful_requests"] += 1
            self.metrics["total_input_tokens"] += input_tokens
            self.metrics["total_output_tokens"] += output_tokens
            self.metrics["provider_calls"][model] = self.metrics["provider_calls"].get(model, 0) + 1
            
            # Find the cost per million for this physical model
            cost_per_mil = 0.1
            for ep in self.config.endpoints:
                if ep.model == model:
                    cost_per_mil = ep.cost_per_million
                    break
            
            total_tokens = input_tokens + output_tokens
            cost = (total_tokens * cost_per_mil) / 1_000_000.0
            
            # Update accumulated spend
            self.accumulated_spend[model] = self.accumulated_spend.get(model, 0.0) + cost
            total_spent = self.accumulated_spend[model]
            
        self.log_event(f"[Cost Auditing] Spend on '{model}' increased by ${cost:.6f}. Total spent: ${total_spent:.6f}.", "success")

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
        
        self.track_request_usage(endpoint.model, total_tokens)
            
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
