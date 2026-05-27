import os
import yaml
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("proxy.config")

class ModelEndpointConfig(BaseModel):
    """
    Object-oriented definition of a specific physical LLM endpoint.
    Includes rate limits (TPM/RPM), context limits (TPR), and API configurations.
    """
    model_name: str = Field(..., description="Logical virtual model name (e.g. oss-chat-fast)")
    model: str = Field(..., description="Physical model path (e.g. groq/llama3-8b-8192)")
    api_key: Optional[str] = Field(None, description="API Key for the provider")
    api_base: Optional[str] = Field(None, description="Optional custom base URL for the API (e.g. for Ollama)")
    rpm: int = Field(default=1000, description="Requests Per Minute rate limit")
    tpm: int = Field(default=100000, description="Tokens Per Minute rate limit")
    tpr: int = Field(default=8192, description="Tokens Per Request (Context Window) limit")
    cost_per_million: float = Field(default=0.1, description="Average cost in USD per million tokens")
    max_tokens: Optional[int] = Field(None, description="Max response tokens allowed")
    complexity_tier: Optional[str] = Field(None, description="Complexity tier for this endpoint: low, medium, or high")

class PIIShieldSettings(BaseModel):
    """
    Configuration model for dynamic local PII masking and shielding.
    """
    enabled: bool = Field(default=True, description="Enable or disable local PII shielding")
    entities: List[str] = Field(
        default_factory=lambda: ["PERSON", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        description="Standard entity types to scan and redact using Presidio NLP"
    )
    custom_regex_rules: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of custom regex patterns to scan and redact"
    )

class ProxyConfig:
    """
    Loads, parses, and validates the LiteLLM Proxy configuration from YAML and Environment variables.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.endpoints: List[ModelEndpointConfig] = []
        self.pii_shield_settings: PIIShieldSettings = PIIShieldSettings()
        self.routing_strategy: str = "simple-shuffle"
        self.fallback_policy: str = "retry_next_suitable"
        self.num_retries: int = 3
        self.timeout: int = 10
        self.context_window_fallbacks: List[Dict[str, List[str]]] = []
        self.general_fallbacks: List[Dict[str, List[str]]] = []
        
        self.load_config()

    def load_config(self):
        """Reads config.yaml, resolves env vars, and populates OOP configurations."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                raw_data = yaml.safe_load(f) or {}

            # Parse general router settings
            router_settings = raw_data.get("router_settings", {})
            self.routing_strategy = router_settings.get("routing_strategy", "simple-shuffle")
            self.fallback_policy = router_settings.get("fallback_policy", "retry_next_suitable")
            self.num_retries = int(router_settings.get("num_retries", 3))
            self.timeout = int(router_settings.get("timeout", 10))

            # Parse fallbacks
            self.context_window_fallbacks = raw_data.get("context_window_fallbacks", [])
            self.general_fallbacks = raw_data.get("general_fallbacks", [])

            # Parse PII Shield settings
            pii_raw = raw_data.get("pii_shield_settings", {})
            self.pii_shield_settings = PIIShieldSettings(
                enabled=pii_raw.get("enabled", True),
                entities=pii_raw.get("entities", ["PERSON", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]),
                custom_regex_rules=pii_raw.get("custom_regex_rules", [])
            )

            # Parse model list
            model_list = raw_data.get("model_list", [])
            self.endpoints = []
            
            for m in model_list:
                model_name = m.get("model_name")
                litellm_params = m.get("litellm_params", {})
                
                # Resolve env vars in model params
                raw_model = litellm_params.get("model", "")
                raw_key = litellm_params.get("api_key", "")
                raw_base = litellm_params.get("api_base", "")
                
                api_key = None
                if raw_key:
                    if raw_key.startswith("os.environ/"):
                        env_var_name = raw_key.replace("os.environ/", "")
                        api_key = os.environ.get(env_var_name)
                    else:
                        api_key = raw_key

                api_base = None
                if raw_base:
                    if raw_base.startswith("os.environ/"):
                        env_var_name = raw_base.replace("os.environ/", "")
                        api_base = os.environ.get(env_var_name)
                    else:
                        api_base = raw_base

                cost = litellm_params.get("cost_per_million", 0.1)
                tpr = litellm_params.get("tpr", 8192)
                complexity_tier = litellm_params.get("complexity_tier")
                
                # Intelligent tier fallback if not explicitly provided
                if not complexity_tier:
                    if cost >= 0.50:
                        complexity_tier = "high"
                    elif cost >= 0.04:
                        complexity_tier = "medium"
                    else:
                        complexity_tier = "low"

                endpoint = ModelEndpointConfig(
                    model_name=model_name,
                    model=raw_model,
                    api_key=api_key,
                    api_base=api_base,
                    rpm=litellm_params.get("rpm", 1000),
                    tpm=litellm_params.get("tpm", 100000),
                    tpr=tpr,
                    cost_per_million=cost,
                    max_tokens=litellm_params.get("max_tokens"),
                    complexity_tier=complexity_tier
                )
                self.endpoints.append(endpoint)
                
            logger.info(f"Loaded {len(self.endpoints)} endpoints from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise e

    def get_endpoints_for_model(self, virtual_model_name: str) -> List[ModelEndpointConfig]:
        """Returns physical endpoints matching the requested virtual model name."""
        return [e for e in self.endpoints if e.model_name == virtual_model_name]

    def to_litellm_model_list(self) -> List[Dict[str, Any]]:
        """Converts OOP configuration list to the dictionary format expected by LiteLLM Router."""
        litellm_list = []
        for e in self.endpoints:
            params = {
                "model": e.model,
                "rpm": e.rpm,
                "tpm": e.tpm
            }
            if e.api_key:
                params["api_key"] = e.api_key
            if e.api_base:
                params["api_base"] = e.api_base
            if e.max_tokens:
                params["max_tokens"] = e.max_tokens
                
            litellm_list.append({
                "model_name": e.model_name,
                "litellm_params": params
            })
        return litellm_list
