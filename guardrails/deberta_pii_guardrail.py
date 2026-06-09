from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
import asyncio
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm._logging import verbose_proxy_logger

SERVER_NER_MODEL       = "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2"
SERVER_OLLAMA_BASE_URL = "http://localhost:11434"
SERVER_OLLAMA_MODEL    = "mistral"
SERVER_THRESHOLD       = 0.4
SERVER_DEFAULT_ACTION  = "MASK"   # set None to make guardrail fully opt-in
SERVER_DEFAULT_POLICY  = {
    "credit card number":     "MASK",
    "social security number": "MASK",
    "passport number":        "MASK",
    "bank account number":    "MASK",
    "password":               "MASK",
    "api key":                "MASK",
    "person":                 "MASK",
    "email address":          "MASK",
    "phone number":           "MASK",
    "address":                "MASK",
}

@dataclass
class _Config:
    policy: Dict[str, str]
    default_action: Optional[str]
    threshold: float
    rewrite_model: str
    apply_to: str
    labels: List[str]

    def action_for(self, label: str) -> Optional[str]:
        return self.policy.get(label, self.default_action)

def _get_masked_and_adjusted(text: str, mask_ents: list, rewrite_ents: list) -> Tuple[str, list]:
    # All entities sorted ascending by start index
    all_ents = sorted(
        [(e, "mask") for e in mask_ents] + [(e, "rewrite") for e in rewrite_ents],
        key=lambda x: x[0]["start"]
    )
    
    current_shift = 0
    new_text_parts = []
    last_idx = 0
    
    adjusted_rewrite_ents = []
    
    for ent, ent_type in all_ents:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        
        if ent_type == "mask":
            new_text_parts.append(text[last_idx:start])
            token = f"[{label.upper().replace(' ', '_')}]"
            new_text_parts.append(token)
            current_shift += len(token) - (end - start)
            last_idx = end
        else:
            new_start = start + current_shift
            new_end = end + current_shift
            adjusted_rewrite_ents.append({
                "label": label,
                "start": new_start,
                "end": new_end,
                "text": ent.get("text") or text[start:end]
            })
            
    new_text_parts.append(text[last_idx:])
    base = "".join(new_text_parts)
    
    adjusted_rewrite_ents.sort(key=lambda e: e["start"], reverse=True)
    return base, adjusted_rewrite_ents


def canonicalize_label(model_label: str) -> str:
    l = model_label.upper().replace("_", "").replace(" ", "")
    if "FIRSTNAME" in l or "LASTNAME" in l or "SURNAME" in l or "NAME" in l or "PER" in l:
        return "person"
    if "EMAIL" in l:
        return "email address"
    if "PHONE" in l or "TELEPHONE" in l or "MOBILE" in l:
        return "phone number"
    if "CARD" in l or "ACCOUNTNUMBER" in l:  # AI4Privacy uses ACCOUNTNUMBER for credit card/IBAN
        return "credit card number"
    if "SSN" in l or "SOCIALSECURITY" in l:
        return "social security number"
    if "PASSPORT" in l:
        return "passport number"
    if "BANK" in l or "IBAN" in l:
        return "bank account number"
    if "ADDRESS" in l or "STREET" in l or "CITY" in l or "ZIP" in l or "LOC" in l:
        return "address"
    if "PASSWORD" in l:
        return "password"
    if "KEY" in l or "SECRET" in l:
        return "api key"
    return model_label.lower()


active_guardrails = []

def get_ner_model_path() -> str:
    import os
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "finetuned-deberta"))
    if os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    return "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2"

class DeBERTaPIIGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        if self not in active_guardrails:
            active_guardrails.append(self)

    def reload_model(self):
        verbose_proxy_logger.info("[DeBERTa PII] Reloading model pipeline...")
        self._model = None

    def should_run_guardrail(self, data, event_type) -> bool:
        res = super().should_run_guardrail(data, event_type)
        verbose_proxy_logger.debug(f"[DeBERTa PII] should_run_guardrail name={self.guardrail_name} event={event_type} res={res}")
        return res

    @property
    def model(self):
        if self._model is None:
            model_path = get_ner_model_path()
            verbose_proxy_logger.info(f"[DeBERTa PII] Loading {model_path} pipeline")
            from transformers import pipeline
            self._model = pipeline(
                "token-classification",
                model=model_path,
                aggregation_strategy="simple"
            )
            verbose_proxy_logger.info("[DeBERTa PII] Model ready.")
        return self._model

    def _get_config(self, data: dict) -> _Config:
        gc = data.get("guardrail_config")
        if gc is None:
            metadata = data.get("metadata") or data.get("litellm_metadata") or {}
            if isinstance(metadata, dict):
                gc = metadata.get("guardrail_config")
        if gc is None:
            gc = {}
        policy = gc.get("pii_policy", SERVER_DEFAULT_POLICY)
        default_action = gc.get("default_action", SERVER_DEFAULT_ACTION)
        threshold = gc.get("threshold", SERVER_THRESHOLD)
        rewrite_model = gc.get("rewrite_model", SERVER_OLLAMA_MODEL)
        apply_to = gc.get("apply_to", "both")
        labels = list(policy.keys())
        return _Config(
            policy=policy,
            default_action=default_action,
            threshold=threshold,
            rewrite_model=rewrite_model,
            apply_to=apply_to,
            labels=labels
        )

    def detect(self, text: str, cfg: _Config) -> list:
        if not text:
            return []
        
        results = self.model(text)
        entities = []
        for r in results:
            score = float(r.get("score", 0.0))
            if score < cfg.threshold:
                continue
                
            model_label = r.get("entity_group", "")
            label = canonicalize_label(model_label)
            
            if label in cfg.labels or cfg.default_action is not None:
                entities.append({
                    "label": label,
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "score": score,
                    "text": r.get("word", "")
                })
                
        return sorted(entities, key=lambda e: e["start"], reverse=True)

    def apply_mask(self, text: str, entities: list, cfg: _Config) -> str:
        for entity in entities:
            label = entity["label"]
            if cfg.action_for(label) == "MASK":
                start = entity["start"]
                end = entity["end"]
                token = f"[{label.upper().replace(' ', '_')}]"
                text = text[:start] + token + text[end:]
        return text

    async def apply_rewrite(self, text: str, entities: list, cfg: _Config) -> str:
        rewrite_ents = [e for e in entities if cfg.action_for(e["label"]) == "REWRITE"]
        if not rewrite_ents:
            return text
        
        pii_types = ", ".join(sorted({e["label"] for e in rewrite_ents}))
        payload = {
            "model": cfg.rewrite_model,
            "prompt": f"Rewrite the following text to remove all instances of: {pii_types}.\nRules:\n- Preserve the original meaning and tone exactly.\n- Replace each removed item with a natural generic placeholder (e.g. 'the user', 'a contact number', 'their address').\n- Return ONLY the rewritten text, no explanation.\n\nText:\n{text}",
            "stream": False
        }
        
        import httpx
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(f"{SERVER_OLLAMA_BASE_URL}/api/generate", json=payload)
                resp.raise_for_status()
                rewritten = resp.json()["response"].strip()
                return rewritten
        except Exception as exc:
            verbose_proxy_logger.warning(f"[DeBERTa PII] REWRITE failed ({exc}), falling back to MASK")
            masked_text = text
            for entity in rewrite_ents:
                label = entity["label"]
                start = entity["start"]
                end = entity["end"]
                token = f"[{label.upper().replace(' ', '_')}]"
                masked_text = masked_text[:start] + token + masked_text[end:]
            return masked_text

    def check_block(self, entities: list, cfg: _Config) -> Optional[str]:
        blocked = [e for e in entities if cfg.action_for(e["label"]) == "BLOCK"]
        if blocked:
            labels = ", ".join(sorted({e["label"] for e in blocked}))
            return f"PII policy violation: request blocked due to detected {labels}"
        return None

    async def process(self, text: str, cfg: _Config) -> Tuple[str, bool, Optional[str]]:
        if not text:
            return text, False, None
            
        entities = self.detect(text, cfg)
        if not entities:
            return text, False, None
            
        for ent in entities:
            label = ent["label"]
            ent_text = ent.get("text") or text[ent["start"]:ent["end"]]
            score = ent.get("score") or ent.get("probability", 0.0)
            action = cfg.action_for(label)
            verbose_proxy_logger.info(
                f"[DeBERTa PII] Detected entity: label={label}, text={ent_text}, score={score:.3f}, action={action}"
            )
            
        block_reason = self.check_block(entities, cfg)
        if block_reason:
            return text, False, block_reason
            
        mask_ents = [e for e in entities if cfg.action_for(e["label"]) == "MASK"]
        rewrite_ents = [e for e in entities if cfg.action_for(e["label"]) == "REWRITE"]
        
        if mask_ents and rewrite_ents:
            base, adjusted_rewrite_ents = _get_masked_and_adjusted(text, mask_ents, rewrite_ents)
            result = await self.apply_rewrite(base, adjusted_rewrite_ents, cfg)
        elif mask_ents:
            result = self.apply_mask(text, mask_ents, cfg)
        elif rewrite_ents:
            result = await self.apply_rewrite(text, rewrite_ents, cfg)
        else:
            result = text
            
        return result, result != text, None

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type,
    ):
        verbose_proxy_logger.debug(f"[DeBERTa PII] async_pre_call_hook called for {self.guardrail_name}")
        cfg = self._get_config(data)
        verbose_proxy_logger.debug(f"[DeBERTa PII] cfg: {cfg}")
        if cfg.apply_to not in ("input", "both"):
            return data
        
        messages = data.get("messages")
        if not messages:
            return data
        
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                cleaned, was_modified, block_reason = await self.process(content, cfg)
                verbose_proxy_logger.debug(f"[DeBERTa PII] process result: cleaned={cleaned}, was_modified={was_modified}, block_reason={block_reason}")
                if block_reason:
                    import litellm
                    raise litellm.exceptions.BadRequestError(
                        message=block_reason,
                        model=data.get("model", "unknown"),
                        llm_provider="guardrail"
                    )
                if was_modified:
                    msg["content"] = cleaned
                    role = msg.get("role", "unknown")
                    verbose_proxy_logger.info(f"[DeBERTa PII] pre_call sanitised role={role}")
        return data

    async def async_post_call_success_hook(
        self,
        data,
        user_api_key_dict,
        response,
    ):
        verbose_proxy_logger.debug(f"[DeBERTa PII] async_post_call_success_hook called for {self.guardrail_name}")
        try:
            cfg = self._get_config(data)
            if cfg.apply_to not in ("output", "both"):
                return response
            
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError):
                return response
            
            if not isinstance(content, str):
                return response
            
            cleaned, was_modified, block_reason = await self.process(content, cfg)
            if block_reason:
                response.choices[0].message.content = "[Response suppressed: model output contained blocked PII]"
                return response
            
            if was_modified:
                response.choices[0].message.content = cleaned
                verbose_proxy_logger.info("[DeBERTa PII] post_call sanitised response")
                
            return response
        except (AttributeError, IndexError):
            return response


# Dynamic registration for LiteLLM custom guardrail
try:
    from litellm.proxy.guardrails.guardrail_registry import guardrail_initializer_registry

    def initialize_guardrail(litellm_params, guardrail, llm_router=None):
        verbose_proxy_logger.info(f"[DeBERTa PII] initialize_guardrail called for {guardrail.get('guardrail_name')}")
        import litellm
        mode = litellm_params.mode
        default_on = litellm_params.default_on
        if hasattr(litellm_params, "model_dump"):
            extra_params = litellm_params.model_dump(exclude_none=True)
        else:
            extra_params = dict(litellm_params) if litellm_params else {}
        for key in ["guardrail", "mode", "default_on"]:
            extra_params.pop(key, None)
        
        cb = DeBERTaPIIGuardrail(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=mode,
            default_on=default_on,
            **extra_params
        )
        litellm.logging_callback_manager.add_litellm_callback(cb)
        return cb

    guardrail_initializer_registry["custom"] = initialize_guardrail
except ImportError:
    pass


from litellm.integrations.custom_logger import CustomLogger

class DeBERTaRegisterCallback(CustomLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

deberta_register_callback = DeBERTaRegisterCallback()


def get_model_label_ids(id2label: dict, canonicalize_fn, entity_label: str) -> Tuple[Optional[int], Optional[int]]:
    entity_label_clean = entity_label.upper().replace("_", "").replace(" ", "")
    candidates = []
    
    # First attempt: Match by canonicalized label name
    for idx, label in id2label.items():
        if label.startswith("B-"):
            suffix = label[2:]
            if canonicalize_fn(suffix) == entity_label:
                i_idx = None
                for idx2, label2 in id2label.items():
                    if label2 == f"I-{suffix}":
                        i_idx = idx2
                        break
                if i_idx is not None:
                    candidates.append((idx, i_idx))
                    
    # Second attempt: Match exactly by suffix name (e.g. "EMAIL", "FIRSTNAME")
    if not candidates:
        for idx, label in id2label.items():
            if label.startswith("B-"):
                suffix = label[2:]
                suffix_clean = suffix.upper().replace("_", "").replace(" ", "")
                if suffix_clean == entity_label_clean:
                    i_idx = None
                    for idx2, label2 in id2label.items():
                        if label2 == f"I-{suffix}":
                            i_idx = idx2
                            break
                    if i_idx is not None:
                        candidates.append((idx, i_idx))
                        
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0]
        
    return None, None


def train_deberta_model(
    dataset: List[Dict[str, Any]],
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 8
):
    """
    Supervised fine-tuning of the DeBERTa token classification model on custom PII samples.
    """
    import os
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification
    )
    
    base_model_name = "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2"
    
    # If a previous fine-tuned model exists, load from it to accumulate training
    if os.path.exists(os.path.join(output_dir, "config.json")):
        model_name_or_path = output_dir
        verbose_proxy_logger.info(f"[DeBERTa Training] Resuming training from local path: {model_name_or_path}")
    else:
        model_name_or_path = base_model_name
        verbose_proxy_logger.info(f"[DeBERTa Training] Loading base pre-trained model: {model_name_or_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    
    id2label = model.config.id2label
    
    tokenized_inputs = []
    
    for item in dataset:
        text = item.get("text", "")
        entities = item.get("entities", [])
        
        # Tokenize with offsets mapping
        encodings = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding=False
        )
        
        offset_mapping = encodings["offset_mapping"]
        labels = []
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                labels.append(-100) # Ignore special tokens in loss
                continue
                
            assigned_label_id = 0
            
            for ent in entities:
                ent_start = ent.get("start")
                ent_end = ent.get("end")
                ent_label = ent.get("label")
                
                # Check character boundaries overlap
                if start >= ent_start and end <= ent_end:
                    is_start = (start == ent_start)
                    b_id, i_id = get_model_label_ids(id2label, canonicalize_label, ent_label)
                    
                    if is_start and b_id is not None:
                        assigned_label_id = b_id
                    elif i_id is not None:
                        assigned_label_id = i_id
                    elif b_id is not None:
                        assigned_label_id = b_id
                    break
            
            labels.append(assigned_label_id)
            
        encodings["labels"] = labels
        encodings.pop("offset_mapping")
        tokenized_inputs.append(encodings)
        
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            return {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
            
        def __len__(self):
            return len(self.encodings)
            
    train_dataset = NERDataset(tokenized_inputs)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    import transformers
    from packaging import version
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator
    }
    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = Trainer(**trainer_kwargs)
    
    verbose_proxy_logger.info("[DeBERTa Training] Starting training arguments loop...")
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    verbose_proxy_logger.info(f"[DeBERTa Training] Saved fine-tuned checkpoint model to: {output_dir}")


if __name__ == "__main__":
    import logging
    # Set up simple logging for console runner
    logging.basicConfig(level=logging.INFO)

    async def run_tests():
        guardrail = DeBERTaPIIGuardrail()
        scenarios = [
            {
                "label": "REWRITE name+email, MASK card, BLOCK SSN",
                "text": "I'm Alice Chen, alice@corp.com, card 4111-1111-1111-1111, SSN 123-45-6789.",
                "pii_policy": {
                    "person": "REWRITE", "full name": "REWRITE", "email address": "REWRITE",
                    "credit card number": "MASK", "social security number": "BLOCK",
                },
                "default_action": "MASK",
            },
            {
                "label": "BLOCK all",
                "text": "Call me at 415-555-9087 or email bob@example.com.",
                "pii_policy": {"phone number": "BLOCK", "email address": "BLOCK"},
                "default_action": "BLOCK",
            },
            {
                "label": "MASK all",
                "text": "Ship to 123 Main St, passport A98765432, DOB 1990-03-15.",
                "pii_policy": {"address": "MASK", "passport number": "MASK", "date of birth": "MASK"},
                "default_action": "MASK",
            },
            {
                "label": "REWRITE all",
                "text": "My name is John Doe, I live at 45 Oak Ave, reach me at john@mail.com.",
                "pii_policy": {
                    "person": "REWRITE", "full name": "REWRITE",
                    "address": "REWRITE", "email address": "REWRITE",
                },
                "default_action": "REWRITE",
            },
            {
                "label": "No PII — clean pass-through",
                "text": "What is the boiling point of water?",
                "pii_policy": {},
                "default_action": "MASK",
            },
        ]
        
        for sc in scenarios:
            print(f"▶ {sc['label']}")
            print(f"  Input  : {sc['text']}")
            
            data = {
                "guardrail_config": {
                    "pii_policy": sc["pii_policy"],
                    "default_action": sc["default_action"],
                    "threshold": 0.3,
                }
            }
            cfg = guardrail._get_config(data)
            
            try:
                res_text, was_modified, block_reason = await guardrail.process(sc["text"], cfg)
                if block_reason:
                    print(f"  → BLOCKED: {block_reason}")
                elif was_modified:
                    print(f"  → Output : {res_text}")
                else:
                    print("  → No PII detected / passed through unchanged")
            except Exception as e:
                print(f"  → Error  : {e}")
            print()

    asyncio.run(run_tests())
