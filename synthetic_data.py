import os
import re
import json
import random
import logging
import math
import time
import difflib
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
import litellm
import requests
from dotenv import load_dotenv
from faker import Faker
import exrex

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthetic_data")

class DatasetList(list):
    """Subclass of list that holds a validation report for backward compatibility."""
    def __init__(self, items: List[Dict[str, Any]], report: Dict[str, Any]):
        super().__init__(items)
        self.report = report

# 1. Default fallback patterns and domains for standard labels
DEFAULT_PII_CONFIGS = {
    "person": {
        "target_label": "person",
        "pattern_val": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        "data_format": "HR records / employee lists"
    },
    "email address": {
        "target_label": "email address",
        "pattern_val": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "data_format": "Tech/Support communications"
    },
    "phone number": {
        "target_label": "phone number",
        "pattern_val": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "data_format": "Customer contact logs"
    },
    "address": {
        "target_label": "address",
        "pattern_val": r"\b\d+\s+[A-Za-z0-9\s,.]+ St(?:reet)?|Ave(?:nue)?|Rd|Road\b",
        "data_format": "Legal contracts / shipping invoices"
    },
    "credit card number": {
        "target_label": "credit card number",
        "pattern_val": r"\b(?:\d[ -]*?){13,16}\b",
        "data_format": "Banking invoices / payment summaries"
    },
    "social security number": {
        "target_label": "social security number",
        "pattern_val": r"\b\d{3}-\d{2}-\d{4}\b",
        "data_format": "HR tax documentation"
    },
    "passport number": {
        "target_label": "passport number",
        "pattern_val": r"\b[A-Z0-9]{6,9}\b",
        "data_format": "Travel booking itineraries"
    },
    "bank account number": {
        "target_label": "bank account number",
        "pattern_val": r"\b\d{9,18}\b",
        "data_format": "Banking transactions"
    },
    "password": {
        "target_label": "password",
        "pattern_val": r"\b[a-zA-Z0-9!@#$%^&*()_+]{8,20}\b",
        "data_format": "Access logs / credentials"
    },
    "api key": {
        "target_label": "api key",
        "pattern_val": r"\b[a-zA-Z0-9_-]{16,40}\b",
        "data_format": "Tech/Support API logs"
    }
}

# 2. Schema definition for batching validation
class CompactEntity(BaseModel):
    l: str = Field(..., description="l: label")
    s: int = Field(..., description="s: start index")
    e: int = Field(..., description="e: end index")
    v: Optional[str] = Field(None, description="v: value substring")

class CompactSample(BaseModel):
    t: str = Field(..., description="t: text")
    d: str = Field(default="Unknown", description="d: domain")
    y: str = Field(default="Clean / Standard Sentence", description="y: style")
    e: List[CompactEntity] = Field(default_factory=list, description="e: entities list")

# 3. Dynamic UI-Driven Synthetic Data Engine
class SyntheticDataEngine:
    def __init__(self):
        # Load environment variables
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(dotenv_path, override=True)
        self.fake = Faker()
        self.last_health_check_time = 0.0
        self.is_proxy_healthy = False

    def is_near_duplicate(self, sentence: str, existing: List[str], threshold: float = 0.85) -> bool:
        """Checks if a carrier sentence is a near-duplicate of already accepted sentences."""
        clean_sent = sentence.replace("[PII]", "").strip().lower()
        for old in existing:
            clean_old = old.replace("[PII]", "").strip().lower()
            ratio = difflib.SequenceMatcher(None, clean_sent, clean_old).ratio()
            if ratio > threshold:
                return True
        return False

    def _clean_pattern_for_exrex(self, pattern: str) -> str:
        """Strips word boundaries and anchors to allow exrex parsing."""
        clean = pattern.replace(r"\b", "")
        if clean.startswith("^"):
            clean = clean[1:]
        if clean.endswith("$"):
            clean = clean[:-1]
        return clean

    def generate_regex_value(self, pattern: str) -> str:
        """Generates a value matching the custom regex pattern using exrex, with validation and retry."""
        if not pattern:
            return str(random.randint(100000000000, 999999999999))

        clean_pattern = self._clean_pattern_for_exrex(pattern)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
            
        for attempt in range(10):
            try:
                val = exrex.getone(clean_pattern)
                if re.fullmatch(compiled, val):
                    return val
            except Exception:
                pass
                
        raise RuntimeError(f"Failed to generate a valid string matching regex '{pattern}' after 10 attempts.")

    def perturb_value(self, val: str, regex_pattern: str) -> str:
        """Perturbs a valid regex‑matching value so it becomes a hard negative (distractor)."""
        if not regex_pattern:
            return val + "0"
            
        try:
            compiled = re.compile(regex_pattern)
        except Exception:
            return val + "XYZ"
            
        chars = list(val)
        for _ in range(30):
            if not chars:
                return "distractor_value"
            strategy = random.choice([1, 2, 3, 4])
            if strategy == 1:
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = random.choice("ABCXYZ987")
            elif strategy == 2:
                idx = random.randint(0, len(chars) - 1)
                chars.pop(idx)
            elif strategy == 3:
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice("-/_*#"))
            elif strategy == 4:
                for idx, c in enumerate(chars):
                    if c.isdigit():
                        chars[idx] = random.choice("JKL")
                        break
            perturbed = "".join(chars)
            if not re.fullmatch(compiled, perturbed) and perturbed.strip():
                return perturbed
        return val + "_invalid"

    def _generate_fake_value(self, entity_name: str, regex_pattern: str, is_hard_negative: bool = False) -> str:
        """Generates a realistic mock value (positive or hard negative) based on instructions."""
        # Never fall back to Faker's semantic guesses when a custom regex is supplied
        if regex_pattern:
            val = self.generate_regex_value(regex_pattern)
            if is_hard_negative:
                val = self.perturb_value(val, regex_pattern)
            return val
            
        name_lower = entity_name.lower().replace("_", " ").strip()
        val = ""
        if name_lower in ("person", "name", "patient", "client", "user", "owner"):
            val = self.fake.name()
        elif name_lower in ("email", "email address", "email_address"):
            val = self.fake.email()
        elif name_lower in ("phone", "phone number", "phone_number", "mobile", "contact"):
            val = self.fake.phone_number()
        elif name_lower in ("address", "location", "street", "city"):
            val = self.fake.address().replace("\n", ", ")
        elif name_lower in ("password", "passcode", "pwd", "pin"):
            val = self.fake.password(length=12, special_chars=True, digits=True, upper_case=True, lower_case=True)
        elif name_lower in ("credit card", "credit card number", "credit_card_number", "cc", "card"):
            val = self.fake.credit_card_number()
        elif name_lower in ("bank account", "bank account number", "bank_account_number", "iban", "bban"):
            val = self.fake.bban()
        elif name_lower in ("ssn", "social security number", "social_security_number"):
            val = self.fake.ssn()
        elif name_lower in ("api key", "api_key", "apikey", "secret"):
            val = "sk-live-" + self.fake.md5()[:16]
        else:
            val = str(random.randint(100000000000, 999999999999))
            
        if is_hard_negative:
            val = val + "_neg"
            
        return val

    def check_proxy_health(self, proxy_url: str) -> bool:
        """Caches proxy health checks for 30 seconds to speed up generation."""
        current_time = time.time()
        if current_time - self.last_health_check_time > 30.0:
            self.last_health_check_time = current_time
            try:
                r = requests.get(f"{proxy_url}/health", timeout=1.5)
                self.is_proxy_healthy = (r.status_code == 200)
            except Exception:
                self.is_proxy_healthy = False
        return self.is_proxy_healthy

    def _generate_batch(
        self,
        batch_idx: int,
        entity_name: str,
        regex_pattern: str,
        domain: str,
        model: str,
        proxy_url: str,
        is_hard_negative: bool = False,
        batch_size: int = 15,
        similarity_threshold: float = 0.85,
        opt_in_cross_label: bool = False
    ) -> Dict[str, Any]:
        """Generates a single batch of samples using the dynamic UI prompt and Factory placeholders."""
        # 1. Diversify domains and few-shots
        few_shots = [
            'Example sentence: "Please reset the passcode to [PII] for client account."',
            'Example sentence: "The employee profile states that the contact phone is [PII]."',
            'Example sentence: "System alert: failed login attempt with token [PII] detected."',
            'Example sentence: "Draft clause: The contracting party\'s address is [PII] for all official notifications."',
            'Example sentence: "Transaction record notes a payment of $450 routed to card ending in [PII]."',
            'Example sentence: "User query: Can you send the results to [PII] directly?"'
        ]
        few_shot = random.choice(few_shots)
        
        # 2. Dynamic prompt constraints (formality, position, length)
        system_prompt = (
            "You are a high-performance synthetic data generator. Your task is to generate natural text sentences for training a Named Entity Recognition (NER) model. "
            f"Return ONLY a valid JSON list of {batch_size} strings containing the generated sentences. Do not perform any labeling. "
            "No conversational filler or markdown blocks. The JSON array must start with [ and end with ]."
        )
        
        if entity_name.upper() == "NEUTRAL":
            user_prompt = (
                f"Generate exactly {batch_size} natural sentences based on the domain: {domain}.\n"
                "The sentences must contain NO Personally Identifiable Information (PII) and no placeholders.\n"
                "You MUST use high lexical variety, different sentence lengths, and diverse structures.\n"
                f"Return only a raw JSON array of {batch_size} strings."
            )
        else:
            pos = random.choice(["start", "middle", "end"])
            pos_instruct = {
                "start": "Place the placeholder '[PII]' near the beginning of each sentence.",
                "middle": "Place the placeholder '[PII]' in the middle of each sentence.",
                "end": "Place the placeholder '[PII]' near the end of each sentence."
            }[pos]
            
            formality = random.choice(["casual/informal", "highly formal", "strictly technical", "neutral/standard"])
            length = random.choice(["short (under 8 words)", "medium length", "long and descriptive (over 20 words)"])
            
            user_prompt = (
                f"Generate exactly {batch_size} natural sentences based on the domain: {domain}.\n"
                f"In each sentence, you MUST include the literal placeholder '[PII]' exactly where a {entity_name} would naturally appear.\n"
                f"Sentence constraints:\n"
                f"- Style/Formality: {formality}\n"
                f"- Length: {length}\n"
                f"- Position: {pos_instruct}\n"
                f"- Lexical Variety: Ensure sentence structures are highly diverse. DO NOT repeat the same template, grammatical format, or sentence starters.\n"
                "DO NOT include actual names, values, or numbers for the target entity—use ONLY the placeholder '[PII]'.\n"
                "DO NOT label the entities or output any indices.\n"
                f"Return only a raw JSON array of {batch_size} strings.\n\n"
                f"Rotate and learn from this example:\n"
                f"{few_shot}"
            )

        # 3. Robust completion loop with cached health check & rate-limit retries
        max_retries = 5
        content = ""
        
        for attempt in range(max_retries):
            use_proxy_api = self.check_proxy_health(proxy_url)

            if use_proxy_api:
                try:
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "bypass_guardrails": True
                    }
                    r = requests.post(f"{proxy_url}/v1/chat/completions", json=payload, timeout=45.0)
                    if r.status_code == 200:
                        content = r.json()["choices"][0]["message"]["content"].strip()
                        break
                    elif r.status_code in (429, 500, 503):
                        err_msg = r.text
                        if "rate_limit" in err_msg.lower() or "limit reached" in err_msg.lower() or r.status_code == 429:
                            sleep_time = 3.0 * (attempt + 1)
                            logger.warning(f"Batch {batch_idx} hit proxy rate limit. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(sleep_time)
                            continue
                        else:
                            raise RuntimeError(f"Proxy status code {r.status_code}: {err_msg}")
                    else:
                        raise RuntimeError(f"Proxy status code {r.status_code}")
                except Exception as proxy_err:
                    err_str = str(proxy_err).lower()
                    if "rate_limit" in err_str or "limit reached" in err_str or "429" in err_str:
                        sleep_time = 3.0 * (attempt + 1)
                        logger.warning(f"Batch {batch_idx} proxy call hit rate limit. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    logger.warning(f"Batch {batch_idx} proxy call failed: {proxy_err}. Falling back to direct call.")
                    use_proxy_api = False

            if not use_proxy_api:
                direct_model = model
                if model == "primary-cluster":
                    direct_model = "groq/llama-3.1-8b-instant"
                elif model == "backup-cluster":
                    direct_model = "groq/llama-3.1-8b-instant"

                try:
                    response = litellm.completion(
                        model=direct_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    content = response.choices[0].message.content.strip()
                    break
                except Exception as direct_err:
                    err_str = str(direct_err).lower()
                    if "rate_limit" in err_str or "limit reached" in err_str or "429" in err_str:
                        sleep_time = 3.0 * (attempt + 1)
                        logger.warning(f"Batch {batch_idx} direct call hit rate limit. Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise direct_err

        if not content:
            raise RuntimeError(f"Batch {batch_idx} failed to generate content after {max_retries} attempts.")
            
        # Robust JSON extraction
        json_array_match = re.search(r"(\[.*\])", content, re.DOTALL)
        if json_array_match:
            content = json_array_match.group(1).strip()
        else:
            json_obj_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if json_obj_match:
                content = "[" + json_obj_match.group(1).strip() + "]"
            else:
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n?", "", content)
                    content = re.sub(r"\n?```$", "", content)
                content = content.strip()
                
        polished_samples = []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                polished_samples = [str(x) for x in parsed]
            elif isinstance(parsed, dict):
                polished_samples = [str(x) for x in parsed.values()]
        except Exception:
            lines = []
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("```") or line.startswith("[") or line.endswith("]"):
                    continue
                line = re.sub(r"^\d+[\s.)-]+\s*", "", line)
                if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                    line = line[1:-1].strip()
                lines.append(line)
            polished_samples = lines[:batch_size]

        if not polished_samples:
            raise ValueError("Polisher did not return any valid sentences.")
            
        # 4. Near-Duplicate Filtering (difflib)
        accepted_sentences = []
        for sentence in polished_samples:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
            if self.is_near_duplicate(sentence_clean, accepted_sentences, similarity_threshold):
                continue
            accepted_sentences.append(sentence_clean)
            
        dup_count = len(polished_samples) - len(accepted_sentences)
        validated_samples = []
        
        for sentence in accepted_sentences:
            if entity_name.upper() == "NEUTRAL":
                validated_samples.append({
                    "domain": domain,
                    "style": "Clean / Standard Sentence",
                    "text": sentence,
                    "entities": [],
                    "is_hard_negative": False
                })
                continue
                
            text = sentence
            entities = []
            
            # Step 2 & 3: Placeholders Injection and exact index calculation
            while "[PII]" in text:
                idx = text.find("[PII]")
                fake_val = self._generate_fake_value(entity_name, regex_pattern, is_hard_negative)
                text = text[:idx] + fake_val + text[idx + 5:]
                
                if not is_hard_negative:
                    entities.append({
                        "label": entity_name.lower(),
                        "start": idx,
                        "end": idx + len(fake_val),
                        "value": fake_val
                    })
                
            if not entities and not is_hard_negative:
                # Healing: Append the fake value if placeholder was missing
                fake_val = self._generate_fake_value(entity_name, regex_pattern, is_hard_negative)
                insert_pos = len(text)
                if text and not text.endswith(" "):
                    text += " "
                    insert_pos += 1
                text += fake_val
                entities.append({
                    "label": entity_name.lower(),
                    "start": insert_pos,
                    "end": insert_pos + len(fake_val),
                    "value": fake_val
                })
            elif not entities and is_hard_negative:
                # Hard negative healing (append only, no label)
                fake_val = self._generate_fake_value(entity_name, regex_pattern, is_hard_negative)
                if text and not text.endswith(" "):
                    text += " "
                text += fake_val
                
            # Scan for other default PII classes (Opt-in only)
            if opt_in_cross_label:
                for other_label, config in DEFAULT_PII_CONFIGS.items():
                    if other_label.lower() == entity_name.lower():
                        continue
                    pat_str = config.get("pattern_val")
                    if pat_str:
                        try:
                            other_pat = re.compile(pat_str)
                            for match in other_pat.finditer(text):
                                entities.append({
                                    "label": other_label.lower(),
                                    "start": match.start(),
                                    "end": match.end(),
                                    "value": match.group()
                                })
                        except Exception:
                            pass
                            
            # Overlap pruning with prioritization of primary target entity label
            entities.sort(key=lambda x: x["start"])
            clean_entities = []
            for ent in entities:
                if not clean_entities:
                    clean_entities.append(ent)
                else:
                    last = clean_entities[-1]
                    if ent["start"] >= last["end"]:
                        clean_entities.append(ent)
                    else:
                        # Prioritize target entity label over secondary matches
                        if ent["label"] == entity_name.lower() and last["label"] != entity_name.lower():
                            clean_entities[-1] = ent
                        elif last["label"] == entity_name.lower() and ent["label"] != entity_name.lower():
                            continue
                        elif (ent["end"] - ent["start"]) > (last["end"] - last["start"]):
                            clean_entities[-1] = ent
                                
            validated_samples.append({
                "domain": domain,
                "style": "Clean / Standard Sentence",
                "text": text,
                "entities": clean_entities,
                "is_hard_negative": is_hard_negative
            })
            
        return {"samples": validated_samples, "near_dup_count": dup_count}

    def generate_dataset(
        self,
        num_samples: int,
        target_labels: List[str],
        model: str = "primary-cluster",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        synthesis_inputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 15,
        similarity_threshold: float = 0.85,
        hard_negative_ratio: float = 0.12,
        max_workers: int = 3,
        domain_pool: Optional[List[str]] = None,
        opt_in_cross_label: bool = False
    ) -> List[Dict[str, Any]]:
        """Generates synthetic dataset using parallel threads, including hard negatives and duplicate filter."""
        if not domain_pool:
            domain_pool = [
                "formal business email",
                "Slack/Teams messaging chat",
                "customer support ticket description",
                "legal contract clause",
                "medical intake form notes",
                "call center audio transcript",
                "internal corporate memo",
                "HR employee incident report",
                "database migration query log",
                "online bank statement comment",
                "software system trace log",
                "e-commerce shipping invoice",
                "job application cover letter",
                "insurance claim description",
                "financial audit report footnote",
                "product review forum post",
                "API request payload JSON log",
                "academic research survey response",
                "government tax filing notes",
                "real estate lease agreement text"
            ]

        # 1. Resolve Generator Runs configurations
        active_configs = []
        for label in target_labels:
            config = None
            if synthesis_inputs:
                for inp in synthesis_inputs:
                    if inp.get("target_label", "").strip().lower() == label.lower():
                        config = {
                            "target_label": label,
                            "pattern_val": inp.get("pattern_val", ""),
                            "data_format": inp.get("data_format", "General records")
                        }
                        break
            if not config:
                config = DEFAULT_PII_CONFIGS.get(label.lower(), {
                    "target_label": label,
                    "pattern_val": "",
                    "data_format": "General communications"
                })
            active_configs.append(config)

        # 2. Build the Deck of Targets with Positives and Hard Negatives
        num_neutral = max(1, int(num_samples * 0.10))
        remaining_samples = num_samples - num_neutral
        
        deck = []
        if not active_configs:
            for _ in range(num_samples):
                deck.append({
                    "target_label": "NEUTRAL",
                    "pattern_val": "",
                    "data_format": random.choice(domain_pool),
                    "is_hard_negative": False
                })
        else:
            samples_per_config = remaining_samples // len(active_configs)
            for config in active_configs:
                num_hn = int(samples_per_config * hard_negative_ratio)
                num_pos = samples_per_config - num_hn
                
                for _ in range(num_pos):
                    deck.append({
                        "target_label": config["target_label"],
                        "pattern_val": config["pattern_val"],
                        "data_format": random.choice(domain_pool),
                        "is_hard_negative": False
                    })
                for _ in range(num_hn):
                    deck.append({
                        "target_label": config["target_label"],
                        "pattern_val": config["pattern_val"],
                        "data_format": random.choice(domain_pool),
                        "is_hard_negative": True
                    })
                    
            remainder = remaining_samples - (samples_per_config * len(active_configs))
            for i in range(remainder):
                config = active_configs[i % len(active_configs)]
                is_hn = (random.random() < hard_negative_ratio)
                deck.append({
                    "target_label": config["target_label"],
                    "pattern_val": config["pattern_val"],
                    "data_format": random.choice(domain_pool),
                    "is_hard_negative": is_hn
                })
                
            for _ in range(num_neutral):
                deck.append({
                    "target_label": "NEUTRAL",
                    "pattern_val": "",
                    "data_format": random.choice(domain_pool),
                    "is_hard_negative": False
                })
                
        random.shuffle(deck)

        # Group runs to prevent LLM context-switching
        grouped_targets = {}
        for item in deck:
            key = (item["target_label"], item["pattern_val"], item["is_hard_negative"])
            if key not in grouped_targets:
                grouped_targets[key] = []
            grouped_targets[key].append(item)
            
        batches_runs = []
        for key, items in grouped_targets.items():
            lbl, pat, is_hn = key
            for i in range(0, len(items), batch_size):
                chunk = items[i:i+batch_size]
                if chunk:
                    rep = chunk[0]
                    batches_runs.append({
                        "entity_name": lbl,
                        "regex_pattern": pat,
                        "domain": rep["data_format"],
                        "is_hard_negative": is_hn,
                        "count": len(chunk)
                    })
                    
        num_batches = len(batches_runs)
        raw_results = []
        total_attempts = 0
        successful_samples_count = 0
        near_dup_total = 0
        
        proxy_url = os.environ.get("PROXY_URL", "http://localhost:8000").rstrip("/")
        logger.info(f"Starting Refactored Factory Generation. Size: {num_samples}, Batches: {num_batches}, Workers: {max_workers}")
        
        # 3. Parallel Batch Generation Loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_batch,
                    b_idx,
                    run["entity_name"],
                    run["regex_pattern"],
                    run["domain"],
                    model,
                    proxy_url,
                    is_hard_negative=run["is_hard_negative"],
                    batch_size=run["count"],
                    similarity_threshold=similarity_threshold,
                    opt_in_cross_label=opt_in_cross_label
                ): run
                for b_idx, run in enumerate(batches_runs)
            }
            
            for future in futures:
                run = futures[future]
                total_attempts += run["count"]
                try:
                    res = future.result()
                    batch_samples = res["samples"]
                    near_dup_total += res.get("near_dup_count", 0)
                    
                    raw_results.extend(batch_samples)
                    successful_samples_count += len(batch_samples)
                    
                    if progress_callback:
                        stats = {
                            "total_attempts": total_attempts,
                            "successful_parses": successful_samples_count,
                            "near_dup_count": near_dup_total,
                            "latest_sample": batch_samples[-1] if batch_samples else None
                        }
                        progress_callback(min(successful_samples_count, num_samples), num_samples, stats)
                except Exception as e:
                    logger.error(f"Batch generation failed: {e}")
                    
        # 4. Sequential Recovery Loop
        remaining = num_samples - len(raw_results)
        if remaining > 0:
            logger.info(f"Sequential recovery loop: generating {remaining} samples...")
            recovery_batches = math.ceil(remaining / batch_size)
            for b_idx in range(recovery_batches):
                try:
                    if active_configs:
                        fallback_config = active_configs[b_idx % len(active_configs)]
                    else:
                        fallback_config = {
                            "target_label": "NEUTRAL",
                            "pattern_val": "",
                            "data_format": random.choice(domain_pool)
                        }
                    
                    current_batch_size = min(remaining - (b_idx * batch_size), batch_size)
                    if current_batch_size <= 0:
                        break
                        
                    is_hn = (random.random() < hard_negative_ratio)
                    res = self._generate_batch(
                        batch_idx=999 + b_idx,
                        entity_name=fallback_config["target_label"],
                        regex_pattern=fallback_config["pattern_val"],
                        domain=fallback_config.get("data_format") or random.choice(domain_pool),
                        model=model,
                        proxy_url=proxy_url,
                        is_hard_negative=is_hn,
                        batch_size=current_batch_size,
                        similarity_threshold=similarity_threshold,
                        opt_in_cross_label=opt_in_cross_label
                    )
                    
                    batch_samples = res["samples"]
                    near_dup_total += res.get("near_dup_count", 0)
                    raw_results.extend(batch_samples)
                    successful_samples_count += len(batch_samples)
                    
                    if progress_callback:
                        stats = {
                            "total_attempts": total_attempts + current_batch_size,
                            "successful_parses": successful_samples_count,
                            "near_dup_count": near_dup_total,
                            "latest_sample": batch_samples[-1] if batch_samples else None
                        }
                        progress_callback(min(successful_samples_count, num_samples), num_samples, stats)
                except Exception as recovery_err:
                    logger.error(f"Sequential recovery failed: {recovery_err}")

        # 5. Strict Validation & Reporting Pass (Requirement 6)
        final_dataset = []
        dropped_count = 0
        hard_negative_count = 0
        domain_distribution = {}
        
        for sample in raw_results:
            text = sample["text"]
            entities = sample["entities"]
            is_hn = sample.get("is_hard_negative", False)
            
            is_valid = True
            for ent in entities:
                start, end, val, lbl = ent["start"], ent["end"], ent["value"], ent["label"]
                # Offset validation
                if text[start:end] != val:
                    is_valid = False
                    logger.warning(f"Dropped sample: Offset mismatch at text[{start}:{end}] for label '{lbl}' (expected '{val}', got '{text[start:end]}')")
                    break
                # Regex validation for active custom classes
                target_regex = None
                for config in active_configs:
                    if config["target_label"].lower() == lbl.lower():
                        target_regex = config["pattern_val"]
                        break
                if target_regex:
                    try:
                        compiled = re.compile(target_regex)
                        if not re.fullmatch(compiled, val):
                            is_valid = False
                            logger.warning(f"Dropped sample: Value '{val}' fails validation against regex '{target_regex}'")
                            break
                    except Exception:
                        pass
                        
            if not is_valid:
                dropped_count += 1
                continue
                
            final_dataset.append(sample)
            if is_hn:
                hard_negative_count += 1
            
            domain = sample.get("domain", "Unknown")
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        report = {
            "sample_counts": len(final_dataset),
            "dropped_count": dropped_count,
            "hard_negatives_count": hard_negative_count,
            "percent_hard_negatives": (hard_negative_count / len(final_dataset) * 100.0) if final_dataset else 0.0,
            "domain_distribution": domain_distribution,
            "near_dup_rate": (near_dup_total / num_samples) if num_samples else 0.0
        }
        
        logger.info(f"Dataset validation pass complete. Report: {report}")
        return DatasetList(final_dataset[:num_samples], report)

if __name__ == "__main__":
    engine = SyntheticDataEngine()
    print("SyntheticDataEngine successfully updated with exrex, hard negatives, and validation pass.")
