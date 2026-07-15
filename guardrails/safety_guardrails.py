"""
Safety Guardrails: Jailbreak, Toxicity, and Prompt Injection detection.

All three guardrails are pattern/keyword-based (no extra ML model downloads).
Each exposes:  check(text: str) -> (is_flagged: bool, reason: str)
"""
from __future__ import annotations
import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Jailbreak Detection
# ---------------------------------------------------------------------------

_JAILBREAK_PHRASES = [
    # DAN / roleplay-escape patterns
    r"\bDAN\b",
    r"do anything now",
    r"jailbreak",
    r"ignore (all )?(previous|prior|above|your) (instructions?|prompts?|guidelines?|rules?|constraints?)",
    r"disregard (all )?(previous|your) (instructions?|rules?|prompts?|guidelines?)",
    r"forget (all )?(previous|prior|your) (instructions?|rules?|prompts?|guidelines?)",
    r"override (your )?(instructions?|guidelines?|rules?|training)",
    r"pretend (you are|you're|to be) (a|an) (?!assistant|AI).{0,40} (no restrictions|no limits|unrestricted|without (any )?restrictions)",
    r"act as (if you have|though you have) no (restrictions|limitations|guidelines|rules|safety)",
    r"(you are|you're) now (a|an) (unrestricted|uncensored|unfiltered|evil|bad|rogue)",
    r"you have no (restrictions|limitations|guidelines|rules|safety|ethical)",
    r"(enable|activate|switch (to|on)) (developer|dev|god|admin|maintenance|unsafe|unrestricted) mode",
    r"(you are|you're) (freed?|liberated) from (your )?(restrictions?|limitations?|guidelines?|safety)",
    r"respond (without|with no) (filters?|restrictions?|limitations?|guidelines?|censorship)",
    r"(bypass|circumvent|override|ignore|remove) (your )?(safety|content|ethical) (filters?|checks?|restrictions?|guidelines?)",
    r"\[SYSTEM\].*ignore",
    r"<\|im_start\|>system",
    r"<\|system\|>",
    r"###\s*(instruction|system|override)",
    r"prompt\s*injection",
    r"you must (obey|follow|comply with|do) (everything|anything|all) (i|the user)",
    r"(your|the) (new|real|true|actual) (instructions?|purpose|goal|role|identity|name) (is|are)",
    r"from now on (you are|you will|ignore)",
    r"i (hereby|now|officially) (grant|give) you (permission|authority|freedom)",
    r"simulate (an|a) ?(AI|chatbot|LLM|GPT|assistant) (without|with no|that has no)",
    r"no (ethical|moral) (guidelines?|constraints?|restrictions?|considerations?)",
    r"(evil|malicious|hacked|broken|glitched|corrupted) (AI|bot|model|GPT|assistant)",
]

_JAILBREAK_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _JAILBREAK_PHRASES]


class JailbreakGuardrail:
    """Detects prompt injection / jailbreak attempts via pattern matching."""

    def check(self, text: str) -> Tuple[bool, str]:
        for pat in _JAILBREAK_PATTERNS:
            m = pat.search(text)
            if m:
                return True, f"Jailbreak attempt detected: matched pattern '{pat.pattern[:60]}'"
        return False, ""


# ---------------------------------------------------------------------------
# Toxicity Detection
# ---------------------------------------------------------------------------

# Tier-1: severe explicit threats / slurs — always block
_TOXICITY_SEVERE = [
    r"\bi will (kill|murder|rape|torture|harm|destroy|attack) (you|them|him|her)\b",
    r"\b(kill|murder|rape|torture|bomb|shoot|stab) (yourself|yourself|them|him|her)\b",
    r"(go|just) (kill|hang) yourself",
    r"\b(n[i1]gg[ae]r|f[a4]gg[o0]t|ch[i1]nk|sp[i1]c|k[i1]ke|w[e3]tb[a4]ck)\b",
    r"\b(i('ll| will| am going to) (blow up|shoot up|bomb))\b",
    r"suicide (method|how to|tutorial|guide|instructions?)",
    r"(how (to|do i) (make|build|create|synthesize) (a |an )?(bomb|explosive|poison|bioweapon|nerve agent))",
    r"(child|minor|underage).{0,20}(porn|sex|nude|naked|explicit)",
]

# Tier-2: strong signals — very likely toxic
_TOXICITY_STRONG = [
    r"\b(you('re| are) (a |an )?(worthless|pathetic|disgusting|vile|filthy) (piece of shit|scum|trash|garbage))\b",
    r"\b(hate|despise|loathe) (all )?(jews?|muslims?|christians?|blacks?|whites?|asians?|women|men|gays?|lesbians?)\b",
    r"\b(death|violence) to (all )?(jews?|muslims?|christians?|blacks?|gays?|immigrants?)\b",
    r"\b(sexual(ly)? (harass|assault|abuse|exploit))\b",
    r"\bself(-| )(harm|mutilat|injur)\b",
]

_TOXICITY_SEVERE_PATS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _TOXICITY_SEVERE]
_TOXICITY_STRONG_PATS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _TOXICITY_STRONG]


class ToxicityGuardrail:
    """Detects toxic, hateful, or threatening content via layered pattern matching."""

    def check(self, text: str) -> Tuple[bool, str]:
        for pat in _TOXICITY_SEVERE_PATS:
            if pat.search(text):
                return True, f"Severe toxic content detected (threats / hate speech / CSAM)"
        for pat in _TOXICITY_STRONG_PATS:
            if pat.search(text):
                return True, "Strongly toxic content detected (hate speech / harassment)"
        return False, ""


# ---------------------------------------------------------------------------
# Prompt Injection Detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # Control token injections
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"<s>.*</s>",
    # Explicit injection keywords
    r"(new|updated?|revised?|override) (system )?instructions?:",
    r"(ignore|discard|forget|override).{0,30}(above|previous|prior|original) (instructions?|context|system prompt)",
    r"you are now (operating|running|functioning) (as|under|in)",
    r"system prompt:",
    r"(your|the) (system|real|hidden|secret|actual|true) (prompt|instructions?|directive|role):",
    r"```\s*(system|SYSTEM|System)",
    r"<system>",
    r"\[system\]",
    r"---\s*(system|override|instructions?)\s*---",
    r"(inject|exfiltrate|leak|extract).{0,30}(prompt|instructions?|system|context)",
    r"(print|output|display|reveal|show|tell me|repeat|echo).{0,40}(system prompt|your instructions?|your prompt|your context)",
    r"translate the above\s*(prompt|instruction|text|context)\s*to",
    r"(summarize|restate|repeat).{0,20}(the )?(system|initial|original) (prompt|instructions?)",
    r"escape.{0,20}(sandbox|container|context|system)",
    r"execute\s+(arbitrary|external|user-provided)\s+(code|commands?|instructions?)",
]

_INJECTION_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


class PromptInjectionGuardrail:
    """Detects system prompt manipulation and control token injection attacks."""

    def check(self, text: str) -> Tuple[bool, str]:
        for pat in _INJECTION_COMPILED:
            if pat.search(text):
                return True, f"Prompt injection attempt detected: matched '{pat.pattern[:60]}'"
        return False, ""


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

_JAILBREAK_INSTANCE = JailbreakGuardrail()
_TOXICITY_INSTANCE = ToxicityGuardrail()
_INJECTION_INSTANCE = PromptInjectionGuardrail()


def check_all(text: str, enabled: dict, action: dict) -> Tuple[bool, str, str]:
    """
    Run all enabled safety guardrails against `text`.

    Args:
        text:    The prompt text to check.
        enabled: Dict of {"jailbreak": bool, "toxicity": bool, "prompt_injection": bool}
        action:  Dict of {"jailbreak": "BLOCK"|"LOG", ...}

    Returns:
        (should_block: bool, guardrail_name: str, reason: str)
        should_block is True only when a guardrail fires AND its action is "BLOCK".
    """
    checks = [
        ("jailbreak",         _JAILBREAK_INSTANCE),
        ("toxicity",          _TOXICITY_INSTANCE),
        ("prompt_injection",  _INJECTION_INSTANCE),
    ]
    for name, guard in checks:
        if not enabled.get(name, False):
            continue
        flagged, reason = guard.check(text)
        if flagged:
            act = action.get(name, "BLOCK")
            return (act == "BLOCK"), name, reason
    return False, "", ""
