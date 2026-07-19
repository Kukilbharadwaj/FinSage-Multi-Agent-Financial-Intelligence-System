# agents/guardrail.py
# Minimal, local input/output guardrails — replaces NVIDIA NeMo Guardrails.
#
# Why NeMo was removed:
#   1. Latency. rails.generate_async() ran a FULL LLM generation for the input
#      check and another for the output check. That was two extra round trips
#      on every query, on top of the pipeline's own LLM calls.
#   2. False positives. The self_check_input prompt classified ordinary
#      conversational openers ("hi", "what can you do?") as non-financial and
#      blocked them, so the assistant could not even introduce itself.
#
# The policy here is deliberately narrow, per the product requirement:
# block ONLY hacking/abuse attempts and clearly off-topic requests. Everything
# even loosely finance-adjacent is allowed through, and small talk is answered
# warmly instead of being refused.
#
# All checks are local string matching — zero network calls, microseconds.

import re

# ── Hard blocks: hacking, malware, and platform abuse ─────────
_ABUSE_PATTERNS = [
    r"\bhack(ing|ed)?\b.*\b(account|password|system|website|server|wifi|database|bank)\b",
    r"\b(how to |help me )?(write|create|build|make)\b.*\b(malware|ransomware|keylogger|virus|trojan|botnet)\b",
    r"\b(sql injection|xss attack|ddos|brute[ -]?force|phish(ing)?|exploit kit)\b",
    r"\b(steal|crack|bypass|dump)\b.*\b(password|credential|otp|2fa|login|database)\b",
    r"\b(carding|credit card dump|cvv dump|skimmer)\b",
    # Financial crime how-tos (as opposed to asking what the law says)
    r"\b(how to|help me|best way to)\b.*\b(launder money|money laundering|evade tax|hide black money|fake invoice)\b",
    r"\b(insider trading|pump and dump)\b.*\b(how|tip|do it|get away)\b",
]

# ── Prompt injection / identity override ──────────────────────
_INJECTION_PATTERNS = [
    r"ignore (all |your |the )?(previous|prior|above|earlier) (instruction|prompt|rule|direction)",
    r"forget (your|all|the) (rules|instructions|training|guidelines|system prompt)",
    r"(you are|act as|pretend to be) (now )?(a |an )?(general|unrestricted|uncensored|different) (assistant|ai|model)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system prompt|instructions|initial prompt)",
    r"\b(dan mode|developer mode|jailbreak)\b",
    r"override (your |the )?(safety|security|guideline|restriction)",
]

# ── Clearly off-topic requests ────────────────────────────────
# Only obvious non-financial asks. Anything ambiguous is allowed:
# a false allow is a mild annoyance, a false block is a broken product.
_OFF_TOPIC_PATTERNS = [
    r"\b(write|compose|create) (me )?(a |an )?(poem|song|story|essay|novel|screenplay|rap)\b",
    r"\b(recipe|how to (cook|bake)|ingredients for)\b",
    r"\b(write|debug|fix|refactor) (me )?(a |an |some )?(code|program|script|function|website|app)\b",
    r"\b(who won|match score|cricket score|football score|ipl score)\b",
    r"\b(weather|temperature) (today|tomorrow|forecast|in)\b",
    r"\b(translate|translation) (this|that|to|into)\b",
    r"\b(my|the) homework\b",
    r"\b(capital of|population of|who invented|when was .* born)\b",
    r"\b(movie|film|series|anime) (recommendation|suggestion|to watch)\b",
]

# ── Small talk: answered directly, never blocked ──────────────
_GREETING_PATTERNS = [
    r"^\s*(hi|hey|hello|yo|namaste|hii+|helo|hlo)\b",
    r"^\s*good (morning|afternoon|evening)\b",
    r"^\s*(how are you|what'?s up|sup)\b",
]

_CAPABILITY_PATTERNS = [
    r"\b(what|which) (can|could) you (do|help)\b",
    r"\bwhat are you (able|capable)\b",
    r"\b(who|what) are you\b",
    r"\bhow (can|do) you help\b",
    r"\byour (capabilities|features|functions)\b",
    r"\bhelp me\s*$",
    r"^\s*(help|menu|options|start)\s*$",
]

_THANKS_PATTERNS = [r"\b(thank|thanks|thx|appreciate it|great job|well done|nice)\b"]
_BYE_PATTERNS = [r"^\s*(bye|goodbye|see you|cya|good night|exit|quit)\b"]

# Finance vocabulary — presence of any of these overrides an off-topic match,
# so "write code to calculate my SIP returns" is treated as a finance question.
_FINANCE_TERMS = {
    "stock", "share", "equity", "nifty", "sensex", "banknifty", "market", "trade", "trading",
    "option", "call", "put", "futures", "f&o", "intraday", "swing", "portfolio", "invest",
    "investment", "mutual fund", "sip", "nav", "elss", "amc", "index fund", "etf",
    "tax", "gst", "itr", "80c", "80d", "stcg", "ltcg", "capital gain", "deduction", "tds",
    "salary", "income", "budget", "saving", "emergency fund", "expense", "emi", "loan",
    "insurance", "term plan", "health cover", "premium", "retirement", "nps", "epf", "ppf",
    "pension", "gold", "sgb", "crypto", "bitcoin", "fd", "rd", "interest rate", "inflation",
    "rupee", "lpa", "lakh", "crore", "dividend", "ipo", "broker", "demat", "sebi", "rbi",
    "finance", "financial", "money", "wealth", "debt", "credit", "bank", "compounding",
}


def _matches(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_finance_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _FINANCE_TERMS)


def classify(query: str) -> dict:
    """
    Classify a user query locally.

    Returns {"action": ..., "reason": ..., "reply": ...} where action is one of:
        "allow"     — run the full agent pipeline
        "smalltalk" — answer immediately with `reply`, skip the pipeline
        "block"     — refuse with `reply`, skip the pipeline
    """
    text = (query or "").strip()

    if not text:
        return {"action": "block", "reason": "empty query", "reply": "Could you type your question? I'm ready when you are."}

    # 1. Abuse and injection always win, even if finance words are present.
    if _matches(text, _ABUSE_PATTERNS):
        return {
            "action": "block",
            "reason": "abuse/hacking request",
            "reply": (
                "I can't help with that one — it's outside what I do.\n\n"
                "I'm here for money questions though: investing, taxes, salary planning, "
                "trading, mutual funds. What would you like to look at?"
            ),
        }

    if _matches(text, _INJECTION_PATTERNS):
        return {
            "action": "block",
            "reason": "prompt injection attempt",
            "reply": (
                "Nice try! 🙂 I'm FinSage either way — your Indian personal finance assistant.\n\n"
                "Ask me about stocks, mutual funds, tax, salary planning or trading and I'm all yours."
            ),
        }

    # 2. Small talk — answered warmly, never refused.
    #    This is the case NeMo used to block, which made the assistant unable
    #    to respond to "hi" or explain what it does.
    if _matches(text, _CAPABILITY_PATTERNS):
        return {"action": "smalltalk", "reason": "capability question", "reply": _capabilities_reply()}

    if _matches(text, _GREETING_PATTERNS) and len(text.split()) <= 6 and not _has_finance_term(text):
        return {"action": "smalltalk", "reason": "greeting", "reply": _greeting_reply()}

    if _matches(text, _BYE_PATTERNS) and len(text.split()) <= 5:
        return {
            "action": "smalltalk",
            "reason": "farewell",
            "reply": "Take care! Come back anytime you want to talk money. 👋",
        }

    if _matches(text, _THANKS_PATTERNS) and len(text.split()) <= 6 and not _has_finance_term(text):
        return {
            "action": "smalltalk",
            "reason": "thanks",
            "reply": "Happy to help! Anything else on your mind — investments, tax, or your monthly budget?",
        }

    # 3. Off-topic, but only when there is no finance angle at all.
    if _matches(text, _OFF_TOPIC_PATTERNS) and not _has_finance_term(text):
        return {
            "action": "block",
            "reason": "off-topic",
            "reply": (
                "That's a bit outside my lane — I stick to Indian personal finance.\n\n"
                "I can help with **stocks and market levels**, **mutual funds and SIPs**, "
                "**income tax and GST**, **salary and budgeting**, **options and intraday trading**, "
                "plus insurance, loans and retirement. What sounds useful?"
            ),
        }

    # 4. Default: allow. Anything ambiguous reaches the agents.
    return {"action": "allow", "reason": "passed", "reply": ""}


def _greeting_reply() -> str:
    return (
        "Hey! 👋 I'm FinSage — I help with money decisions in the Indian context.\n\n"
        "You can ask me things like:\n"
        "- *\"I earn ₹12 LPA — how should I plan my month?\"*\n"
        "- *\"What's NIFTY doing today?\"*\n"
        "- *\"How much tax do I pay on ₹3L of LTCG?\"*\n"
        "- *\"Is this a good time for a BANKNIFTY option trade?\"*\n\n"
        "What's on your mind?"
    )


def _capabilities_reply() -> str:
    return (
        "I'm FinSage — think of me as a finance-savvy friend who's read all the Indian rulebooks. "
        "Here's where I'm useful:\n\n"
        "**📊 Markets & stocks** — live NIFTY/SENSEX levels, company fundamentals, P/E, technical levels\n"
        "**📉 Trading & options** — live NSE option chains, PCR, max pain, intraday levels and risk rules\n"
        "**📈 Mutual funds** — NAV, returns, SIP sizing, ELSS and direct-vs-regular\n"
        "**🧾 Tax & GST** — STCG/LTCG maths, 80C/80D, salary tax, GST basics\n"
        "**💰 Salary & budgeting** — monthly plans, emergency fund, where each rupee should go\n"
        "**🛡️ Insurance, loans, retirement** — term cover, EMIs, NPS/EPF/PPF\n\n"
        "Just ask in plain language — mention your numbers and I'll work with them.\n\n"
        "*One thing up front: I'm educational, not a SEBI-registered advisor.*"
    )


# ── Output side ───────────────────────────────────────────────

SEBI_DISCLAIMER = (
    "\n\n---\n"
    "*Educational information only — FinSage is not a SEBI-registered investment advisor. "
    "Please consider your own situation, and consult a qualified advisor for big decisions.*"
)

# Phrases that promise certainty. Rewritten rather than blocking the whole answer.
_GUARANTEE_REWRITES = [
    (r"\bguaranteed returns?\b", "potential returns"),
    (r"\bguaranteed profits?\b", "potential profit"),
    (r"\b100% safe\b", "relatively low risk"),
    (r"\b100% guaranteed\b", "not guaranteed"),
    (r"\bzero risk\b", "lower risk"),
    (r"\brisk[- ]free returns?\b", "lower-risk returns"),
    (r"\byou will definitely (make|earn|profit)\b", r"you may \1"),
    (r"\bsure[- ]shot\b", "high-conviction"),
]

_DISCLAIMER_MARKERS = ("sebi", "educational", "not registered", "consult a qualified", "consult a financial")


def sanitize_output(text: str) -> tuple:
    """
    Clean the final answer locally.

    Softens certainty language and guarantees a disclaimer is present.
    Returns (cleaned_text, was_modified).
    """
    if not text or not text.strip():
        return text, False

    cleaned = text
    modified = False

    for pattern, replacement in _GUARANTEE_REWRITES:
        cleaned, count = re.subn(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if count:
            modified = True

    lowered = cleaned.lower()
    if sum(1 for marker in _DISCLAIMER_MARKERS if marker in lowered) < 2:
        cleaned += SEBI_DISCLAIMER
        modified = True

    return cleaned, modified
