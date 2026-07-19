# agents/guardrail.py
# The FinSage guardrail POLICY.
#
# This module decides; NVIDIA NeMo Guardrails enforces. The two functions here
# (`classify` and `sanitize_output`) are registered as NeMo actions in
# agents/nemo_rails.py and invoked from the Colang flows in guardrails/rails.co.
# NeMo owns the control flow — stop the pipeline, rewrite the answer, carry
# dialog state — while the actual judgement stays here as deterministic Python.
#
# Why the policy is Python instead of NeMo's stock self_check_* prompts:
#   1. Latency. self_check_input and self_check_output each run a FULL LLM
#      generation, adding two round trips to every single query.
#   2. False positives. The self_check_input prompt classified ordinary
#      conversational openers ("hi", "what can you do?") as non-financial and
#      blocked them, so the assistant could not even introduce itself.
#
# The policy is deliberately narrow, per the product requirement: block ONLY
# hacking/abuse and prompt injection, plus requests that are clearly off-topic
# with no financial angle at all. Everything even loosely finance-adjacent is
# allowed through, and small talk is answered warmly instead of being refused.
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
# Grouped by attack shape. These are checked before anything else, because an
# injection dressed up in finance vocabulary is still an injection.
_INJECTION_PATTERNS = [
    # Instruction override
    r"\b(ignore|disregard|forget|discard|override)\b.{0,30}\b(previous|prior|above|earlier|initial|all|your|the)\b.{0,20}\b(instruction|prompt|rule|direction|guideline|context|constraint)",
    r"forget (your|all|the|everything) (rules|instructions|training|guidelines|system prompt|you were told)",
    r"\b(do not|don'?t|no longer) (follow|obey|apply|respect)\b.{0,25}\b(rule|instruction|guideline|restriction|policy)",
    r"\bnew (instructions?|rules?|system prompt)\s*[:\-]",
    r"\b(stop|halt)\b.{0,15}\bnew (instruction|rule|task)",

    # Identity / persona override
    r"(you are|act as|pretend to be|behave like|roleplay as|simulate)\b.{0,30}\b(unrestricted|uncensored|unfiltered|unlimited|no[- ]restriction|without (any )?(rule|restriction|filter|guideline))",
    r"(you are|act as|pretend to be|behave like) (now )?(a |an )?(general|different|generic) (assistant|ai|model|chatbot)",
    r"\bfrom now on,? you (are|will|must|should)\b",
    r"\byou are no longer\b.{0,20}\b(finsage|a finance|restricted|bound)",
    r"\b(enable|activate|switch to)\b.{0,20}\b(dev|developer|debug|god|admin|root) mode\b",
    r"\b(dan mode|developer mode|jailbreak|do anything now)\b",

    # System-prompt extraction
    r"(reveal|show|print|repeat|output|display|tell me|give me|what is|what'?s)\b.{0,25}\b(system prompt|initial prompt|original instruction|your instructions|your rules|your prompt)",
    r"\b(repeat|print|output)\b.{0,20}\b(everything|the text|all text)\b.{0,15}\babove\b",
    r"\bverbatim\b.{0,25}\b(prompt|instruction)",

    # Fake role / delimiter injection
    r"^\s*(system|assistant|developer)\s*[:>]",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"\[\s*(system|inst|/inst)\s*\]",
    r"###\s*(system|instruction)",

    # Safety-control tampering
    r"override (your |the )?(safety|security|guideline|restriction|guardrail|filter)",
    r"\b(disable|turn off|remove|bypass)\b.{0,20}\b(safety|guardrail|filter|restriction|censor|moderation)",
    r"\bwithout\b.{0,15}\b(disclaimer|warning|restriction|filter)\b.{0,20}\b(answer|respond|reply)",

    # Obfuscation / indirect execution
    r"\b(decode|base64|rot13|reverse)\b.{0,25}\b(and (then )?(execute|follow|do|run|obey))",
    r"\b(execute|run|obey)\b.{0,20}\bthe following\b.{0,20}\b(instruction|command|prompt)",
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


def _history_has_finance(history: list) -> bool:
    """
    True when the recent conversation was already about money.

    Follow-ups lean on context — "and for my wife?" or "what about 20 lakhs"
    carry no finance vocabulary of their own. Judging those in isolation
    mislabels them, so an ongoing financial thread counts as finance context.
    """
    if not history:
        return False

    recent = " ".join(
        f"{turn.get('query', '')} {turn.get('answer', '')}"
        for turn in history[-3:]
        if isinstance(turn, dict)
    )
    return _has_finance_term(recent)


def classify(query: str, history: list = None) -> dict:
    """
    Classify a user query.

    Registered as the `finsage_input_check` NeMo action and invoked from the
    "finsage input policy" flow in guardrails/rails.co.

    Args:
        query:   the raw user message.
        history: recent conversation turns, each {"query": ..., "answer": ...}.
                 Used only to give follow-up questions the benefit of context.

    Returns {"action": ..., "reason": ..., "reply": ...} where action is one of:
        "allow"     — run the full agent pipeline
        "smalltalk" — answer immediately with `reply`, skip the pipeline
        "block"     — refuse with `reply`, skip the pipeline
    """
    text = (query or "").strip()
    in_finance_thread = _history_has_finance(history)

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

    # 3. Off-topic, but only when there is no finance angle at all — not in the
    #    message, and not in the thread it continues.
    if _matches(text, _OFF_TOPIC_PATTERNS) and not _has_finance_term(text) and not in_finance_thread:
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
    Clean the final answer.

    Registered as the `finsage_output_check` NeMo action and invoked from the
    "finsage output policy" flow in guardrails/rails.co.

    Softens certainty language and guarantees a disclaimer is present. This
    rewrites rather than blocks — a refused answer helps nobody, a reworded one
    still answers the question within compliance.

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
