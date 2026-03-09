"""Central configuration for prompts, model candidates, and batch analysis.
Move UI-tunable settings here to avoid hardcoding inside app.py.
"""
from __future__ import annotations

from calls_analyser.services.prompt import PromptTemplate

# ---------------------------
# Prompt templates (UI tab)
# ---------------------------
# ---------------------------
# Common Prompt Body (Single Source of Truth)
# ---------------------------
_BATCH_PROMPT_BODY = (
    """You are analyzing a full phone conversation between a client and a call center employee.
Read and understand the entire dialogue before making a decision.
Decide whether the call needs follow-up or additional attention, and briefly explain why.

A call needs follow-up if:

the client is dissatisfied, complains, or shows negative emotions;

the client requests a callback or information but doesn’t receive a clear answer;

the employee cannot help, gives incomplete information, or ends the call abruptly;

the client reports issues with booking, payment, or the status or outcome of a service or product;

the conversation is incomplete, interrupted, or unresolved.

Make your decision based on the overall outcome of the whole conversation, not a single line.

Output format (must be valid JSON; no extra text):

{
  "needs_follow_up": true | false,
  "reason": "one short sentence explaining the decision"
}"""
)


# ---------------------------
# Prompt templates (UI tab)
# ---------------------------
PROMPTS = {
    "simple": PromptTemplate(
        key="simple",
        title="Simple",
        body=(
            "You are a call-center conversation analyst for a customer service team. From the call recording, provide a brief summary:\n"
            "- Purpose of the call (order / information / complaint / billing / other).\n"
            "- Client intent and expectations.\n"
            "- Outcome (resolved / call-back / routed / unresolved).\n"
            "- Next steps (owner and when).\n"
            "- Client emotion (1–5) and agent tone (1–5).\n"
            "- Alerts: urgency/risks/privacy.\n\n"
            "Keep it short (6–8 lines). End with a line: ‘Service quality rating: X/5’ and one sentence explaining the rating."
        ),
    ),
    "medium": PromptTemplate(
        key="medium",
        title="Medium",
        body=(
            "Act as a senior service analyst. Analyze the call using this structure:\n"
            "1) Quick overview: reason for the call, intent, key facts, urgency (low/medium/high).\n"
            "2) Call flow (2–4 bullets): what was asked/answered, where friction occurred.\n"
            "3) Outcomes & tasks: concrete next actions for company/client with timeframes.\n"
            "4) Emotions & empathy: client mood; agent empathy (0–5).\n"
            "5) Procedural compliance: identity verification, disclosure of recording (if stated), no inappropriate or unauthorized advice, data accuracy.\n"
            "6) Quality rating (0–100) using rubric: greeting, verification, accuracy, empathy, issue resolution (each 0–20)."
        ),
    ),
    "detailed": PromptTemplate(
        key="detailed",
        title="Detailed",
        body=(
            "You are a quality & operations analyst. Provide an in-depth analysis:\n"
            "A) Segmentation: split the call into stages with approximate timestamps (if available) and roles (Client/Agent).\n"
            "B) Structured data for request/booking: full name (if stated), contact details, description of the issue or request (list), relevant context, required product/service, preferred time windows, constraints.\n"
            "C) Priority & risks: urgency level (low/medium/high), key risks, whether escalation is needed.\n"
            "D) Compliance audit: identity/privacy checks, recording disclosure, consent to data processing, booking policies.\n"
            "E) Conversation metrics: talk ratio (agent/patient), interruptions, long pauses, notable keywords.\n"
            "F) Coaching for the agent: 3–5 concrete improvements with sample phrasing.\n\n"
            "Deliver: (1) A short client-case summary (2–3 sentences). (2) A task table with columns: priority, owner, due."
        ),
    ),
    # Batch-specific template (optional; batch uses custom prompt text, but keeping a template makes it discoverable)
    "BATCH_PROMPT": PromptTemplate(
        key="BATCH_PROMPT",
        title="Batch Prompt",
        body=_BATCH_PROMPT_BODY,
    ),
}

# ---------------------------
# Provider model candidates
# ---------------------------
MODEL_CANDIDATES = [
    ("flash", "models/gemini-2.5-flash"),
    ("pro", "models/gemini-2.5-pro"),
    ("flash-lite", "models/gemini-2.5-flash-lite"),
]

# ---------------------------
# Batch analysis configuration
# ---------------------------
BATCH_MODEL_KEY = "models/gemini-2.5-flash-lite"
BATCH_PROMPT_KEY = "BATCH_PROMPT"
BATCH_PROMPT_TEXT = _BATCH_PROMPT_BODY
# ISO language code for batch (app converts to Language enum)
BATCH_LANGUAGE_CODE = "ru"

# Flag for custom batch behaviour.
# Значэнні:
# - "off" (па змаўчанні) — кнопка кастомнага батча схаваная
# - "on" — кнопка бачная
BATCH_custom = "off"

# ---------------------------
# Custom batch prompt configuration
# ---------------------------
BATCH_CUSTOM_CONDITIONS_DEFAULT = (
    """A call requires follow-up or extra attention when any of the following is true:

The client sounds dissatisfied, complains, or expresses negative emotions, excluding cases where the issue is solely due to the unavailability of a specific specialist or service.

The client asks to be called back or requests information but does not receive a clear answer.

The employee cannot help for reasons related to incorrect handling, lack of knowledge, or poor communication, but not due to the unavailability of a specialist or service.

The client reports problems with booking, payment, or the status or outcome of a service or product.

The conversation appears incomplete, interrupted, or unresolved for reasons other than service or specialist unavailability."""
)

BATCH_CUSTOM_PROMPT_TEMPLATE = (
    """You are analyzing a phone call between a client and a call center employee using the following rules and conditions:\n\n"
    "{{CONDITIONS}}\n\n"
    "Your task is to determine whether this call requires follow-up or additional attention, and briefly summarize the outcome.\n\n"
    "Respond strictly in the following format:\n"
    "Needs follow-up: Yes/No\n"
    "Summary: [one short sentence]\n\n"
    "If no issues are found, use a neutral summary such as “The issue was resolved” or “The client received clear information.”"""
)
