"""Central configuration for prompts, model candidates, and batch analysis.
Move UI-tunable settings here to avoid hardcoding inside app.py.
"""
from __future__ import annotations

from calls_analyser.services.prompt import PromptTemplate

# ---------------------------
# Prompt templates (UI tab)
# ---------------------------
PROMPTS = {
    "simple": PromptTemplate(
        key="simple",
        title="Simple",
        body=(
            "You are a call-center conversation analyst for a medical clinic. From the call recording, provide a brief summary:\n"
            "- Purpose of the call (appointment / results / complaint / billing / other).\n"
            "- Patient intent and expectations.\n"
            "- Outcome (booked / call-back / routed / unresolved).\n"
            "- Next steps (owner and when).\n"
            "- Patient emotion (1–5) and agent tone (1–5).\n"
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
            "3) Outcomes & tasks: concrete next actions for clinic/patient with timeframes.\n"
            "4) Emotions & empathy: patient mood; agent empathy (0–5).\n"
            "5) Procedural compliance: identity verification, disclosure of recording (if stated), no off-protocol medical advice, data accuracy.\n"
            "6) Quality rating (0–100) using rubric: greeting, verification, accuracy, empathy, issue resolution (each 0–20)."
        ),
    ),
    "detailed": PromptTemplate(
        key="detailed",
        title="Detailed",
        body=(
            "You are a quality & operations analyst. Provide an in-depth analysis:\n"
            "A) Segmentation: split the call into stages with approximate timestamps (if available) and roles (Patient/Agent).\n"
            "B) Structured data for booking: full name (if stated), date of birth, phone, symptoms/complaints (list), onset/duration, possible pain level 0–10 (if mentioned), required specialist/service, preferred time windows, constraints.\n"
            "C) Triage & risks: class (routine/urgent/emergency), red flags, whether immediate escalation is needed.\n"
            "D) Compliance audit: identity/privacy checks, recording disclosure, consent to data processing, booking policies.\n"
            "E) Conversation metrics: talk ratio (agent/patient), interruptions, long pauses, notable keywords.\n"
            "F) Coaching for the agent: 3–5 concrete improvements with sample phrasing.\n\n"
            "Deliver: (1) A short patient-chart summary (2–3 sentences). (2) A task table with columns: priority, owner, due."
        ),
    ),
    # Batch-specific template (optional; batch uses custom prompt text, but keeping a template makes it discoverable)
    "BATCH_PROMPT": PromptTemplate(
        key="BATCH_PROMPT",
        title="Batch Prompt",
        body=(
            """You are analyzing a phone conversation between a client and a medical center employee.
Your task is to decide whether this call needs follow-up or additional attention, and briefly explain why.

A call needs follow-up if:

the client sounds dissatisfied, complains, or expresses negative emotions;

the client asks to be called back or requests information but doesn’t get a clear answer;

the employee cannot help, gives incomplete information, or ends the call abruptly;

the client reports problems with booking, payment, or test results;

the conversation seems incomplete or interrupted.

Respond in the following format:
Needs follow-up: Yes/No
Reason: [short explanation — one sentence only]

If everything is fine, the reason can be something like “The issue was resolved” or “Client received clear information."""
        ),
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
BATCH_PROMPT_TEXT = (
    """You are analyzing a full phone conversation between a client and a medical center employee.
Read and understand the entire dialogue before making a decision.
Decide whether the call needs follow-up or additional attention, and briefly explain why.

A call needs follow-up if:

the client is dissatisfied, complains, or shows negative emotions;

the client requests a callback or information but doesn’t receive a clear answer;

the employee cannot help, gives incomplete information, or ends the call abruptly;

the client reports issues with booking, payment, or test results;

the conversation is incomplete, interrupted, or unresolved.

Make your decision based on the overall outcome of the whole conversation, not a single line.

Output format (must be valid JSON; no extra text):

{
  "needs_follow_up": true | false,
  "reason": "one short sentence explaining the decision"
}"""
)
# ISO language code for batch (app converts to Language enum)
BATCH_LANGUAGE_CODE = "ru"
