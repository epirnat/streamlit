"""
Triage copilot: takes the RAW TEXT of an incoming customer message and returns
  - a predicted category
  - a suggested priority
  - an automation level (auto-reply / draft for review / human agent)
  - an escalation recommendation
  - a draft reply

Works fully offline with keyword rules (no API key needed). In production, the same
interface could be backed by a trained classifier or an LLM service.

Input format: just the message text (optionally channel / segment / region as context).
The structured columns (category, priority, ...) are what this module PRODUCES.
"""
import re

import pandas as pd

from analysis import REPEATABILITY

# ----------------------------------------------------------------------------- rules
CATEGORY_KEYWORDS = {
    "Order status": ["where is my order", "tracking", "status of order", "current status",
                     "expected delivery date", "still shows", "processing", "shipped"],
    "Delivery delay": ["late", "delayed", "has not arrived", "not arrived", "promised delivery",
                       "installation is on hold", "firm new date", "carrier"],
    "Quote request": ["quote", "quotation", "pricing", "lead time", "framework agreement"],
    "Invoice / billing": ["invoice", "vat", "charged twice", "refund", "accounting", "does not match our agreed"],
    "Returns / RMA": ["return", "rma", "wrong item", "unused", "ordered too many"],
    "Technical question": ["operating pressure", "compatible", "wiring", "configuration", "datasheet",
                           "selection guidance", "fault codes", "errors out", "refrigerant", "commissioning"],
    "Warranty / defect": ["warranty", "defective", "failed after", "leaking", "fail on startup", "replacement"],
    "Account / portal": ["log in", "login", "password", "portal does not show", "add a new user", "account settings"],
}

URGENT_WORDS = ["urgent", "urgently", "asap", "escalate", "on hold", "immediately", "critical", "down", "stopped"]

BASE_PRIORITY = {  # sensible default per category
    "Order status": "Medium", "Delivery delay": "High", "Quote request": "Medium",
    "Invoice / billing": "Medium", "Returns / RMA": "Medium", "Technical question": "High",
    "Warranty / defect": "High", "Account / portal": "Medium",
}
LEVELS = ["Low", "Medium", "High", "Critical"]

REPLY_TEMPLATES = {
    "Order status": "Hello,\n\nThank you for your message. Order {po} is currently {status}. "
                    "You can follow it live in the customer portal, and we will notify you of any change.\n\nKind regards,\nCustomer Service",
    "Delivery delay": "Hello,\n\nWe are sorry about the delay on {po}. We have asked our logistics team for a confirmed "
                      "delivery date and will update you within 4 working hours.\n\nKind regards,\nCustomer Service",
    "Quote request": "Hello,\n\nThank you for your request. To prepare an accurate quote we need the product codes, "
                     "quantities, and delivery address. We will return a quote within 2 working days.\n\nKind regards,\nCustomer Service",
    "Invoice / billing": "Hello,\n\nThank you for letting us know. Our billing team is reviewing {po} and will send a "
                         "corrected invoice or confirmation within 2 working days.\n\nKind regards,\nCustomer Service",
    "Returns / RMA": "Hello,\n\nWe are sorry for the inconvenience. We have started a return authorization for {po}. "
                     "Your RMA number and return instructions will follow shortly.\n\nKind regards,\nCustomer Service",
    "Technical question": "Hello,\n\nThank you for your question. We have passed it to a product specialist, who will reply "
                          "with detailed guidance. Please include model numbers and any error codes.\n\nKind regards,\nTechnical Support",
    "Warranty / defect": "Hello,\n\nWe are sorry to hear this. Please send photos, the serial number, and the date of "
                         "installation so we can start the warranty case right away.\n\nKind regards,\nCustomer Service",
    "Account / portal": "Hello,\n\nWe can help with that. For security, we will send a verification link to the registered "
                        "email address. If you do not receive it within 10 minutes, reply here.\n\nKind regards,\nCustomer Service",
}


# ----------------------------------------------------------------------------- helpers
def _extract_po(text: str) -> str:
    m = re.search(r"PO-\d+", text)
    return m.group(0) if m else "your order"


def classify(text: str) -> tuple[str, float]:
    """Keyword scoring. Returns (category, confidence 0-1)."""
    t = text.lower()
    scores = {c: sum(kw in t for kw in kws) for c, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[best] == 0:
        return "Technical question", 0.0  # unknown -> route to a human specialist
    return best, round(scores[best] / total, 2)


def suggest_priority(text: str, category: str, segment: str | None = None) -> str:
    level = LEVELS.index(BASE_PRIORITY[category])
    t = text.lower()
    if any(w in t for w in URGENT_WORDS):
        level += 1
    if segment == "OEM":  # key accounts get bumped
        level += 1
    return LEVELS[min(level, 3)]


def automation_level(category: str, confidence: float) -> str:
    rep = REPEATABILITY[category]
    if confidence < 0.5:
        return "Human agent"
    if rep >= 0.85:
        return "Auto-reply"
    if rep >= 0.60:
        return "Draft for review"
    return "Human agent"


def escalation(category: str, priority: str, level: str) -> str:
    if priority == "Critical":
        return "Escalate to senior agent immediately"
    if category in ("Warranty / defect", "Technical question") and priority == "High":
        return "Route to specialist queue"
    if level == "Human agent":
        return "Assign to agent"
    return "No escalation needed"


# ----------------------------------------------------------------------------- main entry points
def triage_ticket(text: str, segment: str | None = None) -> dict:
    category, conf = classify(text)
    priority = suggest_priority(text, category, segment)
    level = automation_level(category, conf)
    po = _extract_po(text)
    reply = REPLY_TEMPLATES[category].format(po=po, status="being processed and prepared for shipping")
    return {
        "category": category,
        "confidence": conf,
        "priority": priority,
        "automation_level": level,
        "escalation": escalation(category, priority, level),
        "draft_reply": reply,
        "method": "rules",
    }


def triage_batch(df: pd.DataFrame, text_col: str = "message_text", segment_col: str | None = "customer_segment") -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        seg = r[segment_col] if segment_col and segment_col in df.columns else None
        rows.append(triage_ticket(str(r[text_col]), seg))
    preds = pd.DataFrame(rows).add_prefix("predicted_")  # avoid clashing with existing columns
    return pd.concat([df.reset_index(drop=True), preds], axis=1)


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    df = pd.read_csv("data/tickets.csv")
    out = triage_batch(df.head(2000))
    acc = (out["category"] == out["predicted_category"]).mean()
    print(f"Category accuracy on 2,000 synthetic tickets: {acc:.1%}")
    print(out["predicted_automation_level"].value_counts(normalize=True).round(2).to_dict())
    print("\nExample:")
    print(triage_ticket("Order PO-123456 is 9 days late and our installation is on hold. Please escalate.", "OEM"))
