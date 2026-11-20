# utils.py
import re
import uuid
from datetime import datetime

def extract_entities(text: str) -> dict:
    """Regex se Order ID, Transaction ID, aur Amount extract karta hai."""
    entities = {
        "order_id": None,
        "transaction_id": None,
        "amount": None
    }
    
    # Order ID detection (e.g., ORD-12345, ORDER #98765)
    order_match = re.search(r'\b(?:order|ord)[\s#:-]*([a-zA-Z0-9]{4,10})\b', text, re.IGNORECASE)
    if order_match:
        entities["order_id"] = order_match.group(1).upper()

    # Transaction/Txn ID detection (e.g., TXN-9988, TXN123456)
    txn_match = re.search(r'\b(?:txn|transaction)[\s#:-]*([a-zA-Z0-9]{5,15})\b', text, re.IGNORECASE)
    if txn_match:
        entities["transaction_id"] = txn_match.group(1).upper()

    # Amount extraction (e.g., $50, ₹1500, Rs. 500, 1000 INR)
    amount_match = re.search(r'(?:[\$₹]|rs\.?|inr)\s?(\d+(?:,\d+)*(?:\.\d{1,2})?)', text, re.IGNORECASE)
    if amount_match:
        entities["amount"] = amount_match.group(0)

    return entities

def analyze_sentiment_and_tone(text: str) -> str:
    """Rule-based negative tone analyzer."""
    clean = text.lower()
    angry_words = ["worst", "scam", "pathetic", "useless", "terrible", "lawyer", "police", "fraud", "sue"]
    if any(word in clean for word in angry_words):
        return "Highly Frustrated / Threatening"
    elif any(word in clean for word in ["angry", "bad", "disappointed", "slow", "error", "issue"]):
        return "Negative"
    return "Neutral"
def assign_routing_queue(category: str, urgency: str) -> str:
    """Smart routing queue define karta hai."""
    if urgency == "Critical":
        return "🚨 Escalation Desk (L3 / Senior Ops)"
    
    routing_map = {
        "Account Security": "🔒 CyberSec & Fraud Prevention Unit",
        "Billing": "💳 Accounts & Finance Desk",
        "Technical Issues": "💻 L2 Technical Support Queue",
        "Product Quality": "📦 QA & Inventory Operations",
        "Service Delivery": "🚚 Logistics & Dispatch Control"
    }
    return routing_map.get(category, "General Support Tier-1")

def process_complaint_pipeline(text: str, category: str, confidence: float) -> dict:
    """
    Final Triaging Engine: ML classification aur rules ko combine karke
    complete structured JSON payload return karta hai.
    """
    clean_text = text.lower()

    # Heuristic checks
    critical_words = ["hacked", "breached", "unauthorized", "fraud", "stolen", "compromised", "police", "legal"]
    high_words = ["refund", "charged twice", "lost money", "double deduction", "cheated"]
    
    # Priority & Urgency determination
    if category == "Account Security" or any(re.search(rf"\b{w}\b", clean_text) for w in critical_words):
        urgency = "Critical"
        priority_score = 5
    elif any(re.search(rf"\b{w}\b", clean_text) for w in high_words):
        urgency = "High"
        priority_score = 4
    elif any(w in clean_text for w in ["delayed", "crash", "bug", "broken", "missing"]):
        urgency = "Medium"
        priority_score = 3
    else:
        urgency = "Low"
        priority_score = 2

    entities = extract_entities(text)
    sentiment = analyze_sentiment_and_tone(text)
    assigned_queue = assign_routing_queue(category, urgency)

    # Suggested Auto Actions
    action_map = {
        "Critical": "Freeze associated account credentials / Trigger immediate L3 incident alert.",
        "High": "Validate financial ledger transaction and request payment gateway trace.",
        "Medium": "Forward device logs and error trace to technical engineering.",
        "Low": "Dispatch standard SLA resolution email within 24-48 business hours."
    }

    # Structured JSON Payload
    structured_ticket = {
        "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_category": category,
        "confidence_score": round(confidence, 2),
        "priority_score": priority_score,
        "urgency_level": urgency,
        "customer_sentiment": sentiment,
        "extracted_entities": entities,
        "routing_queue": assigned_queue,
        "suggested_action": action_map.get(urgency, "Review manually."),
        "draft_reply": f"Hello, we have registered your ticket regarding '{category}'. Our team has routed this to '{assigned_queue}'. Current SLA priority is level {priority_score}."
    }

    return structured_ticket