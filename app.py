# app.py
import os
import joblib
import pandas as pd
import streamlit as st
from utils import process_complaint_pipeline

st.set_page_config(page_title="AI Ticket Triaging System", page_icon="⚙️", layout="wide")

MODEL_PATH = os.path.join("models", "complaint_model.joblib")

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)
pipeline = load_pipeline()

# Session State for ticket logging (Dashboard ke liye)
if "ticket_history" not in st.session_state:
    st.session_state["ticket_history"] = []

st.title("Automated Ticket Triaging & Smart Routing Engine")
st.caption("AI-driven Customer Complaint Processing • Real-time Triage, Routing & Analytics")

if pipeline is None:
    st.error("Model file nahi mili! Terminal me `python train.py` run karein.")
    st.stop()

# Multi-tab layout: Triaging Screen vs Manager Analytics Dashboard
tab1, tab2 = st.tabs(["📥 Live Ticket Triaging", "📊 Manager Analytics Dashboard"])

# ================= TAB 1: TRIAGING & CLASSIFICATION =================
with tab1:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        complaint_input = st.text_area(
            "Enter incoming customer complaint:",
            placeholder="Example: My account was hacked! Please reverse the unauthorized transaction TXN-98442 of $450 immediately.",
            height=130
        )

    with col_meta:
        st.markdown("**Quick Test Templates:**")
        if st.button("Billing & Money Double Charged"):
            complaint_input = "I was charged twice on my card for $120. Order ID is ORD-8831. Refund my money immediately!"
        if st.button("Security & Account Breach"):
            complaint_input = "My account was hacked and unauthorized login changed my email! Lock my account."

    if st.button("Process & Route Ticket", type="primary"):
        if not complaint_input.strip():
            st.warning("Please enter complaint text.")
        else:
            # 1. ML Model Prediction
            predicted_category = pipeline.predict([complaint_input])[0]
            probs = pipeline.predict_proba([complaint_input])[0]
            confidence = max(probs) * 100

            # 2. Rule & Heuristic Triaging
            ticket_data = process_complaint_pipeline(complaint_input, predicted_category, confidence)
            
            # Save into memory history
            st.session_state["ticket_history"].append(ticket_data)

            st.divider()
            st.subheader(f"Ticket Processed: `{ticket_data['ticket_id']}`")

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Category", ticket_data["predicted_category"])
            m2.metric("Urgency", ticket_data["urgency_level"])
            m3.metric("Priority Score", f"{ticket_data['priority_score']} / 5")
            m4.metric("Confidence", f"{ticket_data['confidence_score']}%")

            # Smart Routing & Entities
            c_route, c_entities = st.columns(2)
            with c_route:
                st.markdown(f"**Assigned Smart Queue:** `{ticket_data['routing_queue']}`")
                st.markdown(f"**Customer Sentiment:** `{ticket_data['customer_sentiment']}`")
                st.markdown(f"**Suggested Next Action:** `{ticket_data['suggested_action']}`")

            with c_entities:
                st.markdown("**Extracted Entities (Regex):**")
                st.json(ticket_data["extracted_entities"])

            # AI Draft Reply & JSON Output
            st.subheader("Agent Suggested Reply")
            st.info(ticket_data["draft_reply"])

            with st.expander("Show Structured JSON Output (API Payload)"):
                st.json(ticket_data)

# ================= TAB 2: MANAGER ANALYTICS DASHBOARD =================
with tab2:
    st.subheader("Support Operations & Real-Time Analytics")
    history = st.session_state["ticket_history"]

    if not history:
        st.info("Abhi tak koi ticket process nahi hua hai. 'Live Ticket Triaging' tab se kuch tickets run karein.")
    else:
        df_history = pd.DataFrame(history)

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Total Tickets Processed", len(df_history))
        critical_count = len(df_history[df_history["urgency_level"] == "Critical"])
        col_kpi2.metric("Critical / P5 Incidents", critical_count)
        avg_confidence = df_history["confidence_score"].mean()
        col_kpi3.metric("Avg Classifier Confidence", f"{avg_confidence:.1f}%")

        st.divider()

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("### Category Distribution")
            cat_counts = df_history["predicted_category"].value_counts()
            st.bar_chart(cat_counts)

        with col_chart2:
            st.markdown("### Urgency Levels")
            urgency_counts = df_history["urgency_level"].value_counts()
            st.bar_chart(urgency_counts)

        st.markdown("### Raw Ticket Stream Log")
        st.dataframe(df_history[["ticket_id", "timestamp", "predicted_category", "priority_score", "urgency_level", "routing_queue"]])