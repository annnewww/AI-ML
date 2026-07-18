import joblib
import streamlit as st
import time

import Common_Function

# --- Layout Setup ---
st.set_page_config(page_title="Intelligent Ticket Router", layout="wide")
st.title("🤖 Intelligent Incident Classification & Routing")
st.markdown("Automating IT Service Desk Triage using Machine Learning")

feature_extraction = joblib.load("ticket_classifier_model.pkl")
encoder = joblib.load("ticket_encoder.pkl")

# --- User Input ---
ticket_text = st.text_area(
    "Describe your issue: ",
    placeholder="Type or paste the user complaint here..."
)

if st.button("Analyze and Route Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter some text first!")
        st.stop()
    else:
        with st.spinner("Processing text and predicting routing targets..."):
            # Mock processing delay for the demo
            time.sleep(0.8)

            # Placeholder for your model prediction logic
            # (In reality, you would call your trained pipeline or LLM API here)
            cleaned_input = Common_Function.clean_ticket_text(ticket_text)
            numeric_prediction = feature_extraction.predict([cleaned_input])
            text_prediction = encoder.inverse_transform(numeric_prediction)
            predicted_priority = text_prediction[0][0]
            assigned_team = text_prediction[0][1]
            confidence_score = 0.94

    # --- Display Results ---
    st.success("Analysis Complete!")

    col1, col2 = st.columns(2)
    try:
        with col1:
            st.metric(label="Assigned Team", value=assigned_team)
        with col2:
            st.metric(label="Priority Level", value=predicted_priority)
    except Exception as e:
        print("")

    #st.progress(confidence_score)
    #st.caption(f"Model Confidence: {confidence_score * 100:.1f}%")

    with st.bottom:
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Created with ❤️ by Anurag Shukla"
            "</div>",
            unsafe_allow_html=True
        )