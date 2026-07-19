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

def determine_priority_directly(text):
    # Your custom list of urgency words
    high_urgency = {'critical', 'blocking', 'down', 'broken', 'stopped', 'urgent', 'failure', 'crash'}
    medium_urgency = {'error', 'issue', 'warning', 'slow', 'failed'}

    # Tokenize and lowercase the input text words
    words = set(text.lower().split())

    # Count structural keyword intersections
    high_matches = len(words.intersection(high_urgency))
    medium_matches = len(words.intersection(medium_urgency))

    # Assign target structural priority tier
    if high_matches >= 1:
        return "🔴 High"
    elif medium_matches >= 1:
        return "🟡 Medium"
    else:
        return "🟢 Low"

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

            cleaned_input = Common_Function.clean_ticket_text(ticket_text)
            calculated_priority = determine_priority_directly(cleaned_input)
            numeric_prediction = feature_extraction.predict([cleaned_input])
            text_prediction = encoder.inverse_transform(numeric_prediction)
            assigned_team = text_prediction[0]

    # --- Display Results ---
    st.success("Analysis Complete!")

    col1, col2 = st.columns(2)
    try:
        with col1:
            st.metric(label="Assigned Team", value=assigned_team)
        with col2:
            st.metric(label="Calculated Priority", value=calculated_priority)
    except Exception as e:
        print("exception occured", e)


    with st.bottom:
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Created with ❤️ by Anurag Shukla"
            "</div>",
            unsafe_allow_html=True
        )


