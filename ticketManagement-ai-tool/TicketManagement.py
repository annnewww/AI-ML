import joblib
import streamlit as st

import Common_Function
import os
import joblib
# --- Layout Setup ---
st.set_page_config(page_title="Intelligent Ticket Router", layout="wide")
st.title("🤖 Intelligent Incident Classification & Routing")
st.markdown("Automating IT Service Desk Triage using Machine Learning")



# Get the directory where the current Python script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ticket_classifier_model.pkl")
encoder_path = os.path.join(BASE_DIR, "ticket_encoder.pkl")

# If the .pkl files are in the same folder as TicketManagement.py:
feature_extraction = joblib.load(model_path)
encoder = joblib.load(encoder_path)

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

with st.bottom:
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Created with ❤️ by Anurag Shukla"
        "</div>",
        unsafe_allow_html=True
    )

if st.button("Analyze and Route Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter some text first!")
        st.stop()
    elif len(ticket_text) < 30:
        st.warning(
            "It seems like your message may be incomplete. Could you please share what you'd like help with? "
            "I'm here to assist with any questions related to ticket creation.\n\n"
            "Thanks, Your personal Support assistant. (If my answers are not satisfying, please log a ticket with ticketManagement Support.)"
        )
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





