import joblib
import Common_Function

# 1. Load the saved model AND the saved encoder
feature_extraction = joblib.load("ticketManagement-ai-tool/ticket_classifier_model.pkl")
encoder = joblib.load("ticketManagement-ai-tool/ticket_encoder.pkl")  # <-- Loads the exact original mapping

# 2. Get user input
issue_desc = input("Describe your issue: ")
cleaned_input = Common_Function.clean_ticket_text(issue_desc)

# 3. Predict numerical codes
numeric_prediction = feature_extraction.predict([cleaned_input])

# 4. Convert numbers back to original text labels safely
text_prediction = encoder.inverse_transform(numeric_prediction)

predicted_team = text_prediction[0][0]
predicted_priority = text_prediction[0][1]

print("\n--- Prediction Results ---")
print("Predicted Team     : ", predicted_team)
print("Predicted Priority : ", predicted_priority)