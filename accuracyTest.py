import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Common_Function

# ==========================================
# 1. LOAD MODEL, ENCODER, AND TEST DATA
# ==========================================
feature_extraction = joblib.load("ticketManagement-ai-tool/ticket_classifier_model.pkl")
encoder = joblib.load("ticketManagement-ai-tool/ticket_encoder.pkl")

# Load the original dataset to recreate the test split
df_pd = pd.read_csv("C:/Users/ACIPLE1398/Downloads/archive_dataset/IT_Support_Ticket_Data.csv")
df_pd = df_pd.where((pd.notnull(df_pd)), '')
df_pd['cleaned_desc'] = df_pd['Body'].apply(Common_Function.clean_ticket_text)

X = df_pd["cleaned_desc"]
Y = df_pd[["Department", "Priority"]]

# Encode the targets using the loaded encoder
Y_encoded = encoder.transform(Y)

# Recreate the exact same test split using random_state=8
_, X_test, _, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=8)


# ==========================================
# 2. CHECK MODEL ACCURACY
# ==========================================
# Method A: Quick Built-in Score
# This returns the subset accuracy (both labels must be correct for a row to count as correct)
subset_accuracy = feature_extraction.score(X_test, Y_test)

# Method B: Detailed Element-wise Accuracy (Optional but helpful)
Y_pred = feature_extraction.predict(X_test)
# Department accuracy is column 0, Priority accuracy is column 1
dept_accuracy = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
priority_accuracy = accuracy_score(Y_test[:, 1], Y_pred[:, 1])

print("--- MODEL ACCURACY METRICS ---")
print(f"Overall Exact-Match Accuracy: {subset_accuracy * 100:.2f}%")
print(f"Department-only Accuracy:     {dept_accuracy * 100:.2f}%")
print(f"Priority-only Accuracy:       {priority_accuracy * 100:.2f}%")
print("-------------------------------\n")

"""
# ==========================================
# 3. RUN LIVE USER PREDICTION
# ==========================================
issue_desc = input("Describe your issue: ")
cleaned_input = Common_Function.clean_ticket_text(issue_desc)

numeric_prediction = feature_extraction.predict([cleaned_input])
text_prediction = encoder.inverse_transform(numeric_prediction)

predicted_team = text_prediction[0][0]
predicted_priority = text_prediction[0][1]

print("\n--- Prediction Results ---")
print("Predicted Team     : ", predicted_team)
print("Predicted Priority : ", predicted_priority)
"""