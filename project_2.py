import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.svm import LinearSVC

import Common_Function

pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
df_pd = pd.read_csv("C:/Users/ACIPLE1398/Downloads/archive_dataset/IT_Support_Ticket_Data.csv")


le = preprocessing.LabelEncoder()
Department = le.fit_transform(list(df_pd["Department"]))
df_pd = df_pd.drop(columns=["Priority","Tags"])
df_pd = df_pd.where((pd.notnull(df_pd)),'')

df_pd['cleaned_desc'] = df_pd['Body'].apply(Common_Function.clean_ticket_text)
dfpredict = "cleaned_desc"
X = df_pd[dfpredict].tolist()
y= list(Department)
joblib.dump(le, 'ticketManagement-ai-tool/ticket_encoder.pkl')

X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size=0.2,random_state=26 )
feature_extraction = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True,ngram_range=(1, 2),max_features=10000)),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42))
])
feature_extraction.fit(X_train,Y_train)
joblib.dump(feature_extraction, 'ticketManagement-ai-tool/ticket_classifier_model.pkl')
predictions = feature_extraction.predict(X_test)

"""Check the model performance
accuracy = accuracy_score(Y_test, predictions)
#accu_on_testingData = feature_extraction.score(X_test,Y_test)
print("accuracy=", accuracy)
if accuracy > best:
    best = accuracy


issue_desc=input("Describe your issue= ")
cleaned_input = Common_Function.clean_ticket_text(issue_desc)
predictionOnUserInput_encoded = feature_extraction.predict([cleaned_input])
predictionOnUserInput_text = le.inverse_transform(predictionOnUserInput_encoded)
print("predictionOnUserInput_text = ", predictionOnUserInput_text )
"""
"""
#X = df_pd[dfpredict]
#print(X.head())
#print("My actual columns are X:", X.head())

#Y = df_pd.drop(columns = ["Body",dfpredict,"Tags"])
Y = df_pd[["Department","Priority"]]
#print("My actual columns are Y:", Y.columns.tolist())

encoder = OrdinalEncoder()
Y_encoded = encoder.fit_transform(Y)
print("Y_encoded", Y_encoded)
joblib.dump(encoder, 'ticket_encoder.pkl')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y_encoded, test_size=0.2,random_state=8 )

feature_extraction = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('classifier', MultiOutputClassifier(LogisticRegression(solver='lbfgs',max_iter=10000,class_weight='balanced')))
])

feature_extraction.fit(X_train,Y_train)


print("\n🔍 --- RUNNING LIVE MOCK TESTS ---")
test_tickets = [
    "DATABASE CRASH: Oracle production database connection timeout error alert.",
    "Hey, I forgot my login credentials and need a password reset ASAP.",
    "The UI is completely broken on the homepage, buttons are not clickable."
]

# Run predictions
mock_predictions_encoded = feature_extraction.predict(test_tickets)
mock_predictions_text = encoder.inverse_transform(mock_predictions_encoded)

for ticket, pred in zip(test_tickets, mock_predictions_text):
    print(f"\n🎫 Ticket: {ticket}")
    print(f"🤖 Predicted: Team -> {pred[0]}, Priority -> {pred[1]}")
print("-----------------------------------\n")

feature_extraction = joblib.load("ticket_classifier_model.pkl")
accu_on_testingData = feature_extraction.score(X_test,Y_test)
#print("Accuracy on testing = ",accu_on_testingData)

joblib.dump(feature_extraction, 'ticket_classifier_model.pkl')

#print("Accuracy on testing=" , accu_on_testingData)

issue_desc=input("Describe your issue= ")
cleaned_input = Common_Function.clean_ticket_text(issue_desc)

# 1. Model predicts numerical codes (e.g., [[0., 1., 0.]])
numeric_prediction = feature_extraction.predict([cleaned_input])

# 2. Convert those numbers back to real text words!
text_prediction = encoder.inverse_transform(numeric_prediction)

predicted_team = text_prediction[0][0]
predicted_priority = text_prediction[0][1]
print("predicted_team = ", predicted_team)
print("predicted_priority = ",predicted_priority)
"""