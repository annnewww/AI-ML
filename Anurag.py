#import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame


#load the dataset

df = pd.read_csv("C:/Users/ACIPLE1398/Downloads/archive/spam.csv")
#print(df)


#data preprocessing

data = df.where((pd.notnull(df)),'')
#print(data.head())
#print(data.info())
#print(data.shape)

#setting values of spam mail as 0 and ham mail as 1

data.loc[data['Category'] == 'spam','Category'] = '0'
data.loc[data['Category'] == 'ham','Category'] = '1'
#print(data)

#assign Column Message_body(input) and Label(target/output) to X & Y

X = data['Message']
Y = data['Category']

#print(X.head())
#print(X.shape)
#print(Y.head())
#print(Y.shape)
#print(X.info())
#print(Y.info())


# splitting testing & training data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=8)
#print(Y.shape)
#print(Y_train.shape)
#print(Y_test.shape)


feature_extraction = TfidfVectorizer()
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#train the model
my_model = LogisticRegression()
my_model.fit(X_train_features,Y_train)

#prediction on training data
prediction_on_training_data = my_model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
#print("accuracy_on_training_data=", accuracy_on_training_data)

#prediction on test data

prediction_on_testing_data = my_model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test,prediction_on_testing_data)

#print("accuracy_on_testing_data=",accuracy_on_testing_data)

#Classify mail is spam or ham
input_data = [input("Enter your mail=")]
feature_extraction_on_inputData = feature_extraction.transform(input_data)
prediction_on_input_data = my_model.predict(feature_extraction_on_inputData)

#print(prediction_on_input_data)

# Display result
if prediction_on_input_data[0] == 0:
    print("The email is classified as: SPAM")
else:
    print("The email is classified as: HAM")



from sklearn.metrics import confusion_matrix

# Generate confusion matrix on the test data
cm = confusion_matrix(Y_test, prediction_on_testing_data)

# Print confusion matrix to terminal
print("\nConfusion Matrix:")
print(cm)

# Optionally, print the matrix with more readable labels
print("\nConfusion Matrix (Formatted):")
print(f"True Negative: {cm[0][0]}  False Positive: {cm[0][1]}")
print(f"False Negative: {cm[1][0]}  True Positive: {cm[1][1]}")


