import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
try:
    df = pd.read_csv('C:/Users/anurag4.shukla/Pictures/Personal DATA/spam.csv',encoding='latin1')
    #print(df)

    data = df.where((pd.notnull(df)),'')
    #print(data.head(10))
    #print(data.info())
    #print(data.shape)
    data.loc[data['v1'] == 'ham','v1'] = 1
    data.loc[data['v1'] == 'spam','v1'] = 0
    a = data['v2']
    b = data['v1']
    #print(a)
    #print(b)
    a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.3,random_state=8)
    #print(b.shape)
    #print(b_test.shape)
    #print(b_train.shape)

    feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
    a_train_features =  feature_extraction.fit_transform(a_train)
    a_test_features =   feature_extraction.transform(a_test)

    b_train = b_train.astype('int')
    b_test = b_test.astype('int')
    #print(a_train_features)
    model = LogisticRegression()
    model.fit(a_train_features,b_train)
    prediction_on_training_data = model.predict(a_train_features)
    accuracy_on_training_data = accuracy_score(b_train,prediction_on_training_data)
    #print("accuracy on training data: ",accuracy_on_training_data)

    prediction_on_testing_data = model.predict(a_test_features)
    accuracy_on_testing_data = accuracy_score(b_test,prediction_on_testing_data)
    #print("accuracy on testing data: ",accuracy_on_testing_data)

    #now test your mail

    input_data = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
    input_data_features = feature_extraction.transform(input_data)
    prediction_on_input_data = model.predict(input_data_features)
    print("Your mail is = ",prediction_on_input_data)





except Exception as e:
    print("exception=",e)
