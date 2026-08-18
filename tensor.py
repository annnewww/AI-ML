import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model

data = pandas.read_csv("C:/Users/ACIPLE1398/Downloads/student+performance/student-mat.csv", sep = ";")
upd_data = data[["G1","G2","G3","studytime","freetime","absences","health"]]
#print("before operation")
#print(upd_data.head())
#print(data[["health","absences","studytime","failures","traveltime","age"]])

predict = "G3"

X = numpy.array(upd_data.drop(columns = [predict]))
#print("After operation=")
#print(X)

Y = numpy.array(data[predict])

X_test,X_train,Y_test,Y_train = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

model_1 = linear_model.LinearRegression()

model_1.fit(X_train,Y_train)
accuracy = model_1.score(X_test,Y_test)
print("model accuracy = ",accuracy)

prediction = model_1.predict(X_test)

for x in range(len(prediction)):
    print(prediction[x], X_test[x], Y_test[x])