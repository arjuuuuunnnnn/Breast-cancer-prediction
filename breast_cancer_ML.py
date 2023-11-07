# to predict whether the breast cancer is Malignant or Benign
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")
# print(data['Unnamed: 32'].head())
# print(data.columns)
data = data.dropna(axis=1)
# print(data.columns)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'].values)
# print(data.head())

features = []
for i in data.columns:
    if i != "diagnosis":
        features.append(i)
# print(features)
target = "diagnosis"
X = data[features]
Y = data[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Selecting model based on accuracy
# def models(X_train, Y_train):
#     log = LogisticRegression(random_state=0)
#     log.fit(X_train, Y_train)
#
#     tree = DecisionTreeClassifier(random_state=0, criterion="entropy")
#     tree.fit(X_train, Y_train)
#
#     forest_classifier = RandomForestClassifier(random_state=0, criterion="entropy", n_estimators=10)
#     forest_classifier.fit(X_train, Y_train)
#
#     return log, tree, forest_classifier
#
# model = models(X_train, Y_train)
# for i in range(len(model)):
#     print("model ", i)
#     print("accuracy : ", accuracy_score(Y_test, model[i].predict(X_test)))
# By this I got that Random Forest Classifier is having high accuracy score and I can use that

model = RandomForestClassifier(random_state=0, criterion="entropy")
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
print(accuracy_score(Y_test, prediction))

predictions = []
for i in prediction:
    if i == 1:
        predictions.append("M")
    else:
        predictions.append("B")
print(predictions)
