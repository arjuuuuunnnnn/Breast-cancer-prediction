import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.models import Sequential
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")
data = data.dropna(axis=1)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'].values)

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

model = Sequential([
    layers.Dense(64, input_shape=(31,), activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
    # for binary final output we use sigmoid activation
])

model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)
prediction = model.predict(X_test)
# print(prediction)
predictions = []
for i in prediction:
    if i < 0.5:
        predictions.append(0)
    else:
        predictions.append(1)
print(predictions)
print("Accuracy : ", accuracy_score(Y_test, predictions))
