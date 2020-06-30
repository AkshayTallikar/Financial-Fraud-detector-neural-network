import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.utils import shuffle
from helpers import resize_to_fit
data = pd.read_csv("PS_20174392719_1491204439457_log.csv", date_parser=True)
print(data.head())

data = data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', ]]

predict2 = 'isFraud'
X = np.array(data.drop([predict2], 1))
y = np.array(data[predict2])
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2) 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=512)

results = model.evaluate(X_test, y_test)
print(results)