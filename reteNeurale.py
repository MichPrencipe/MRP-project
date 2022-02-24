# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:15:32 2021

@author: Michele
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split

lista = [x for x in "ABCDEFGHIKLMNOPQRSTUVWXYZ"]
lista.append("t_eff")
df = pd.read_csv("input.csv") 
df = df.drop(index = df[df["t_eff"] == 0].index)

X = df.drop(columns = ["t_eff"])
y = df["t_eff"]


"""splitto i modelli""" 
X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=1/3, random_state=42)


def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def print_eval(X, y, model):
    preds = model.predict(X)
    print("   Mean squared error: {:.5}".format(mean_squared_error(y, preds)))
    print("       Relative error: {:.5%}".format(relative_error(y, preds)))
    print("R-squared coefficient: {:.5}".format(r2_score(y, preds)))

"""creo un modello di regressione lineare semplice e lo addestro"""
lrm = Pipeline([
    ("scale",  StandardScaler()),
    ("linreg", LinearRegression())
])
lrm.fit(X_train, y_train)
print_eval(X_val, y_val, lrm)

"""influenza dei parametri sul tempo"""
print(pd.Series(lrm.named_steps["linreg"].coef_, X_train.columns))


"""PROVA CON TENSORFLOW"""
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))

train_dataset = train_dataset.shuffle(len(X)).batch(5)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))

val_dataset = val_dataset.shuffle(len(X)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape = [X.shape[1]]),
        tf.keras.layers.Dense(1)
      ]) 
  
  model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())
  return model

tf.keras.backend.clear_session()
model = get_compiled_model()

history = model.fit(train_dataset , epochs = 100)

model2 = get_compiled_model()

history2 = model2.fit(val_dataset, epochs = 100)

val_hist = history2.history["loss"]
loss_hist = history.history["loss"]
plt.plot(range(100), loss_hist)
plt.grid()
plt.xlabel("epochs");plt.ylabel("loss")
plt.plot(range(100), val_hist)
plt.legend(['val loss', 'train loss'], loc='upper left')
     


