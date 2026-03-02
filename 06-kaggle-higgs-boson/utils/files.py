import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

test = pd.read_csv('test.csv')
dataset = pd.read_csv('training.csv')

# Matriz X contiene los datos de entrenamiento
X = dataset.copy()
X = X.drop('Weight', axis=1)
X = X.drop('Label', axis=1)
X.set_index('EventId')

# Serie y contiene los labels
y = dataset['Label']

# Los pesos los usaremos durante el entrenamiento
event_id = dataset['EventId']
weights = dataset['Weight']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
     X, y_encoded, weights, test_size=0.20, random_state=4546546)

X_val, X_test, y_val, y_test, weights_val, weights_test  = train_test_split(X_test, y_test, weights_test,  test_size=0.25, random_state=4546546)