#import comet_ml in the top of your file
#from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
#sns.set_theme(style="whitegrid")

import keras_tuner as kt
import tensorflow as tf

from sklearn.model_selection import train_test_split

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sampleSubmission.csv')

df_train['datetime'] = pd.to_datetime(df_train['datetime'])

df_train['hour'] = df_train['datetime'].dt.hour
df_train['weekday'] = df_train['datetime'].dt.weekday
#df_train['month'] = df_train['datetime'].dt.month 
df_train['year'] = df_train['datetime'].dt.year

y_train_full = df_train['count']
df_train = df_train.drop(['datetime', 'casual', 'registered', 'count'], axis=1) # hay que eliminarlas ya que tiene relación directa con la columna objetivo y no aparecen en el conjunto de *test*.

#

X_train, X_val, y_train, y_val = train_test_split(df_train, y_train_full, train_size=0.9, random_state=42)

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)
X_train = norm_layer(X_train)

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_val)
X_val = norm_layer(X_val)

#

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=8)
    n_neurons = []
    for i in range(n_hidden):
        n_neurons.append(hp.Int(f"n_neurons{i}", min_value=16, max_value=50, default=35))
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=0.0033138)
   
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_hidden):
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(n_neurons[i], activation="relu"))
        
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(loss="msle", optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate))
    
    return model

hyperband = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=80,
    overwrite=True,
    directory="nn_model",
    project_name="nn_model_hyperband",
    seed=42
)

early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
hyperband.search(X_train, y_train, epochs=80, validation_data=(X_val, y_val), callbacks=[early])

#

best_model = hyperband.get_best_models()[0]
best_model.save("./best_hyperband")
#best_model = tf.keras.models.load_model('./best_hyperband')
best_model.evaluate(X_val, y_val)

#

from sklearn.model_selection import train_test_split

X = df_test.copy()

X['datetime'] = pd.to_datetime(X['datetime'])
X['hour'] = X['datetime'].dt.hour
X['weekday'] = X['datetime'].dt.weekday
#X['month'] = X['datetime'].dt.month
X['year'] = X['datetime'].dt.month

# "count" es la columna objetivo, "casual" y "registered" son parte del objetivo
# pero solo queremos predecir la cantidad total de bicicletas alquiladas.
X = X.drop(columns=["datetime"])

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X)
X = norm_layer(X)

y = pd.DataFrame()
y["datetime"] = df_test["datetime"]
y["count"] = np.round(best_model.predict(X)).astype(int)

y.to_csv("submission.csv", index=False)
