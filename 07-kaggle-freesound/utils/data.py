import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import constantes as const
import numpy as np


class DataManager():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test = None
        self.encoder = None
        self.event_id = None
        self.X = None
        self.y = None
        self.weights = None

    def preparar_conjuntos(self):
        test = pd.read_csv('higgs-boson/test.csv')
        dataset = pd.read_csv('higgs-boson/training.csv')

        test.replace(-999, np.nan, inplace=True)
        dataset.replace(-999, np.nan, inplace=True)

        # Matriz X contiene los datos de entrenamiento
        X = dataset.copy()
        X = X.drop('Weight', axis=1)
        X = X.drop('Label', axis=1)
        X.set_index('EventId')

        # Serie y contiene los labels
        y = dataset['Label']
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Los pesos los usaremos durante el entrenamiento
        event_id = dataset['EventId']
        weights = dataset['Weight']

        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=const.RANDOM_STATE)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.weights_train = weights_train
        self.weights_test = weights_test
        self.test = test
        self.event_id = event_id
        self.X = X
        self.y = y
        self.weights = weights

    def preparar_conjuntos_ROS(self):
        test = pd.read_csv('higgs-boson/test.csv')
        dataset = pd.read_csv('higgs-boson/training.csv')

        test.replace(-999, np.nan, inplace=True)
        dataset.replace(-999, np.nan, inplace=True)

        # Matriz X contiene los datos de entrenamiento
        X = dataset.copy()
        #X = X.drop('Weight', axis=1)
        X = X.drop('Label', axis=1)
        X.set_index('EventId')

        # Serie y contiene los labels
        y = dataset['Label']
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        # Los pesos los usaremos durante el entrenamiento
        event_id = dataset['EventId']
        ros =  RandomOverSampler(sampling_strategy= 'not majority', random_state=42 )
        X_res, y_res = ros.fit_resample(X,y)
        y = y_res
        weights = X_res['Weight']
        X_res = X_res.drop(['Weight'], axis=1)
        X = X_res
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=const.RANDOM_STATE)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test = test
        self.X = X
        self.y = y


    def get_train_data(self):
        return self.X_train, self.y_train, self.weights_train

    def get_test_data(self):
        return self.X_test, self.y_test, self.weights_test

    def get_test(self):
        return self.test

    def get_encoder(self):
        return self.encoder

    def get_event_id(self):
        return self.event_id

    def get_X_y(self):
        return self.X, self.y