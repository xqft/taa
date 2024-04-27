import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from skopt import BayesSearchCV

# Obtengo los datos

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sampleSubmission.csv')

# Pre-proceso
def get_datetime_cols(X):
    X['datetime'] = pd.to_datetime(X['datetime'])

    X['hour'] = X['datetime'].dt.hour
    X['weekday'] = X['datetime'].dt.weekday
    X['month'] = X['datetime'].dt.month
    X['year'] = X['datetime'].dt.year

    X.drop(columns=["datetime"], inplace=True)

    return X

X = df_train.copy()
y = df_train["count"]

X = X.drop(columns=["count", "casual", "registered"])
get_datetime_cols(X)

# Defino el estimador

optimization_metric = root_mean_squared_error
evaluation_metric = 'neg_root_mean_squared_log_error'

boost = xgb.sklearn.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=5,
    eval_metric=optimization_metric,
    random_state=42,
    max_depth=5,
    learning_rate=0.496
)

transf_target_regressor = TransformedTargetRegressor(
    regressor=boost,
    func=np.log1p, inverse_func=np.expm1
)

model = Pipeline([
    #('imputer', SimpleImputer()), # SimpleImputer no parece funcionar correctamente con XGBoost
    #('datetime_preprocess', FunctionTransformer(get_datetime_cols)), # FunctionTransformer idem
    ('transf_target_regressor', transf_target_regressor)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, random_state=42)

# Busqueda de hiperparametros
search_hyp = input("Utilizar hiperparametros ya encontrados? (sino realiza bayesian search) (y/n): ")
if search_hyp == "n":
        param_distributions = {
            'transf_target_regressor__regressor__max_depth': [None] + list(range(12)),
            'transf_target_regressor__regressor__learning_rate': np.linspace(0.01, 0.5, 100),
        }

        model = BayesSearchCV(
            model,
            param_distributions,
            scoring=evaluation_metric,
            refit=True,
            return_train_score=True,
            cv=3,
            n_iter=50,
            n_points=2
        )

        model.fit(X_train, y_train, transf_target_regressor__eval_set=[(X_val, y_val)])

        print("Mejor RMSLE en validacion:", -model.best_score_)
        print("Mejores parametros:", model.best_params_)
elif search_hyp == "y":
    model.fit(X_train, y_train, transf_target_regressor__eval_set=[(X_val, y_val)])
else:
    raise Exception("Respuesta invalida")


search_hyp = input("Generar submission? (y/n): ")
if search_hyp == "y":
    X_test = df_test.copy()
    submission = pd.DataFrame()
    submission["datetime"] = df_test["datetime"]
    get_datetime_cols(X_test)

    submission["count"] = np.round(model.predict(X_test)).astype(int)
    submission.to_csv("submission.csv", index=False)
elif search_hyp != "n":
    raise Exception("Respuesta invalida")
