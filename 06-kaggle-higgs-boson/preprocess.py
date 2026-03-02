from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

class FillWithPrevious(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.ffill().bfill()

def std_pipeline():
    std_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    return std_pipeline


def create_fullpipeline():
    std_attribs = ['PRI_jet_leading_eta', 'PRI_jet_subleading_eta', 'DER_mass_MMC', 'DER_prodeta_jet_jet']
    uniform_attribs = ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi']
    cat_attribs = ['PRI_jet_num']
    power_law_atrribs = ['PRI_jet_leading_pt','PRI_jet_subleading_pt','DER_mass_jet_jet','PRI_jet_subleading_pt','DER_deltaeta_jet_jet', 'PRI_jet_leading_pt']
    twobin_attribs = ['DER_lep_eta_centrality']

    std_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    uniform_pipeline = Pipeline([
        ('imputer', FillWithPrevious())
    ])

    power_law_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), 
        ('log_transform', FunctionTransformer(np.log1p)),
    ])

    binary_pipeline = Pipeline([
        ('imputer', FillWithPrevious()),
        ('twobins', KBinsDiscretizer(encode='ordinal', strategy='uniform'))
    ])

    full_pipeline = ColumnTransformer([
        ("std", std_pipeline, std_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
        ("uni", uniform_pipeline, uniform_attribs),
        ("plaw", power_law_pipeline, power_law_atrribs),
        ('twobins', binary_pipeline, twobin_attribs)
    ], remainder=std_pipeline)

    return full_pipeline