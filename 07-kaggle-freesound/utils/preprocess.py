from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
import constantes as const


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.std_attribs = const.std_attribs
        self.uniform_attribs = const.uniform_attribs
        self.cat_attribs = const.cat_attribs
        self.power_law_attribs = const.power_law_atrribs
        self.twobin_attribs = const.twobin_attribs
        self.feature_names_ = None

        # Standard pipeline
        self.std_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        # Pipeline for uniform distributions
        self.uniform_pipeline = Pipeline([
            ('imputer', FillWithPrevious())
        ])

        # Pipeline for power law distributions
        self.power_law_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('log_transform', FunctionTransformer(np.log1p)),
        ])

        # Pipeline for binary attributes
        self.binary_pipeline = Pipeline([
            ('imputer', FillWithPrevious()),
            ('twobins', KBinsDiscretizer(encode='ordinal', strategy='uniform',
                                         subsample=None))
        ])

        default_imputer = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('std_scaler', StandardScaler()),
        ])

        # Full pipeline
        self.full_pipeline = ColumnTransformer([
            ("std", self.std_pipeline, self.std_attribs),
            ("cat", OneHotEncoder(), self.cat_attribs),
            ("uni", self.uniform_pipeline, self.uniform_attribs),
            ("plaw", self.power_law_pipeline, self.power_law_attribs),
            ('twobins', self.binary_pipeline, self.twobin_attribs)
        ], remainder=default_imputer)

    def fit(self, X, y=None):
        # Fit the full pipeline
        self.full_pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        # Transform the data
        self.feature_names_ = self.get_feature_names(self.full_pipeline)
        return self.full_pipeline.transform(X)

    def fit_transform(self, X, y=None):
        # Fit and transform
        return self.full_pipeline.fit_transform(X, y)

    def get_feature_names(self, transformer):
        # Función para extraer los nombres de las características después de las transformaciones
        output_features = []
        # Iterar sobre los transformadores en el ColumnTransformer
        for name, trans, column, in transformer.transformers_[:-1]: # Excluyendo el último transformador 'remainder'
            if hasattr(trans, 'get_feature_names'):
                # Si el transformador es OneHotEncoder, obtiene los nombres de las características de esta manera
                if isinstance(trans, Pipeline):
                    # Si el transformador es un Pipeline, accede al último paso para los nombres de las características
                    output_features.extend(trans.steps[-1][1].get_feature_names(input_features=column))
                else:
                    output_features.extend(trans.get_feature_names(input_features=column))
            else:
                # Si el transformador no modifica los nombres de las características (como SimpleImputer)
                output_features.extend(column)
        return output_features



# Helper class for filling with the previous value
class FillWithPrevious(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.ffill().bfill()
