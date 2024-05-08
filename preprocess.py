import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import umap

# Méthode de sklearn:
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class Preprocess():

    def __init__(self, df, numeric_features, categorial_features):
        self.df = df
        self.numeric_features = numeric_features
        self.categorial_features = categorial_features

    def preprocessing_pipeline(self):
        preprocessing_pipeline = ColumnTransformer(transformers = [
            ('numeric', make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), self.numeric_features),
            ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), self.categorical_features)
        ])

        return Preprocess(preprocessing_pipeline.fit_transform(self.df), self.numeric_features, self.categorical_features)
    
    def remove_na_columns(self, treshold_rate):
        removed_col = []
        new_df = self.data.copy()
        new_categorical_features = self.categorical_features.copy()
        new_numeric_features = self.numeric_features.copy()
        for col in self.df.columns :
            if self.df[col].isna().sum() > treshold_rate * self.df.shape[0]:

                if col in self.numeric_features:
                    new_numeric_features.remove(col)
                elif col in self.categorial_features:
                    new_categorical_features.remove(col)

                removed_col.append(col)
                new_df = new_df.drop(col, axis = 1)

        print (f'The number of removed columns {len(removed_col)}')
        return Preprocess(new_df, new_numeric_features, new_categorical_features)