import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from training.train_utils import DATA_FILE_PATH, MODEL_PATH, MODEL_DIR




df = (
    pd.read_csv(DATA_FILE_PATH)
    .drop_duplicates()
    .drop(columns=['name', 'edition'])
)



X = df.drop(columns=['selling_price'])
y = df['selling_price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = [col for col in X_train.columns if col not in num_cols]

freq_cols = [col for col in cat_cols if X_train[col].nunique() >= 50]
ohe_cols = [col for col in cat_cols if col not in freq_cols]



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_ = None  # sklearn-fitted attribute

    def fit(self, X, y=None):
        self.freq_maps_ = []
        for col_idx in range(X.shape[1]):
            values, counts = np.unique(X[:, col_idx], return_counts=True)
            freqs = counts / counts.sum()
            self.freq_maps_.append(dict(zip(values, freqs)))
        return self

    def transform(self, X):
        X_out = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            freq_map = self.freq_maps_[col_idx]
            X_out[:, col_idx] = [
                freq_map.get(val, 0) for val in X[:, col_idx]
            ]
        return X_out


num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Separate pipelines for low-cardinality (one-hot) and high-cardinality (frequency) categorical cols
ohe_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

freq_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('freq', FrequencyEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('ohe', ohe_pipe, ohe_cols),
    ('freq', freq_pipe, freq_cols)
])


regressor = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=6, n_jobs=-1)

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

rf_model.fit(X_train, y_train)


os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, MODEL_PATH)