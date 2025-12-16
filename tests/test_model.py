
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.data_processing import build_training_pipeline

def test_training_pipeline_logic():
    # Mock Data
    X = pd.DataFrame({
        'ChannelId': ['Web', 'Mobile', 'Web'],
        'ProductCategory': ['A', 'B', 'A'],
        'Amount': [100, 50, 200]
    })
    y = pd.Series([0, 1, 0])
    
    # Build
    cat_cols = ['ChannelId', 'ProductCategory']
    num_cols = ['Amount']
    preprocessor = build_training_pipeline(cat_cols, num_cols)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=10))
    ])
    
    # Fit
    pipeline.fit(X, y)
    
    # Predict
    preds = pipeline.predict(X)
    assert len(preds) == 3

def test_data_splitting():
    # Verify train_test_split logic preserves stratification
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X = np.random.rand(8, 2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    
    assert len(y_train) == 4
    assert len(y_test) == 4
    assert sum(y_train) == 2 # 50% ones
    assert sum(y_test) == 2 # 50% ones
