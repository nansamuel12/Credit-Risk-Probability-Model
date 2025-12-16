
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
