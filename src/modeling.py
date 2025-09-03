import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
from sklearn.inspection import permutation_importance
import joblib

def train_baseline_model(X_train, y_train, preprocessor):
    """
    Train a baseline logistic regression model
    """
    from sklearn.pipeline import Pipeline
    
    logit = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    logit.fit(X_train, y_train)
    return logit

def train_random_forest(X_train, y_train, preprocessor, param_grid=None):
    """
    Train a Random Forest model with hyperparameter tuning
    """
    from sklearn.pipeline import Pipeline
    
    if param_grid is None:
        param_grid = {
            'rf__max_depth': [6, 10, 20],
            'rf__n_estimators': [200, 500]
        }
    
    rf_pipe = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestClassifier(
            random_state=42, class_weight='balanced', n_jobs=-1
        ))
    ])
    
    grid = GridSearchCV(
        rf_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    
    return grid

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    results = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return results

def calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """
    Calculate permutation importance for model interpretation
    """
    # Transform test set with preprocessor
    X_test_transformed = model.named_steps['pre'].transform(X_test)
    
    # Run permutation importance on the model
    perm_res = permutation_importance(
        model.named_steps['rf'] if hasattr(model, 'named_steps') else model,
        X_test_transformed,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )
    
    # Build DataFrame
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_res.importances_mean,
        'std': perm_res.importances_std
    }).sort_values('importance', ascending=False)
    
    return perm_df

def save_model(model, filepath):
    """
    Save trained model to disk
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load trained model from disk
    """
    return joblib.load(filepath)
