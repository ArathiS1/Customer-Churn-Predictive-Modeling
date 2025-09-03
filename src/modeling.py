from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, 
                           confusion_matrix, average_precision_score)
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def train_baseline_model(X: pd.DataFrame, y: pd.Series, preprocessor, 
                        test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Train and evaluate baseline logistic regression model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    logit = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    logit.fit(X_train, y_train)
    y_pred_proba = logit.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    results = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'model': logit
    }
    
    return results, X_test, y_test

def train_random_forest(X: pd.DataFrame, y: pd.Series, preprocessor, 
                       param_grid: Dict[str, Any], cv_folds: int = 5,
                       scoring: str = 'roc_auc', test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
    """Train and optimize Random Forest model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    rf_pipe = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestClassifier(
            n_estimators=200, random_state=random_state, 
            class_weight='balanced', n_jobs=-1
        ))
    ])
    
    grid = GridSearchCV(
        rf_pipe, param_grid, cv=cv_folds, scoring=scoring, 
        n_jobs=-1, verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    y_pred_proba = grid.predict_proba(X_test)[:, 1]
    
    results = {
        'best_score': grid.best_score_,
        'best_params': grid.best_params_,
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_avg_precision': average_precision_score(y_test, y_pred_proba),
        'best_estimator': grid.best_estimator_
    }
    
    return results, X_test, y_test

def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """Evaluate trained model on test set."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def calculate_feature_importance(model, X_test, y_test, feature_names) -> pd.DataFrame:
    """Calculate permutation importance for features."""
    X_test_transformed = model.named_steps['pre'].transform(X_test)
    
    perm_res = permutation_importance(
        model.named_steps['rf'],
        X_test_transformed,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_res.importances_mean,
        'std': perm_res.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df

def create_lift_table(X_test, y_pred_proba) -> pd.DataFrame:
    """Create decile-based lift table."""
    df_test = X_test.copy()
    df_test['churn_prob'] = y_pred_proba
    df_test['decile'] = pd.qcut(df_test['churn_prob'], 10, labels=False, duplicates='drop')
    
    lift_table = df_test.groupby('decile').agg(
        avg_prob=('churn_prob', 'mean'),
        count=('churn_prob', 'size')
    ).sort_index(ascending=True)
    
    return lift_table
