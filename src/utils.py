import pandas as pd
import numpy as np

def create_lift_table(X_test, y_pred_proba, deciles=10):
    """
    Create a lift table for model evaluation
    """
    df_test = X_test.copy()
    df_test['churn_prob'] = y_pred_proba
    df_test['decile'] = pd.qcut(df_test['churn_prob'], deciles, labels=False, duplicates='drop')
    
    # Average churn probability and count per decile
    lift_table = df_test.groupby('decile').agg(
        avg_prob=('churn_prob', 'mean'),
        count=('churn_prob', 'size')
    ).sort_index(ascending=True)
    
    return lift_table

def get_high_risk_customers(X_test, y_pred_proba, deciles=10):
    """
    Identify high-risk customers in the top decile
    """
    df_test = X_test.copy()
    df_test['churn_prob'] = y_pred_proba
    df_test['decile'] = pd.qcut(df_test['churn_prob'], deciles, labels=False, duplicates='drop')
    
    top_decile = df_test[df_test['decile'] == df_test['decile'].max()]
    return top_decile
