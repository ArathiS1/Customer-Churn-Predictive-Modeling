import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from typing import List

def plot_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot distributions of numeric features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df['MonthlyCharges'], bins=50, ax=axes[0])
    axes[0].set_title('Monthly Charges Distribution')
    
    sns.histplot(df['TotalCharges'], bins=50, ax=axes[1])
    axes[1].set_title('Total Charges Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_categorical_churn(df: pd.DataFrame, column: str, save_path: str = None):
    """Plot churn rate by categorical variable."""
    plt.figure(figsize=(10, 6))
    
    if column == 'Contract':
        pd.crosstab(df[column], df['churn_flag'], normalize='index').plot(
            kind='bar', stacked=True, ax=plt.gca()
        )
        plt.legend(title='Churn', labels=['No', 'Yes'])
    else:
        churn_rate = df.groupby(column)['churn_flag'].mean().sort_values(ascending=False)
        churn_rate.plot(kind='bar')
    
    plt.title(f'Churn Rate by {column}')
    plt.ylabel('Proportion' if column == 'Contract' else 'Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Plot correlation heatmap for numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(num_df.corr(), annot=True, cmap='vlag', center=0, fmt='.2f')
    plt.title('Correlation Heatmap (Numeric Features)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
    """Plot feature importance from permutation importance."""
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.gca().invert_yaxis()
    plt.title('Permutation Importance (Top 20 Features)')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_summary(model, X_test, feature_names, save_path: str = None):
    """Plot SHAP summary plot."""
    X_test_trans = model.named_steps['pre'].transform(X_test)
    
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()
    
    explainer = shap.TreeExplainer(model.named_steps['rf'])
    shap_values = explainer.shap_values(X_test_trans)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test_trans, feature_names=feature_names, max_display=20, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_tenure_churn(df: pd.DataFrame, save_path: str = None):
    """Plot churn rate by tenure buckets."""
    bins = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    tenure_churn = df.groupby('tenure_bucket')['churn_flag'].mean()
    
    plt.figure(figsize=(10, 6))
    tenure_churn.plot(kind='bar')
    plt.title('Churn Rate by Tenure Bucket (months)')
    plt.ylabel('Churn Rate')
    plt.xlabel('Tenure (months)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
