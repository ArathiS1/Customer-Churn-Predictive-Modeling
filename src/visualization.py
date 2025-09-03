import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap

def setup_visualization_style():
    """
    Set up consistent visualization style
    """
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def plot_churn_distribution(df, target_col='churn_flag', save_path=None):
    """
    Plot distribution of churn vs non-churn customers
    """
    plt.figure(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xlabel('Churn (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_numeric_distribution(df, column, save_path=None):
    """
    Plot distribution of a numeric column
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=50)
    plt.title(f'Distribution of {column}')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_categorical_churn(df, categorical_col, target_col='churn_flag', save_path=None):
    """
    Plot churn rate by categorical variable
    """
    plt.figure(figsize=(10, 6))
    churn_rate = df.groupby(categorical_col)[target_col].mean().sort_values(ascending=False)
    churn_rate.plot(kind='bar')
    plt.title(f'Churn Rate by {categorical_col}')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_correlation_heatmap(df, numeric_cols, save_path=None):
    """
    Plot correlation heatmap for numeric variables
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='vlag', center=0, square=True)
    plt.title('Correlation Heatmap (Numeric Features)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Plot feature importance from permutation importance
    """
    plt.figure(figsize=(10, 12))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Permutation Importance (Top {})'.format(top_n))
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_shap_summary(shap_values, features, feature_names, save_path=None):
    """
    Plot SHAP summary plot
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, feature_names=feature_names, max_display=20, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
