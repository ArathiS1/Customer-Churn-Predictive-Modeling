import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(url=None, file_path=None):
    """
    Load the Telco Customer Churn dataset from a URL or local file
    """
    if file_path:
        df = pd.read_csv(file_path)
    elif url:
        df = pd.read_csv(url)
    else:
        raise ValueError("Either url or file_path must be provided")
    
    return df

def clean_data(df):
    """
    Clean the raw dataset
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean TotalCharges which sometimes is whitespace -> numeric with NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Simple imputation for numeric columns
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    df_clean['tenure'] = pd.to_numeric(df_clean['tenure'], errors='coerce').fillna(0)
    df_clean['MonthlyCharges'] = pd.to_numeric(df_clean['MonthlyCharges'], errors='coerce')
    
    # Feature engineering
    df_clean['avg_monthly_calc'] = df_clean['TotalCharges'] / np.where(df_clean['tenure']>0, df_clean['tenure'], np.nan)
    df_clean['avg_monthly_calc'] = df_clean['avg_monthly_calc'].fillna(df_clean['MonthlyCharges'])
    
    # Target mapping
    df_clean['churn_flag'] = df_clean['Churn'].map({'Yes':1, 'No':0}).astype(int)
    
    # Remove duplicate customers
    df_clean = df_clean.drop_duplicates(subset='customerID')
    
    # Remove invalid rows
    df_clean = df_clean[(df_clean['MonthlyCharges'] >= 0) & (df_clean['TotalCharges'] >= 0)]
    
    return df_clean

def get_feature_definitions():
    """
    Define feature sets for modeling
    """
    # Choose typical numeric & categorical cols from Telco dataset
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'avg_monthly_calc']
    
    # categorical columns
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    id_cols = ['customerID']
    target_col = 'churn_flag'
    
    return num_cols, cat_cols, id_cols, target_col

def create_preprocessing_pipeline(num_cols, cat_cols):
    """
    Create the preprocessing pipeline
    """
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    return preproc
