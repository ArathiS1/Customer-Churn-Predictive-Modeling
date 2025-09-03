import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List

def load_data(url: str) -> pd.DataFrame:
    """Load dataset from URL."""
    df = pd.read_csv(url)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw data."""
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Feature engineering
    df['avg_monthly_calc'] = df['TotalCharges'] / np.where(df['tenure']>0, df['tenure'], np.nan)
    df['avg_monthly_calc'] = df['avg_monthly_calc'].fillna(df['MonthlyCharges'])
    
    # Target mapping
    df['churn_flag'] = df['Churn'].map({'Yes':1, 'No':0}).astype(int)
    
    # Remove duplicates and invalid rows
    df = df.drop_duplicates(subset='customerID')
    df = df[(df['MonthlyCharges'] >= 0) & (df['TotalCharges'] >= 0)]
    
    return df

def create_preprocessor(numeric_strategy: str = "median", 
                       categorical_strategy: str = "constant",
                       fill_value: str = "Unknown") -> ColumnTransformer:
    """Create preprocessing pipeline."""
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'avg_monthly_calc']
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy=categorical_strategy, fill_value=fill_value)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    return preproc, num_cols, cat_cols

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare features and target for modeling."""
    preproc, num_cols, cat_cols = create_preprocessor()
    feature_cols = num_cols + cat_cols
    
    X = df[feature_cols].copy()
    y = df['churn_flag'].copy()
    
    return X, y, num_cols, cat_cols
