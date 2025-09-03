from setuptools import setup, find_packages

setup(
    name="churn-prediction-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "shap>=0.42.1",
        "imbalanced-learn>=0.10.1",
        "xgboost>=1.7.6",
        "pyyaml>=6.0"
    ],
)
