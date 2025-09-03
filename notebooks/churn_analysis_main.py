import pandas as pd
import numpy as np
from src.preprocessing import load_data, clean_data, create_preprocessor, prepare_features
from src.modeling import (train_baseline_model, train_random_forest, 
                         calculate_feature_importance, create_lift_table)
from src.visualization import (plot_distributions, plot_categorical_churn,
                              plot_correlation_heatmap, plot_feature_importance,
                              plot_shap_summary, plot_tenure_churn)
from src.utils import load_config, save_model, save_results, save_plot
import yaml

def main():
    # Load configuration
    config = load_config('config/parameters.yaml')
    
    # Load and clean data
    print("Loading data...")
    df = load_data(config['data']['url'])
    df = clean_data(df)
    
    # EDA Visualizations
    print("Creating EDA visualizations...")
    plot_distributions(df, 'results/images/distributions.png')
    plot_categorical_churn(df, 'Contract', 'results/images/churn_by_contract.png')
    plot_categorical_churn(df, 'InternetService', 'results/images/churn_by_internet.png')
    plot_correlation_heatmap(df, 'results/images/correlation_heatmap.png')
    plot_tenure_churn(df, 'results/images/churn_by_tenure.png')
    
    # Prepare features
    X, y, num_cols, cat_cols = prepare_features(df)
    preprocessor, _, _ = create_preprocessor()
    feature_names = preprocessor.get_feature_names_out()
    
    # Train baseline model
    print("Training baseline model...")
    baseline_results, X_test, y_test = train_baseline_model(
        X, y, preprocessor, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_results, X_test, y_test = train_random_forest(
        X, y, preprocessor,
        param_grid={
            'rf__max_depth': config['model']['max_depth'],
            'rf__n_estimators': config['model']['n_estimators']
        },
        cv_folds=config['model']['cv_folds'],
        scoring=config['model']['scoring'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Feature importance
    print("Calculating feature importance...")
    importance_df = calculate_feature_importance(
        rf_results['best_estimator'], X_test, y_test, feature_names
    )
    importance_df.to_csv('results/feature_importance.csv', index=False)
    
    # Lift table
    y_pred_proba = rf_results['best_estimator'].predict_proba(X_test)[:, 1]
    lift_table = create_lift_table(X_test, y_pred_proba)
    lift_table.to_csv('results/lift_table.csv')
    
    # SHAP analysis
    print("Creating SHAP plots...")
    plot_shap_summary(
        rf_results['best_estimator'], X_test, feature_names,
        'results/images/shap_summary_plot.png'
    )
    
    plot_feature_importance(
        importance_df, top_n=20, 
        save_path='results/images/feature_importance_plot.png'
    )
    
    # Save model
    save_model(rf_results['best_estimator'], 'models/best_random_forest.pkl')
    
    # Save results
    all_results = {
        'baseline': baseline_results,
        'random_forest': rf_results
    }
    save_results(all_results, 'results/model_results.json')
    
    print("Analysis completed successfully!")
    print(f"Best Random Forest AUC: {rf_results['test_roc_auc']:.4f}")
    print(f"Top 5 features: {list(importance_df['feature'].head(5))}")

if __name__ == "__main__":
    main()
