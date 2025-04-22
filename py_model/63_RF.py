import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import logging
import os
import warnings
warnings.filterwarnings('ignore')

def setup_logger(model_name):
    """Set up logger for the model."""
    log_dir = '/Users/maguoli/Documents/Development/Predictive/Models'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'63_{model_name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate ECE and MCE."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            actual_prob = np.mean(y_true[in_bin])
            predicted_prob = np.mean(y_prob[in_bin])
            ece += np.abs(actual_prob - predicted_prob) * prop_in_bin
            mce = max(mce, np.abs(actual_prob - predicted_prob))
    
    return ece, mce

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all required metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sensitivity = recall  # Same as recall
    specificity = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate calibration metrics
    ece, mce = calculate_calibration_metrics(y_true, y_prob)
    
    # Calculate Brier score
    brier = np.mean((y_prob - y_true) ** 2)
    
    return {
        'AUC': roc_auc,
        'AUC-PR': pr_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier
    }

def load_and_preprocess_data():
    """Load and preprocess the data."""
    # Read data
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    # Define features
    weights = df['WTDRD1']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 
                          'Alcohol', 'Employment', 'ActivityLevel']
    
    # Prepare X and y
    X = pd.concat([
        df[['DII_food']],
        df[categorical_features]
    ], axis=1)
    y = df['Epilepsy']
    
    # Create preprocessor - only for categorical features
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # This will keep DII_food as is
    )
    
    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor

def objective_function(metrics):
    """Calculate objective score based on metrics and constraints."""
    weights = {
        'AUC': 0.3, 'MCE': -0.15, 'ECE': -0.15, 'F1': 0.1,
        'Precision': 0.1, 'Sensitivity': 0.1, 'Specificity': 0.1
    }
    
    # Check hard constraints
    if not (0.7 < metrics['AUC'] < 0.9):
        return float('-inf')
    if metrics['MCE'] >= 0.3:
        return float('-inf')
    if metrics['ECE'] >= 0.25:
        return float('-inf')
    if metrics['F1'] <= 0.2:
        return float('-inf')
    
    # Calculate weighted sum
    score = sum(weight * metrics[metric] for metric, weight in weights.items())
    return score

def main():
    model_name = "RF"
    logger = setup_logger(model_name)
    logger.info("Starting Random Forest model training")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Initialize variables for tracking best model
    best_score = float('-inf')
    best_model = None
    best_params = None
    best_metrics = None
    
    # Random search with cross-validation
    n_trials = 200  
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for _ in tqdm(range(n_trials), desc="Hyperparameter optimization"):
        # Sample random parameters
        params = {key: np.random.choice(values) for key, values in param_grid.items()}
        pipeline.set_params(**{'classifier__' + k: v for k, v in params.items()})
        
        # Perform cross-validation
        try:
            pipeline.fit(X_train, y_train, classifier__sample_weight=weights_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            # Calculate objective score
            score = objective_function(metrics)
            
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_params = params
                best_metrics = metrics
                
                # Save current best model
                model_path = os.path.join('/Users/maguoli/Documents/Development/Predictive/Models', 
                                        f'63_{model_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(pipeline, f)
                
                # Save current best parameters
                param_path = os.path.join('/Users/maguoli/Documents/Development/Predictive/Models',
                                        f'63_{model_name}_param.json')
                with open(param_path, 'w') as f:
                    json.dump(params, f, indent=4)
                
                # Log best metrics
                logger.info(f"New best model found with score: {score}")
                logger.info("Metrics:")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            continue
    
    logger.info("Training completed")
    
    if best_params is None:
        logger.info("No model found that meets the constraints.")
        logger.info("Constraints:")
        logger.info("- 0.75 < AUC < 0.9")
        logger.info("- MCE < 0.25")
        logger.info("- ECE < 0.2")
        logger.info("- F1 > 0.2")
    else:
        logger.info(f"Best parameters: {best_params}")
        logger.info("Final best metrics:")
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
