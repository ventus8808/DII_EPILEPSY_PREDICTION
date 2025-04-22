import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import json
import pickle
import logging
from datetime import datetime
from tqdm import tqdm
import os

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate ECE and MCE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            prob_true = np.mean(y_true[in_bin])
            prob_pred = np.mean(y_prob[in_bin])
            ece += np.abs(prob_true - prob_pred) * np.sum(in_bin) / len(y_true)
            mce = max(mce, np.abs(prob_true - prob_pred))
    
    return float(ece), float(mce)

def calculate_metrics(y_true, y_pred_proba):
    """Calculate all required metrics"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # ROC-AUC
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = float(auc(recall, precision))
    
    # Other metrics with zero_division parameter
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    precision_val = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Calibration metrics
    ece, mce = calculate_calibration_metrics(y_true, y_pred_proba)
    
    # Brier score
    brier = float(np.mean((y_pred_proba - y_true) ** 2))
    
    return {
        'AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Sensitivity': sensitivity,
        'Precision': precision_val,
        'Recall': sensitivity,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier
    }

def custom_scorer(estimator, X, y):
    """Custom scorer for model evaluation"""
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y, y_pred_proba)
    
    # Check hard constraints with more lenient thresholds
    if not (0.7 < metrics['AUC'] < 0.95 and 
            metrics['MCE'] < 0.3 and 
            metrics['ECE'] < 0.25):
        return float('-inf')
    
    # Calculate weighted score
    weights = {
        'AUC': 0.3, 'MCE': -0.15, 'ECE': -0.15, 'F1': 0.1,
        'Precision': 0.1, 'Sensitivity': 0.1, 'Specificity': 0.1
    }
    
    score = sum(weights[k] * metrics[k] for k in weights.keys() if k in metrics)
    return float(score)

def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    weights = df['WTDRD1']
    covariables = ['Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
                   'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df['DII_food'],
        df[covariables]
    ], axis=1)
    y = df['Epilepsy']
    
    return X, y, weights

def main():
    # Setup logging
    model_name = "CatBoost"
    log_file = f"/Users/maguoli/Documents/Development/Predictive/Models/64_{model_name}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    X, y, weights = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define parameter space
    param_dist = {
        'iterations': [100, 200, 500, 1000, 1500, 2000],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        'depth': [3, 4, 5, 6, 7, 8, 10, 12],
        'l2_leaf_reg': [0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'scale_pos_weight': [1, 3, 5, 7, 10, 15, 20],
        'min_data_in_leaf': [1, 3, 5, 7, 10, 15, 20, 30],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'random_strength': [1, 2, 3, 5, 7, 10],
        'bagging_temperature': [0.0, 0.5, 1.0, 2.0],
        'od_type': ['IncToDec', 'Iter'],
        'od_wait': [10, 20, 30, 50],
        'leaf_estimation_method': ['Newton', 'Gradient'],
        'bootstrap_type': ['Bernoulli', 'MVS'],
        'max_ctr_complexity': [1, 2, 3, 4]
    }
    
    # Initialize base model
    base_model = CatBoostClassifier(
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    n_iter = 500  # Number of search iterations
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    all_params = []
    all_scores = []
    best_score = float('-inf')
    best_model = None
    best_params = None
    
    # Perform random search with manual progress tracking
    print(f"Starting Random Search with {n_iter} iterations...")
    with tqdm(total=n_iter, desc="Parameter Search Progress") as pbar:
        for i in range(n_iter):
            # Randomly sample parameters
            params = {k: np.random.choice(v) for k, v in param_dist.items()}
            # Convert numpy types to Python types for JSON serialization
            params = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v 
                     for k, v in params.items()}
            
            # Initialize and train model
            model = CatBoostClassifier(**params, random_seed=42, verbose=False)
            
            try:
                # Perform cross-validation
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    w_fold_train = weights_train.iloc[train_idx]
                    
                    model.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
                    score = custom_scorer(model, X_fold_val, y_fold_val)
                    if score != float('-inf'):
                        scores.append(score)
                
                # Calculate mean score if any valid scores exist
                if scores:
                    mean_score = float(np.mean(scores))
                    all_params.append(params)
                    all_scores.append(mean_score)
                    
                    # Update best model if current is better
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params.copy()
                        print(f"\nNew best score: {best_score:.4f}")
                        print("Parameters:", best_params)
                
            except Exception as e:
                logging.error(f"Error in iteration {i}: {str(e)}")
            
            pbar.update(1)
    
    # Train final model if best parameters were found
    if best_params is not None:
        try:
            final_model = CatBoostClassifier(**best_params, random_seed=42, verbose=False)
            final_model.fit(X_train, y_train, sample_weight=weights_train)
            
            # Evaluate on test set
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, y_pred_proba)
            
            # Save model and parameters
            with open(f'/Users/maguoli/Documents/Development/Predictive/Models/64_{model_name}_model.pkl', 'wb') as f:
                pickle.dump(final_model, f)
            
            with open(f'/Users/maguoli/Documents/Development/Predictive/Models/64_{model_name}_param.json', 'w') as f:
                json.dump(best_params, f)
            
            # Log and print results
            logging.info(f"Best model found! Metrics: {metrics}")
            print("\nFinal model metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        except Exception as e:
            print("Error training final model:", str(e))
            logging.error(f"Error training final model: {str(e)}")
    else:
        print("No model found meeting the constraints")
        logging.info("No model found meeting the constraints")

if __name__ == "__main__":
    main()
