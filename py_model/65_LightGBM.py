import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
from tqdm import tqdm
import json
import pickle
import logging
import random
import os
from datetime import datetime

def setup_logger(model_name):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    log_file = f'/Users/maguoli/Documents/Development/Predictive/Models/65_{model_name}.log'
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

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
            ece += np.abs(prob_true - prob_pred) * np.mean(in_bin)
            mce = max(mce, np.abs(prob_true - prob_pred))
    
    return ece, mce

def calculate_metrics(y_true, y_pred_proba):
    # Adjust threshold based on ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)
    
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate specificity
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate calibration metrics
    ece, mce = calculate_calibration_metrics(y_true, y_pred_proba)
    
    # Calculate Brier score
    brier = np.mean((y_pred_proba - y_true) ** 2)
    
    return {
        'AUC': auc_roc,
        'AUC_PR': auc_pr,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Recall': sensitivity,
        'F1': f1,
        'Specificity': specificity,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
        'Threshold': optimal_threshold
    }

def evaluate_model(params, X_train, X_test, y_train, y_test, weights_train):
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(0)  # Disable logging
    ]
    
    model = lgb.train(
        params, 
        train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=callbacks
    )
    
    # Use best iteration for prediction
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    metrics = calculate_metrics(y_test, y_pred_proba)
    return metrics, model

def check_constraints(metrics):
    return (0.7 < metrics['AUC'] < 0.95 and 
            metrics['MCE'] < 0.25 and 
            metrics['ECE'] < 0.2 and 
            metrics['F1'] > 0.1)

def calculate_weighted_score(metrics, weights):
    score = 0
    for metric, weight in weights.items():
        if metric in metrics:
            score += metrics[metric] * weight
    return score

def main():
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Setup logger
    logger = setup_logger('LightGBM')
    logger.info('Starting LightGBM model training')
    
    # Load data
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    # Print class distribution
    print("\nClass Distribution:")
    print(df['Epilepsy'].value_counts(normalize=True))
    
    # Feature engineering
    weights = df['WTDRD1']
    covariables = ['Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
                   'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df['DII_food'],
        df[covariables]
    ], axis=1)
    y = df['Epilepsy']
    
    # Calculate positive class weight
    pos_weight = len(y[y==0]) / len(y[y==1])
    print(f"\nPositive class weight: {pos_weight}")
    
    # Split dataset
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define parameter space for random search
    param_space = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': lambda: random.randint(15, 63),
        'max_depth': lambda: random.randint(4, 8),
        'learning_rate': lambda: random.uniform(0.001, 0.01),
        'feature_fraction': lambda: random.uniform(0.7, 0.9),
        'bagging_fraction': lambda: random.uniform(0.7, 0.9),
        'bagging_freq': lambda: random.randint(2, 5),
        'min_child_samples': lambda: random.randint(20, 50),
        'scale_pos_weight': pos_weight,
        'min_child_weight': lambda: random.uniform(0.001, 0.01),
        'min_split_gain': lambda: random.uniform(0.1, 0.3),
        'reg_alpha': lambda: random.uniform(0.1, 0.5),
        'reg_lambda': lambda: random.uniform(0.1, 0.5),
        'verbose': -1
    }
    
    weights_metrics = {
        'AUC': 0.3,
        'MCE': -0.15,
        'ECE': -0.15,
        'F1': 0.1,
        'Precision': 0.1,
        'Sensitivity': 0.1,
        'Specificity': 0.1
    }
    
    best_score = float('-inf')
    best_metrics = None
    best_params = None
    best_model = None
    found_valid_model = False
    
    # Random search with cross-validation
    n_trials = 10  
    for trial in tqdm(range(n_trials), desc='Random Search Trials'):
        # Generate random parameters
        current_params = {k: v() if callable(v) else v for k, v in param_space.items()}
        print(f"\nTrial {trial + 1} parameters:")
        for k, v in current_params.items():
            if k != 'verbose':  # Don't print verbose parameter
                print(f"{k}: {v}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            weights_fold_train = weights_train.iloc[train_idx]
            
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_metrics, _ = evaluate_model(
                current_params, X_fold_train, X_fold_val, 
                y_fold_train, y_fold_val, weights_fold_train
            )
            cv_metrics.append(fold_metrics)
        
        # Average CV metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in cv_metrics])
            for metric in cv_metrics[0].keys()
        }
        
        # Check constraints and calculate score
        if check_constraints(avg_metrics):
            found_valid_model = True
            current_score = calculate_weighted_score(avg_metrics, weights_metrics)
            
            if current_score > best_score:
                # Train final model on full training set
                test_metrics, model = evaluate_model(
                    current_params, X_train, X_test, y_train, y_test, weights_train
                )
                
                if check_constraints(test_metrics):
                    best_score = current_score
                    best_metrics = test_metrics
                    best_params = current_params
                    best_model = model
                    
                    # Save current best model and parameters
                    model_path = '/Users/maguoli/Documents/Development/Predictive/Models/65_LightGBM_model.pkl'
                    param_path = '/Users/maguoli/Documents/Development/Predictive/Models/65_LightGBM_param.json'
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    with open(param_path, 'w') as f:
                        json.dump({k: v for k, v in current_params.items() if k != 'verbose'}, f, indent=4)
                    
                    # Log and print best metrics
                    logger.info(f'New best model found in trial {trial + 1}')
                    logger.info(f'Metrics: {test_metrics}')
                    print(f'\nNew best model found in trial {trial + 1}')
                    print('\nTest set metrics:')
                    print('-' * 40)
                    for metric, value in test_metrics.items():
                        if metric != 'Threshold':  # Don't print threshold
                            print(f'{metric:12}: {value:.4f}')
                    print('-' * 40)
    
    if not found_valid_model:
        logger.info('No model found meeting the constraints')
        print('\nNo model found meeting the constraints')
    else:
        logger.info('Training completed')
        logger.info(f'Final best metrics: {best_metrics}')
        logger.info(f'Final best parameters: {best_params}')
        print('\nTraining completed successfully!')
        print('\nFinal best metrics:')
        print('-' * 40)
        for metric, value in best_metrics.items():
            if metric != 'Threshold':  # Don't print threshold
                print(f'{metric:12}: {value:.4f}')
        print('-' * 40)

if __name__ == '__main__':
    main()
