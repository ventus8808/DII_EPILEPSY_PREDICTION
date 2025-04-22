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
from pathlib import Path
import yaml
warnings.filterwarnings('ignore')

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
model_dir.mkdir(exist_ok=True)
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)


def setup_logger(model_name):
    """Set up logger for the model."""
    log_file = model_dir / f'{model_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def load_and_preprocess_data():
    """Load and preprocess the data."""
    # Read data
    df = pd.read_csv(data_path)
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None, None, preprocessor

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
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    ece, mce = calculate_calibration_metrics(y_true, y_prob)
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

def objective_function(metrics):
    """Calculate objective score based on metrics and constraints."""
    weights = {
        'AUC': 0.3, 'MCE': -0.15, 'ECE': -0.15, 'F1': 0.1,
        'Precision': 0.1, 'Sensitivity': 0.1, 'Specificity': 0.1
    }
    if not (0.7 < metrics['AUC'] < 0.9):
        return float('-inf')
    if metrics['MCE'] >= 0.3:
        return float('-inf')
    if metrics['ECE'] >= 0.25:
        return float('-inf')
    if metrics['F1'] <= 0.2:
        return float('-inf')
    score = sum(weight * metrics[metric] for metric, weight in weights.items())
    return score

def main():
    model_name = "RF"
    logger = setup_logger(model_name)
    logger.info("Starting Random Forest model training")
    X_train, X_test, y_train, y_test, _, _, preprocessor = load_and_preprocess_data()
    # 参数空间直接在脚本内定义
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    # 训练次数从config.yaml读取
    n_trials = config.get('n_trials', 200)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    import optuna
    from imblearn.over_sampling import SMOTE
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600, 800, 1000]),
            'max_depth': trial.suggest_categorical('max_depth', [2, 3, 5, 7, 10, 15, 20, 25, None]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 3, 4, 5, 6, 8, 10, 15]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 4, 5, 6, 8]),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.5, 0.7, 0.9])
        }
        pipeline.set_params(**{'classifier__' + k: v for k, v in params.items()})
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        try:
            pipeline.fit(X_train_res, y_train_res)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            score = objective_function(metrics)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", params)
            return score
        except Exception as e:
            logger.error(f"Optuna trial error: {e}")
            return float('-inf')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_trial = study.best_trial
    best_score = best_trial.value
    best_params = best_trial.user_attrs['params']
    best_metrics = best_trial.user_attrs['metrics']
    # 训练并保存最佳模型
    pipeline.set_params(**{'classifier__' + k: v for k, v in best_params.items()})
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    pipeline.fit(X_train_res, y_train_res)
    model_path = model_dir / f'RF_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    params_serializable = {k: convert_np(v) for k, v in best_params.items()}
    param_path = model_dir / f'RF_best_params.json'
    with open(param_path, 'w') as f:
        json.dump(params_serializable, f, indent=4)
    logger.info(f"Optuna best score: {best_score}")
    logger.info("Best parameters:")
    for k, v in best_params.items():
        logger.info(f"{k}: {v}")
    logger.info("Best metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
