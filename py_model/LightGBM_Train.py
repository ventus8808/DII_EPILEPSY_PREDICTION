import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, log_loss, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 1. 配置读取
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
model_dir.mkdir(exist_ok=True)
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)
plot_dir = Path(config['plot_dir']) if 'plot_dir' in config else Path('plots')
plot_dir.mkdir(exist_ok=True)

# 2. 日志
def setup_logger(model_name):
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

# 3. 数据加载与预处理
def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
    features = ['DII_food'] + numeric_features + categorical_features
    X = df[features]
    y = df['Epilepsy']
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['DII_food'] + numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        weights_train = weights_test = None
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, features

# 4. 指标与目标函数
def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if weights is None:
        weights = np.ones_like(y_true)
    ece = 0
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            bin_weights = weights[in_bin]
            bin_total_weight = np.sum(bin_weights)
            if bin_total_weight > 0:
                actual_prob = np.average(y_true[in_bin], weights=bin_weights)
                predicted_prob = np.average(y_prob[in_bin], weights=bin_weights)
                bin_weight = bin_total_weight / np.sum(weights)
                ece += np.abs(actual_prob - predicted_prob) * bin_weight
                mce = max(mce, np.abs(actual_prob - predicted_prob))
    return ece, mce

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    ece, mce = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.mean((y_prob - y_true) ** 2)
    logloss = log_loss(y_true, y_prob)
    return {
        'AUC': roc_auc,
        'AUC-PR': pr_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
        'LogLoss': logloss
    }

def load_objective_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    weights = config.get('objective_weights', {})
    constraints = config.get('objective_constraints', {})
    return weights, constraints

def objective_function(metrics, config_path='config.yaml'):
    weights, constraints = load_objective_config(config_path)
    # 硬约束
    if 'AUC_min' in constraints and metrics.get('AUC', 0) < constraints['AUC_min']:
        return float('-inf')
    if 'AUC_max' in constraints and metrics.get('AUC', 1) > constraints['AUC_max']:
        return float('-inf')
    if 'ECE_max' in constraints and metrics.get('ECE', 0) >= constraints['ECE_max']:
        return float('-inf')
    if 'F1_min' in constraints and metrics.get('F1', 0) <= constraints['F1_min']:
        return float('-inf')
    if 'Sensitivity_min' in constraints and metrics.get('Sensitivity', 0) < constraints['Sensitivity_min']:
        return float('-inf')
    if 'Specificity_min' in constraints and metrics.get('Specificity', 0) < constraints['Specificity_min']:
        return float('-inf')
    # 线性加权
    score = sum(weights.get(metric, 0) * metrics.get(metric, 0) for metric in weights)
    return score

# 5. Optuna + SMOTE 优化
def objective(trial):
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, features = load_and_preprocess_data()
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    if weights_train is not None:
        weights_train_res = weights_train.iloc[X_train_res.index]
    else:
        weights_train_res = None
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5.0),
        'random_state': 42,
        'verbosity': -1
    }
    lgb_train = lgb.Dataset(X_train_res, y_train_res, weight=weights_train_res)
    lgb_valid = lgb.Dataset(X_test, y_test, weight=weights_test, reference=lgb_train)
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_valid], early_stopping_rounds=50, verbose_eval=False)
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    score = objective_function(metrics)
    trial.set_user_attr('params', params)
    trial.set_user_attr('metrics', metrics)
    return score

# 6. 主流程
def main():
    model_name = "LightGBM"
    logger = setup_logger(model_name)
    logger.info("Starting LightGBM model training")
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, features = load_and_preprocess_data()
    n_trials = config.get('n_trials', 100)
    study = optuna.create_study(direction='maximize')
    # 断点恢复机制：每发现更优模型立即保存
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    for trial in range(n_trials):
        optuna_trial = study.ask()
        score = objective(optuna_trial)
        study.tell(optuna_trial, score)
        trial_metrics = optuna_trial.user_attrs['metrics']
        trial_params = optuna_trial.user_attrs['params']
        if score > best_score:
            best_score = score
            best_params = trial_params
            best_metrics = trial_metrics
            # 训练并保存当前最佳模型
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            if weights_train is not None:
                weights_train_res = weights_train.iloc[X_train_res.index]
            else:
                weights_train_res = None
            lgb_train = lgb.Dataset(X_train_res, y_train_res, weight=weights_train_res)
            model = lgb.train(best_params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train], verbose_eval=False)
            model_path = model_dir / f'LightGBM_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            # 保存最佳参数
            def convert_np(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            params_serializable = {k: convert_np(v) for k, v in best_params.items()}
            param_path = model_dir / f'LightGBM_best_params.json'
            with open(param_path, 'w') as f:
                json.dump(params_serializable, f, indent=4)
            # 保存最佳指标
            metrics_path = model_dir / f'LightGBM_best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
            logger.info(f"[Trial {trial+1}] New best score: {best_score}")
            logger.info("Best parameters:")
            for k, v in best_params.items():
                logger.info(f"{k}: {v}")
            logger.info("Best metrics:")
            for metric, value in best_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
    logger.info(f"Training finished. Best score: {best_score}")
    logger.info("Final best parameters:")
    for k, v in best_params.items():
        logger.info(f"{k}: {v}")
    logger.info("Final best metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
