import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
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
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
model_dir.mkdir(exist_ok=True)
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)
plot_dir = Path(config['plot_dir']) if 'plot_dir' in config else Path('plots')
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

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
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor

# 4. 指标与目标函数
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, log_loss
)

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if weights is None:
        weights = np.ones_like(y_true)
    ece = 0
    mce = 0
    bin_metrics = []
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
                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'actual_prob': float(actual_prob),
                    'predicted_prob': float(predicted_prob),
                    'bin_weight': float(bin_weight)
                })
    return ece, mce, bin_metrics

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
    precision = precision_score(y_true, y_pred, sample_weight=weights)
    recall = recall_score(y_true, y_pred, sample_weight=weights)
    specificity = recall_score(y_true, y_pred, pos_label=0, sample_weight=weights)
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    ece, mce, _ = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    logloss = log_loss(y_true, y_prob, sample_weight=weights)
    return {
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc,
        'Sensitivity': recall,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
        'LogLoss': logloss
    }

def objective_function(metrics):
    weights = {
        'AUC-ROC': 0.3, 'MCE': -0.15, 'ECE': -0.15, 'F1': 0.1,
        'AUC-PR': 0.1, 'Sensitivity': 0.1, 'Brier': -0.1, 'LogLoss': -0.1
    }
    # 约束
    if metrics['AUC-ROC'] < 0.7:
        return float('-inf')
    if metrics['MCE'] >= 0.3:
        return float('-inf')
    if metrics['ECE'] >= 0.25:
        return float('-inf')
    if metrics['F1'] <= 0.2:
        return float('-inf')
    score = sum(weight * metrics[metric] for metric, weight in weights.items())
    return score

# 5. Optuna + SMOTE 优化
def objective(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600, 800, 1000]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10, 15, 20, 25, 30, None]),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
        'use_label_encoder': False,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))
    ])
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    try:
        pipeline.fit(X_train_res, y_train_res)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
        score = objective_function(metrics)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", params)
        return score
    except Exception as e:
        logger.error(f"Optuna trial error: {e}")
        return float('-inf')

# 6. 可视化函数（与RF_Plot.py风格一致）
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 300

def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(str(plot_dir / f"XGB_ROC.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'auc': float(roc_auc)
    }, str(plot_data_dir / f"XGB_ROC_data.json"))

def plot_pr_curve(y_true, y_prob, weights, model_name):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(str(plot_dir / f"XGB_PR.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'precision': [float(x) for x in precision],
        'recall': [float(x) for x in recall],
        'auc': float(pr_auc)
    }, str(plot_data_dir / f"XGB_PR_data.json"))

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    ece, mce, bin_metrics = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    plt.figure()
    bin_centers = [(bm['bin_lower'] + bm['bin_upper']) / 2 for bm in bin_metrics]
    actual_probs = [bm['actual_prob'] for bm in bin_metrics]
    predicted_probs = [bm['predicted_prob'] for bm in bin_metrics]
    plt.plot(bin_centers, predicted_probs, 's-', label='Predicted')
    plt.plot(bin_centers, actual_probs, 'o-', label='Actual')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('Calibration Curve')
    plt.text(0.05, 0.95, f'ECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier:.3f}', bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
    plt.legend(loc='lower right')
    plt.savefig(str(plot_dir / f"XGB_Calibration.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'bin_metrics': bin_metrics,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier)
    }, str(plot_data_dir / f"XGB_Calibration_data.json"))

def plot_decision_curve(y_true, y_prob, weights, model_name):
    thresholds = np.linspace(0.01, 0.3, 30)
    prevalence = np.sum(weights[y_true == 1]) / np.sum(weights) if weights is not None else np.mean(y_true)
    net_benefits_model = []
    net_benefits_all = []
    for threshold in thresholds:
        def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
            if weights is None:
                weights = np.ones_like(y_true)
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            weights = np.array(weights)
            predictions = (y_prob >= threshold).astype(int)
            true_pos = predictions & y_true.astype(bool)
            false_pos = predictions & ~y_true.astype(bool)
            n_total = len(y_true)
            tp_rate = np.sum(true_pos) / n_total
            fp_rate = np.sum(false_pos) / n_total
            net_benefit = tp_rate - fp_rate * (threshold/(1-threshold))
            return net_benefit
        nb_model = calculate_net_benefit(y_true, y_prob, threshold, weights)
        net_benefits_model.append(nb_model)
        nb_all = prevalence - (1 - prevalence) * (threshold/(1-threshold))
        net_benefits_all.append(nb_all)
    net_benefits_none = np.zeros_like(thresholds)
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, net_benefits_model, 'b-', label='Model', linewidth=2.5)
    plt.plot(thresholds, net_benefits_all, 'g--', label='Treat All', linewidth=2)
    plt.plot(thresholds, net_benefits_none, 'r:', label='Treat None', linewidth=2)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='upper right')
    plt.savefig(str(plot_dir / f"XGB_DCA.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'thresholds': [float(t) for t in thresholds],
        'net_benefit_model': [float(nb) for nb in net_benefits_model],
        'net_benefit_all': [float(nb) for nb in net_benefits_all],
        'net_benefit_none': [float(nb) for nb in net_benefits_none]
    }, str(plot_data_dir / f"XGBoost_DCA_data.json"))

# 7. 主流程
def main():
    global X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, logger
    model_name = "XGBoost"
    logger = setup_logger(model_name)
    logger.info("Starting XGBoost model training")
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    n_trials = config.get('n_trials', 200)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_trial = study.best_trial
    best_score = best_trial.value
    best_params = best_trial.user_attrs['params']
    best_metrics = best_trial.user_attrs['metrics']
    # 训练并保存最佳模型
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**best_params))
    ])
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    pipeline.fit(X_train_res, y_train_res)
    model_path = model_dir / f'XGBoost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
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
    param_path = model_dir / f'XGBoost_best_params.json'
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
