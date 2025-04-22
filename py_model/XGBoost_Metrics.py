import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import yaml
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score, log_loss)

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
result_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
result_dir.mkdir(exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess data (路径已适配)."""
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
    features = ['DII_food'] + numeric_features + categorical_features
    X = df[features]
    y = df['Epilepsy']
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)],
        remainder='passthrough'
    )
    from sklearn.model_selection import train_test_split
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

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if weights is None:
        weights = np.ones_like(y_true)
    ece = 0
    mce = 0
    bin_metrics = []
    observed = []
    expected = []
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
                obs_pos = np.sum(y_true[in_bin] * bin_weights)
                obs_neg = np.sum((1 - y_true[in_bin]) * bin_weights)
                exp_pos = np.sum(y_prob[in_bin] * bin_weights)
                exp_neg = np.sum((1 - y_prob[in_bin]) * bin_weights)
                if exp_pos > 1e-10 and exp_neg > 1e-10:
                    observed.append([obs_neg, obs_pos])
                    expected.append([exp_neg, exp_pos])
                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'actual_prob': float(actual_prob),
                    'predicted_prob': float(predicted_prob),
                    'bin_weight': float(bin_weight)
                })
    if len(observed) >= 2:
        from scipy.stats import chi2_contingency
        chi2, p_value = chi2_contingency(np.array(observed), np.array(expected))[:2]
    else:
        chi2, p_value = np.nan, np.nan
    return ece, mce, bin_metrics, chi2, p_value

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss
)

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred, sample_weight=weights)
    precision = precision_score(y_true, y_pred, sample_weight=weights)
    recall = recall_score(y_true, y_pred, sample_weight=weights)
    sensitivity = recall
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    logloss = log_loss(y_true, y_prob, sample_weight=weights)
    ece, mce, _, _, _ = calculate_calibration_metrics(y_true, y_prob, weights)
    kappa = cohen_kappa_score(y_true, y_pred, sample_weight=weights)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    else:
        specificity = float('nan')
        npv = float('nan')
    youden = recall + specificity - 1 if not (np.isnan(recall) or np.isnan(specificity)) else float('nan')

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "NPV": npv,
        "F1 Score": f1,
        "Youden's Index": youden,
        "Cohen's Kappa": kappa,
        "AUC-ROC": roc_auc,
        "AUC-PR": pr_auc,
        "Log Loss": logloss,
        "Brier": brier,
        "ECE": ece,
        "MCE": mce
    }

def main():
    # 加载模型
    model_path = model_dir / 'XGBoost_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # 加载数据
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # 计算指标
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    # 保存指标
    metrics_path = result_dir / 'XGBoost_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()

