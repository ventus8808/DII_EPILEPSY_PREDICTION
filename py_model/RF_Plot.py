import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score,
                           roc_auc_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import yaml

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 300

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess data (路径已适配)."""
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    X = pd.concat([
        df[['DII_food']],
        df[categorical_features]
    ], axis=1)
    y = df['Epilepsy']
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

def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(str(plot_dir / f"{model_name}_ROC.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'auc': float(roc_auc)
    }, str(plot_data_dir / f"{model_name}_ROC_data.json"))

def plot_pr_curve(y_true, y_prob, weights, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(str(plot_dir / f"{model_name}_PR.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'precision': [float(x) for x in precision],
        'recall': [float(x) for x in recall],
        'auc': float(pr_auc)
    }, str(plot_data_dir / f"{model_name}_PR_data.json"))

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    ece, mce, bin_metrics, chi2, p_value = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    actual_probs = [m['actual_prob'] for m in bin_metrics]
    predicted_probs = [m['predicted_prob'] for m in bin_metrics]
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(predicted_probs, actual_probs, 'ro-', label='Model calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.text(0.05, 0.95, f'ECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier:.3f}\nChi2: {chi2:.3f}\np-value: {p_value:.3f}', bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
    plt.legend(loc='lower right')
    plt.savefig(str(plot_dir / f"{model_name}_Calibration.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'bin_metrics': bin_metrics,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'chi2': float(chi2),
        'p_value': float(p_value)
    }, str(plot_data_dir / f"{model_name}_Calibration_data.json"))

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
    plt.savefig(str(plot_dir / f"{model_name}_DCA.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'thresholds': [float(t) for t in thresholds],
        'net_benefit_model': [float(nb) for nb in net_benefits_model],
        'net_benefit_all': [float(nb) for nb in net_benefits_all],
        'net_benefit_none': [float(nb) for nb in net_benefits_none]
    }, str(plot_data_dir / f"{model_name}_DCA_data.json"))

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
    precision = precision_score(y_true, y_pred, sample_weight=weights)
    recall = recall_score(y_true, y_pred, sample_weight=weights)
    sensitivity = recall
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    ece, mce, _, _, _ = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    return {
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier
    }

def main():
    # 加载模型
    model_path = model_dir / 'RF_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # 加载数据
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # 计算指标
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    # 绘图
    model_name = "RF"
    plot_roc_curve(y_test, y_prob, weights_test, model_name)
    plot_pr_curve(y_test, y_prob, weights_test, model_name)
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    plot_decision_curve(y_test, y_prob, weights_test, model_name)

if __name__ == "__main__":
    main()
