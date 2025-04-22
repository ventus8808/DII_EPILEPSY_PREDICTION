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
from pathlib import Path
import yaml

# Set global style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_ROC.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': float(roc_auc)}, str(plot_data_dir / f"{model_name}_ROC_data.json"))

def plot_pr_curve(y_true, y_prob, weights, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_PR.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({'recall': recall.tolist(), 'precision': precision.tolist(), 'pr_auc': float(pr_auc)}, str(plot_data_dir / f"{model_name}_PR_data.json"))

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

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_calibration.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({'prob_pred': prob_pred.tolist(), 'prob_true': prob_true.tolist()}, str(plot_data_dir / f"{model_name}_calibration_data.json"))

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
    plt.plot(thresholds, net_benefits_model, label='Model')
    plt.plot(thresholds, net_benefits_all, label='Treat All')
    plt.plot(thresholds, net_benefits_none, label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_DCA.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'thresholds': [float(t) for t in thresholds],
        'net_benefit_model': [float(nb) for nb in net_benefits_model],
        'net_benefit_all': [float(nb) for nb in net_benefits_all],
        'net_benefit_none': [float(nb) for nb in net_benefits_none]
    }, str(plot_data_dir / f"{model_name}_DCA_data.json"))

def plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name="CatBoost"):
    """绘制学习曲线：训练集和测试集AUC随迭代次数变化，正方形坐标系，Times New Roman，12号字"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    n_trees = model.tree_count_ if hasattr(model, 'tree_count_') else model.get_tree_count()
    train_auc_list = []
    test_auc_list = []
    eval_steps = list(range(0, n_trees+1, max(1, n_trees//100)))
    for ntree in eval_steps:
        y_train_prob = model.predict_proba(X_train, ntree_start=0, ntree_end=ntree)[:, 1]
        y_test_prob = model.predict_proba(X_test, ntree_start=0, ntree_end=ntree)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc = roc_auc_score(y_test, y_test_prob)
        train_auc_list.append(train_auc)
        test_auc_list.append(test_auc)
    plt.figure(figsize=(8, 8))
    plt.plot(eval_steps, train_auc_list, label="Training set", color="green")
    plt.plot(eval_steps, test_auc_list, label="Test set", color="pink")
    plt.xlabel("Number of Trees (Iterations)")
    plt.ylabel("AUC-ROC")
    plt.title(f"Learning Curve (AUC) - {model_name}")
    plt.xlim(0, n_trees)
    plt.ylim(0.5, 1)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_learning_curve.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'steps': eval_steps,
        'train_auc': train_auc_list,
        'test_auc': test_auc_list
    }, str(plot_data_dir / f"{model_name}_learning_curve.json"))

def plot_threshold_curve(y_true, y_prob, model_name="CatBoost"):
    """绘制不同阈值下的sensitivity, specificity, precision, f1曲线，指定颜色和字体"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
    thresholds = np.linspace(0, 1, 101)
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    f1_list = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        f1_list.append(f1)
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(thresholds, sensitivity_list, label="Sensitivity")
    plt.plot(thresholds, specificity_list, label="Specificity")
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, f1_list, label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Curve - {model_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{model_name}_threshold_curve.png"), bbox_inches='tight', dpi=300)
    plt.close()
    save_plot_data({
        'thresholds': thresholds.tolist(),
        'sensitivity': sensitivity_list,
        'specificity': specificity_list,
        'precision': precision_list,
        'f1': f1_list
    }, str(plot_data_dir / f"{model_name}_threshold_curve.json"))

def main():
    # 加载模型
    model_path = model_dir / 'CatBoost_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # 加载数据
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    # 加载特征顺序和类别特征索引
    with open(model_dir / 'CatBoost_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    features = feature_info['features']
    cat_feature_indices = feature_info['cat_feature_indices']
    X = df[features]
    y = df['Epilepsy']
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
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # 绘图
    model_name = "CatBoost"
    plot_roc_curve(y_test, y_prob, weights_test, model_name)
    plot_pr_curve(y_test, y_prob, weights_test, model_name)
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    plot_decision_curve(y_test, y_prob, weights_test, model_name)
    # 新增：绘制学习曲线
    plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name)
    # 新增：绘制阈值曲线
    plot_threshold_curve(y_test, y_prob, model_name)

if __name__ == "__main__":
    main()



def plot_calibration(y_true, y_prob, output_dir, model_name="CatBoost"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_calibration_curve.png'))
    plt.close()
