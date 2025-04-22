import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, accuracy_score,
                           balanced_accuracy_score, cohen_kappa_score, log_loss)
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    """Calculate calibration metrics with sample weights."""
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.array(weights)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize metrics
    bin_metrics = []
    ece = 0.0
    mce = 0.0
    
    # Create bins for Hosmer-Lemeshow test
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    
    # Calculate observed and expected frequencies for each bin
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    total = np.zeros(n_bins)
    
    for i in range(len(y_true)):
        bin_idx = bin_indices[i]
        weight = weights[i]
        total[bin_idx] += weight
        observed[bin_idx] += y_true[i] * weight
        expected[bin_idx] += y_prob[i] * weight
    
    # Calculate chi-square statistic
    chi2_stat = 0
    valid_bins = []
    
    for i in range(n_bins):
        if total[i] > 0:
            observed_bin = observed[i]
            expected_bin = expected[i]
            
            # For each bin, calculate observed and expected counts
            o_pos = observed_bin
            o_neg = total[i] - observed_bin
            e_pos = expected_bin
            e_neg = total[i] - expected_bin
            
            # Only include bins with at least 5 expected positive and negative cases
            if e_pos >= 5 and e_neg >= 5:
                valid_bins.append([o_pos, o_neg, e_pos, e_neg])
    
    # Calculate chi-square and p-value if we have valid bins
    if len(valid_bins) >= 2:
        # Convert to array for chi2_contingency
        valid_bins = np.array(valid_bins)
        observed_table = valid_bins[:, 0:2]
        expected_table = valid_bins[:, 2:4]
        
        # Calculate chi-square statistic manually
        chi2_stat = np.sum((observed_table - expected_table) ** 2 / expected_table)
        # 使用scipy.stats.chi2_dist的cdf函数
        p_value = 1 - chi2_dist.cdf(chi2_stat, df=len(valid_bins) - 2)
    else:
        chi2_stat = np.nan
        p_value = np.nan
    
    # Calculate calibration metrics for each bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples in this bin
        bin_mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        if np.any(bin_mask):
            bin_weights = weights[bin_mask]
            bin_y_true = y_true[bin_mask]
            bin_y_prob = y_prob[bin_mask]
            
            # Calculate weighted statistics
            bin_total = np.sum(bin_weights)
            bin_actual = np.sum(bin_y_true * bin_weights) / bin_total
            bin_predicted = np.sum(bin_y_prob * bin_weights) / bin_total
            
            # Calculate calibration error for this bin
            bin_error = np.abs(bin_predicted - bin_actual)
            bin_size = np.sum(bin_mask)
            
            # Update ECE and MCE
            bin_weight = np.sum(bin_weights) / np.sum(weights)
            ece += bin_weight * bin_error
            mce = max(mce, bin_error)
            
            # Store bin metrics
            bin_metrics.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_size': int(bin_size),
                'actual_prob': float(bin_actual),
                'predicted_prob': float(bin_predicted),
                'bin_error': float(bin_error),
                'bin_weight': float(bin_weight)
            })
    
    # Calculate Brier score
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    return ece, mce, bin_metrics, brier, chi2_stat, p_value

def calculate_classification_metrics(y_true, y_pred, y_prob, weights=None):
    """
    计算分类模型的各种性能指标
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    y_prob: 预测概率
    weights: 样本权重
    
    返回:
    包含所有指标的字典
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    tn, fp, fn, tp = cm.ravel()
    
    # 基本指标
    metrics = {}
    
    # 1. 准确率 (Accuracy) - most basic metric
    metrics['Accuracy'] = accuracy_score(y_true, y_pred, sample_weight=weights)
    
    # 2. 灵敏度/召回率 (Sensitivity/Recall)
    metrics['Sensitivity'] = recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 3. 特异度 (Specificity)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 4. 阳性预测值/精确率 (PPV/Precision)
    metrics['Precision'] = precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 5. 阴性预测值 (NPV)
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # 6. F1 Score - harmonic mean of precision and recall
    metrics['F1_Score'] = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 7. Youden's Index - simple combination of sensitivity and specificity
    metrics['Youdens_Index'] = metrics['Sensitivity'] + metrics['Specificity'] - 1
    
    # 8. Cohen's Kappa - agreement beyond chance
    metrics['Cohens_Kappa'] = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    
    # 9. Log Loss - probabilistic metric
    metrics['Log_Loss'] = log_loss(y_true, y_prob, sample_weight=weights)
    
    # 10. AUC-ROC - area under curve metric
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    # 11. AUC-PR - area under curve metric for imbalanced data
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    metrics['AUC-PR'] = auc(recall_curve, precision_curve)
    
    return metrics

def plot_roc_curve(fpr, tpr, auc_value, model_name):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 8), dpi=300)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_ROC.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/72_{model_name}_ROC_data.json', 'w') as f:
        json.dump(data, f)

def plot_pr_curve(precision, recall, auc_value, model_name):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(8, 8), dpi=300)
    
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AUC = {auc_value:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_PR.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/72_{model_name}_PR_data.json', 'w') as f:
        json.dump(data, f)

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
    ece, mce, bin_metrics, brier, hl_stat, p_value = calculate_calibration_metrics(y_true, y_prob, weights)
    
    # Extract data for plotting
    bin_centers = [(m['bin_lower'] + m['bin_upper'])/2 for m in bin_metrics]
    actual_probs = [m['actual_prob'] for m in bin_metrics]
    predicted_probs = [m['predicted_prob'] for m in bin_metrics]
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(predicted_probs, actual_probs, 'ro-', label='Model calibration')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title(f'Calibration Curve\nECE: {ece:.3f}, MCE: {mce:.3f}, Brier: {brier:.3f}, HL Chi2: {hl_stat:.3f}, HL p-value: {p_value:.3f}')
    plt.legend(loc='lower right')
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_Calibration.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    data = {
        'bin_metrics': bin_metrics,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'hl_stat': float(hl_stat),
        'p_value': float(p_value)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/72_{model_name}_Calibration_data.json', 'w') as f:
        json.dump(data, f, indent=4)

def plot_decision_curve(y_true, y_prob, weights, model_name):
    """Plot Decision Curve Analysis."""
    # Create threshold array (avoid 0 and 1)
    thresholds = np.linspace(0.01, 0.3, 30)  # Focus on more relevant threshold range
    
    # Calculate prevalence (weighted)
    prevalence = np.sum(weights[y_true == 1]) / np.sum(weights)
    print(f"Disease prevalence: {prevalence:.4f}")
    print(f"Prediction range: {y_prob.min():.4f} to {y_prob.max():.4f}")
    print(f"Mean prediction: {y_prob.mean():.4f}")
    
    # Calculate net benefit for different strategies
    net_benefits_model = []
    net_benefits_all = []
    
    for threshold in thresholds:
        # Model strategy
        nb_model = calculate_net_benefit(y_true, y_prob, threshold, weights)
        net_benefits_model.append(nb_model)
        
        # Treat-all strategy
        nb_all = prevalence - (1 - prevalence) * (threshold/(1-threshold))
        net_benefits_all.append(nb_all)
        
        # Print debug info for a few thresholds
        if threshold in [0.01, 0.1, 0.2]:
            print(f"\nAt threshold {threshold}:")
            print(f"Model net benefit: {nb_model:.4f}")
            print(f"Treat-all net benefit: {nb_all:.4f}")
            predictions = (y_prob >= threshold).astype(int)
            true_pos = predictions & y_true.astype(bool)
            false_pos = predictions & ~y_true.astype(bool)
            n_pos = np.sum(predictions)
            n_true = np.sum(y_true)
            print(f"Number of positive predictions: {n_pos}")
            print(f"Number of true positives in dataset: {n_true}")
            print(f"True positives at this threshold: {np.sum(true_pos)}")
            print(f"False positives at this threshold: {np.sum(false_pos)}")
            print(f"True positive rate: {np.sum(true_pos)/len(y_true):.4f}")
            print(f"False positive rate: {np.sum(false_pos)/len(y_true):.4f}")
    
    # Convert to numpy arrays for easier operations
    net_benefits_model = np.array(net_benefits_model)
    net_benefits_all = np.array(net_benefits_all)
    
    # Treat-none is always 0
    net_benefits_none = np.zeros_like(thresholds)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot with adjusted line styles and colors
    plt.plot(thresholds, net_benefits_model, 'b-', label='Model', linewidth=2.5)
    plt.plot(thresholds, net_benefits_all, 'g--', label='Treat All', linewidth=2)
    plt.plot(thresholds, net_benefits_none, 'r:', label='Treat None', linewidth=1.5)
    
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits based on prevalence
    max_benefit = max(prevalence * 1.2, max(net_benefits_model))
    plt.ylim(bottom=-0.01, top=max_benefit * 1.1)
    
    # Add text with prevalence information
    plt.text(0.02, max_benefit, f'Disease Prevalence: {prevalence:.3f}', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_DCA.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    data = {
        'thresholds': thresholds.tolist(),
        'net_benefits_model': net_benefits_model.tolist(),
        'net_benefits_all': net_benefits_all.tolist(),
        'net_benefits_none': net_benefits_none.tolist(),
        'prevalence': float(prevalence)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/72_{model_name}_DCA_data.json', 'w') as f:
        json.dump(data, f, indent=4)

def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
    """Calculate net benefit for a given threshold."""
    if weights is None:
        weights = np.ones_like(y_true)
    
    # Convert to numpy arrays for consistent operations
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    weights = np.array(weights)
    
    # Calculate predictions at threshold
    predictions = (y_prob >= threshold).astype(int)
    
    # Calculate true positives and false positives
    true_pos = predictions & y_true.astype(bool)
    false_pos = predictions & ~y_true.astype(bool)
    
    # Calculate rates using weights
    total_weight = np.sum(weights)
    tp_rate = np.sum(weights[true_pos]) / total_weight
    fp_rate = np.sum(weights[false_pos]) / total_weight
    
    # Calculate net benefit
    net_benefit = tp_rate - fp_rate * (threshold/(1-threshold))
    return net_benefit

def load_weighted_ensemble_model(model_name="WeightedEnsemble"):
    """加载加权集成模型及其权重"""
    # 加载各个基础模型
    base_models = {
        'XGB': {'file': '62_XGB_model.pkl', 'metrics_file': '62_XGB_classification_metrics.json'},
        'RF': {'file': '63_RF_model.pkl', 'metrics_file': '63_RF_metrics.json'},
        'CatBoost': {'file': '64_CatBoost_model.pkl', 'metrics_file': '64_CatBoost_metrics.json'},
        'LightGBM': {'file': '65_LightGBM_model.pkl', 'metrics_file': '65_LightGBM_metrics.json'}
    }
    
    models = {}
    model_metrics = {}
    
    for model_key, model_info in base_models.items():
        model_path = f'/Users/maguoli/Documents/Development/Predictive/Models/{model_info["file"]}'
        metrics_path = f'/Users/maguoli/Documents/Development/Predictive/Model metrics/{model_info["metrics_file"]}'
        
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                models[model_key] = pickle.load(f)
            
            # 加载指标
            with open(metrics_path, 'r') as f:
                model_metrics[model_key] = json.load(f)
            
            print(f"Successfully loaded {model_key} model and metrics")
        except Exception as e:
            print(f"Error loading {model_key} model or metrics: {e}")
    
    # 加载集成模型权重
    weights_path = f'/Users/maguoli/Documents/Development/Predictive/Models/72_{model_name}_weights.json'
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        print(f"Successfully loaded ensemble weights")
    except Exception as e:
        print(f"Error loading ensemble weights: {e}")
        # 如果无法加载权重，使用平均权重
        weights = {model_key: 1/len(models) for model_key in models.keys()}
    
    return models, weights

def weighted_ensemble_predict(models, weights, X):
    """使用加权集成模型进行预测"""
    predictions = {}
    
    # 获取每个模型的预测概率
    for model_name, model in models.items():
        if model_name == 'LightGBM':
            # LightGBM的Booster对象使用不同的预测方法
            predictions[model_name] = model.predict(X)
        else:
            predictions[model_name] = model.predict_proba(X)[:, 1]
    
    # 计算加权平均
    weighted_pred = np.zeros(len(X))
    for model_name, pred in predictions.items():
        weighted_pred += weights[model_name] * pred
    
    return weighted_pred

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    
    # 加载模型
    model_name = "WeightedEnsemble"  # 修改为与保存权重时使用的相同名称
    models, weights = load_weighted_ensemble_model(model_name)
    
    # 加载数据
    print("Loading data...")
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    weights_data = df['WTDRD1']
    covariables = ['Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
                   'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df['DII_food'],
        df[covariables]
    ], axis=1)
    y = df['Epilepsy']
    
    # 分割数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights_data, test_size=0.2, stratify=y, random_state=42
    )
    
    # 使用集成模型进行预测
    print("Making predictions with weighted ensemble...")
    y_prob = weighted_ensemble_predict(models, weights, X_test)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 计算性能指标
    print("Calculating performance metrics...")
    classification_metrics = calculate_classification_metrics(y_test, y_pred, y_prob, weights_test)
    
    # 计算校准指标
    ece, mce, bin_metrics, brier, hl_stat, p_value = calculate_calibration_metrics(y_test, y_prob, weights_test)
    
    # 组织指标按类别
    basic_metrics = {
        'Accuracy': classification_metrics['Accuracy']
    }
    
    classification_performance = {
        'Sensitivity': classification_metrics['Sensitivity'],
        'Specificity': classification_metrics['Specificity'],
        'Precision': classification_metrics['Precision'],
        'NPV': classification_metrics['NPV'],
        'F1_Score': classification_metrics['F1_Score'],
        'Youdens_Index': classification_metrics['Youdens_Index'],
        'Cohens_Kappa': classification_metrics['Cohens_Kappa']
    }
    
    probabilistic_metrics = {
        'AUC-ROC': classification_metrics['AUC-ROC'],
        'AUC-PR': classification_metrics['AUC-PR'],
        'Log_Loss': classification_metrics['Log_Loss'],
        'Brier': brier,
        'ECE': ece,
        'MCE': mce,
        'HL_Chi2': hl_stat,
        'HL_pvalue': p_value
    }
    
    # 合并所有指标
    all_metrics = {**basic_metrics, **classification_performance, **probabilistic_metrics}
    
    # 按类别打印指标
    print("\nBasic Metrics:")
    for metric in ['Accuracy']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    print("\nClassification Performance Metrics:")
    for metric in ['Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_Score', 'Youdens_Index', 'Cohens_Kappa']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    print("\nProbabilistic Metrics:")
    for metric in ['AUC-ROC', 'AUC-PR', 'Log_Loss', 'Brier', 'ECE', 'MCE', 'HL_Chi2', 'HL_pvalue']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    # 计算指标总数
    total_metrics = len(all_metrics)
    print(f"\nTotal number of metrics calculated: {total_metrics}")
    
    # 保存指标到JSON文件
    metrics_file_path = f'/Users/maguoli/Documents/Development/Predictive/Model metrics/72_{model_name}_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file_path}")
    
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=weights_test)
    roc_auc = all_metrics['AUC-ROC']
    
    # 计算PR曲线数据
    precision, recall, _ = precision_recall_curve(y_test, y_prob, sample_weight=weights_test)
    pr_auc = all_metrics['AUC-PR']
    
    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, model_name)
    print(f"ROC curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_ROC.png")
    
    # 绘制PR曲线
    plot_pr_curve(precision, recall, pr_auc, model_name)
    print(f"PR curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_PR.png")
    
    # 绘制校准曲线
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    print(f"Calibration curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_Calibration.png")
    
    # 绘制决策曲线
    plot_decision_curve(y_test, y_prob, weights_test, model_name)
    print(f"Decision curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/72_{model_name}_DCA.png")
    
    print("\nWeighted ensemble evaluation completed successfully!")

if __name__ == "__main__":
    main()
