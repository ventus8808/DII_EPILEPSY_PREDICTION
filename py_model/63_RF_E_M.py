import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, accuracy_score, log_loss)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import chi2 as chi2_dist, chi2_contingency

def load_and_preprocess_data():
    """Load and preprocess data exactly as in the training script."""
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    weights = df['WTDRD1']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 
                          'Alcohol', 'Employment', 'ActivityLevel']
    
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
    
    # Use same random state as training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    """Calculate calibration metrics with sample weights."""
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is not None:
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
        weight = 1.0 if weights is None else weights[i]
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
            bin_weights = None if weights is None else weights[bin_mask]
            bin_y_true = y_true[bin_mask]
            bin_y_prob = y_prob[bin_mask]
            
            # Calculate weighted statistics
            if bin_weights is not None:
                bin_total = np.sum(bin_weights)
                bin_actual = np.sum(bin_y_true * bin_weights) / bin_total
                bin_predicted = np.sum(bin_y_prob * bin_weights) / bin_total
            else:
                bin_actual = np.mean(bin_y_true)
                bin_predicted = np.mean(bin_y_prob)
            
            # Calculate calibration error for this bin
            bin_error = np.abs(bin_predicted - bin_actual)
            bin_size = np.sum(bin_mask)
            
            # Update ECE and MCE
            if weights is not None:
                bin_weight = np.sum(bin_weights) / np.sum(weights)
            else:
                bin_weight = bin_size / len(y_true)
            
            ece += bin_weight * bin_error
            mce = max(mce, bin_error)
            
            # Store bin metrics
            bin_metrics.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_size': int(bin_size),
                'actual_prob': float(bin_actual),
                'predicted_prob': float(bin_predicted),
                'bin_error': float(bin_error)
            })
    
    # Calculate Brier score
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    return ece, mce, bin_metrics, chi2_stat, p_value

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
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    tn, fp, fn, tp = cm.ravel()
    
    # 基本指标
    metrics = {}
    
    # 0. Basic confusion matrix elements
    metrics['TP'] = float(tp)
    metrics['TN'] = float(tn)
    metrics['FP'] = float(fp)
    metrics['FN'] = float(fn)
    
    # 1. 准确率 (Accuracy) - most basic metric
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # 2. 灵敏度/召回率 (Sensitivity/Recall)
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['Recall'] = metrics['Sensitivity']  # Alias
    
    # 3. 特异度 (Specificity)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 4. 阳性预测值/精确率 (PPV/Precision)
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['PPV'] = metrics['Precision']  # Alias
    
    # 5. 阴性预测值 (NPV)
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # 6. 假阳性率 (FPR)
    metrics['FPR'] = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # 7. 假阴性率 (FNR)
    metrics['FNR'] = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    # 8. F1 Score - harmonic mean of precision and recall
    if (metrics['Precision'] + metrics['Sensitivity']) > 0:
        metrics['F1_Score'] = 2 * (metrics['Precision'] * metrics['Sensitivity']) / (metrics['Precision'] + metrics['Sensitivity'])
    else:
        metrics['F1_Score'] = 0
    
    # 9. Youden's Index - simple combination of sensitivity and specificity
    metrics['Youdens_Index'] = metrics['Sensitivity'] + metrics['Specificity'] - 1
    
    # 10. 阳性似然比 (PLR) - more complex ratio
    if metrics['FPR'] > 0:
        metrics['PLR'] = metrics['Sensitivity'] / metrics['FPR']
    else:
        metrics['PLR'] = float('inf') if metrics['Sensitivity'] > 0 else 0
    
    # 11. 阴性似然比 (NLR)
    if metrics['Specificity'] > 0:
        metrics['NLR'] = metrics['FNR'] / metrics['Specificity']
    else:
        metrics['NLR'] = float('inf') if metrics['FNR'] > 0 else 0
    
    # 12. Cohen's Kappa - agreement beyond chance
    # 计算观察一致性
    p_o = metrics['Accuracy']
    # 计算随机一致性
    p_positive = (tp + fn) / (tp + tn + fp + fn)  # 实际阳性的比例
    p_negative = (tn + fp) / (tp + tn + fp + fn)  # 实际阴性的比例
    p_predicted_positive = (tp + fp) / (tp + tn + fp + fn)  # 预测为阳性的比例
    p_predicted_negative = (tn + fn) / (tp + tn + fp + fn)  # 预测为阴性的比例
    p_e = p_positive * p_predicted_positive + p_negative * p_predicted_negative
    
    # 计算Kappa
    if p_e < 1:
        metrics['Cohens_Kappa'] = (p_o - p_e) / (1 - p_e)
    else:
        metrics['Cohens_Kappa'] = 0
    
    # 13. Log Loss - probabilistic metric
    metrics['Log_Loss'] = log_loss(y_true, y_prob, sample_weight=weights)
    
    # 14. AUC-ROC - area under curve metric
    metrics['AUC_ROC'] = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    # 15. AUC-PR - area under curve metric for imbalanced data
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    metrics['AUC_PR'] = auc(recall_curve, precision_curve)
    
    return metrics

def calculate_metrics(y_true, y_pred, y_prob, weights):
    """Calculate all evaluation metrics."""
    # Calculate basic classification metrics
    classification_metrics = calculate_classification_metrics(y_true, y_pred, y_prob, weights)
    
    # Calculate calibration metrics
    ece, mce, bin_metrics, chi2_stat, p_value = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    # Organize metrics by category
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
        'AUC-ROC': classification_metrics['AUC_ROC'],
        'AUC-PR': classification_metrics['AUC_PR'],
        'Log_Loss': classification_metrics['Log_Loss'],
        'Brier': brier,
        'ECE': ece,
        'MCE': mce,
        'HL_Chi2': chi2_stat,
        'HL_pvalue': p_value
    }
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **classification_performance, **probabilistic_metrics}
    
    return all_metrics

def main():
    # Create metrics directory if it doesn't exist
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    
    # Load the trained model
    with open('/Users/maguoli/Documents/Development/Predictive/Models/63_RF_model.pkl', 'rb') as f:
        import pickle
        model = pickle.load(f)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    
    # Get predictions on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    
    # Print metrics by category
    print("\nBasic Metrics:")
    for metric in ['Accuracy']:
        print(f"{metric}: {metrics[metric]:.4f}")
    
    print("\nClassification Performance Metrics:")
    for metric in ['Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_Score', 'Youdens_Index', 'Cohens_Kappa']:
        print(f"{metric}: {metrics[metric]:.4f}")
    
    print("\nProbabilistic Metrics:")
    for metric in ['AUC-ROC', 'AUC-PR', 'Log_Loss', 'Brier', 'ECE', 'MCE', 'HL_Chi2', 'HL_pvalue']:
        print(f"{metric}: {metrics[metric]:.4f}")
    
    # Calculate total number of metrics
    total_metrics = len(metrics)
    print(f"\nTotal number of metrics calculated: {total_metrics}")
    
    # Save metrics to JSON file
    metrics_file_path = '/Users/maguoli/Documents/Development/Predictive/Model metrics/63_RF_metrics.json'
    with open(metrics_file_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_file_path}")

if __name__ == "__main__":
    main()
