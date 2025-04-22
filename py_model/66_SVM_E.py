import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib
from scipy.stats import chi2_contingency
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Constants
MODEL_NAME = 'SVM'
SCRIPT_NUMBER = '66'
DPI = 300
FIG_SIZE = (8, 8)

def load_and_preprocess_data():
    # Read CSV file
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    # Feature engineering
    weights = df['WTDRD1']
    numeric_features = ['Age', 'BMI']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df[['DII_food']],
        df[numeric_features],
        df[categorical_features]
    ], axis=1)
    y = df['Epilepsy']
    
    # Split dataset with same random state as training
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, weights_test

def create_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('minmax', MinMaxScaler(feature_range=(-1, 1)))
    ])
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def plot_roc_curve(y_true, y_prob, weights):
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/{SCRIPT_NUMBER}_{MODEL_NAME}_ROC.png',
                bbox_inches='tight')
    plt.close()
    
    # Save data
    data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(roc_auc)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{SCRIPT_NUMBER}_{MODEL_NAME}_ROC_data.json', 'w') as f:
        json.dump(data, f)
    
    return roc_auc

def plot_pr_curve(y_true, y_prob, weights):
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/{SCRIPT_NUMBER}_{MODEL_NAME}_PR.png',
                bbox_inches='tight')
    plt.close()
    
    # Save data
    data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(pr_auc)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{SCRIPT_NUMBER}_{MODEL_NAME}_PR_data.json', 'w') as f:
        json.dump(data, f)
    
    return pr_auc

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    """Calculate calibration metrics with sample weights."""
    # 根据实际预测范围设置bin边界
    max_pred = max(y_prob)
    bin_boundaries = np.linspace(0, np.ceil(max_pred * 20) / 20, n_bins + 1)  # 向上取整到最近的0.05
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    
    ece = 0.0
    mce = 0.0
    bin_metrics = []
    total_weight = np.sum(weights)
    
    # 调试信息
    print("\nCalibration Bin Details:")
    print("-" * 70)
    print("Bin Range      | Samples | Weight | Pred Prob | True Prob | |Diff|")
    print("-" * 70)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找出在当前bin中的样本
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            bin_weights = weights[in_bin]
            bin_total_weight = np.sum(bin_weights)
            
            if bin_total_weight > 0:
                # 计算加权平均的预测概率和实际概率
                actual_prob = np.average(y_true[in_bin], weights=bin_weights)
                predicted_prob = np.average(y_prob[in_bin], weights=bin_weights)
                
                # 计算该bin的权重（相对于总样本）
                bin_weight = bin_total_weight / total_weight
                
                # 计算校准误差
                calibration_error = np.abs(actual_prob - predicted_prob)
                ece += calibration_error * bin_weight
                mce = max(mce, calibration_error)
                
                # 保存bin的信息
                bin_metrics.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'actual_prob': actual_prob,
                    'predicted_prob': predicted_prob,
                    'bin_weight': bin_weight,
                    'n_samples': np.sum(in_bin),
                    'calibration_error': calibration_error
                })
                
                # 打印调试信息
                print(f"{bin_lower:4.3f}-{bin_upper:4.3f} | {np.sum(in_bin):7d} | {bin_weight:.3f} | "
                      f"{predicted_prob:.3f} | {actual_prob:.3f} | {calibration_error:.3f}")
    
    print("-" * 70)
    print(f"Final ECE: {ece:.3f}, MCE: {mce:.3f}")
    print(f"Total samples: {len(y_true)}, Total weight: {total_weight:.3f}")
    
    return ece, mce, bin_metrics

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
    ece, mce, bin_metrics = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    # Extract data for plotting
    bin_centers = [(m['bin_lower'] + m['bin_upper'])/2 for m in bin_metrics]
    actual_probs = [m['actual_prob'] for m in bin_metrics]
    predicted_probs = [m['predicted_prob'] for m in bin_metrics]
    n_samples = [int(m['n_samples']) for m in bin_metrics]
    calibration_errors = [m['calibration_error'] for m in bin_metrics]
    
    # Create figure with two subplots side by side
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(FIG_SIZE[0]*2, FIG_SIZE[1]), dpi=DPI)
    
    # Plot in main axes (full range)
    ax_main.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax_main.plot(predicted_probs, actual_probs, 'r-', alpha=0.3)
    
    # 根据校准误差调整点的颜色
    scatter = ax_main.scatter(predicted_probs, actual_probs, 
                            s=50, alpha=0.8, 
                            c=calibration_errors, 
                            cmap='YlOrRd',
                            label='Operating points')
    plt.colorbar(scatter, ax=ax_main, label='Calibration Error')
    
    # Add sample size and error information to scatter points
    for i, (x, y, n, err) in enumerate(zip(predicted_probs, actual_probs, n_samples, calibration_errors)):
        if n > 0:  # Only add text if there are samples
            ax_main.annotate(f'n={n}\nerr={err:.3f}', 
                           (x, y), 
                           xytext=(5, 5), 
                           textcoords='offset points', 
                           fontsize=8)
    
    ax_main.set_xlim([0.0, 1.0])
    ax_main.set_ylim([0.0, 1.0])
    ax_main.set_xlabel('Mean predicted probability')
    ax_main.set_ylabel('True probability')
    ax_main.set_title(f'Full Range Calibration Curve\nECE: {ece:.3f}, MCE: {mce:.3f}, Brier: {brier:.3f}')
    ax_main.legend(loc='lower right')
    
    # Plot in zoom axes (actual prediction range)
    margin = 0.01  # Add small margin
    min_prob = max(0, min(y_prob) - margin)
    max_prob = min(1, max(y_prob) + margin)
    
    ax_zoom.plot([min_prob, max_prob], [min_prob, max_prob], 'k--', label='Perfectly calibrated')
    ax_zoom.plot(predicted_probs, actual_probs, 'r-', linewidth=2, label='Calibration curve')
    scatter_zoom = ax_zoom.scatter(predicted_probs, actual_probs, 
                                 s=50, alpha=0.8,
                                 c=calibration_errors,
                                 cmap='YlOrRd',
                                 label='Operating points')
    plt.colorbar(scatter_zoom, ax=ax_zoom, label='Calibration Error')
    
    # Add sample size information to zoom plot
    for i, (x, y, n, err) in enumerate(zip(predicted_probs, actual_probs, n_samples, calibration_errors)):
        if n > 0:  # Only add text if there are samples
            ax_zoom.annotate(f'n={n}', 
                           (x, y), 
                           xytext=(5, 5), 
                           textcoords='offset points', 
                           fontsize=8)
    
    ax_zoom.set_xlim([min_prob, max_prob])
    ax_zoom.set_ylim([min_prob, max_prob])
    ax_zoom.set_xlabel('Mean predicted probability')
    ax_zoom.set_ylabel('True probability')
    ax_zoom.set_title(f'Zoom to Prediction Range\n[{min(y_prob):.3f}, {max(y_prob):.3f}]')
    ax_zoom.legend(loc='lower right')
    
    # Add prediction range text
    plt.figtext(0.02, 0.02, 
                f'Mean prediction: {np.mean(y_prob):.3f}', 
                fontsize=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/{SCRIPT_NUMBER}_{MODEL_NAME}_Calibration.png',
                bbox_inches='tight')
    plt.close()
    
    # Save data
    data = {
        'bin_metrics': [{
            'bin_lower': float(m['bin_lower']),
            'bin_upper': float(m['bin_upper']),
            'actual_prob': float(m['actual_prob']),
            'predicted_prob': float(m['predicted_prob']),
            'bin_weight': float(m['bin_weight']),
            'n_samples': int(m['n_samples']),
            'calibration_error': float(m['calibration_error'])
        } for m in bin_metrics],
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'prediction_range': {
            'min': float(min(y_prob)),
            'max': float(max(y_prob)),
            'mean': float(np.mean(y_prob))
        }
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{SCRIPT_NUMBER}_{MODEL_NAME}_Calibration_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    return ece, mce, brier

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
    
    # Calculate rates with weights
    total_weight = np.sum(weights)
    tp_rate = np.sum(weights[true_pos]) / total_weight
    fp_rate = np.sum(weights[false_pos]) / total_weight
    
    # Calculate net benefit
    net_benefit = tp_rate - fp_rate * (threshold/(1-threshold))
    return net_benefit

def plot_decision_curve(y_true, y_prob, weights, model_name):
    """Plot Decision Curve Analysis."""
    # Create threshold array (avoid 0 and 1)
    thresholds = np.linspace(0.01, 0.3, 30)  # Focus on more relevant threshold range
    
    # Calculate prevalence (weighted)
    prevalence = np.sum(weights[y_true == 1]) / np.sum(weights)
    print(f"\nDisease prevalence: {prevalence:.4f}")
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
    
    # Convert to numpy arrays for easier operations
    net_benefits_model = np.array(net_benefits_model)
    net_benefits_all = np.array(net_benefits_all)
    
    # Treat-none is always 0
    net_benefits_none = np.zeros_like(thresholds)
    
    # Create the plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
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
    
    # Save the plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/{SCRIPT_NUMBER}_{MODEL_NAME}_DCA.png',
                bbox_inches='tight')
    plt.close()
    
    # Save the original data
    data = {
        'thresholds': thresholds.tolist(),
        'net_benefits_model': net_benefits_model.tolist(),
        'net_benefits_all': net_benefits_all.tolist(),
        'net_benefits_none': net_benefits_none.tolist(),
        'prevalence': prevalence
    }
    
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{SCRIPT_NUMBER}_{MODEL_NAME}_DCA_data.json', 'w') as f:
        json.dump(data, f)

def main():
    # Create directories if they don't exist
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, weights_test = load_and_preprocess_data()
    
    # Load model
    model = joblib.load('/Users/maguoli/Documents/Development/Predictive/Models/66_SVM_model.pkl')
    
    # Get predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    print("\nModel Performance Metrics:")
    print("-" * 50)
    
    # ROC-AUC
    roc_auc = plot_roc_curve(y_test, y_prob, weights_test)
    print(f"AUC-ROC: {roc_auc:.3f}")
    
    # PR-AUC
    pr_auc = plot_pr_curve(y_test, y_prob, weights_test)
    print(f"AUC-PR: {pr_auc:.3f}")
    
    # Other metrics
    sensitivity = recall_score(y_test, y_pred, sample_weight=weights_test)
    precision = precision_score(y_test, y_pred, sample_weight=weights_test)
    recall = recall_score(y_test, y_pred, sample_weight=weights_test)
    f1 = f1_score(y_test, y_pred, sample_weight=weights_test)
    
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Calibration metrics
    ece, mce, brier = plot_calibration_curve(y_test, y_prob, weights_test, MODEL_NAME)
    print(f"ECE: {ece:.3f}")
    print(f"MCE: {mce:.3f}")
    print(f"Brier Score: {brier:.3f}")
    
    # Decision curve analysis
    plot_decision_curve(y_test, y_prob, weights_test, MODEL_NAME)

if __name__ == "__main__":
    main()
