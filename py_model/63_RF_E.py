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
from scipy.stats import chi2_contingency
import numpy as np

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 300

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
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    if weights is None:
        weights = np.ones_like(y_true)
    
    ece = 0
    mce = 0
    bin_metrics = []
    
    # For Hosmer-Lemeshow test
    observed = []
    expected = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
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
                
                # For Hosmer-Lemeshow test
                obs_pos = np.sum(y_true[in_bin] * bin_weights)
                obs_neg = np.sum((1 - y_true[in_bin]) * bin_weights)
                exp_pos = np.sum(y_prob[in_bin] * bin_weights)
                exp_neg = np.sum((1 - y_prob[in_bin]) * bin_weights)
                
                # Only add to chi-square calculation if expected values are not too close to zero
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
    
    # Calculate Hosmer-Lemeshow chi-square test if we have enough valid bins
    if len(observed) >= 2:  # Need at least 2 bins for the test
        chi2, p_value = chi2_contingency(np.array(observed), np.array(expected))[:2]
    else:
        chi2, p_value = np.nan, np.nan
    
    return ece, mce, bin_metrics, chi2, p_value

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
    
    # Calculate rates
    n_total = len(y_true)
    tp_rate = np.sum(true_pos) / n_total
    fp_rate = np.sum(false_pos) / n_total
    
    # Calculate net benefit
    net_benefit = tp_rate - fp_rate * (threshold/(1-threshold))
    return net_benefit

def calculate_dca_metrics(y_true, y_prob, weights=None, threshold_min=0.01, threshold_max=0.99, n_thresholds=100):
    """Calculate Decision Curve Analysis metrics."""
    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds)
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    if weights is None:
        weights = np.ones_like(y_true)
    
    total_weight = np.sum(weights)
    
    for threshold in thresholds:
        # Model strategy
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum(((y_pred == 1) & (y_true == 1)) * weights)
        fp = np.sum(((y_pred == 1) & (y_true == 0)) * weights)
        net_benefit = (tp/total_weight - (fp/total_weight) * (threshold/(1-threshold)))
        net_benefit_model.append(float(net_benefit))
        
        # Treat all strategy
        net_all = np.sum(y_true * weights)/total_weight - (np.sum((1-y_true) * weights)/total_weight) * (threshold/(1-threshold))
        net_benefit_all.append(float(net_all))
        
        # Treat none strategy
        net_benefit_none.append(0.0)
    
    return {
        'thresholds': [float(t) for t in thresholds],
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': net_benefit_none
    }

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
    
    # Create directories if they don't exist
    plot_dir = '/Users/maguoli/Documents/Development/Predictive/plot'
    original_data_dir = os.path.join(plot_dir, 'original_data')
    os.makedirs(original_data_dir, exist_ok=True)
    
    # Save the plot with modified filename
    plt.savefig(os.path.join(plot_dir, f'63_{model_name}_DCA.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the original data with modified filename
    data = {
        'thresholds': thresholds.tolist(),
        'net_benefits_model': net_benefits_model.tolist(),
        'net_benefits_all': net_benefits_all.tolist(),
        'net_benefits_none': net_benefits_none.tolist(),
        'prevalence': prevalence
    }
    
    save_plot_data(data, os.path.join(original_data_dir, f'63_{model_name}_DCA_data.json'))

def save_plot_data(data, filename):
    """Save plot data as JSON."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name):
    """Plot and save ROC curve."""
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
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/63_{model_name}_ROC.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'auc': float(roc_auc)
    }, f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/63_{model_name}_ROC_data.json')

def plot_pr_curve(y_true, y_prob, weights, model_name):
    """Plot and save Precision-Recall curve."""
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
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/63_{model_name}_PR.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'precision': [float(x) for x in precision],
        'recall': [float(x) for x in recall],
        'auc': float(pr_auc)
    }, f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/63_{model_name}_PR_data.json')

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
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
    
    # Add metrics to plot
    plt.text(0.05, 0.95, f'ECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier:.3f}\n' + 
             f'Chi2: {chi2:.3f}\np-value: {p_value:.3f}', 
             bbox=dict(facecolor='white', alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.legend(loc='lower right')
    
    # Save plot
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/63_{model_name}_Calibration.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'bin_metrics': bin_metrics,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'chi2': float(chi2),
        'p_value': float(p_value)
    }, f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/63_{model_name}_Calibration_data.json')

def calculate_metrics(y_true, y_pred, y_prob, weights):
    """Calculate all evaluation metrics."""
    precision = precision_score(y_true, y_pred, sample_weight=weights)
    recall = recall_score(y_true, y_pred, sample_weight=weights)
    sensitivity = recall
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate calibration metrics
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
    # Create plot directory if it doesn't exist
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    
    # Load the trained model
    with open('/Users/maguoli/Documents/Development/Predictive/Models/63_RF_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    
    # Get predictions on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate and save plots
    model_name = "RF"
    plot_roc_curve(y_test, y_prob, weights_test, model_name)
    plot_pr_curve(y_test, y_prob, weights_test, model_name)
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    plot_decision_curve(y_test, y_prob, weights_test, model_name)

if __name__ == "__main__":
    main()
