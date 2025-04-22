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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 300

def load_and_preprocess_data():
    """Load and preprocess data exactly as in the training script."""
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    weights = df['WTDRD1']
    numeric_features = ['Age', 'BMI']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 
                          'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df[['DII_food']],
        df[numeric_features],
        df[categorical_features]
    ], axis=1)
    y = df['Epilepsy']
    
    # Create preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Age', 'BMI', 'DII_food']),
            ('cat', categorical_transformer, categorical_features)
        ])
    
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
        weights = np.ones_like(y_true, dtype=float)
    
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
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'actual_prob': actual_prob,
                    'predicted_prob': predicted_prob,
                    'bin_weight': bin_weight,
                    'bin_total': bin_total_weight
                })
    
    return ece, mce, bin_metrics

def save_plot_data(data, filename):
    """Save plot data as JSON."""
    # Convert numpy arrays to lists
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        elif isinstance(value, np.float64):
            data[key] = float(value)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to file
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
    plt.savefig(os.path.join('/Users/maguoli/Documents/Development/Predictive/plot', f'{model_name}_ROC.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }, os.path.join('/Users/maguoli/Documents/Development/Predictive/plot/original_data', f'{model_name}_ROC_data.json'))

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
    plt.savefig(os.path.join('/Users/maguoli/Documents/Development/Predictive/plot', f'{model_name}_PR.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'auc': pr_auc
    }, os.path.join('/Users/maguoli/Documents/Development/Predictive/plot/original_data', f'{model_name}_PR_data.json'))

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
    ece, mce, bin_metrics = calculate_calibration_metrics(y_true, y_prob, weights)
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    # Extract data for plotting
    bin_centers = [(m['bin_lower'] + m['bin_upper'])/2 for m in bin_metrics]
    actual_probs = [m['actual_prob'] for m in bin_metrics]
    predicted_probs = [m['predicted_prob'] for m in bin_metrics]
    
    # Calculate Hosmer-Lemeshow test
    observed = np.array([m['bin_total'] * m['actual_prob'] for m in bin_metrics])
    expected = np.array([m['bin_total'] * m['predicted_prob'] for m in bin_metrics])
    total_obs = np.array([m['bin_total'] for m in bin_metrics])
    
    # Calculate H-L statistic
    hl_stat = np.sum(((observed - expected) ** 2) / (expected * (1 - np.array(predicted_probs))))
    df = len(bin_metrics) - 2  # degrees of freedom = number of groups - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, df)
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(predicted_probs, actual_probs, 'ro-', label='Model calibration')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title(f'Calibration Curve\nECE: {ece:.3f}, MCE: {mce:.3f}, Brier: {brier:.3f}\nH-L χ²: {hl_stat:.3f}, df: {df}, p: {p_value:.3f}')
    plt.legend(loc='lower right')
    
    # Save plot
    plt.savefig(os.path.join('/Users/maguoli/Documents/Development/Predictive/plot', f'{model_name}_Calibration.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save data
    save_plot_data({
        'bin_metrics': bin_metrics,
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'hosmer_lemeshow_chi2': hl_stat,
        'hosmer_lemeshow_df': df,
        'hosmer_lemeshow_pvalue': p_value
    }, os.path.join('/Users/maguoli/Documents/Development/Predictive/plot/original_data', f'{model_name}_Calibration_data.json'))

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
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, f'{model_name}_DCA.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the original data
    data = {
        'thresholds': thresholds.tolist(),
        'net_benefits_model': net_benefits_model.tolist(),
        'net_benefits_all': net_benefits_all.tolist(),
        'net_benefits_none': net_benefits_none.tolist(),
        'prevalence': prevalence
    }
    
    save_plot_data(data, os.path.join(original_data_dir, f'{model_name}_DCA_data.json'))

def main():
    model_name = "62_XGB"

    # Load model
    model_path = model_dir / f"{model_name}_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate metrics using RF_Metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics
    metrics_path = result_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)

    # Generate plots using RF_Plot
    plot_roc_curve(y_test, y_prob, weights_test, model_name)
    plot_pr_curve(y_test, y_prob, weights_test, model_name)
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    plot_decision_curve(y_test, y_prob, weights_test, model_name)

if __name__ == "__main__":
    main()
