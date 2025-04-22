import pandas as pd
import numpy as np
import json
import pickle
import os
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss
from sklearn.metrics import log_loss, cohen_kappa_score
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate ECE and MCE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            prob_true = np.mean(y_true[in_bin])
            prob_pred = np.mean(y_prob[in_bin])
            ece += np.abs(prob_true - prob_pred) * np.sum(in_bin) / len(y_true)
            mce = max(mce, np.abs(prob_true - prob_pred))
    
    return float(ece), float(mce)

def calculate_metrics(y_true, y_pred_proba):
    """Calculate all required metrics"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Basic classification metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_val = float(precision_score(y_true, y_pred, zero_division=0))
    recall_val = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix based metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((recall_val + specificity) / 2)
    
    # ROC and PR curves
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = float(auc(recall, precision))
    
    # Calibration metrics
    ece, mce = calculate_calibration_metrics(y_true, y_pred_proba)
    brier = float(brier_score_loss(y_true, y_pred_proba))
    log_loss_val = float(log_loss(y_true, y_pred_proba))
    
    # Other metrics
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision_val,
        'recall': recall_val,
        'sensitivity': recall_val,
        'specificity': specificity,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ece': ece,
        'mce': mce,
        'brier_score': brier,
        'log_loss': log_loss_val,
        'kappa': kappa
    }

def save_curve_data(y_true, y_prob, model_name):
    """Save ROC and PR curve data to JSON files"""
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(auc(fpr, tpr))
    }
    
    # PR curve data
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_data = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'auc': float(auc(recall, precision))
    }
    
    # Save ROC curve data
    with open(f'/Users/maguoli/Documents/Development/Predictive/ROC data/{model_name}_roc.json', 'w') as f:
        json.dump(roc_data, f)
    
    # Save PR curve data
    with open(f'/Users/maguoli/Documents/Development/Predictive/PR data/{model_name}_pr.json', 'w') as f:
        json.dump(pr_data, f)

def load_base_models():
    """Load all base models and their validation metrics"""
    base_models = {
        'XGB': {'file': '62_XGB_model.pkl', 'metrics_file': '62_XGB_classification_metrics.json'},
        'RF': {'file': '63_RF_model.pkl', 'metrics_file': '63_RF_metrics.json'},
        'CatBoost': {'file': '64_CatBoost_model.pkl', 'metrics_file': '64_CatBoost_metrics.json'},
        'LightGBM': {'file': '65_LightGBM_model.pkl', 'metrics_file': '65_LightGBM_metrics.json'}
    }
    
    loaded_models = {}
    model_metrics = {}
    
    for name, files in base_models.items():
        try:
            # Load model
            model_path = f"/Users/maguoli/Documents/Development/Predictive/Models/{files['file']}"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metrics
            metrics_path = f"/Users/maguoli/Documents/Development/Predictive/Model metrics/{files['metrics_file']}"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            loaded_models[name] = model
            model_metrics[name] = metrics
            
            print(f"Successfully loaded {name} model and metrics")
            
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
    
    return loaded_models, model_metrics

def calculate_weights(model_metrics, metric='roc_auc', method='proportional'):
    """
    Calculate weights for each model based on specified method
    
    Parameters:
    -----------
    model_metrics : dict
        Dictionary containing metrics for each model
    metric : str
        Metric to use for weight calculation (default: 'roc_auc')
    method : str
        Method to use for weight calculation:
        - 'proportional': Weights proportional to metric values
        - 'rank': Weights based on rank of models
        - 'softmax': Weights using softmax of metric values
        - 'equal': Equal weights for all models
    
    Returns:
    --------
    dict
        Dictionary of model weights
    """
    # Extract the specified metric for each model
    metric_values = {}
    for model_name, metrics in model_metrics.items():
        # Handle different metric naming conventions
        if metric in metrics:
            metric_values[model_name] = metrics[metric]
        elif metric.upper() in metrics:
            metric_values[model_name] = metrics[metric.upper()]
        elif 'AUC' in metrics and metric == 'roc_auc':
            metric_values[model_name] = metrics['AUC']
        elif 'AUC_ROC' in metrics and metric == 'roc_auc':
            metric_values[model_name] = metrics['AUC_ROC']
        else:
            print(f"Warning: Metric {metric} not found for model {model_name}")
            metric_values[model_name] = 0.5  # Default value
    
    # Calculate weights based on the specified method
    if method == 'equal':
        # Equal weights for all models
        n_models = len(model_metrics)
        weights = {model: 1.0 / n_models for model in model_metrics}
    
    elif method == 'proportional':
        # Weights proportional to the metric values
        total = sum(metric_values.values())
        weights = {model: value / total for model, value in metric_values.items()}
    
    elif method == 'rank':
        # Weights based on rank (better models get higher weights)
        sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        ranks = {model: i+1 for i, (model, _) in enumerate(sorted_models)}
        total_rank = sum(ranks.values())
        # Invert ranks so better models get higher weights
        weights = {model: (len(ranks) - rank + 1) / (len(ranks) * (len(ranks) + 1) / 2) 
                  for model, rank in ranks.items()}
    
    elif method == 'softmax':
        # Weights using softmax function (exponential normalization)
        # Temperature parameter controls how much to favor better models
        temperature = 1.0
        values = np.array(list(metric_values.values())) / temperature
        exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
        softmax_values = exp_values / np.sum(exp_values)
        weights = {model: float(softmax_values[i]) for i, model in enumerate(metric_values.keys())}
    
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    return weights

def optimize_weights_grid_search(models, X_val, y_val, metric_func=roc_auc_score, n_points=5):
    """
    Optimize ensemble weights using grid search on validation data
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
    metric_func : function
        Metric function to maximize (default: roc_auc_score)
    n_points : int
        Number of points to use in grid search for each model
    
    Returns:
    --------
    dict
        Optimized model weights
    """
    print("Optimizing weights using grid search...")
    
    # Get predictions from each model
    predictions = {}
    for name, model in models.items():
        try:
            predictions[name] = predict_with_sklearn_model(model, X_val)
        except Exception as e:
            print(f"Error getting predictions from {name}: {str(e)}")
            return {model: 1.0/len(models) for model in models}  # Equal weights if error
    
    # For 4 models, we need 3 degrees of freedom (last weight is determined by others)
    model_names = list(models.keys())
    
    # Create grid of weights that sum to 1
    best_score = -np.inf
    best_weights = {model: 1.0/len(models) for model in models}  # Start with equal weights
    
    # For 4 models, we can use a 3D grid search
    weight_points = np.linspace(0, 1, n_points)
    
    # Progress bar
    total_iterations = n_points**3
    with tqdm(total=total_iterations) as pbar:
        for w1 in weight_points:
            for w2 in weight_points:
                for w3 in weight_points:
                    # Ensure weights sum to 1
                    w4 = 1 - w1 - w2 - w3
                    if w4 < 0:
                        pbar.update(1)
                        continue
                    
                    # Create weight dictionary
                    weights = {
                        model_names[0]: w1,
                        model_names[1]: w2,
                        model_names[2]: w3,
                        model_names[3]: w4
                    }
                    
                    # Calculate weighted prediction
                    y_pred = np.zeros(len(X_val))
                    for name, pred in predictions.items():
                        y_pred += weights[name] * pred
                    
                    # Calculate score
                    try:
                        score = metric_func(y_val, y_pred)
                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()
                    except:
                        # Some metrics might fail for certain weight combinations
                        pass
                    
                    pbar.update(1)
    
    print(f"Best score: {best_score:.4f}")
    return best_weights

def optimize_weights_with_validation_set(models, X_val, y_val, initial_weights=None, 
                                        metric_func=roc_auc_score, n_iterations=100, 
                                        learning_rate=0.01):
    """
    Optimize ensemble weights using gradient-based optimization on validation data
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
    initial_weights : dict
        Initial weights for optimization (default: equal weights)
    metric_func : function
        Metric function to maximize (default: roc_auc_score)
    n_iterations : int
        Number of optimization iterations
    learning_rate : float
        Learning rate for optimization
    
    Returns:
    --------
    dict
        Optimized model weights
    """
    print("Optimizing weights using validation set...")
    
    # Get predictions from each model
    predictions = {}
    model_names = list(models.keys())
    for name, model in models.items():
        try:
            predictions[name] = predict_with_sklearn_model(model, X_val)
        except Exception as e:
            print(f"Error getting predictions from {name}: {str(e)}")
            return {model: 1.0/len(models) for model in models}  # Equal weights if error
    
    # Initialize weights
    if initial_weights is None:
        weights = np.ones(len(models)) / len(models)  # Equal weights
    else:
        weights = np.array([initial_weights[name] for name in model_names])
    
    # Simple optimization loop
    best_weights = weights.copy()
    best_score = -np.inf
    
    for iteration in range(n_iterations):
        # Calculate weighted prediction
        y_pred = np.zeros(len(X_val))
        for i, name in enumerate(model_names):
            y_pred += weights[i] * predictions[name]
        
        # Calculate score
        try:
            score = metric_func(y_val, y_pred)
            
            # Update best weights if score improved
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
            
            # Simple gradient estimation for each weight
            gradients = []
            for i in range(len(weights)):
                # Slightly increase this weight
                temp_weights = weights.copy()
                temp_weights[i] += 0.01
                temp_weights = temp_weights / np.sum(temp_weights)  # Normalize
                
                # Calculate new prediction
                temp_pred = np.zeros(len(X_val))
                for j, name in enumerate(model_names):
                    temp_pred += temp_weights[j] * predictions[name]
                
                # Calculate new score
                temp_score = metric_func(y_val, temp_pred)
                
                # Approximate gradient
                gradient = (temp_score - score) / 0.01
                gradients.append(gradient)
            
            # Update weights using gradients
            weights += learning_rate * np.array(gradients)
            
            # Ensure weights are non-negative
            weights = np.maximum(weights, 0)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            continue
    
    # Convert best weights back to dictionary
    optimized_weights = {name: float(best_weights[i]) for i, name in enumerate(model_names)}
    print(f"Best score: {best_score:.4f}")
    
    return optimized_weights

def predict_with_sklearn_model(model, X):
    """Get probability predictions from a scikit-learn compatible model"""
    try:
        return model.predict_proba(X)[:, 1]
    except:
        # For models that don't have predict_proba
        try:
            return model.predict(X)
        except:
            raise ValueError("Model doesn't support either predict_proba or predict")

def weighted_ensemble_predict(models, weights, X):
    """Make weighted predictions using all base models"""
    predictions = {}
    
    # Get predictions from each model
    for name, model in models.items():
        predictions[name] = predict_with_sklearn_model(model, X)
    
    # Calculate weighted average
    weighted_pred = np.zeros(len(X))
    for name, pred in predictions.items():
        weighted_pred += weights[name] * pred
    
    return weighted_pred

def main():
    """Main function to run the weighted ensemble model."""
    # 设置集成模型名称
    ensemble_model_name = "WeightedEnsemble"
    
    # Setup directories
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Models', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/ROC data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/PR data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Predictions', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Plots', exist_ok=True)
    
    # Setup logging
    model_name = "72_weighted_ensemble"
    log_file = f"/Users/maguoli/Documents/Development/Predictive/Models/{model_name}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    print("Loading data...")
    X_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/X_test.csv')
    y_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/y_test.csv').values.ravel()
    
    # Load training data to use as validation set for weight optimization
    X_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/X_train.csv')
    y_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/y_train.csv').values.ravel()
    
    # Use training data as validation set
    X_val = X_train
    y_val = y_train
    
    # Load base models and their metrics
    print("Loading base models...")
    base_models, model_metrics = load_base_models()
    
    # Choose weight optimization method
    weight_methods = {
        '1': {'name': 'proportional', 'description': 'Proportional to metric values'},
        '2': {'name': 'rank', 'description': 'Based on model ranking'},
        '3': {'name': 'softmax', 'description': 'Using softmax function'},
        '4': {'name': 'equal', 'description': 'Equal weights'},
        '5': {'name': 'grid_search', 'description': 'Grid search optimization on training data'},
        '6': {'name': 'gradient', 'description': 'Gradient-based optimization on training data'}
    }
    
    print("\nWeight optimization methods:")
    for key, method in weight_methods.items():
        print(f"{key}: {method['description']}")
    
    # 可以在这里选择不同的权重优化方法
    # 1: proportional - 基于指标值的比例分配权重
    # 2: rank - 基于模型排名分配权重
    # 3: softmax - 使用softmax函数，更强调性能好的模型
    # 4: equal - 平均分配权重
    # 5: grid_search - 使用网格搜索在训练集上优化权重
    # 6: gradient - 使用梯度优化在训练集上优化权重
    weight_method = '6'  # 使用梯度优化权重
    
    # Calculate weights based on selected method
    print(f"\nUsing weight method: {weight_methods[weight_method]['description']}")
    
    if weight_method in ['1', '2', '3', '4']:
        # Use simple weighting methods based on metrics
        weights = calculate_weights(
            model_metrics, 
            metric='roc_auc', 
            method=weight_methods[weight_method]['name']
        )
    elif weight_method == '5':
        # Use grid search optimization
        weights = optimize_weights_grid_search(
            base_models, 
            X_val, 
            y_val, 
            metric_func=roc_auc_score,
            n_points=5
        )
    elif weight_method == '6':
        # Use gradient-based optimization
        # First get initial weights from proportional method
        initial_weights = calculate_weights(model_metrics, metric='roc_auc', method='proportional')
        weights = optimize_weights_with_validation_set(
            base_models,
            X_val,
            y_val,
            initial_weights=initial_weights,
            metric_func=roc_auc_score,
            n_iterations=100
        )
    
    # Print weights
    print("\nModel weights:")
    for model_name, weight in weights.items():
        print(f"{model_name}: {weight:.4f}")
    
    # 保存权重到JSON文件
    weights_file_path = f'/Users/maguoli/Documents/Development/Predictive/Models/72_{ensemble_model_name}_weights.json'
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Models', exist_ok=True)
    with open(weights_file_path, 'w') as f:
        json.dump(weights, f, indent=4)
    print(f"Weights saved to: {weights_file_path}")
    
    # Make predictions with weighted ensemble
    print("\nMaking predictions with weighted ensemble...")
    y_pred_proba = weighted_ensemble_predict(base_models, weights, X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(y_test, y_pred_proba)
    
    # Save predictions
    pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_pred_proba
    }).to_csv(f'/Users/maguoli/Documents/Development/Predictive/Predictions/{ensemble_model_name}_predictions.csv', index=False)
    
    # Save curve data
    save_curve_data(y_test, y_pred_proba, ensemble_model_name)
    
    # Save metrics
    with open(f'/Users/maguoli/Documents/Development/Predictive/Model metrics/{ensemble_model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create bar chart of model weights
    plt.figure(figsize=(10, 6))
    models = list(weights.keys())
    model_weights = list(weights.values())
    
    plt.bar(models, model_weights, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Weight')
    plt.title('Model Weights in Weighted Ensemble')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/Plots/{ensemble_model_name}_weights.png')
    
    # Log and print results
    logging.info(f"Weighted ensemble created! Metrics: {metrics}")
    print("\nWeighted ensemble metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nWeighted ensemble model completed. Files saved with prefix {ensemble_model_name}")

if __name__ == "__main__":
    main()
