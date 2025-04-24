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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
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
    """Load all base models"""
    base_models = {
        'XGB': {'file': '62_XGB_model.pkl'},
        'RF': {'file': '63_RF_model.pkl'},
        'CatBoost': {'file': '64_CatBoost_model.pkl'},
        'LightGBM': {'file': '65_LightGBM_model.pkl'}
    }
    
    loaded_models = {}
    
    for name, files in base_models.items():
        try:
            # Load model
            model_path = f"/Users/maguoli/Documents/Development/Predictive/Models/{files['file']}"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            loaded_models[name] = model
            print(f"Successfully loaded {name} model")
            
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
    
    return loaded_models

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

def generate_meta_features(models, X):
    """Generate meta-features by getting predictions from all base models"""
    meta_features = np.zeros((X.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        if name == 'LightGBM':
            # LightGBM的Booster对象使用不同的预测方法
            meta_features[:, i] = model.predict(X)
        else:
            meta_features[:, i] = predict_with_sklearn_model(model, X)
    
    return meta_features

def main():
    # 设置集成模型名称
    ensemble_model_name = "BlendingEnsemble"
    
    # Setup directories
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Models', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/ROC data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/PR data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Predictions', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Plots', exist_ok=True)
    
    # Setup logging
    log_file = f"/Users/maguoli/Documents/Development/Predictive/Models/73_{ensemble_model_name}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    print("Loading data...")
    X_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/X_train.csv')
    X_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/X_test.csv')
    y_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/y_train.csv').values.ravel()
    y_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/Data/y_test.csv').values.ravel()
    
    # Split training data into training and blending sets
    print("Splitting training data for blending...")
    X_train_base, X_blend, y_train_base, y_blend = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    
    # Load base models
    print("Loading base models...")
    base_models = load_base_models()
    
    # Generate meta-features on blending set
    print("Generating meta-features on blending set...")
    meta_features_blend = generate_meta_features(base_models, X_blend)
    
    # Generate meta-features on test set
    print("Generating meta-features on test set...")
    meta_features_test = generate_meta_features(base_models, X_test)
    
    # Try different meta-learners
    meta_learners = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    best_meta_learner = None
    best_meta_learner_name = None
    best_score = -1
    
    print("Training and evaluating meta-learners...")
    for name, learner in meta_learners.items():
        # Train meta-learner
        learner.fit(meta_features_blend, y_blend)
        
        # Evaluate on test set
        y_pred_proba = learner.predict_proba(meta_features_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred_proba)
        
        print(f"\n{name} meta-learner metrics:")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        
        # Update best meta-learner
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_meta_learner = learner
            best_meta_learner_name = name
    
    print(f"\nBest meta-learner: {best_meta_learner_name} with ROC-AUC: {best_score:.4f}")
    
    # Use the best meta-learner for final predictions
    y_pred_proba = best_meta_learner.predict_proba(meta_features_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate final metrics
    metrics = calculate_metrics(y_test, y_pred_proba)
    
    # Save predictions
    pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_pred_proba
    }).to_csv(f'/Users/maguoli/Documents/Development/Predictive/Predictions/73_{ensemble_model_name}_predictions.csv', index=False)
    
    # Save curve data
    save_curve_data(y_test, y_pred_proba, ensemble_model_name)
    
    # Save meta-learner
    with open(f'/Users/maguoli/Documents/Development/Predictive/Models/73_{ensemble_model_name}_metamodel.pkl', 'wb') as f:
        pickle.dump(best_meta_learner, f)
    
    # Save meta-learner info
    meta_info = {
        'meta_learner_type': best_meta_learner_name,
        'base_models': list(base_models.keys())
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/Models/73_{ensemble_model_name}_info.json', 'w') as f:
        json.dump(meta_info, f, indent=4)
    
    # Save metrics
    with open(f'/Users/maguoli/Documents/Development/Predictive/Model metrics/73_{ensemble_model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create feature importance plot for meta-learner
    if best_meta_learner_name == 'LogisticRegression':
        plt.figure(figsize=(10, 6))
        coefficients = best_meta_learner.coef_[0]
        feature_names = list(base_models.keys())
        
        # Sort by absolute coefficient value
        sorted_idx = np.argsort(np.abs(coefficients))
        plt.barh([feature_names[i] for i in sorted_idx], coefficients[sorted_idx])
        plt.xlabel('Coefficient Value')
        plt.ylabel('Base Model')
        plt.title('Meta-Learner Coefficients')
        plt.tight_layout()
        plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/Plots/73_{ensemble_model_name}_coefficients.png')
    
    elif best_meta_learner_name == 'XGBoost':
        plt.figure(figsize=(10, 6))
        feature_names = list(base_models.keys())
        xgb.plot_importance(best_meta_learner, importance_type='weight', feature_names=feature_names)
        plt.title('Meta-Learner Feature Importance')
        plt.tight_layout()
        plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/Plots/73_{ensemble_model_name}_feature_importance.png')
    
    # Log and print results
    logging.info(f"Blending ensemble created with {best_meta_learner_name} meta-learner! Metrics: {metrics}")
    print("\nFinal blending ensemble metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nBlending ensemble model completed. Files saved with prefix {ensemble_model_name}")

if __name__ == "__main__":
    main()
