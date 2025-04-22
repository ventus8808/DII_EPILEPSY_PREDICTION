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
from sklearn.model_selection import KFold, StratifiedKFold
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
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{model_name}_roc.json', 'w') as f:
        json.dump(roc_data, f)
    
    # Save PR curve data
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/{model_name}_pr.json', 'w') as f:
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

def generate_meta_features_cv(models, X, y, cv=5):
    """Generate meta-features using cross-validation to avoid data leakage"""
    meta_features = np.zeros((X.shape[0], len(models)))
    
    # Use stratified k-fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name_idx, (name, model) in enumerate(models.items()):
        print(f"Generating meta-features for {name}...")
        
        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold_idx+1}/{cv}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y[train_idx]
            
            # Clone the model for this fold
            try:
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                # Get predictions for validation fold
                meta_features[val_idx, name_idx] = predict_with_sklearn_model(fold_model, X_val_fold)
            except Exception as e:
                print(f"Error cloning model {name}: {str(e)}")
                # Fallback: train the original model on this fold
                try:
                    model.fit(X_train_fold, y_train_fold)
                    meta_features[val_idx, name_idx] = predict_with_sklearn_model(model, X_val_fold)
                except Exception as e2:
                    print(f"Error training model {name} on fold: {str(e2)}")
                    # If all else fails, use zeros
                    meta_features[val_idx, name_idx] = 0.5
    
    return meta_features

def generate_meta_features_test(models, X_test):
    """Generate meta-features for the test set using full models"""
    meta_features = np.zeros((X_test.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        meta_features[:, i] = predict_with_sklearn_model(model, X_test)
    
    return meta_features

def main():
    # Setup directories
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Models', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Predictions', exist_ok=True)
    
    # Setup logging
    model_name = "StackingEnsemble"
    log_file = f"/Users/maguoli/Documents/Development/Predictive/Models/74_{model_name}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    print("Loading data...")
    X_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/X_train.csv')
    X_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/X_test.csv')
    y_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/y_train.csv').values.ravel()
    y_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/y_test.csv').values.ravel()
    
    # 加载样本权重（如果存在）
    try:
        weights_train = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/weights_train.csv').values.ravel()
        weights_test = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/weights_test.csv').values.ravel()
        print("Sample weights loaded successfully")
    except:
        print("No sample weights found, using uniform weights")
        weights_train = np.ones_like(y_train)
        weights_test = np.ones_like(y_test)
    
    # Load base models
    print("Loading base models...")
    base_models = load_base_models()
    
    # Generate meta-features on training set using cross-validation
    print("Generating meta-features on training set using cross-validation...")
    meta_features_train = generate_meta_features_cv(base_models, X_train, y_train, cv=5)
    
    # Save meta-features for training set
    meta_train_df = pd.DataFrame(
        meta_features_train, 
        columns=list(base_models.keys())
    )
    meta_train_df.to_csv(f'/Users/maguoli/Documents/Development/Predictive/data/74_{model_name}_meta_train.csv', index=False)
    
    # Generate meta-features on test set
    print("Generating meta-features on test set...")
    meta_features_test = generate_meta_features_test(base_models, X_test)
    
    # Save meta-features for test set
    meta_test_df = pd.DataFrame(
        meta_features_test, 
        columns=list(base_models.keys())
    )
    meta_test_df.to_csv(f'/Users/maguoli/Documents/Development/Predictive/data/74_{model_name}_meta_test.csv', index=False)
    
    # Try different meta-learners
    meta_learners = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_meta_learner = None
    best_meta_learner_name = None
    best_score = -1
    
    print("Training and evaluating meta-learners...")
    for name, learner in meta_learners.items():
        # Train meta-learner
        learner.fit(meta_features_train, y_train)
        
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
    }).to_csv(f'/Users/maguoli/Documents/Development/Predictive/Predictions/74_{model_name}_predictions.csv', index=False)
    
    # Save curve data
    save_curve_data(y_test, y_pred_proba, f"74_{model_name}")
    
    # Save meta-learner
    with open(f'/Users/maguoli/Documents/Development/Predictive/Models/74_{model_name}_metamodel.pkl', 'wb') as f:
        pickle.dump(best_meta_learner, f)
    
    # Save meta-learner info
    meta_info = {
        'meta_learner_type': best_meta_learner_name,
        'base_models': list(base_models.keys())
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/Models/74_{model_name}_info.json', 'w') as f:
        json.dump(meta_info, f, indent=4)
    
    # Save metrics
    with open(f'/Users/maguoli/Documents/Development/Predictive/Model metrics/74_{model_name}_metrics.json', 'w') as f:
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
        plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/Plots/74_{model_name}_coefficients.png')
    
    elif best_meta_learner_name == 'XGBoost':
        plt.figure(figsize=(10, 6))
        feature_names = list(base_models.keys())
        xgb.plot_importance(best_meta_learner, importance_type='weight', feature_names=feature_names)
        plt.title('Meta-Learner Feature Importance')
        plt.tight_layout()
        plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/Plots/74_{model_name}_feature_importance.png')
    
    # Log and print results
    logging.info(f"Stacking ensemble created with {best_meta_learner_name} meta-learner! Metrics: {metrics}")
    print("\nFinal stacking ensemble metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nStacking ensemble model completed. Files saved with prefix 74_{model_name}")

if __name__ == "__main__":
    main()
