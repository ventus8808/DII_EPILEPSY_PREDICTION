import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import signal
import joblib
import warnings
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    f1_score, 
    precision_score, 
    recall_score, 
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# Set model name and script number
MODEL_NAME = 'SVM'
SCRIPT_NUMBER = '66'

# Logging configuration
log_dir = '/Users/maguoli/Documents/Development/Predictive/Models'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f'{SCRIPT_NUMBER}_{MODEL_NAME}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Data loading
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
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, weights

# Preprocessing pipeline
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

# Evaluation metrics
def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics['AUC-PR'] = auc(recall, precision)
    
    # Calibration metrics
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    metrics['ECE'] = np.mean(np.abs(prob_true - prob_pred))
    metrics['Brier Score'] = brier_score_loss(y_true, y_prob)
    
    return metrics

# Parameter space for random search
def get_random_params():
    params = {
        'C': np.exp(random.uniform(np.log(1e-3), np.log(1e3))),  
        'kernel': random.choice(['rbf', 'linear', 'poly', 'sigmoid']),  
        'gamma': random.choice(['scale', 'auto']) if random.random() < 0.5 else np.exp(random.uniform(np.log(1e-4), np.log(1e1))),  
        'class_weight': random.choice(['balanced', None]),  
        'max_iter': random.choice([5000, 10000]),  
        'tol': random.choice([1e-4, 1e-3])  
    }
    
    # 对于多项式核，添加degree参数
    if params['kernel'] == 'poly':
        params['degree'] = random.randint(2, 5)
    
    return params

# Evaluation function
def evaluate_model(params, X_train, y_train):
    try:
        with time_limit(100):  
            preprocessor = create_preprocessor(
                numeric_features=['Age', 'BMI'], 
                categorical_features=['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', SVC(
                    C=params['C'], 
                    kernel=params['kernel'], 
                    gamma=params['gamma'],
                    class_weight=params['class_weight'],
                    probability=True, 
                    random_state=42,
                    max_iter=params['max_iter'],  
                    tol=params['tol']  
                ))
            ])
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            cv_metrics = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                pipeline.fit(X_train_fold, y_train_fold)
                y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
                y_pred = pipeline.predict(X_val_fold)
                
                metrics = calculate_metrics(y_val_fold, y_pred, y_pred_proba)
                cv_metrics.append(metrics)
                
                score = (
                    0.4 * metrics['AUC-ROC'] - 
                    0.2 * metrics['ECE'] - 
                    0.2 * metrics['Brier Score'] + 
                    0.2 * metrics['F1 Score']
                )
                cv_scores.append(score)
            
            mean_cv_score = np.mean(cv_scores)
            mean_metrics = {k: np.mean([m[k] for m in cv_metrics]) for k in cv_metrics[0].keys()}
            
            if (0.6 < mean_metrics['AUC-ROC'] < 0.95 and 
                mean_metrics['ECE'] < 0.3 and 
                mean_metrics['F1 Score'] > 0.1):
                return mean_cv_score, pipeline, mean_metrics
            else:
                return -np.inf, pipeline, mean_metrics
                
    except (TimeoutException, Exception) as e:
        logging.warning(f"Model evaluation failed: {str(e)}")
        return -np.inf, None, None

# Main execution
if __name__ == '__main__':
    # Filter convergence warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, weights = load_and_preprocess_data()
    
    best_score = -np.inf
    best_params = None
    best_model = None
    best_metrics = None
    
    # Random search
    n_trials = 200
    for trial in tqdm(range(n_trials), desc='Random Search Trials'):
        # Generate random parameters
        current_params = get_random_params()
        
        # Evaluate model
        score, model, metrics = evaluate_model(current_params, X_train, y_train)
        
        # Update best if better
        if score > best_score:
            best_score = score
            best_params = current_params
            best_model = model
            best_metrics = metrics
            
            # Log and print best results
            print("\nNew best model found!")
            print(f"Trial {trial + 1}/{n_trials}")
            print(f"Parameters: {json.dumps(best_params, indent=2)}")
            print("\nValidation Metrics:")
            for metric, value in best_metrics.items():
                print(f"{metric}: {value:.3f}")
            print(f"Overall Score: {best_score:.3f}")
            
            logging.info(f"New best score: {best_score}")
            logging.info(f"Parameters: {best_params}")
            logging.info(f"Metrics: {best_metrics}")
    
    # Final evaluation on test set
    if best_model is not None:
        print("\n" + "="*50)
        print("Training completed. Best model performance:")
        print("="*50)
        
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        final_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print("\nTest Set Metrics:")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_score': float(best_score),
            'validation_metrics': {k: float(v) for k, v in best_metrics.items()},
            'test_metrics': {k: float(v) for k, v in final_metrics.items()}
        }
        
        # Save model and results
        model_path = os.path.join(log_dir, f'{SCRIPT_NUMBER}_{MODEL_NAME}_model.pkl')
        results_path = os.path.join(log_dir, f'{SCRIPT_NUMBER}_{MODEL_NAME}_results.json')
        
        joblib.dump(best_model, model_path)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\nModel saved to:", model_path)
        print("Results saved to:", results_path)
        
        logging.info("Training completed successfully")
        logging.info(f"Final test metrics: {final_metrics}")
        logging.info(f"Model saved to: {model_path}")
    else:
        print("\nNo valid model found during random search")
        logging.warning("No valid model found during random search")
