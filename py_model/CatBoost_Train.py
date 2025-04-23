import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, log_loss
from imblearn.over_sampling import SMOTE
import optuna
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 1. 配置读取
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
model_dir.mkdir(exist_ok=True)
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)
plot_dir = Path(config['plot_dir']) if 'plot_dir' in config else Path('plots')
plot_dir.mkdir(exist_ok=True)

# 2. 日志
def setup_logger(model_name):
    log_file = model_dir / f'{model_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# 3. 数据加载和预处理
# 全局特征定义，供 main() 和特征保存使用
categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
numeric_features = [col for col in ['Age', 'BMI'] if col in pd.read_csv(data_path).columns]
features = ['DII_food'] + numeric_features + categorical_features

def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    X = df[features]
    y = df['Epilepsy']
    # CatBoost 需要类别特征的列索引
    cat_feature_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        weights_train = weights_train.reset_index(drop=True)
        weights_test = weights_test.reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        weights_train = weights_test = None
    # 重新计算 cat_feature_indices，保证和 features 顺序一致
    cat_feature_indices = [features.index(col) for col in categorical_features if col in features]
    return X_train, X_test, y_train, y_test, weights_train, weights_test, cat_feature_indices

# 4. 目标函数配置和指标
import yaml

def load_objective_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    weights = config.get('objective_weights', {})
    constraints = config.get('objective_constraints', {})
    return weights, constraints

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sensitivity = recall
    specificity = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    ece = np.mean(np.abs(y_prob - y_true))
    mce = np.max(np.abs(y_prob - y_true))
    brier = np.mean((y_prob - y_true) ** 2)
    logloss = log_loss(y_true, y_prob)
    return {
        'AUC': roc_auc,
        'AUC-PR': pr_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
        'LogLoss': logloss
    }

def objective_function(metrics, config_path='config.yaml'):
    weights, constraints = load_objective_config(config_path)
    # 硬约束
    if 'AUC_min' in constraints and metrics.get('AUC', 0) < constraints['AUC_min']:
        return float('-inf')
    if 'AUC_max' in constraints and metrics.get('AUC', 1) > constraints['AUC_max']:
        return float('-inf')
    if 'ECE_max' in constraints and metrics.get('ECE', 0) >= constraints['ECE_max']:
        return float('-inf')
    if 'F1_min' in constraints and metrics.get('F1', 0) <= constraints['F1_min']:
        return float('-inf')
    if 'Sensitivity_min' in constraints and metrics.get('Sensitivity', 0) < constraints['Sensitivity_min']:
        return float('-inf')
    if 'Specificity_min' in constraints and metrics.get('Specificity', 0) < constraints['Specificity_min']:
        return float('-inf')
    # 线性加权
    score = sum(weights.get(metric, 0) * metrics.get(metric, 0) for metric in weights)
    return score

# 5. Optuna目标函数
# 注意：主流程 optuna_objective 里调用 calculate_metrics + objective_function

# 5. 主流程
def main():
    model_name = "CatBoost"
    logger = setup_logger(model_name)
    logger.info("Starting CatBoost model training")
    X_train, X_test, y_train, y_test, weights_train, weights_test, cat_feature_indices = load_and_preprocess_data()
    smote = SMOTE(random_state=42)
    if weights_train is not None:
        Xy = X_train.copy()
        Xy['__label__'] = y_train
        Xy['__weight__'] = weights_train
        X_res, y_res = smote.fit_resample(Xy.drop(['__label__'], axis=1), Xy['__label__'])
        weights_train_res = X_res['__weight__'].reset_index(drop=True)
        X_train_res = X_res.drop(['__weight__'], axis=1)
        y_train_res = y_res.reset_index(drop=True)
    else:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        weights_train_res = None
    def optuna_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'cat_features': cat_feature_indices
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train_res, y_train_res, sample_weight=weights_train_res, cat_features=cat_feature_indices)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        score = objective_function(metrics)
        # 记录所有关键指标到 Optuna trial
        for k, v in metrics.items():
            trial.set_user_attr(k, float(v))
        return score
    n_trials = config['n_trials'] if 'n_trials' in config else 30
    study = optuna.create_study(direction='maximize')
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    for trial in range(n_trials):
        optuna_trial = study.ask()
        score = optuna_objective(optuna_trial)
        study.tell(optuna_trial, score)
        trial_metrics = {k: optuna_trial.user_attrs[k] for k in optuna_trial.user_attrs if k not in ('params',)}
        trial_params = optuna_trial.params if hasattr(optuna_trial, 'params') and optuna_trial.params else optuna_trial.user_attrs.get('params', {})
        if score > best_score:
            best_score = score
            best_params = trial_params
            best_metrics = trial_metrics
            # 训练并保存当前最佳模型
            best_params['random_seed'] = 42
            best_params['verbose'] = False
            best_params['auto_class_weights'] = 'Balanced'
            best_params['cat_features'] = cat_feature_indices
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(X_train_res, y_train_res, sample_weight=weights_train_res, cat_features=cat_feature_indices)
            # 保存模型
            model_path = model_dir / f'CatBoost_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            # 保存最佳参数
            def convert_np(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            params_serializable = {k: convert_np(v) for k, v in best_params.items()}
            param_path = model_dir / f'CatBoost_best_params.json'
            with open(param_path, 'w') as f:
                json.dump(params_serializable, f, indent=4)
            # 保存特征顺序和类别特征索引
            feature_info = {
                'features': features,
                'cat_feature_indices': cat_feature_indices
            }
            with open(model_dir / 'CatBoost_feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=4)
            # 保存最佳指标
            metrics_path = model_dir / f'CatBoost_best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
            logger.info(f"[Trial {trial+1}] New best score: {best_score}")
            logger.info("Best parameters:")
            for k, v in best_params.items():
                logger.info(f"{k}: {v}")
            logger.info("Best metrics:")
            for metric, value in best_metrics.items():
                logger.info(f"{metric}: {value}")
    logger.info(f"Training finished. Best score: {best_score}")
    logger.info("Final best parameters:")
    for k, v in best_params.items():
        logger.info(f"{k}: {v}")
    logger.info("Final best metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()
