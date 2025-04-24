import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
import optuna
import traceback
from imblearn.over_sampling import SMOTE

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if weights is None:
        weights = np.ones_like(y_true)
    ece = 0
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            bin_weights = weights[in_bin]
            bin_total_weight = np.sum(bin_weights)
            actual_prob = np.average(y_true[in_bin], weights=bin_weights)
            predicted_prob = np.average(y_prob[in_bin], weights=bin_weights)
            bin_weight = bin_total_weight / np.sum(weights)
            ece += np.abs(actual_prob - predicted_prob) * bin_weight
            mce = max(mce, np.abs(actual_prob - predicted_prob))
    return ece, mce

def calculate_metrics(y_true, y_pred, y_prob, weights=None, n_bins=10):
    accuracy = accuracy_score(y_true, y_pred, sample_weight=weights)
    precision = precision_score(y_true, y_pred, sample_weight=weights)
    recall = recall_score(y_true, y_pred, sample_weight=weights)
    sensitivity = recall
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    logloss = log_loss(y_true, y_prob, sample_weight=weights)
    brier = brier_score_loss(y_true, y_prob, sample_weight=weights)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    else:
        specificity = float('nan')
        npv = float('nan')
    youden = recall + specificity - 1 if not (np.isnan(recall) or np.isnan(specificity)) else float('nan')
    kappa = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    ece, mce = calculate_calibration_metrics(y_true, y_prob, weights, n_bins)
    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "NPV": npv,
        "F1 Score": f1,
        "Youden's Index": youden,
        "Cohen's Kappa": kappa,
        "AUC-ROC": roc_auc,
        "AUC-PR": pr_auc,
        "Log Loss": logloss,
        "Brier": brier,
        "ECE": ece,
        "MCE": mce
    }

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
            X, y, weights, test_size=0.3, random_state=42, stratify=y
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        weights_train = weights_train.reset_index(drop=True)
        weights_test = weights_test.reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
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
    logger.info(f"Starting CatBoost model training")
    logger.info("[并行调参说明] 本脚本支持Optuna多进程/多机并行调参，只需多开终端运行本脚本即可，Optuna会自动分配trial。")
    X_train, X_test, y_train, y_test, weights_train, weights_test, cat_feature_indices = load_and_preprocess_data()
    # 只在训练集做 SMOTE，test/val 集绝不采样
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
    # test set 绝不做采样，且所有划分都 stratify=y，保证分层
    # 变量提前初始化，避免作用域报错
    best_params = None
    best_metrics = None
    best_score = float('-inf')
    import concurrent.futures
    import traceback
    def optuna_objective(trial):
        # 动态参数空间（以历史最优参数为中心微调）
        def get_suggested_param(trial, name, default_low, default_high, best=None, pct=0.2, is_int=False, log=False):
            if best is not None:
                low = max(default_low, best * (1 - pct))
                high = min(default_high, best * (1 + pct))
                if is_int:
                    low = int(round(low)); high = int(round(high))
            else:
                low, high = default_low, default_high
            if is_int:
                return trial.suggest_int(name, low, high)
            else:
                return trial.suggest_float(name, low, high, log=log)
        # 若有历史最优参数，动态调整参数空间
        best_params_for_search = best_params if best_params is not None else {}
        params = {
            'iterations': get_suggested_param(trial, 'iterations', 200, 800, best_params_for_search.get('iterations'), is_int=True),
            'learning_rate': get_suggested_param(trial, 'learning_rate', 0.02, 0.2, best_params_for_search.get('learning_rate'), log=True),
            'depth': get_suggested_param(trial, 'depth', 4, 8, best_params_for_search.get('depth'), is_int=True),
            'l2_leaf_reg': get_suggested_param(trial, 'l2_leaf_reg', 2.0, 8.0, best_params_for_search.get('l2_leaf_reg')),
            'early_stopping_rounds': 50,
            'random_seed': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'cat_features': cat_feature_indices
        }
        # 5折交叉验证并行
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        def train_fold(fold, train_idx, val_idx):
            try:
                # 每个fold再做一次SMOTE，保证分层且仅在train fold内采样
                X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
                y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
                w_tr = weights_train_res.iloc[train_idx] if weights_train_res is not None else None
                smote_fold = SMOTE(random_state=42)
                if w_tr is not None:
                    Xy_tr = X_tr.copy()
                    Xy_tr['__label__'] = y_tr
                    Xy_tr['__weight__'] = w_tr
                    X_res_fold, y_res_fold = smote_fold.fit_resample(Xy_tr.drop(['__label__'], axis=1), Xy_tr['__label__'])
                    w_tr_res = X_res_fold['__weight__'].reset_index(drop=True)
                    X_tr_res = X_res_fold.drop(['__weight__'], axis=1)
                    y_tr_res = y_res_fold.reset_index(drop=True)
                else:
                    X_tr_res, y_tr_res = smote_fold.fit_resample(X_tr, y_tr)
                    w_tr_res = None
                model = CatBoostClassifier(**params)
                model.fit(X_tr_res, y_tr_res, sample_weight=w_tr_res, cat_features=cat_feature_indices)
                y_prob = model.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                metrics = calculate_metrics(y_val, y_pred, y_prob)
                score = objective_function(metrics)
                return metrics, score
            except Exception as e:
                logger.error(f"[Trial {getattr(trial,'number','?')}][Fold {fold+1}/5] Exception: {e}\n{traceback.format_exc()}")
                return None, float('-inf')
        metrics_list = []
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_res, y_train_res)):
            metrics, score = train_fold(fold, train_idx, val_idx)
            if metrics is not None:
                metrics_list.append(metrics)
                scores.append(score)
        # 只输出5折平均值日志
        if metrics_list:
            avg_metrics_3 = {k: (round(np.mean([m[k] for m in metrics_list]), 3) if isinstance(metrics_list[0][k], float) else metrics_list[0][k]) for k in metrics_list[0]}
            logger.info(f"[Trial {getattr(trial,'number','?')}] 5-fold mean metrics: {avg_metrics_3}")
        # 计算平均指标和分数
        if metrics_list:
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            avg_score = float(np.mean(scores))
        else:
            avg_metrics = {}
            avg_score = float('-inf')
        # 记录所有关键指标到 Optuna trial
        for k, v in avg_metrics.items():
            trial.set_user_attr(k, float(v))
        return avg_score
    from tqdm import tqdm
    n_trials = config['n_trials'] if 'n_trials' in config else 30
    # Optuna study SQLite resume
    optuna_db_dir = Path(__file__).parent.parent / "optuna" / "CatBoost"
    optuna_db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = optuna_db_dir / "CatBoost_optuna.db"
    study = optuna.create_study(
        direction='maximize',
        study_name="catboost_optuna",
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True
    )
    # 先检查是否有历史最佳参数，若有则用其在当前训练集做5折CV得初始分数
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    best_param_path = model_dir / 'CatBoost_best_params.json'
    if best_param_path.exists():
        logger.info('Found existing CatBoost_best_params.json, evaluating initial score...')
        with open(best_param_path, 'r') as f:
            prev_params = json.load(f)
        prev_params['random_seed'] = 42
        prev_params['verbose'] = False
        prev_params['auto_class_weights'] = 'Balanced'
        prev_params['cat_features'] = cat_feature_indices
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list = []
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_res, y_train_res)):
            X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
            y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
            w_tr = weights_train_res.iloc[train_idx] if weights_train_res is not None else None
            model = CatBoostClassifier(**prev_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=cat_feature_indices)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = calculate_metrics(y_val, y_pred, y_prob)
            score = objective_function(metrics)
            metrics_list.append(metrics)
            scores.append(score)
            
        avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        best_score = float(np.mean(scores))
        best_params = prev_params
        best_metrics = avg_metrics
        logger.info(f"Initial best score from CatBoost_best_params.json: {best_score}")
        logger.info(f"Initial best metrics: {avg_metrics}")
    else:
        logger.info('No existing CatBoost_best_params.json found, starting from scratch.')
    for trial in tqdm(range(n_trials), desc='Optuna Trials'):
        optuna_trial = study.ask()
        score = optuna_objective(optuna_trial)
        study.tell(optuna_trial, score)
        trial_metrics = {k: optuna_trial.user_attrs[k] for k in optuna_trial.user_attrs if k not in ('params',)}
        trial_params = optuna_trial.params if hasattr(optuna_trial, 'params') and optuna_trial.params else optuna_trial.user_attrs.get('params', {})

        # 日志结构优化：时间戳、trial编号、AUC-ROC、AUC-PR、sensitivity、specificity、precision、score
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        auc_roc = trial_metrics.get('AUC-ROC', float('nan'))
        auc_pr = trial_metrics.get('AUC-PR', float('nan'))
        sens = trial_metrics.get('Sensitivity', float('nan'))
        spec = trial_metrics.get('Specificity', float('nan'))
        prec = trial_metrics.get('Precision', float('nan'))
        score_val = score if score is not None else float('nan')
        logger.info(f"[{now_str}][Trial {trial+1}/{n_trials}] AUC-ROC={auc_roc:.3f} AUC-PR={auc_pr:.3f} Sens={sens:.3f} Spec={spec:.3f} Prec={prec:.3f} Score={score_val:.3f}")

        if score > best_score:
            best_score = score
            best_params = trial_params
            best_metrics = trial_metrics
            # 训练并保存当前最佳模型
            best_params['random_seed'] = 42
            best_params['verbose'] = False
            best_params['auto_class_weights'] = 'Balanced'
            best_params['cat_features'] = cat_feature_indices
            # 指定 train_dir 到模型目录下
            final_model = CatBoostClassifier(**best_params, train_dir=str(model_dir / "catboost_info"))
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
    if best_params is not None:
        logger.info("Final best parameters:")
        for k, v in best_params.items():
            logger.info(f"{k}: {v}")
        logger.info("Final best metrics (CV mean):")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.3f}")
            else:
                logger.info(f"{metric}: {value}")
        # 用最终最优参数在训练集做采样训练模型，并在test set评估
        logger.info("Evaluating on held-out test set (never seen during training/tuning)...")
        # 重新采样整个训练集
        smote_final = SMOTE(random_state=42)
        if weights_train is not None:
            Xy_train = X_train.copy()
            Xy_train['__label__'] = y_train
            Xy_train['__weight__'] = weights_train
            X_res_final, y_res_final = smote_final.fit_resample(Xy_train.drop(['__label__'], axis=1), Xy_train['__label__'])
            weights_train_res_final = X_res_final['__weight__'].reset_index(drop=True)
            X_train_res_final = X_res_final.drop(['__weight__'], axis=1)
            y_train_res_final = y_res_final.reset_index(drop=True)
        else:
            X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train, y_train)
            weights_train_res_final = None
        # 用最优参数训练模型
        best_params['random_seed'] = 42
        best_params['verbose'] = False
        best_params['auto_class_weights'] = 'Balanced'
        best_params['cat_features'] = cat_feature_indices
        final_model = CatBoostClassifier(**best_params)
        final_model.fit(X_train_res_final, y_train_res_final, sample_weight=weights_train_res_final, cat_features=cat_feature_indices)
        # 在test set评估
        y_prob_test = final_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
        logger.info("Test set metrics (never seen during training/tuning):")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        # 保存模型和参数到 model_dir
        model_path = model_dir / 'CatBoost_best_model.cbm'
        final_model.save_model(model_path)
        best_param_path = model_dir / 'CatBoost_best_params.json'
        with open(best_param_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        # 保存最终测试集指标到 output_dir
        test_metrics_path = output_dir / 'CatBoost_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        logger.warning("No valid model found that meets the constraints.")
        logger.warning("Check your objective constraints and data.")

if __name__ == "__main__":
    main()
