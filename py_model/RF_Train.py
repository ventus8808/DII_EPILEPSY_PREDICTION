import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
import optuna
import traceback
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 忽略警告
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
    
    # 创建预处理器 - 处理类别特征
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)],
        remainder='passthrough'  # 保留数值特征
    )
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 分割数据
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
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor


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

def calculate_metrics(y_true, y_pred, y_prob, weights=None):
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
    ece, mce = calculate_calibration_metrics(y_true, y_prob, weights)
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

def load_objective_config(config_path='config.yaml', constraint_type='objective_constraints'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    weights = config.get('objective_weights', {})
    
    # 根据不同阶段使用不同的约束
    if constraint_type == 'cv':
        constraints = config.get('cv_constraints', {})
    elif constraint_type == 'test':
        constraints = config.get('test_constraints', {})
    else:  # 默认使用原有约束
        constraints = config.get('objective_constraints', {})
        
    return weights, constraints

def objective_function(metrics, config_path='config.yaml', constraint_type='objective_constraints'):
    """Calculate objective score based on metrics and config constraints.
    
    Args:
        metrics: 评估指标字典
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv', 'test' 或默认的 'objective_constraints'
    """
    weights, constraints = load_objective_config(config_path, constraint_type)
    
    # 创建约束检查失败的原因列表，用于调试
    failed_constraints = []
    
    # 硬约束检查
    if 'AUC_min' in constraints and metrics.get('AUC-ROC', 0) < constraints['AUC_min']:
        failed_constraints.append(f"AUC过低: {metrics.get('AUC-ROC', 0):.4f} < {constraints['AUC_min']}")
    if 'AUC_max' in constraints and metrics.get('AUC-ROC', 1) > constraints['AUC_max']:
        failed_constraints.append(f"AUC过高: {metrics.get('AUC-ROC', 0):.4f} > {constraints['AUC_max']}")
    if 'MCE_max' in constraints and metrics.get('MCE', 0) >= constraints['MCE_max']:
        failed_constraints.append(f"MCE过高: {metrics.get('MCE', 0):.4f} >= {constraints['MCE_max']}")
    if 'ECE_max' in constraints and metrics.get('ECE', 0) >= constraints['ECE_max']:
        failed_constraints.append(f"ECE过高: {metrics.get('ECE', 0):.4f} >= {constraints['ECE_max']}")
    if 'F1_min' in constraints and metrics.get('F1 Score', 0) <= constraints['F1_min']:
        failed_constraints.append(f"F1过低: {metrics.get('F1 Score', 0):.4f} <= {constraints['F1_min']}")
    if 'Sensitivity_min' in constraints and metrics.get('Sensitivity', 0) < constraints['Sensitivity_min']:
        failed_constraints.append(f"敏感度过低: {metrics.get('Sensitivity', 0):.4f} < {constraints['Sensitivity_min']}")
    if 'Specificity_min' in constraints and metrics.get('Specificity', 0) < constraints['Specificity_min']:
        failed_constraints.append(f"特异度过低: {metrics.get('Specificity', 0):.4f} < {constraints['Specificity_min']}")
    
    # 如果有约束检查失败，返回负无穷和失败原因
    if failed_constraints:
        return float('-inf'), failed_constraints
    
    # 线性加权计算分数
    score = sum(weights.get(metric, 0) * metrics.get(metric, 0) for metric in weights if metric in metrics)
    
    return score, []  # 返回分数和空的失败原因列表

def main():
    model_name = "RF"
    logger = setup_logger(model_name)
    logger.info(f"Starting Random Forest model training")
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
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
    # 变量提前初始化，避免作用域报错
    best_score = float('-inf')
    best_params = None
    best_metrics = None
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
            # 减少树的数量上限，更多关注质量而非数量
            'n_estimators': get_suggested_param(trial, 'n_estimators', 100, 800, best_params_for_search.get('n_estimators'), is_int=True),
            # 控制树的深度，避免过拟合，保持与原先的研究一致
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10, 15, 20, None]),
            # 增加树的稳定性
            'min_samples_split': get_suggested_param(trial, 'min_samples_split', 2, 30, best_params_for_search.get('min_samples_split'), is_int=True),
            'min_samples_leaf': get_suggested_param(trial, 'min_samples_leaf', 2, 15, best_params_for_search.get('min_samples_leaf'), is_int=True),
            # 阻止过拟合，保持与原先研究一致的值范围
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.5, 0.7, 0.9])
        }
        
        # 在rf_optuna_balanced研究中使用更全面的参数空间
        if getattr(trial, 'study', None) and hasattr(trial.study, '_study_id') and trial.study._study_id and 'rf_optuna_balanced' in (getattr(trial.study, '_study_name', '') or ''):
            # 重置基本参数为更强的抗过拟合配置
            params = {
                # 控制树数量，不过多
                'n_estimators': get_suggested_param(trial, 'n_estimators', 50, 400, best_params_for_search.get('n_estimators'), is_int=True),
                # 限制树的深度防止过拟合
                'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10, 15]),
                # 增加每个节点要求的样本数
                'min_samples_split': get_suggested_param(trial, 'min_samples_split', 5, 50, best_params_for_search.get('min_samples_split'), is_int=True),
                'min_samples_leaf': get_suggested_param(trial, 'min_samples_leaf', 5, 25, best_params_for_search.get('min_samples_leaf'), is_int=True),
                # 限制特征选择
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                # 类别权重不使用None选项，强制平衡
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}]),
                # 总是使用OOB评分来改善校准
                'oob_score': True,
                # 尝试不同的决策函数
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                # 总是使用bootstrap采样
                'bootstrap': True,
                # 增大剪枝力度范围
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.01, 0.1),
                # 添加最大叶子节点数量限制
                'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [50, 100, 200, None])
            }
        # 创建Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, **params))
        ])
        # 5折交叉验证并行
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        def train_fold(fold, train_idx, val_idx):
            try:
                # 每个fold再做一次SMOTE，保证分层且仅在train fold内采样
                X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
                y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
                w_tr = weights_train_res.iloc[train_idx] if weights_train_res is not None else None
                # 检查验证集中的类别分布
                if len(np.unique(y_val)) < 2:
                    logger.warning(f"[Trial {getattr(trial,'number','?')}][Fold {fold+1}/5] y_val仅有单一类别，跳过该fold")
                    return None, float('-inf'), ["y_val仅有单一类别"]
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
                pipeline.fit(X_tr_res, y_tr_res, **({'classifier__sample_weight': w_tr_res} if w_tr_res is not None else {}))
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                # 使用验证集的样本权重，而非训练集的权重
                w_val = weights_train_res.iloc[val_idx] if weights_train_res is not None else None
                metrics = calculate_metrics(y_val, y_pred, y_prob, weights=w_val)
                # 更新目标函数以使用交叉验证约束
                score, failed_reasons = objective_function(metrics, constraint_type='cv')
                return metrics, score, failed_reasons
            except Exception as e:
                logger.error(f"[Trial {getattr(trial,'number','?')}][Fold {fold+1}/5] Exception: {e}\n{traceback.format_exc()}")
                return None, float('-inf'), [f"Exception: {str(e)}"]
        
        # 执行5折交叉验证
        metrics_list = []
        scores = []
        cv_failed_reasons_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_res, y_train_res)):
            metrics, score, failed_reasons = train_fold(fold, train_idx, val_idx)
            if metrics is not None:
                metrics_list.append(metrics)
                scores.append(score)
                if failed_reasons:  # 收集失败的原因
                    cv_failed_reasons_list.extend(failed_reasons)
        
        # 只输出5折平均值日志
        if not metrics_list:
            logger.warning(f"[Trial {getattr(trial,'number','?')}] 无有效交叉验证结果, 原因: {cv_failed_reasons_list}")
            return float('-inf')
            
        avg_metrics_3 = {k: (round(np.mean([m[k] for m in metrics_list]), 3) if isinstance(metrics_list[0][k], float) else metrics_list[0][k]) for k in metrics_list[0]}
        logger.info(f"[Trial {getattr(trial,'number','?')}] 5-fold mean metrics: {avg_metrics_3}")
        
        # 计算平均指标和分数
        avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        avg_cv_score = float(np.mean(scores))
        
        # 如果存在样本失败原因，返回负无穷
        if cv_failed_reasons_list:
            logger.warning(f"[Trial {getattr(trial,'number','?')}] 交叉验证失败原因: {cv_failed_reasons_list}")
            return float('-inf')
        
        # 保存交叉验证指标到trial属性
        for k, v in avg_metrics.items():
            trial.set_user_attr(f"cv_{k}", float(v))
        
        # 在过了交叉验证约束后，训练完整模型并在测试集上评估
        logger.info(f"[Trial {getattr(trial,'number','?')}] 交叉验证通过，开始在测试集上评估...")
        
        try:
            # 使用全部训练数据训练最终模型
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, **params))
            ])
            # 使用 SMOTE 处理过的数据进行训练
            if weights_train_res is not None:
                full_pipeline.fit(X_train_res, y_train_res, classifier__sample_weight=weights_train_res)
            else:
                full_pipeline.fit(X_train_res, y_train_res)
            
            # 在测试集上评估
            test_probs = full_pipeline.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)
            test_metrics = calculate_metrics(y_test, test_preds, test_probs, weights=weights_test)
            
            # 使用测试集约束进行评估
            test_score, test_failed_reasons = objective_function(test_metrics, constraint_type='test')
            
            # 记录测试集指标
            for k, v in test_metrics.items():
                trial.set_user_attr(f"test_{k}", float(v) if isinstance(v, (int, float)) else v)
            
            # 如果测试集约束检查失败
            if test_failed_reasons:
                logger.warning(f"[Trial {getattr(trial,'number','?')}] 测试集约束检查失败: {test_failed_reasons}")
                return float('-inf')
            
            # 输出测试集结果
            test_metrics_3 = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in test_metrics.items()}
            logger.info(f"[Trial {getattr(trial,'number','?')}] 测试集指标: {test_metrics_3}")
            
            # 由于本次修改要求同时满足交叉验证和测试集约束，因此返回交叉验证分数
            return avg_cv_score
        except Exception as e:
            logger.error(f"[Trial {getattr(trial,'number','?')}] 测试集评估错误: {e}\n{traceback.format_exc()}")
            return float('-inf')
    # 先检查是否有历史最佳参数，若有则用其在当前训练集做5折CV得初始分数
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    best_param_path = model_dir / 'RF_best_params.json'
    if best_param_path.exists():
        logger.info('Found existing RF_best_params.json, evaluating initial score...')
        with open(best_param_path, 'r') as f:
            prev_params = json.load(f)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list = []
        scores = []
        cv_failed_reasons_list = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_res, y_train_res)):
            try:
                X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
                y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
                w_tr = weights_train_res.iloc[train_idx] if weights_train_res is not None else None
                if len(np.unique(y_val)) < 2:
                    logger.warning(f"[评估历史参数][Fold {fold+1}/5] y_val仅有单一类别，跳过该fold")
                    continue
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42, **prev_params))
                ])
                pipeline.fit(X_tr, y_tr, **({'classifier__sample_weight': w_tr} if w_tr is not None else {}))
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                w_val = weights_train_res.iloc[val_idx] if weights_train_res is not None else None
                metrics = calculate_metrics(y_val, y_pred, y_prob, weights=w_val)
                score, failed_reasons = objective_function(metrics, constraint_type='cv')
                if failed_reasons:
                    cv_failed_reasons_list.extend(failed_reasons)
                    continue
                metrics_list.append(metrics)
                scores.append(score)
            except Exception as e:
                logger.error(f"[Initial eval][Fold {fold+1}/5] Exception: {e}")
        if metrics_list:
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            best_score = float(np.mean(scores))
            best_params = prev_params
            best_metrics = avg_metrics
            
            # 测试集上评估历史最优参数
            logger.info("在测试集上评估历史最优参数...")
            try:
                # 使用历史最优参数训练模型
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42, **prev_params))
                ])
                # 使用全部训练数据
                if weights_train_res is not None:
                    pipeline.fit(X_train_res, y_train_res, classifier__sample_weight=weights_train_res)
                else:
                    pipeline.fit(X_train_res, y_train_res)
                
                # 评估测试集性能
                test_probs = pipeline.predict_proba(X_test)[:, 1]
                test_preds = (test_probs >= 0.5).astype(int)
                test_metrics = calculate_metrics(y_test, test_preds, test_probs, weights=weights_test)
                
                # 检查是否符合测试集约束
                test_score, test_failed_reasons = objective_function(test_metrics, constraint_type='test')
                
                test_metrics_3 = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in test_metrics.items()}
                logger.info(f"历史最优参数在测试集上的指标: {test_metrics_3}")
                
                if test_failed_reasons:
                    logger.warning(f"历史最优参数在测试集上不符合约束: {test_failed_reasons}")
                    logger.info("由于历史最优参数在测试集上不满足约束，将重新从头开始")
                    best_score = float('-inf')
                    best_params = None
                    best_metrics = None
            except Exception as e:
                logger.error(f"在测试集上评估历史最优参数时出错: {e}\n{traceback.format_exc()}")
            
            logger.info(f"Initial best score from RF_best_params.json: {best_score}")
            logger.info(f"Initial best metrics: {avg_metrics}")
    else:
        logger.info('No existing RF_best_params.json found, starting from scratch.')
    
    # Optuna study SQLite resume
    n_trials = config.get('n_trials', 30)
    optuna_db_dir = Path(__file__).parent.parent / "optuna" / "RF"
    optuna_db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = optuna_db_dir / "RF_optuna.db"
    study_name = "rf_optuna_balanced"  # 使用新的研究名称，避免与旧研究参数空间冲突
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True
    )
    for trial in tqdm(range(n_trials), desc='Optuna Trials'):
        optuna_trial = study.ask()
        # objective_function 现在返回 (score, failed_reasons) 元组
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
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, **best_params))
            ])
            # 用最优参数在全训练集上训练模型
            final_model = pipeline
            final_model.fit(X_train_res, y_train_res, **({'classifier__sample_weight': weights_train_res} if weights_train_res is not None else {}))
            # 保存模型
            model_path = model_dir / 'RF_model.pkl'
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
            param_path = model_dir / 'RF_best_params.json'
            with open(param_path, 'w') as f:
                json.dump(params_serializable, f, indent=4)
            # 保存特征信息
            feature_info = {
                'features': features,
                'cat_feature_indices': [features.index(col) for col in categorical_features if col in features]
            }
            with open(model_dir / 'RF_feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=4)
            # 保存最佳指标
            metrics_path = model_dir / 'RF_best_metrics.json'
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
        final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, **best_params))
        ])
        final_model.fit(X_train_res_final, y_train_res_final, **({'classifier__sample_weight': weights_train_res_final} if weights_train_res_final is not None else {}))
        # 在test set评估
        y_prob_test = final_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights=weights_test if weights_test is not None else None)
        logger.info("Test set metrics (never seen during training/tuning):")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        # 保存模型和参数到 model_dir
        model_path = model_dir / 'RF_best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        best_param_path = model_dir / 'RF_best_params.json'
        with open(best_param_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        # 保存最终测试集指标到 output_dir
        test_metrics_path = output_dir / 'RF_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        logger.warning("No valid model found that meets the constraints.")
        logger.warning("Check your objective constraints and data.")

if __name__ == "__main__":
    main()
