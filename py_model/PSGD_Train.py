import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
import optuna
import traceback
from imblearn.over_sampling import SMOTE

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
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

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

# 3. 数据加载与预处理
# 全局特征定义，供 main() 和特征保存使用
categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
numeric_features = [col for col in ['Age', 'BMI'] if col in pd.read_csv(data_path).columns]
features = ['DII_food'] + numeric_features + categorical_features

# 全局预处理器定义 - 用于基本配置
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
numeric_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))  # 默认使用2次多项式
])

def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    X = df[features]
    y = df['Epilepsy']
    
    # 使用全局预处理器创建初始预处理管道
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['DII_food'] + numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
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

# 4. 指标与目标函数
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
    specificity = recall_score(y_true, y_pred, sample_weight=weights, pos_label=0)
    f1 = f1_score(y_true, y_pred, sample_weight=weights)
    
    # ROC曲线和AUC
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    # PR曲线和AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=weights).ravel()
    
    # 计算平均预测值
    avg_pred = np.average(y_pred, weights=weights)
    
    # Kappa系数
    kappa = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    
    # 对数损失和Brier分数
    log_loss_value = log_loss(y_true, y_prob, sample_weight=weights)
    brier = brier_score_loss(y_true, y_prob, sample_weight=weights)
    
    # 计算校准度量
    ece, mce = calculate_calibration_metrics(y_true, y_prob, weights, n_bins)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'avg_pred': avg_pred,
        'kappa': kappa,
        'log_loss': log_loss_value,
        'brier': brier,
        'ece': ece,
        'mce': mce
    }
    
    return metrics_dict

def load_objective_config(config_path='config.yaml', constraint_type='objective_constraints'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    constraints = {}
    if constraint_type in config:
        constraints = config[constraint_type]
    
    return constraints

def objective_function(metrics, config_path='config.yaml', constraint_type='objective_constraints'):
    """
    定制目标函数，使用config中的硬约束及自定义权重
    
    Args:
        metrics: 评估指标字典
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv', 'test' 或默认的 'objective_constraints'
        
    Returns:
        tuple: (score, failed_reasons) - 评分和失败原因列表
    """
    # 获取正确的配置部分（确保使用正确的约束类型：cv或test）
    constraints = load_objective_config(config_path, constraint_type)
    failed_reasons = []
    
    # 首先检查硬约束
    for metric, constraint in constraints.items():
        if not isinstance(constraint, dict):
            continue
            
        if 'min' in constraint and metrics[metric] < constraint['min']:
            failed_reasons.append(f"{metric}={metrics[metric]:.4f} < {constraint['min']} (min)")
            
        if 'max' in constraint and metrics[metric] > constraint['max']:
            failed_reasons.append(f"{metric}={metrics[metric]:.4f} > {constraint['max']} (max)")
    
    # 如果有任何约束未满足，返回负无穷分数
    if failed_reasons:
        return float('-inf'), failed_reasons
    
    # 计算目标函数值（带权重的评分）
    score = 0.0
    weights_sum = 0.0
    
    # 正确获取objective_weights而不是使用constraints
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用专门的权重部分
    weights = config.get('objective_weights', {})
    
    for metric, weight in weights.items():
        if metric in metrics and isinstance(weight, (int, float)):
            score += metrics[metric] * weight
            weights_sum += abs(weight)  # 使用绝对值，因为有些权重可能是负的
    
    # 默认情况，如果没有指定权重，则使用 roc_auc
    if weights_sum == 0:
        return metrics.get('roc_auc', 0.0), failed_reasons
        
    return score / weights_sum, failed_reasons

# 5. 主函数
def main():
    # 设置日志
    logger = setup_logger("PSGD")
    logger.info("Starting PSGD (Polynomial SGD) model training...")
    
    # 检查是否存在已有的最佳参数文件
    previous_best_params = None
    param_path = model_dir / 'PSGD_best_params.json'
    
    if param_path.exists():
        try:
            logger.info(f"Found existing parameter file: {param_path}")
            with open(param_path, 'r') as f:
                previous_best_params = json.load(f)
            logger.info(f"Loaded previous best parameters as optimization starting point.")
        except Exception as e:
            logger.warning(f"Failed to load previous parameters: {e}")
            previous_best_params = None
    
    # 加载和预处理数据
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # 最佳参数、分数和指标初始化
    best_params = None
    best_score = float('-inf')
    best_metrics = None
    
    # 从config中读取n_trials，保持与其他模型一致
    n_trials = config.get('n_trials', 100)
    cv_folds = config.get('cv_folds', 5)
    
    # 全局化categorical_transformer变量以便optuna_objective函数使用
    global categorical_transformer
    
    try:
        def optuna_objective(trial):
            # 获取建议的参数，尝试不同的正则化强度和损失函数
            def get_suggested_param(trial, name, default_low, default_high, best=None, pct=0.2, is_int=False, log=False):
                if best is None:
                    low, high = default_low, default_high
                else:
                    range_val = best * pct
                    low = max(default_low, best - range_val)
                    high = min(default_high, best + range_val)
                
                if is_int:
                    return trial.suggest_int(name, int(low), int(high), log=log)
                else:
                    return trial.suggest_float(name, low, high, log=log)
                    
            # 如果存在先前的最佳参数，尝试使用它们来设置起始值
            def suggest_with_prior(trial, name, suggestion_func, *args, **kwargs):
                if previous_best_params and name in previous_best_params:
                    prior_value = previous_best_params[name]
                    try:
                        # 尝试使用先前的参数值作为起点
                        if trial.number == 0:  # 只在第一次试验时使用先前的参数
                            return prior_value
                    except Exception:
                        pass  # 如果无法使用先前的参数，就使用正常的建议流程
                return suggestion_func(*args, **kwargs)
            
            # 1. 搜索多项式特征程度和配置
            polynomial_degree = suggest_with_prior(trial, 'polynomial_degree', 
                                              trial.suggest_int, 'polynomial_degree', 1, 4)  # 增加到最大4阶
            interaction_only = suggest_with_prior(trial, 'interaction_only', 
                                             trial.suggest_categorical, 'interaction_only', [True, False])  # 是否只使用交互项
            include_bias = suggest_with_prior(trial, 'include_bias', 
                                         trial.suggest_categorical, 'include_bias', [True, False])  # 是否包含偏置项
            
            # 2. 创建动态数值处理器
            # 使用局部变量避免与全局变量冲突
            trial_numeric_transformer = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(
                    degree=polynomial_degree, 
                    interaction_only=interaction_only,
                    include_bias=include_bias
                ))
            ])
            
            # 3. 构建动态预处理器
            # 使用全局的categorical_transformer
            trial_preprocessor = ColumnTransformer([
                ('num', trial_numeric_transformer, ['DII_food'] + numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
            # 4. 是否添加特征选择
            use_feature_selection = suggest_with_prior(trial, 'use_feature_selection',
                                               trial.suggest_categorical, 'use_feature_selection', [True, False])
            
            # 5. 超参数搜索空间
            # 只使用支持predict_proba的损失函数
            
            # 正则化参数细化
            penalty = suggest_with_prior(trial, 'penalty',
                                   trial.suggest_categorical, 'penalty', ['l2', 'l1', 'elasticnet', None])
            
            params = {
                'loss': suggest_with_prior(trial, 'loss', trial.suggest_categorical, 'loss', ['log_loss', 'modified_huber']),
                'penalty': penalty,  # 动态选择正则化类型
                'alpha': suggest_with_prior(trial, 'alpha', trial.suggest_float, 'alpha', 1e-6, 10.0, log=True),
                'learning_rate': suggest_with_prior(trial, 'learning_rate', trial.suggest_categorical, 'learning_rate', 
                                                  ['optimal', 'constant', 'invscaling', 'adaptive']),
                'class_weight': suggest_with_prior(trial, 'class_weight', trial.suggest_categorical, 'class_weight', ['balanced', None]),
                'max_iter': suggest_with_prior(trial, 'max_iter', trial.suggest_int, 'max_iter', 100, 2000),
                'random_state': 42,
                'average': suggest_with_prior(trial, 'average', trial.suggest_categorical, 'average', [True, False]),
                'shuffle': suggest_with_prior(trial, 'shuffle', trial.suggest_categorical, 'shuffle', [True, False]),
                'fit_intercept': suggest_with_prior(trial, 'fit_intercept', trial.suggest_categorical, 'fit_intercept', [True, False])
            }
            
            # 如果使用弹性网络正则化，添加l1_ratio参数
            if penalty == 'elasticnet':
                params['l1_ratio'] = suggest_with_prior(trial, 'l1_ratio', trial.suggest_float, 'l1_ratio', 0.0, 1.0)
            
            # 学习率参数优化 - 扩展范围和精度
            if params['learning_rate'] != 'optimal':
                params['eta0'] = suggest_with_prior(trial, 'eta0', trial.suggest_float, 'eta0', 0.0001, 1.0, log=True)
            
            # 更精细的学习率衰减参数
            if params['learning_rate'] == 'invscaling':
                params['power_t'] = suggest_with_prior(trial, 'power_t', trial.suggest_float, 'power_t', 0.01, 2.0)
                
            # 如果是自适应学习率，添加弹性参数
            if params['learning_rate'] == 'adaptive':
                params['n_iter_no_change'] = suggest_with_prior(trial, 'n_iter_no_change_adaptive', 
                                                          trial.suggest_int, 'n_iter_no_change_adaptive', 3, 50)
                params['tol'] = suggest_with_prior(trial, 'tol_adaptive', 
                                              trial.suggest_float, 'tol_adaptive', 1e-6, 1e-2, log=True)
            
            # 增强早停策略并统一处理不同的损失函数
            
            # 解耦性适用于所有主要损失函数的早停策略
            params['early_stopping'] = suggest_with_prior(trial, 'early_stopping', 
                                                   trial.suggest_categorical, 'early_stopping', [True, False])
            
            if params['early_stopping']:
                params['validation_fraction'] = suggest_with_prior(trial, 'validation_fraction', 
                                                            trial.suggest_float, 'validation_fraction', 0.1, 0.4)
                params['n_iter_no_change'] = suggest_with_prior(trial, 'n_iter_no_change', 
                                                          trial.suggest_int, 'n_iter_no_change', 3, 50)
                params['tol'] = suggest_with_prior(trial, 'tol', 
                                              trial.suggest_float, 'tol', 1e-6, 1e-2, log=True)
            
            # 针对不同的损失函数设置特殊参数
            if params['loss'] == 'modified_huber':
                # modified_huber可以设置epsilon
                params['epsilon'] = suggest_with_prior(trial, 'epsilon', trial.suggest_float, 'epsilon', 0.01, 0.5)
            
            # 添加批量大小参数
            batch_size = suggest_with_prior(trial, 'batch_size', trial.suggest_categorical, 'batch_size', 
                                        [1, 16, 32, 64, 128, 256, 512, 'auto'])
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            fold_metrics = []
            train_smote_fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                # 分割训练集和验证集
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                if weights_train is not None:
                    weights_train_fold = weights_train.iloc[train_idx]
                    weights_val_fold = weights_train.iloc[val_idx]
                else:
                    weights_train_fold = weights_val_fold = None
                
                # 应用SMOTE过采样平衡训练集
                # 为SMOTE添加k_neighbors参数优化
                smote_k = suggest_with_prior(trial, 'smote_k', trial.suggest_int, 'smote_k', 
                                     3, min(10, len(y_train_fold) - 1))  # 确保不超过样本数
                smote = SMOTE(random_state=42, k_neighbors=smote_k)
                # 处理权重
                if weights_train_fold is not None:
                    X_train_fold_with_weights = X_train_fold.copy()
                    X_train_fold_with_weights['__weight__'] = weights_train_fold
                    X_train_fold_with_weights['__label__'] = y_train_fold
                    
                    # 重采样包含权重的DataFrame
                    X_res, y_res = smote.fit_resample(
                        X_train_fold_with_weights.drop('__label__', axis=1), 
                        X_train_fold_with_weights['__label__']
                    )
                    
                    # 提取重采样后的权重和特征
                    weights_res = X_res['__weight__'].reset_index(drop=True)
                    X_res = X_res.drop('__weight__', axis=1).reset_index(drop=True)
                    y_res = y_res.reset_index(drop=True)
                else:
                    # 不带权重的SMOTE重采样
                    X_res, y_res = smote.fit_resample(X_train_fold, y_train_fold)
                    weights_res = None
                
                # 构建管道
                pipeline_steps = [('preprocessor', trial_preprocessor)]
                
                # 如果选择使用特征选择，添加到管道中
                if use_feature_selection:
                    percentile = suggest_with_prior(trial, 'percentile', trial.suggest_int, 'percentile', 50, 100)  # 选择特征百分比
                    pipeline_steps.append(('feature_selection', SelectPercentile(f_classif, percentile=percentile)))
                
                pipeline_steps.append(('classifier', SGDClassifier(**params)))
                model = Pipeline(pipeline_steps)
                
                # 训练模型
                model.fit(
                    X_res, y_res,
                    **({'classifier__sample_weight': weights_res} if weights_res is not None else {})
                )
                
                # 评估验证集
                y_val_prob = model.predict_proba(X_val_fold)[:, 1]
                y_val_pred = model.predict(X_val_fold)
                
                # 计算指标
                val_metrics = calculate_metrics(
                    y_val_fold, y_val_pred, y_val_prob, 
                    weights=weights_val_fold
                )
                fold_metrics.append(val_metrics)
                
            # 计算平均指标
            avg_metrics = {}
            for metric in fold_metrics[0].keys():
                avg_metrics[metric] = np.mean([fm[metric] for fm in fold_metrics])
            
            # 计算目标函数分数
            score, failed_reasons = objective_function(avg_metrics, config_path='config.yaml', constraint_type='cv')
            
            # 更新参数信息，供后续使用
            trial.set_user_attr('metrics', avg_metrics)
            trial.set_user_attr('params', params)
            trial.set_user_attr('failed_reasons', failed_reasons)  # 保存失败原因
            
            return score
        
        # 创建Optuna学习器
        study = optuna.create_study(direction='maximize')
        logger.info(f"Running Optuna optimization for {n_trials} trials...")
        
        # 如果有先前的最佳参数，添加特殊日志
        if previous_best_params:
            logger.info(f"Using previous best parameters as starting point for optimization")
            # 记录问出来源
            study.set_user_attr('previous_best_score', previous_best_params.get('score', 'unknown'))
            study.set_user_attr('previous_best_params', previous_best_params)
        
        # 使用tqdm包装optimize过程，显示整体优化进度
        from tqdm.auto import tqdm
        with tqdm(total=n_trials, desc="Optuna优化进度", ncols=100) as pbar:
            # 定义回调函数来更新进度条
            def tqdm_callback(study, trial):
                pbar.update(1)
                best_value = study.best_value
                best_trial = study.best_trial.number
                current_value = trial.value if trial.value is not None else float('-inf')
                # 更新进度条描述，显示当前最佳分数和本次试验分数
                pbar.set_postfix({'最佳分数': f"{best_value:.4f}(#{best_trial})", 
                                '当前分数': f"{current_value:.4f}"})
            
            # 添加回调函数到optimize过程
            study.optimize(optuna_objective, n_trials=n_trials, callbacks=[tqdm_callback])
        
        # 获取最佳试验
        best_trial = study.best_trial
        best_params = best_trial.user_attrs['params']
        best_metrics = best_trial.user_attrs['metrics']
        best_score = best_trial.value
        
        logger.info("Optimization completed.")
        logger.info(f"Best score: {best_score}")
        
        # 打印最佳参数和指标
        logger.info("Best parameters:")
        for k, v in best_params.items():
            logger.info(f"{k}: {v}")
        
        logger.info("Best metrics:")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
        
        # 保存最佳参数
        def convert_np(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(i) for i in obj]
            return obj
        
        params_serializable = {k: convert_np(v) for k, v in best_params.items()}
        param_path = model_dir / f'PSGD_best_params.json'
        with open(param_path, 'w') as f:
            json.dump(params_serializable, f, indent=4)
            
        # 保存特征顺序和信息
        feature_info = {
            'features': features,
            'cat_feature_indices': [features.index(col) for col in categorical_features if col in features],
            'polynomial_degree': best_params.get('polynomial_degree', 2),  # 记录使用的多项式程度
            'use_feature_selection': best_params.get('use_feature_selection', False),
            'percentile': best_params.get('percentile', None) if best_params.get('use_feature_selection', False) else None
        }
        with open(model_dir / 'PSGD_feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
            
        # 保存最佳指标
        metrics_path = model_dir / f'PSGD_best_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        logger.error(traceback.format_exc())
    
    # 如果成功找到最佳模型，在测试集上进行评估
    if best_params is not None:
        logger.info("Final best parameters:")
        for k, v in best_params.items():
            logger.info(f"{k}: {v}")
        logger.info("Final best metrics (CV mean):")
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
                
        # 用最终最优参数在训练集做采样训练模型，并在test set评估
        logger.info("评估在所有数据集上（从未在训练/调整期间使用）...")
        
        # 使用最佳参数设置SMOTE
        best_smote_k = best_params.get('smote_k', 5)  # 默认为5如果没有最佳值
        smote_final = SMOTE(random_state=42, k_neighbors=best_smote_k)
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
        # 确保不重复传入参数
        model_params = best_params.copy()
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        # 生成最终的多项式程度配置
        final_polynomial_degree = best_params.get('polynomial_degree', 2)  # 默认为2如果没有最佳值
        final_numeric_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=final_polynomial_degree, include_bias=False))
        ])
        
        final_preprocessor = ColumnTransformer([
            ('num', final_numeric_transformer, ['DII_food'] + numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        # 构建最终管道
        final_pipeline_steps = [('preprocessor', final_preprocessor)]
        
        # 根据最佳参数决定是否添加特征选择
        if best_params.get('use_feature_selection', False):
            best_percentile = best_params.get('percentile', 80)  # 默认值
            final_pipeline_steps.append(('feature_selection', SelectPercentile(f_classif, percentile=best_percentile)))
        
        final_pipeline_steps.append(('classifier', SGDClassifier(**model_params)))
        final_pipeline = Pipeline(final_pipeline_steps)
        
        final_pipeline.fit(
            X_train_res_final, y_train_res_final,
            **({'classifier__sample_weight': weights_train_res_final} if weights_train_res_final is not None else {})
        )
        
        # 在test set评估
        y_pred_test = final_pipeline.predict(X_test)
        y_prob_test = final_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
        
        # 检查测试集指标是否满足test约束
        test_score, test_failed_reasons = objective_function(test_metrics, config_path='config.yaml', constraint_type='test')
        
        logger.info("Test set metrics (never seen during training/tuning):")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 显示测试集约束检查结果
        if test_failed_reasons:
            logger.warning("Test set metrics did not satisfy test constraints:")
            for reason in test_failed_reasons:
                logger.warning(f"- {reason}")
        else:
            logger.info("Test set metrics satisfied all test constraints!")
            logger.info(f"Test set objective score: {test_score:.4f}")
            
        # 保存模型和参数到 model_dir
        model_path = model_dir / 'PSGD_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)
            
        # 保存最终测试集指标到 output_dir
        test_metrics_path = output_dir / 'PSGD_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        logger.warning("No valid model found that meets the constraints.")
        logger.warning("Check your objective constraints and data.")

if __name__ == "__main__":
    main()
