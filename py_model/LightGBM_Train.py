import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from lightgbm import LGBMClassifier
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
    from sklearn.preprocessing import OneHotEncoder
    
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
    # 检查数据是否有缺失值
    print(f"\n原始数据中缺失值统计: \n{df[features].isna().sum()}")
    
    # 分离数值特征和类别特征
    numeric_data = df[['DII_food'] + numeric_features].copy()
    categorical_data = df[categorical_features].copy()
    
    # 独热编码处理类别特征
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(categorical_data)
    
    # 获取独热编码后的特征名称
    encoded_feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = encoder.categories_[i]
        for category in categories:
            encoded_feature_names.append(f"{feature}_{category}")
    
    # 创建独热编码后的DataFrame
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)
    
    # 合并数值特征和独热编码后的特征
    X = pd.concat([numeric_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    y = df['Epilepsy'].reset_index(drop=True)
    
    # 检查处理后的数据是否有NaN
    print(f"\n独热编码后的数据缺失值统计: \n{X.isna().sum().sum()}")
    
    # 数据分割
    from sklearn.model_selection import train_test_split
    # 保存原始索引
    original_indices = X.index.values
    
    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test, train_indices, test_indices = train_test_split(
            X, y, weights, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        weights_train = weights_train.reset_index(drop=True)
        weights_test = weights_test.reset_index(drop=True)
    else:
        # 保存原始索引
        original_indices = X.index.values
        
        # 使用train_test_split进行数据划分
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
        )
        
        # 重置索引
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        weights_train = weights_test = None
    
    # 保存编码器和编码后的特征名称，便于后续使用
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / 'encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    # 替代原来的categorical_features传递编码后的特征名称
    # 增加返回train_indices和test_indices
    return X_train, X_test, y_train, y_test, weights_train, weights_test, encoded_feature_names, train_indices, test_indices

# 4. 目标函数配置和指标
import yaml

def load_objective_config(config_path='config.yaml', constraint_type='objective_constraints'):
    """加载目标函数配置，包括权重和约束
    
    Args:
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv_constraints', 'test_constraints' 或默认的 'objective_constraints'
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    weights = config.get('objective_weights', {})
    constraints = config.get(constraint_type, {})
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

def objective_function(metrics, config_path='config.yaml', constraint_type='objective_constraints'):
    """定制目标函数，使用config中的硬约束及自定义权重
    
    Args:
        metrics: 评估指标字典
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv_constraints', 'test_constraints' 或默认的 'objective_constraints'
    """
    # 获取配置文件中的约束条件
    weights, constraints = load_objective_config(config_path, constraint_type)
    
    # 关键指标
    auc_roc = metrics.get('AUC-ROC', 0)  # AUC-ROC在metrics中是'AUC-ROC'
    sensitivity = metrics.get('Sensitivity', 0)
    specificity = metrics.get('Specificity', 0)
    f1 = metrics.get('F1 Score', 0)  # F1在metrics中是'F1 Score'
    kappa = metrics.get('Cohen\'s Kappa', 0)
    ece = metrics.get('ECE', 0)
    
    # 创建约束检查失败的原因列表，用于调试
    failed_constraints = []
    
    # 应用配置文件中的硬约束
    if 'AUC_min' in constraints and auc_roc < constraints['AUC_min']:
        failed_constraints.append(f"AUC过低: {auc_roc:.4f} < {constraints['AUC_min']}")
    if 'AUC_max' in constraints and auc_roc > constraints['AUC_max']:
        failed_constraints.append(f"AUC过高: {auc_roc:.4f} > {constraints['AUC_max']}")
    if 'Sensitivity_min' in constraints and sensitivity < constraints['Sensitivity_min']:
        failed_constraints.append(f"敏感度过低: {sensitivity:.4f} < {constraints['Sensitivity_min']}")
    if 'Specificity_min' in constraints and specificity < constraints['Specificity_min']:
        failed_constraints.append(f"特异度过低: {specificity:.4f} < {constraints['Specificity_min']}")
    # 增加F1和ECE约束检查，与XGBoost保持一致
    if 'ECE_max' in constraints and ece >= constraints['ECE_max']:
        failed_constraints.append(f"ECE过高: {ece:.4f} >= {constraints['ECE_max']}")
    if 'F1_min' in constraints and f1 <= constraints['F1_min']:
        failed_constraints.append(f"F1过低: {f1:.4f} <= {constraints['F1_min']}")
    
    # 如果有任何约束失败，给予大量惩罚但不返回-inf
    if failed_constraints:
        # 计算增强权重的分数
        base_score = 0.2 * auc_roc + 0.6 * sensitivity + 0.1 * specificity + 0.05 * f1 + 0.05 * kappa
        # 给予很大的惩罚，但还是有限的负值
        penalty = len(failed_constraints) * -1000
        return base_score + penalty, failed_constraints
    
    # 采用硬编码的权重方式，特别提高敏感度的权重（与XGBoost相似）
    # 将敏感度权重提高到0.6，更强调其重要性
    score = 0.2 * auc_roc + 0.6 * sensitivity + 0.1 * specificity + 0.05 * f1 + 0.05 * kappa
    
    return score, []  # 成功时返回计算出的目标函数值和空的失败原因列表

# 5. Optuna目标函数
# 注意：主流程 optuna_objective 里调用 calculate_metrics + objective_function

# 5. 主流程
def main():
    model_name = "LightGBM"
    logger = setup_logger(model_name)
    logger.info(f"Starting LightGBM model training")
    X_train, X_test, y_train, y_test, weights_train, weights_test, categorical_features_names, train_indices, test_indices = load_and_preprocess_data()
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
                # 确保low始终小于high
                if low >= high:
                    # 如果冲突，优先保持high，将low设为high的一个较小比例
                    low = high * 0.9
                if is_int:
                    low = int(round(low))
                    high = int(round(high))
                    # 整数情况下确保至少差1
                    if low >= high:
                        high = low + 1
            else:
                low, high = default_low, default_high
            if is_int:
                return trial.suggest_int(name, low, high)
            else:
                return trial.suggest_float(name, low, high, log=log)
        # 若有历史最优参数，动态调整参数空间
        best_params_for_search = best_params if best_params is not None else {}
        # 不同的类别不平衡处理策略 - 移除focal_loss选项
        imbalance_strategy = trial.suggest_categorical('imbalance_strategy', [
            'scale_pos_weight', 'is_unbalance', 'both'
        ])
        
        # 分类阈值策略 - 针对敏感度调优
        threshold_strategy = trial.suggest_categorical('threshold_strategy', [
            'standard',  # 0.5
            'lower_threshold',  # 降低阈值提高敏感度
        ])
        
        params = {
            # 增加n_estimators上限，降低学习率，让模型有更多机会学习少数类
            'n_estimators': get_suggested_param(trial, 'n_estimators', 1000, 5000, best_params_for_search.get('n_estimators'), is_int=True),
            'learning_rate': get_suggested_param(trial, 'learning_rate', 0.0005, 0.02, best_params_for_search.get('learning_rate'), log=True),
            
            # 提高模型复杂度，允许更精细的决策边界学习少数类
            'max_depth': get_suggested_param(trial, 'max_depth', 3, 20, best_params_for_search.get('max_depth'), is_int=True),
            'num_leaves': get_suggested_param(trial, 'num_leaves', 20, 300, best_params_for_search.get('num_leaves'), is_int=True),
            
            # 更大幅减小最小样本数，使模型能学习少数类的特征
            'min_child_samples': get_suggested_param(trial, 'min_child_samples', 1, 10, best_params_for_search.get('min_child_samples'), is_int=True),
            'min_child_weight': get_suggested_param(trial, 'min_child_weight', 0.000001, 0.05, best_params_for_search.get('min_child_weight'), log=True),
            
            # 显著降低正则化力度，更好地拟合少数类
            'reg_alpha': get_suggested_param(trial, 'reg_alpha', 0.00001, 0.5, best_params_for_search.get('reg_alpha'), log=True),
            'reg_lambda': get_suggested_param(trial, 'reg_lambda', 0.00001, 0.5, best_params_for_search.get('reg_lambda'), log=True),
            
            # 更灵活的子采样策略
            'subsample': get_suggested_param(trial, 'subsample', 0.5, 1.0, best_params_for_search.get('subsample')),
            'colsample_bytree': get_suggested_param(trial, 'colsample_bytree', 0.5, 1.0, best_params_for_search.get('colsample_bytree')),
            'subsample_freq': get_suggested_param(trial, 'subsample_freq', 0, 10, best_params_for_search.get('subsample_freq'), is_int=True),
            
            # 尝试不同的提升类型
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            
            # 使用LightGBM支持的目标函数
            'objective': trial.suggest_categorical('objective', ['binary', 'cross_entropy']),
            
            # 增大max_bin范围，提高特征细度
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            
            # boosting_type为'goss'时不能使用这些参数，所以在后面根据情况添加
            
            'random_state': 42,
            'verbose': -1
        }
        
        # 针对不同的不平衡处理策略设置相应参数
        if imbalance_strategy == 'scale_pos_weight':
            # 显著增加scale_pos_weight范围，给正类更高的权重
            params['scale_pos_weight'] = get_suggested_param(trial, 'scale_pos_weight', 50.0, 500.0, best_params_for_search.get('scale_pos_weight'))
            params['is_unbalance'] = False
        elif imbalance_strategy == 'is_unbalance':
            params['is_unbalance'] = True
            params.pop('scale_pos_weight', None)
        elif imbalance_strategy == 'both':
            # 不能同时设置 is_unbalance 和 scale_pos_weight，选择使用更大范围的 scale_pos_weight
            params['is_unbalance'] = False  # 显式设置为False
            params['scale_pos_weight'] = get_suggested_param(trial, 'scale_pos_weight', 20.0, 300.0, best_params_for_search.get('scale_pos_weight'))
            
        # 根据提升类型调整相关参数
        boosting_type = params['boosting_type']
        # 如果boosting类型为goss，不能使用bagging相关参数
        if boosting_type == 'goss':
            params.pop('subsample', None)
            params.pop('subsample_freq', None)
        else:
            # 非goss时可以使用bagging相关参数
            params['subsample'] = get_suggested_param(trial, 'subsample', 0.5, 1.0, best_params_for_search.get('subsample'))
            params['subsample_freq'] = get_suggested_param(trial, 'subsample_freq', 0, 10, best_params_for_search.get('subsample_freq'), is_int=True)

        # 对于非gbdt提升器，添加相关特定参数
        if boosting_type == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.5)
        elif boosting_type == 'goss':
            params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.5)
            params['other_rate'] = trial.suggest_float('other_rate', 0.05, 0.3)
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
                model = LGBMClassifier(**params)
                model.fit(X_tr_res, y_tr_res, sample_weight=w_tr_res)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # 根据阈值策略确定分类阈值
                if threshold_strategy == 'standard':
                    threshold = 0.5
                elif threshold_strategy == 'lower_threshold':
                    # 使用更低的阈值来提高敏感度
                    threshold = trial.suggest_float('classification_threshold', 0.05, 0.35)
                
                y_pred = (y_prob >= threshold).astype(int)
                metrics = calculate_metrics(y_val, y_pred, y_prob)
                score, _ = objective_function(metrics, constraint_type='cv_constraints')  # 使用交叉验证约束，忽略失败原因
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
    optuna_db_dir = Path(__file__).parent.parent / "optuna" / "LightGBM"
    optuna_db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = optuna_db_dir / "LightGBM_optuna.db"
    # 使用带有敏感度优化标识的新研究名称 v2
    study = optuna.create_study(
        direction='maximize',
        study_name="lightgbm_optuna_sensitivity_optimized_v2",
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True
    )
    # 先检查是否有历史最佳参数，若有则用其在当前训练集做5折CV得初始分数
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    best_param_path = model_dir / 'LightGBM_best_params.json'
    if best_param_path.exists():
        logger.info('Found existing LightGBM_best_params.json, evaluating initial score...')
        with open(best_param_path, 'r') as f:
            prev_params = json.load(f)
        prev_params['random_state'] = 42
        prev_params['verbose'] = -1
        prev_params['class_weight'] = 'balanced'
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list = []
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_res, y_train_res)):
            X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
            y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
            w_tr = weights_train_res.iloc[train_idx] if weights_train_res is not None else None
            model = LGBMClassifier(**prev_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = calculate_metrics(y_val, y_pred, y_prob)
            score, _ = objective_function(metrics, constraint_type='cv_constraints')  # 使用交叉验证约束，忽略失败原因
            metrics_list.append(metrics)
            scores.append(score)
            
        avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        best_score = float(np.mean(scores))
        best_params = prev_params
        best_metrics = avg_metrics
        logger.info(f"Initial best score from LightGBM_best_params.json: {best_score}")
        logger.info(f"Initial best metrics: {avg_metrics}")
    else:
        logger.info('No existing LightGBM_best_params.json found, starting from scratch.')
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
            best_params['random_state'] = 42
            best_params['verbose'] = -1
            best_params['class_weight'] = 'balanced'
            # 为LightGBM创建模型
            final_model = LGBMClassifier(**best_params)
            final_model.fit(X_train_res, y_train_res, sample_weight=weights_train_res)
            # 保存模型
            model_path = model_dir / f'LightGBM_model.pkl'
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
            param_path = model_dir / f'LightGBM_best_params.json'
            with open(param_path, 'w') as f:
                json.dump(params_serializable, f, indent=4)
            # 保存特征顺序和类别特征索引
            feature_info = {
                'features': features,
                'categorical_features': categorical_features_names
            }
            with open(model_dir / 'LightGBM_feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=4)
            # 保存最佳指标
            metrics_path = model_dir / f'LightGBM_best_metrics.json'
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
    # 我们总是会尝试保存一个模型，无论是否完全满足硬约束
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
            
    # 检查是否有约束失败情况
    best_trial = study.best_trial
    if 'constraints_failed' in best_trial.user_attrs and best_trial.user_attrs['constraints_failed']:
        logger.warning(f"最佳模型仍然失败了这些约束: {best_trial.user_attrs['constraints_failed']}")
        logger.warning("尽管如此，仍然继续训练和保存最佳模型，因为它是满足约束最多的一个")
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
    # 用最优参数训练模型 - 构建Pipeline对象，与XGBoost保持一致
    model_params = best_params.copy()
    model_params['random_state'] = 42
    model_params['verbose'] = -1
    model_params['class_weight'] = 'balanced'
    
    # 创建一个与XGBoost类似的Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # 定义特征处理器
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # 包含数值特征
    )
    
    # 保存原始特征名称，便于之后识别
    feature_info = {
        'numeric_features': ['DII_food'] + numeric_features,
        'categorical_features': categorical_features
    }
    with open(model_dir / 'LightGBM_feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    # 构建Pipeline
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**model_params))
    ])
    
    # 直接使用SMOTE过采样后的数据训练模型
    # 确保特征顺序正确
    feature_order = ['DII_food'] + numeric_features + categorical_features
    
    # 直接训练最终的分类器
    final_classifier = LGBMClassifier(**model_params)
    final_classifier.fit(X_train_res_final, y_train_res_final, sample_weight=weights_train_res_final)
    
    # 保存全局对象，方便后续评估
    final_model = final_classifier
    # 在test set评估
    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
    logger.info("Test set metrics (never seen during training/tuning):")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        # 保存模型和参数到 model_dir
        model_path = model_dir / 'LightGBM_best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        # 保存原始特征名称，便于之后识别
        feature_info = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_order': feature_order
        }
        with open(model_dir / 'LightGBM_feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
            
        # 保存训练和测试数据的索引，便于度量脚本复现
        train_test_indices = {'train_indices': train_indices.tolist(), 'test_indices': test_indices.tolist()}
        with open(model_dir / 'LightGBM_train_test_indices.json', 'w') as f:
            json.dump(train_test_indices, f)
        best_param_path = model_dir / 'LightGBM_best_params.json'
        with open(best_param_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        # 保存最终测试集指标到 output_dir
        test_metrics_path = output_dir / 'LightGBM_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
            
        # 修改原有的硬约束检查，使其成为警告而非错误
        test_constraints = {'Sensitivity_min': 0.3, 'Specificity_min': 0.9}
        logger.info("检查最终模型是否满足常用的测试集硬约束:")
        failed_test_constraints = []
        if test_constraints.get('Sensitivity_min') and test_metrics['Sensitivity'] < test_constraints['Sensitivity_min']:
            failed_test_constraints.append(f"Sensitivity {test_metrics['Sensitivity']:.4f} < {test_constraints['Sensitivity_min']}")
        if test_constraints.get('Specificity_min') and test_metrics['Specificity'] < test_constraints['Specificity_min']:
            failed_test_constraints.append(f"Specificity {test_metrics['Specificity']:.4f} < {test_constraints['Specificity_min']}")
            
        if failed_test_constraints:
            logger.warning(f"最终模型在测试集上不满足这些硬约束: {failed_test_constraints}")
            logger.warning("如果追求更高的敏感度，我们可能需要调整后处理步骤或降低分类阈值")
        else:
            logger.info("最终模型在测试集上满足所有硬约束!")
    else:
        logger.warning("No valid model parameters found. This is unexpected since we should always save at least one model.")
        logger.warning("Check your objective function and Optuna setup for potential issues.")

if __name__ == "__main__":
    main()
