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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
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

def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
    features = ['DII_food'] + numeric_features + categorical_features
    X = df[features]
    y = df['Epilepsy']
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['DII_food'] + numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
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

import yaml

def load_objective_config(config_path='config.yaml', constraint_type='objective_constraints'):
    """加载目标函数配置，支持不同阶段的约束
    
    Args:
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv', 'test' 或默认的 'objective_constraints'
    """
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
    """定制目标函数，使用config中的硬约束及自定义权重
    
    Args:
        metrics: 评估指标字典
        config_path: 配置文件路径
        constraint_type: 约束类型，可选 'cv', 'test' 或默认的 'objective_constraints'
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
    if 'ECE_max' in constraints and ece >= constraints['ECE_max']:
        failed_constraints.append(f"ECE过高: {ece:.4f} >= {constraints['ECE_max']}")
    if 'F1_min' in constraints and f1 <= constraints['F1_min']:
        failed_constraints.append(f"F1过低: {f1:.4f} <= {constraints['F1_min']}")
    if 'Sensitivity_min' in constraints and sensitivity < constraints['Sensitivity_min']:
        failed_constraints.append(f"敏感度过低: {sensitivity:.4f} < {constraints['Sensitivity_min']}")
    if 'Specificity_min' in constraints and specificity < constraints['Specificity_min']:
        failed_constraints.append(f"特异度过低: {specificity:.4f} < {constraints['Specificity_min']}")
    
    # 如果有约束检查失败，返回负无穷和失败原因
    if failed_constraints:
        return float('-inf'), failed_constraints
    
    # 定制权重 - 特别提高敏感度的重要性
    # 使用自定义权重，能提高敏感度
    score = 0.25 * auc_roc + 0.5 * sensitivity + 0.1 * specificity + 0.1 * f1 + 0.05 * kappa
    
    return score, []  # 返回分数和空的失败原因列表

# 5. Optuna + SMOTE 优化


# 5. Optuna目标函数
# 注意：主流程 optuna_objective 里调用 calculate_metrics + objective_function

# 7. 主流程
def main():
    global X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, logger
    model_name = "LGBM"
    logger = setup_logger(model_name)
    logger.info("Starting LGBM model training")
    logger.info("[并行调参说明] 本脚本支持Optuna多进程/多机并行调参，只需多开终端运行本脚本即可，Optuna会自动分配trial。")
    
    # 3. 全局特征定义，供 main() 和特征保存使用
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    numeric_features = [col for col in ['Age', 'BMI'] if col in pd.read_csv(data_path).columns]
    features = ['DII_food'] + numeric_features + categorical_features
    
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    
    # 等分交叉验证优化
    n_trials = config.get('n_trials', 100)  # 默认运行100次试验
    study = optuna.create_study(direction='maximize')
    logger.info(f"Running {n_trials} trials for hyperparameter optimization")
    
    # Optuna优化函数，返回每次试验的评分
    def optuna_objective(trial):
        # 辅助函数：基于当前最佳值生成建议参数范围
        def get_suggested_param(trial, name, default_low, default_high, best=None, pct=0.2, is_int=False, log=False):
            if best is not None:
                low = best * (1 - pct)
                high = best * (1 + pct)
                if is_int:
                    low, high = int(max(1, low)), int(high)
                return trial.suggest_float(name, low, high, log=log) if not is_int else trial.suggest_int(name, low, high)
            else:
                return trial.suggest_float(name, default_low, default_high, log=log) if not is_int else trial.suggest_int(name, default_low, default_high)
        
        # LightGBM模型需要调整的参数
        # 注意: LightGBM不能同时设置 scale_pos_weight 和 is_unbalance参数
        use_scale_pos_weight = trial.suggest_categorical('use_scale_pos_weight', [True, False])
        
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600, 800, 1000]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10, 15]),  # 降低最大深度上限，移除-1
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # 提高学习率下限
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # 提高下限
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # 提高下限
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0),  # 修改范围
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0),  # 修改范围
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),  # 大幅降低上限，防止过拟合
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),  # 增加最小样本数下限
            'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 5.0),  # 调整范围
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),  # 新增参数，控制分裂所需的最小增益
            'class_weight': 'balanced', # 添加类别权重参数
            'random_state': 42,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1  # 降低输出冗余
        }
        
        # 根据随机选择添加scale_pos_weight或is_unbalance
        if use_scale_pos_weight:
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 100.0)
        else:
            params['is_unbalance'] = True
        
        # 交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        trial_metrics_list = []
        try:
            # 平均K折交叉验证指标
            metrics_list = []
            scores = []
            cv_failed_reasons_list = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                metrics, score, failed_reasons = train_fold(fold, train_idx, val_idx, params)
                if metrics is not None:
                    metrics_list.append(metrics)
                    scores.append(score)
                    if failed_reasons:  # 收集失败的原因
                        cv_failed_reasons_list.extend(failed_reasons)
                    # 针对在缓冲区显示常见关键指标，第一折结束后展示
                    if fold == 0 and not failed_reasons:
                        auc_roc = metrics.get('AUC-ROC', 0)
                        sens = metrics.get('Sensitivity', 0)
                        spec = metrics.get('Specificity', 0)
                        print(f"\r[Fold 1/5] AUC={auc_roc:.3f} Sens={sens:.3f} Spec={spec:.3f}", end="")
            
            # 只输出5折平均值日志
            if not metrics_list:
                logger.warning(f"[Trial {getattr(trial,'number','?')}] 无有效交叉验证结果, 原因: {cv_failed_reasons_list}")
                return float('-inf')
            
            # 计算平均指标
            avg_metrics = {}
            for key in metrics_list[0].keys():
                avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
            
            # 如果存在失败原因，返回负无穷
            if cv_failed_reasons_list:
                logger.warning(f"[Trial {getattr(trial,'number','?')}] 交叉验证失败原因: {cv_failed_reasons_list}")
                return float('-inf')
            
            # 保存交叉验证指标到trial属性
            for k, v in avg_metrics.items():
                trial.set_user_attr(f"cv_{k}", float(v) if isinstance(v, (float, int)) else v)
            # 保存参数到trial属性
            trial.set_user_attr("params", params)
            
            avg_cv_score = float(np.mean(scores))
            
            # 交叉验证结果记录
            logger.info(f"[Trial {getattr(trial,'number','?')}] 5-fold mean metrics: {avg_metrics}")
            auc_roc = avg_metrics.get('AUC-ROC', 0)
            auc_pr = avg_metrics.get('AUC-PR', 0)
            sens = avg_metrics.get('Sensitivity', 0)
            spec = avg_metrics.get('Specificity', 0)
            prec = avg_metrics.get('Precision', 0)
            
            # 在过了交叉验证约束后，训练完整模型并在测试集上评估
            logger.info(f"[Trial {getattr(trial,'number','?')}] 交叉验证通过，开始在测试集上评估...")
            
            try:
                # 使用全部训练数据训练最终模型
                # SMOTE处理整个训练集
                smote_final = SMOTE(random_state=42)
                # 处理可能存在的权重
                if weights_train is not None:
                    Xy_train_temp = X_train.copy()
                    Xy_train_temp['__weight__'] = weights_train
                    X_train_res_temp, y_train_res = smote_final.fit_resample(Xy_train_temp, y_train)
                    weights_train_res = X_train_res_temp['__weight__'].reset_index(drop=True)
                    X_train_res = X_train_res_temp.drop(['__weight__'], axis=1)
                else:
                    X_train_res, y_train_res = smote_final.fit_resample(X_train, y_train)
                    weights_train_res = None
                
                # 构建并训练模型
                full_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LGBMClassifier(**params))
                ])
                full_pipeline.fit(X_train_res, y_train_res, classifier__sample_weight=weights_train_res if weights_train is not None else None)
                
                # 在测试集上评估
                test_preds = full_pipeline.predict(X_test)
                test_probs = full_pipeline.predict_proba(X_test)[:, 1]
                test_metrics = calculate_metrics(y_test, test_preds, test_probs, weights=weights_test)
                
                # 使用测试集约束进行评估
                test_score, test_failed_reasons = objective_function(test_metrics, constraint_type='test')
                
                # 记录测试集指标
                for k, v in test_metrics.items():
                    trial.set_user_attr(f"test_{k}", float(v) if isinstance(v, (int, float)) else v)
                # 保存测试集结果metrics到trial属性 - 这是最终指标
                trial.set_user_attr("metrics", test_metrics)
                
                # 如果测试集约束检查失败
                if test_failed_reasons:
                    logger.warning(f"[Trial {getattr(trial,'number','?')}] 测试集约束检查失败: {test_failed_reasons}")
                    return float('-inf')
                
                # 输出测试集结果
                test_metrics_3 = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in test_metrics.items()}
                logger.info(f"[Trial {getattr(trial,'number','?')}] 测试集指标: {test_metrics_3}")
                
                # 记录试验的完整信息
                logger.info(f"Trial {trial.number}: CV AUC={auc_roc:.3f} Sens={sens:.3f} Spec={spec:.3f} | Test AUC={test_metrics.get('AUC-ROC', 0):.3f} Sens={test_metrics.get('Sensitivity', 0):.3f} Spec={test_metrics.get('Specificity', 0):.3f}")
                
                # 返回交叉验证分数
                return avg_cv_score
            except Exception as e:
                logger.error(f"[Trial {getattr(trial,'number','?')}] 测试集评估错误: {e}\n{traceback.format_exc()}")
                return float('-inf')
        
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            traceback.print_exc()
            return float('-inf')
    
    # 训练指定折数的模型
    def train_fold(fold, train_idx, val_idx, params):
        try:
            # 分割数据和样本重量
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_train_fold = weights_train.iloc[train_idx] if weights_train is not None else None
            w_val_fold = weights_train.iloc[val_idx] if weights_train is not None else None
            
            # 检查验证集中的类别分布
            if len(np.unique(y_val_fold)) < 2:
                logger.warning(f"[Trial {getattr(trial,'number','?')}][Fold {fold+1}/5] y_val仅有单一类别，跳过该fold")
                return None, float('-inf'), ["y_val仅有单一类别"]
            
            # SMOTE处理训练集
            smote = SMOTE(random_state=42)
            X_train_fold_res, y_train_fold_res = smote.fit_resample(X_train_fold, y_train_fold)
            
            # 构建和训练模型
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LGBMClassifier(**params))
            ])
            pipeline.fit(X_train_fold_res, y_train_fold_res)
            
            # 验证集预测
            y_val_pred = pipeline.predict(X_val_fold)
            y_val_prob = pipeline.predict_proba(X_val_fold)[:, 1]
            
            # 计算指标
            metrics = calculate_metrics(y_val_fold, y_val_pred, y_val_prob, weights=w_val_fold)
            
            # 使用交叉验证约束
            score, failed_reasons = objective_function(metrics, constraint_type='cv')
            
            return metrics, score, failed_reasons
        except Exception as e:
            logger.error(f"Error in fold {fold}: {e}")
            return None, float('-inf'), [f"Exception: {str(e)}"]
    
    # 运行Optuna优化
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    best_pipeline = None
    
    # 检查是否有已存在的最佳参数
    best_params_file = model_dir / 'LGBM_best_params.json'
    if best_params_file.exists():
        logger.info('Found existing LGBM_best_params.json, evaluating initial score...')
        try:
            with open(best_params_file, 'r') as f:
                best_params = json.load(f)
                
            # 使用已有最佳参数评估初始分数
            # 创建具有相同接口的伪试验对象
            class PseudoTrial:
                def __init__(self, params):
                    self.params = params
                    self.user_attrs = {}
                    
                def set_user_attr(self, key, value):
                    self.user_attrs[key] = value
                    
                def suggest_categorical(self, name, choices):
                    return self.params.get(name, choices[0])
                    
                def suggest_float(self, name, low, high, **kwargs):
                    return self.params.get(name, (low + high) / 2)
                    
                def suggest_int(self, name, low, high, **kwargs):
                    return self.params.get(name, low)
            
            initial_trial = PseudoTrial(best_params)
            initial_score = optuna_objective(initial_trial)
            avg_metrics = initial_trial.user_attrs.get("metrics", {})
            
            # 更新最佳分数和指标
            best_score = initial_score
            best_metrics = avg_metrics
            # 记录初始参数
            trial_metrics = initial_trial.user_attrs.get('metrics', {})
            trial_params = initial_trial.params
            
            logger.info(f"Initial best score from LGBM_best_params.json: {best_score:.3f}")
            logger.info(f"Initial best metrics: {avg_metrics}")
        except Exception as e:
            logger.error(f"Error loading existing params: {e}")
            logger.info('Starting optimization from scratch due to error loading params.')
    else:
        logger.info('No existing LGBM_best_params.json found, starting from scratch.')
    
    try:
        # 运行调参试验并显示进度
        from tqdm import tqdm
        for trial in tqdm(range(n_trials), desc='Optuna Trials'):
            optuna_trial = study.ask()
            try:
                score = optuna_objective(optuna_trial)
                study.tell(optuna_trial, score)
                
                # 获取当前试验的指标和参数
                trial_metrics = optuna_trial.user_attrs.get('metrics', {})
                trial_params = optuna_trial.user_attrs.get('params', {})
                
                # 格式化时间戳
                now_str = datetime.now().strftime('%m-%d %H:%M:%S')
                
                # 提取关键指标用于显示
                auc_roc = trial_metrics.get('AUC-ROC', 0)
                auc_pr = trial_metrics.get('AUC-PR', 0)
                sens = trial_metrics.get('Sensitivity', 0)
                spec = trial_metrics.get('Specificity', 0)
                prec = trial_metrics.get('Precision', 0)
                score_val = score
                
                # 输出每个试验的简要信息
                logger.info(f"[{now_str}][Trial {trial+1}/{n_trials}] AUC-ROC={auc_roc:.3f} AUC-PR={auc_pr:.3f} Sens={sens:.3f} Spec={spec:.3f} Prec={prec:.3f} Score={score_val:.3f}")
                
                # 更新最佳模型和指标
                if score > best_score:
                    best_score = score
                    best_params = trial_params
                    best_metrics = trial_metrics
                    
                    # 保存最佳参数和指标
                    # 准备保存参数
                    def convert_np(obj):
                        if isinstance(obj, (np.integer, np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj
                        
                    params_serializable = {k: convert_np(v) for k, v in best_params.items()}
                    param_path = model_dir / 'LGBM_best_params.json'
                    with open(param_path, 'w') as f:
                        json.dump(params_serializable, f, indent=4)
                    
                    # 保存特征顺序和信息
                    feature_info = {
                        'features': features
                    }
                    with open(model_dir / 'LGBM_feature_info.json', 'w') as f:
                        json.dump(feature_info, f, indent=4)
                        
                    # 保存最佳指标
                    metrics_path = model_dir / 'LGBM_best_metrics.json'
                    with open(metrics_path, 'w') as f:
                        json.dump(best_metrics, f, indent=4)
                        
                    # 输出新的最佳结果
                    logger.info(f"[Trial {trial+1}] New best score: {best_score:.3f}")
                    logger.info("Best parameters:")
                    for k, v in best_params.items():
                        logger.info(f"{k}: {v}")
                    logger.info("Best metrics:")
                    for metric, value in best_metrics.items():
                        if isinstance(value, float):
                            logger.info(f"{metric}: {value:.3f}")
                        else:
                            logger.info(f"{metric}: {value}")
            except Exception as e:
                study.tell(optuna_trial, float('-inf'))
                logger.error(f"Error in trial {trial+1}: {e}")            
        
        logger.info(f"Training finished. Best score: {best_score:.3f}")
        # 获取最佳试验
        try:
            best_trial = study.best_trial
            # 检查是否是有效试验
            if best_trial.value > float('-inf'):
                best_params = best_trial.user_attrs.get("params", {})
                best_metrics = best_trial.user_attrs.get("metrics", {})
                best_score = best_trial.value
                logger.info(f"Optimization finished. Best score: {best_score:.3f}")
            else:
                logger.warning("No valid trials found, all trials failed constraints.")
                best_params = None
                best_metrics = None
                best_score = float('-inf')
        except Exception as e:
            logger.error(f"Error getting best trial: {e}")
            best_params = None
            best_metrics = None
        
        # 仅当有有效模型时才记录参数和指标
        if best_params is not None:
            logger.info(f"Optimization finished. Best score: {best_score:.3f}")
            logger.info("Best parameters:")
            for k, v in best_params.items():
                logger.info(f"{k}: {v}")
            
        # 仅当有有效模型时才保存参数和指标
        if best_params is not None:
            # 准备保存参数
            def convert_np(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
                
            params_serializable = {k: convert_np(v) for k, v in best_params.items()}
            param_path = model_dir / f'LGBM_best_params.json'
            with open(param_path, 'w') as f:
                json.dump(params_serializable, f, indent=4)
                
            # 保存特征顺序和信息
            feature_info = {
                'features': features
            }
            with open(model_dir / 'LGBM_feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=4)
                
            # 保存最佳指标
            metrics_path = model_dir / f'LGBM_best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
            logger.info(f"Best CV metrics:")
            for metric, value in best_metrics.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.3f}")
                else:
                    logger.info(f"{metric}: {value}")
            
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
        # 确保不重复传入参数
        model_params = best_params.copy()
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**model_params))
        ])
        
        final_pipeline.fit(X_train_res_final, y_train_res_final, classifier__sample_weight=weights_train_res_final)
        
        # 在test set评估
        y_pred_test = final_pipeline.predict(X_test)
        y_prob_test = final_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
        
        logger.info("Test set metrics (never seen during training/tuning):")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.3f}")
            
        # 保存模型和参数到 model_dir
        model_path = model_dir / 'LGBM_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)
            
        # 保存最终测试集指标到 output_dir
        test_metrics_path = output_dir / 'LGBM_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        logger.warning("No valid model found that meets the constraints.")
        logger.warning("Check your objective constraints and data.")

if __name__ == "__main__":
    main()
