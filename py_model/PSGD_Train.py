import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# 确保optuna目录存在
optuna_dir = Path('optuna')
optuna_dir.mkdir(exist_ok=True)

# 确保model目录存在
model_dir = Path('model')
model_dir.mkdir(exist_ok=True)

# 配置日志格式
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器
log_file = model_dir / 'PSGD.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 清空现有的处理器
logger.handlers = []

# 添加处理器到logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 1. 配置读取
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)
plot_dir = Path(config['plot_dir']) if 'plot_dir' in config else Path('plots')
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

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
        raw_constraints = config[constraint_type]
        # 将配置转换为标准格式
        for metric, value in raw_constraints.items():
            if metric.endswith('_min'):
                base_metric = metric[:-4]
                if base_metric not in constraints:
                    constraints[base_metric] = {}
                constraints[base_metric]['min'] = value
            elif metric.endswith('_max'):
                base_metric = metric[:-4]
                if base_metric not in constraints:
                    constraints[base_metric] = {}
                constraints[base_metric]['max'] = value
    
    return constraints

def objective_function(metrics, config_path='config.yaml', constraint_type='objective_constraints', verbose=False):
    """
    计算综合得分，使用config中的硬约束及自定义权重
    
    参数:
    metrics -- 包含各项指标得分的字典
    config_path -- 配置文件路径
    constraint_type -- 约束类型
    verbose -- 是否输出详细日志
    
    返回:
    final_score -- 综合得分
    constraints_met -- 是否满足所有约束条件
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取权重和约束
    weights = config.get('objective_weights', {})
    constraints = load_objective_config(config_path, constraint_type).get('constraints', {})
    
    # 指标名称映射（处理大小写不一致问题）
    metric_mapping = {
        'AUC': 'roc_auc',
        'ECE': 'ece',
        'F1': 'f1',
        'Precision': 'precision',
        'Sensitivity': 'sensitivity',
        'Specificity': 'specificity'
    }
    
    if verbose:
        logger.info("\n===== 目标函数计算 =====")
        logger.info(f"使用的权重: {weights}")
    
    # 计算加权分数
    score = 0.0
    weights_sum = 0.0
    
    for metric, weight in weights.items():
        # 获取标准化的指标名称
        normalized_metric = metric_mapping.get(metric, metric).lower()
        
        # 尝试获取指标值
        metric_value = metrics.get(normalized_metric)
        if metric_value is not None and isinstance(weight, (int, float)):
            score += metric_value * weight
            weights_sum += abs(weight)  # 使用绝对值，因为有些权重可能是负的
            if verbose:
                logger.info(f"指标 {metric}({normalized_metric}): {metric_value:.4f} * {weight:.2f} = {metric_value * weight:.4f}")
    
    # 计算最终得分
    final_score = score / weights_sum if weights_sum > 0 else metrics.get('roc_auc', 0.0)
    
    # 检查是否满足所有约束条件
    constraints_met = True
    for constraint_name, constraint_value in constraints.items():
        if constraint_name in metrics:
            metric_value = metrics[constraint_name]
            if not (constraint_value['min'] <= metric_value <= constraint_value['max']):
                constraints_met = False
                if verbose:
                    logger.warning(f"约束未满足: {constraint_name} = {metric_value:.4f} 不在范围 [{constraint_value['min']}, {constraint_value['max']}] 内")
                break
    
    if verbose:
        logger.info(f"总分: {score:.4f} / 权重和: {weights_sum:.4f} = 最终得分: {final_score:.4f}")
        logger.info(f"约束条件{'全部满足' if constraints_met else '未全部满足'}")
    
    return final_score, constraints_met

# 5. 主函数
def main():
    # 使用已配置好的日志
    logger.info("Starting PSGD (Polynomial SGD) model training...")
    
    # 定义模型保存路径
    model_save_path = model_dir / 'PSGD_model.pkl'
    
    # 定义Optuna数据库路径
    optuna_db_path = optuna_dir / 'PSGD_optuna.db'
    storage = RDBStorage(f'sqlite:///{optuna_db_path}')
    
    # 加载和预处理数据
    logger.info("正在加载和预处理数据...")
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor = load_and_preprocess_data()
    
    # 总是训练新模型
    logger.info("开始训练新模型...")
    
    # 记录现有模型的性能（如果存在）
    existing_model_score = -np.inf
    if model_save_path.exists():
        try:
            with open(model_save_path, 'rb') as f:
                model_data = pickle.load(f)
                if all(key in model_data for key in ['model', 'preprocessor', 'features']):
                    logger.info(f"找到现有模型: {model_save_path}")
                    logger.info(f"最后更新时间: {model_data.get('last_updated', '未知')}")
                    
                    # 评估现有模型
                    existing_model = model_data['model']
                    y_prob = existing_model.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                    metrics = calculate_metrics(y_test, y_pred, y_prob, weights=weights_test)
                    existing_model_score, _ = objective_function(metrics)
                    logger.info(f"现有模型综合得分: {existing_model_score:.4f}")
                    
                    if 'best_params' in model_data:
                        logger.info("现有模型参数:")
                        for k, v in model_data['best_params'].items():
                            logger.info(f"  {k}: {v}")
        except Exception as e:
            logger.warning(f"加载或评估现有模型失败: {e}")
    
    # 定义Optuna目标函数
    def objective(trial):
        # 定义超参数搜索空间 - 优化以提高敏感度
        loss = trial.suggest_categorical('loss', ['log_loss', 'modified_huber'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        
        # 根据penalty参数决定是否使用l1_ratio
        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        else:
            l1_ratio = 0
            
        # 设置learning_rate和power_t
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        power_t = 0.5
        if learning_rate == 'invscaling':
            power_t = trial.suggest_float('power_t', 0.1, 0.5)
        
        # 添加class_weight参数以平衡类别，提高敏感度
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        
        # 提高过采样率，更有利于对极少类标签的检测
        validation_fraction = trial.suggest_float('validation_fraction', 0.05, 0.2)
        
        params = {
            'loss': loss,
            'penalty': penalty,
            'alpha': trial.suggest_float('alpha', 1e-7, 1e-2, log=True),  # 扩大搜索范围
            'l1_ratio': l1_ratio,
            'learning_rate': learning_rate,
            'eta0': trial.suggest_float('eta0', 1e-4, 0.2, log=True),  # 扩大搜索范围
            'power_t': power_t,
            'max_iter': 2000,  # 增加迭代次数
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),  # 动态调整收敛阈值
            'early_stopping': True,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),  # 动态调整早停参数
            'class_weight': class_weight,  # 类别权重
            'random_state': 42
        }
            
        # 创建模型
        model = SGDClassifier(**params)
        
        # 训练模型
        if weights_train is not None:
            sample_weight = weights_train.values
        else:
            sample_weight = None
            
        # 使用交叉验证评估模型
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if weights_train is not None:
                sample_weight_fold = weights_train.iloc[train_idx].values
            else:
                sample_weight_fold = None
                
            model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight_fold)
            
            # 在验证集上评估
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            
            # 计算指标
            metrics = calculate_metrics(y_val_fold, y_pred, y_prob, 
                                     weights=weights_train.iloc[val_idx] if weights_train is not None else None)
            score, _ = objective_function(metrics)
            cv_scores.append(score)
            
        # 返回平均得分
        return np.mean(cv_scores)
        
    # 创建Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='PSGD_hyperparameter_optimization',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5, n_min_trials=5)
    )
        
    # 从配置文件中获取试验次数
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    n_trials = config.get('n_trials', 100)
    
    # 自定义进度条回调
    pbar = tqdm(total=n_trials, desc="超参数优化进度", unit="trial")
    best_score = -np.inf
    
    def progress_callback(study, trial):
        nonlocal best_score
        current_score = study.best_value
        if current_score > best_score:
            best_score = current_score
            pbar.set_postfix({"最佳分数": f"{best_score:.4f}"})
        pbar.update(1)
        
    # 运行优化
    logger.info(f"开始超参数优化，共进行 {n_trials} 次试验...")
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=-1, 
        callbacks=[progress_callback],
        show_progress_bar=False  # 使用自定义进度条
    )
    pbar.close()
    
    # 打印最佳结果
    logger.info(f"\n=== 最佳参数 ===")
    logger.info(f"最佳分数: {study.best_value:.4f}")
    logger.info("最佳参数组合:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
        
    # 使用最佳参数训练最终模型
    best_params = study.best_params
    
    # 确保基本参数存在
    if 'max_iter' not in best_params:
        best_params['max_iter'] = 2000
    if 'early_stopping' not in best_params:
        best_params['early_stopping'] = True
    if 'random_state' not in best_params:
        best_params['random_state'] = 42
    
    # 添加类别权重以提高敏感度
    if 'class_weight' not in best_params:
        best_params['class_weight'] = 'balanced'
        
    # 训练基础模型
    base_model = SGDClassifier(**best_params)
    
    # 训练基础模型
    if weights_train is not None:
        sample_weight = weights_train.values
    else:
        sample_weight = None
    
    logger.info("训练基础SGD模型...")
    base_model.fit(X_train, y_train, sample_weight=sample_weight)
    
    # 对模型进行概率校准
    logger.info("对模型进行概率校准...")
    # 使用isotonic回归和sigmoid校准分别进行试验
    calibrated_model_sigmoid = CalibratedClassifierCV(
        base_model, 
        method='sigmoid',
        cv='prefit'  # 使用预先训练好的模型
    )
    calibrated_model_isotonic = CalibratedClassifierCV(
        base_model, 
        method='isotonic',
        cv='prefit'  # 使用预先训练好的模型
    )
    
    # 对两种校准模型进行训练
    calibrated_model_sigmoid.fit(X_test, y_test, sample_weight=weights_test.values if weights_test is not None else None)
    calibrated_model_isotonic.fit(X_test, y_test, sample_weight=weights_test.values if weights_test is not None else None)
    
    # 评估三种模型
    logger.info("评估基础模型和两种校准模型...")
    
    # 基础模型评估
    y_pred_base = base_model.predict(X_test)
    y_prob_base = base_model.predict_proba(X_test)[:, 1]
    metrics_base = calculate_metrics(
        y_test, y_pred_base, y_prob_base, 
        weights=weights_test if weights_test is not None else None
    )
    base_score, _ = objective_function(metrics_base)
    
    # Sigmoid校准模型评估
    y_pred_sigmoid = calibrated_model_sigmoid.predict(X_test)
    y_prob_sigmoid = calibrated_model_sigmoid.predict_proba(X_test)[:, 1]
    metrics_sigmoid = calculate_metrics(
        y_test, y_pred_sigmoid, y_prob_sigmoid, 
        weights=weights_test if weights_test is not None else None
    )
    sigmoid_score, _ = objective_function(metrics_sigmoid)
    
    # Isotonic校准模型评估
    y_pred_isotonic = calibrated_model_isotonic.predict(X_test)
    y_prob_isotonic = calibrated_model_isotonic.predict_proba(X_test)[:, 1]
    metrics_isotonic = calculate_metrics(
        y_test, y_pred_isotonic, y_prob_isotonic, 
        weights=weights_test if weights_test is not None else None
    )
    isotonic_score, _ = objective_function(metrics_isotonic)
    
    # 打印比较结果
    logger.info(f"基础模型得分: {base_score:.4f}")
    logger.info(f"Sigmoid校准模型得分: {sigmoid_score:.4f}")
    logger.info(f"Isotonic校准模型得分: {isotonic_score:.4f}")
    
    # 强制选择等温回归校准模型
    logger.info("强制选择Isotonic校准模型")
    final_model = calibrated_model_isotonic
    final_metrics = metrics_isotonic
    model_type = "Isotonic校准模型"
    
    # 将最终选择的模型结果保存
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    best_metrics = final_metrics
        
    # 保存模型
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'preprocessor': preprocessor,
            'features': features,
            'best_params': best_params,
            'model_type': model_type,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)
        
    # 计算综合得分
    current_score, _ = objective_function(best_metrics)
        
    # 评估新模型
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_prob, weights=weights_test)
    new_model_score, _ = objective_function(metrics)
        
    # 比较新旧模型性能
    if new_model_score > existing_model_score:
        # 保存新模型
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'preprocessor': preprocessor,
                'features': features,
                'best_params': best_params,
                'model_type': model_type,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        metrics = calculate_metrics(y_test, final_model.predict(X_test), final_model.predict_proba(X_test)[:, 1], weights=weights_test)
        new_model_score, _ = objective_function(metrics)
        
        logger.info(f"新模型训练完成，综合得分: {new_model_score:.4f}")
        logger.info(f"模型已保存至: {model_save_path}")
    else:
        logger.info(f"新模型得分 ({new_model_score:.4f}) 未超过现有模型得分 ({existing_model_score:.4f})，保留原模型")
    
    logger.info("模型训练/加载完成")

if __name__ == "__main__":
    main()
