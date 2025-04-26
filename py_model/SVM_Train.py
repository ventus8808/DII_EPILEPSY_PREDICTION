import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import optuna
import traceback
import signal
import time
import sys
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

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
    """加载并预处理数据"""
    logger = logging.getLogger()
    logger.info("加载数据...")
    df = pd.read_csv(data_path)
    
    # 准备特征和目标变量
    X_raw = df[features].copy()
    y = df['Epilepsy']
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
    # 检查缺失值
    missing = X_raw.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"数据中存在缺失值：\n{missing[missing > 0]}")
        logger.info("删除含有缺失值的行...")
        # 获取缺失行的索引
        missing_indices = set()
        for col in X_raw.columns:
            missing_indices.update(X_raw[X_raw[col].isnull()].index.tolist())
        
        # 删除缺失行
        X_raw = X_raw.drop(index=list(missing_indices))
        y = y.drop(index=list(missing_indices))
        if weights is not None:
            weights = weights.drop(index=list(missing_indices))
        
        logger.info(f"删除了 {len(missing_indices)} 行含缺失值的数据")
    
    # 分离数值和类别特征
    numeric_data = X_raw[['DII_food'] + numeric_features].copy()
    categorical_data = X_raw[categorical_features].copy()
    
    # 独热编码类别特征
    # 兼容旧版scikit-learn
    try:
        # 新版sklearn使用sparse_output
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        # 旧版sklearn不支持sparse_output参数
        encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(categorical_data)
    # 处理旧版scikit-learn返回稀疏矩阵的情况
    if hasattr(encoded_data, 'toarray'):
        encoded_data = encoded_data.toarray()
    
    # 创建编码后的DataFrame
    feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X_raw.index)
    
    # 合并数值特征和独热编码后的特征
    X = pd.concat([numeric_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    # 检查处理后的数据是否有缺失值
    if X.isnull().sum().sum() > 0:
        missing_encoded = X.isnull().sum()
        logger.warning(f"编码后数据中存在缺失值：\n{missing_encoded[missing_encoded > 0]}")
    
    logger.info(f"特征处理完成。特征数量：{X.shape[1]}")
    logger.info(f"目标变量分布：\n{y.value_counts()}")
    
    return X, y, weights, encoder

# 4. 目标函数配置和指标
import yaml

def load_objective_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('objective_weights', {}), config.get('objective_constraints', {})

def objective_function(metrics, config_path='config.yaml'):
    weights, constraints = load_objective_config(config_path)
    score = 0
    for metric, value in metrics.items():
        if metric in weights:
            score += weights[metric] * value
    
    # 临时修改: SVM模型先不强制约束条件，而是添加惩罚项
    failed_constraints = []
    for metric, value in metrics.items():
        if f"{metric}_min" in constraints and value < constraints[f"{metric}_min"]:
            # 添加惩罚，而不是直接返回-inf
            diff = constraints[f"{metric}_min"] - value
            score -= diff * 10  # 根据差距添加惩罚
            failed_constraints.append(f"{metric} ({value:.4f} < {constraints[f'{metric}_min']:.4f})")
        if f"{metric}_max" in constraints and value > constraints[f"{metric}_max"]:
            diff = value - constraints[f"{metric}_max"]
            score -= diff * 10
            failed_constraints.append(f"{metric} ({value:.4f} > {constraints[f'{metric}_max']:.4f})")
    
    if failed_constraints:
        logging.getLogger().debug(f"未满足约束: {', '.join(failed_constraints)}, 分数调整到: {score:.4f}")
    return score

# 5. 主流程
def main():
    # 设置日志
    logger = setup_logger('SVM')
    logger.info("开始SVM模型训练...")
    logger.info(f"特征列表: {features}")
    logger.info(f"类别特征: {categorical_features}")
    
    # 加载数据
    X, y, weights, encoder = load_and_preprocess_data()
    
    # 训练测试分割
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
    
    # 加载配置中的试验次数
    n_trials = config.get('n_trials', 10)  # 默认10次
    logger.info(f"将执行 {n_trials} 次试验")

    # 定义Optuna目标函数
    def optuna_objective(trial):
        # 辅助函数：获取建议的参数
        def get_suggested_param(trial, name, default_low, default_high, best=None, pct=0.2, is_int=False, log=False):
            if best is not None:
                low = best * (1 - pct) if best > 0 else best * (1 + pct)
                high = best * (1 + pct) if best > 0 else best * (1 - pct)
                low, high = min(low, high), max(low, high)
            else:
                low, high = default_low, default_high
            
            if is_int:
                return trial.suggest_int(name, int(low), int(high), log=log)
            else:
                return trial.suggest_float(name, low, high, log=log)
        
        try:
            # 设置超时处理器
            import signal
            from contextlib import contextmanager
            import time
            
            class TimeoutException(Exception): pass
            
            @contextmanager
            def time_limit(seconds):
                def signal_handler(signum, frame):
                    raise TimeoutException(f"操作超时 ({seconds}秒)")
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
            
            # 整个试验设置时间限制
            trial_start_time = time.time()
            trial_max_time = 600  # 10分钟超时
            
            # 初始化交叉验证
            n_splits = 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # 确定试验的超参数
            if best_params is None:
                # 第一次试验，基于提高敏感度来调整参数空间
                params = {
                    # 增大C的范围，大的C值可能有助于提高敏感度
                    'C': trial.suggest_float('C', 0.5, 50.0, log=True),
                    # 添加多项式核，可能更适合不平衡数据
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    # 探索更广的gamma值
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    # 对于多项式核，添加阶数参数
                    'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel', None) == 'poly' else 3,
                    # 关键参数：用于不平衡数据的类别权重
                    'class_weight': trial.suggest_categorical('class_weight', 
                                                            ['balanced', {0:1, 1:10}, {0:1, 1:20}, {0:1, 1:50}]),
                    'probability': True,
                    'max_iter': 1500,  # 增加最大迭代次数
                    'tol': 1e-3,      # 适当的容差
                    'random_state': 42
                }
            else:
                # 使用前一次最佳参数附近搜索，但扩大搜索范围
                params = {
                    # 扩大之前最佳参数周围的搜索范围
                    'C': get_suggested_param(trial, 'C', 0.1, 100.0, best_params.get('C'), 0.5, log=True),
                    # 尝试更多类型的核函数
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel', None) == 'poly' else 3,
                    # 类别权重探索，更加强调对少数类的惩罚
                    'class_weight': trial.suggest_categorical('class_weight', 
                                                          ['balanced', {0:1, 1:10}, {0:1, 1:20}, {0:1, 1:50}]),
                    'probability': True,
                    'max_iter': 1500,
                    'tol': 1e-3,
                    'random_state': 42
                }    
            
            # 训练函数 - 处理一个交叉验证折叠
            def train_fold(fold, train_idx, val_idx):
                # 获取当前折叠的训练和验证数据
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                weights_fold_train = weights_train.iloc[train_idx] if weights_train is not None else None
                weights_fold_val = weights_train.iloc[val_idx] if weights_train is not None else None
                
                # 控制样本量，防止SMOTE生成过多样本
                pos_count = (y_fold_train == 1).sum()
                neg_count = (y_fold_train == 0).sum()

                # 针对提高敏感度的采样策略
                logger.debug(f"[Trial {getattr(trial, 'number', '?')}][折叠 {fold+1}/5] 采样策略: 0.15 (正样本数量: {pos_count}, 负样本数量: {neg_count}, 比例: 1:{neg_count/pos_count:.1f})")
                
                try:
                    # 设置超时保护
                    with time_limit(60):  # 给SMOTE操作设置60秒超时
                        # 使用灵活的SMOTE采样策略，改进对少数类的处理
                        smote_fold = SMOTE(random_state=42, sampling_strategy=0.15, k_neighbors=5)
                        
                        # 处理加权样本的SMOTE
                        if weights_fold_train is not None:
                            # 添加权重列进行SMOTE处理
                            Xy_tr = X_fold_train.copy()
                            Xy_tr['__label__'] = y_fold_train
                            Xy_tr['__weight__'] = weights_fold_train
                            
                            # 进行SMOTE平衡处理
                            X_res_fold, y_res_fold = smote_fold.fit_resample(
                                Xy_tr.drop(['__label__'], axis=1),
                                Xy_tr['__label__']
                            )
                            
                            # 提取权重和特征
                            w_tr_res = X_res_fold['__weight__'].reset_index(drop=True)
                            X_tr_res = X_res_fold.drop(['__weight__'], axis=1)
                            y_tr_res = y_res_fold.reset_index(drop=True)
                        else:
                            # 无权重情况的SMOTE处理
                            X_tr_res, y_tr_res = smote_fold.fit_resample(X_fold_train, y_fold_train)
                            w_tr_res = None
                            
                        # 重命名变量与内部保持一致
                        X_fold_train_res, y_fold_train_res = X_tr_res, y_tr_res
                        weights_fold_train_res = w_tr_res
                except (TimeoutException, Exception) as e:
                    logger.warning(f"[Trial {getattr(trial, 'number', '?')}][折叠 {fold+1}/5] SMOTE处理超时或失败: {str(e)}，使用原始数据...")
                    X_fold_train_res, y_fold_train_res = X_fold_train, y_fold_train
                    weights_fold_train_res = weights_fold_train
                
                # 标准化处理
                scaler = StandardScaler()
                X_fold_train_scaled = scaler.fit_transform(X_fold_train_res)
                X_fold_val_scaled = scaler.transform(X_fold_val)
                
                try:
                    # 训练SVM模型（带超时保护）
                    with time_limit(120):  # 给SVM训练设置120秒超时
                        model = SVC(**params)
                        model.fit(X_fold_train_scaled, y_fold_train_res, sample_weight=weights_fold_train_res)
                    
                        # 预测
                        y_pred = model.predict(X_fold_val_scaled)
                        y_prob = model.predict_proba(X_fold_val_scaled)[:, 1]
                    
                        # 计算指标
                        metrics = calculate_metrics(y_fold_val, y_pred, y_prob, weights_fold_val)
                        return metrics
                except (TimeoutException, Exception) as e:
                    logger.warning(f"[Trial {getattr(trial, 'number', '?')}][折叠 {fold+1}/5] SVM训练或评估超时/失败: {str(e)}")
                    # 返回一组较差的评估指标，使得该组参数不会被选中
                    return {
                        "Accuracy": 0.5,
                        "Sensitivity": 0.0,
                        "Specificity": 1.0,
                        "Precision": 0.0,
                        "NPV": 0.0,
                        "F1 Score": 0.0,
                        "Youden's Index": 0.0,
                        "Cohen's Kappa": 0.0,
                        "AUC-ROC": 0.5,
                        "AUC-PR": 0.0,
                        "Log Loss": 1.0,
                        "Brier": 0.25,
                        "ECE": 0.5,
                        "MCE": 0.5
                    }
            
            # 执行交叉验证，使用tqdm显示进度条
            fold_metrics = []
            folds = list(skf.split(X_train, y_train))
            for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc=f"[Trial {trial.number+1}] 交叉验证", leave=False)):
                # 检查试验是否已经超时
                if time.time() - trial_start_time > trial_max_time:
                    logger.warning(f"[Trial {trial.number+1}] 达到最大时间限制，终止试验")
                    raise TimeoutException("试验总时间超过限制")
                
                # 不再显示每个折叠的进度信息，由tqdm进度条显示
                fold_metric = train_fold(fold, train_idx, val_idx)
                fold_metrics.append(fold_metric)
            
            # 检查是否所有折叠都完成了
            if len(fold_metrics) < n_splits:
                logger.warning(f"[Trial {trial.number+1}] 只完成了 {len(fold_metrics)}/{n_splits} 个折叠，使用已有数据计算平均指标")
                
            # 防止fold_metrics为空的情况
            if not fold_metrics:
                logger.error(f"[Trial {trial.number+1}] 没有任何折叠完成评估，返回失败分数")
                return float('-inf')
                
            # 计算指标
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics]
                avg_metrics[key] = sum(values) / len(values)
            
            # 计算目标分数
            score = objective_function(avg_metrics)
            logger.info(f"[Trial {getattr(trial, 'number', '?')}] 平均指标: AUC = {avg_metrics['AUC-ROC']:.4f}, F1 = {avg_metrics['F1 Score']:.4f}, 分数 = {score:.4f}")
            
            # 取消超时报警
            signal.alarm(0)
            
            return score
        except Exception as e:
            logger.error(f"[Trial {trial.number+1}] 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return float('-inf')
    
    # Optuna优化搜索
    # 定义最佳模型相关变量
    best_score = float('-inf')
    best_params = None
    logger.info(f"开始Optuna优化 ({n_trials} 次试验)...")
    for trial in tqdm(range(n_trials), desc="超参数优化试验"):
        try:
            # 查找可能的最佳超参数
            if trial == 0:
                # 第一次试验，使用默认参数
                best_params = None
                logger.info(f"[Trial {trial+1}] 使用默认参数")
            else:
                # 根据前面的试验进行参数推荐
                logger.info(f"[Trial {trial+1}] 开始参数优化")
            
            # 设置全局超时保护
            def timeout_handler(signum, frame):
                raise Exception(f"[Trial {trial+1}] 全局超时保护触发")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10分钟全局超时保护
            
            study = optuna.create_study(direction='maximize')
            study.optimize(optuna_objective, n_trials=1)
            
            # 获取最佳结果
            trial_score = study.best_value
            trial_params = study.best_params
            
            # 更新全局最佳
            if trial_score > best_score:
                logger.info(f"[Trial {trial+1}] 更新最佳参数...")
                best_score = trial_score
                best_params = {
                    'C': trial_params.get('C', 1.0),
                    'kernel': trial_params.get('kernel', 'rbf'),
                    'gamma': trial_params.get('gamma', 'scale'),
                }
                
                logger.info(f"[Trial {trial+1}] 新的最佳分数: {best_score}")
                logger.info("最佳参数:")
                for k, v in best_params.items():
                    logger.info(f"{k}: {v}")
            signal.alarm(0)  # 确保取消任何超时报警
        except Exception as e:
            logger.error(f"[Trial {trial+1}] 错误: {str(e)}")
            logger.error(traceback.format_exc())
            signal.alarm(0)  # 确保取消任何超时报警
    
    logger.info(f"训练完成。最佳分数: {best_score}")
    
    if best_params is not None:
        logger.info("最终最佳参数:")
        for k, v in best_params.items():
            logger.info(f"{k}: {v}")
        
        # 用最终最优参数在训练集做采样训练模型，并在test set评估
        logger.info("在保留测试集上评估(训练/调优期间从未见过)...")
        
        # 重新采样整个训练集
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        
        # 采用更平衡的SMOTE策略来提高敏感度
        logger.info(f"最终模型SMOTE采样策略 (正样本数量: {pos_count}, 负样本数量: {neg_count}, 比例: 1:{neg_count/pos_count:.1f})")
        
        try:
            # 提高采样策略到0.2，增加正样本比例，同时保持效率
            # 这个值比之前的0.1更高，有助于提高敏感度而不致于数据量过大
            sampling_strategy = 0.2
            logger.info(f"使用提高的采样策略: {sampling_strategy}")
            
            smote_final = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=5)
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
                
            logger.info(f"SMOTE后样本数: {len(y_train_res_final)}, 正样本数量: {(y_train_res_final == 1).sum()}, 负样本数量: {(y_train_res_final == 0).sum()}, 正样本比例: {y_train_res_final.mean():.4f}")
        except Exception as e:
            logger.warning(f"最终SMOTE处理失败: {str(e)}，使用原始数据")
            X_train_res_final, y_train_res_final = X_train, y_train
            weights_train_res_final = weights_train
        
        # 标准化处理
        scaler_final = StandardScaler()
        X_train_scaled = scaler_final.fit_transform(X_train_res_final)
        X_test_scaled = scaler_final.transform(X_test)
        
        # 用最优参数训练模型，并进行调整以提高敏感度
        params = {
            'C': best_params.get('C', 5.0),  # 使用较大的默认值来提高敏感度
            'kernel': best_params.get('kernel', 'rbf'),
            'gamma': best_params.get('gamma', 'scale'),
            'probability': True,
            # 手动调整类别权重，增加对少数类的重视
            'class_weight': best_params.get('class_weight', {0:1, 1:20}),
            'random_state': 42,
            # 添加最大迭代次数限制
            'max_iter': 1500,
            # 设置允许的收敛误差
            'tol': 1e-3,
            # 限制内存使用
            'cache_size': 200
        }
        
        # 对于多项式和余弦核添加特殊参数
        if params['kernel'] == 'poly':
            params['degree'] = best_params.get('degree', 3)
        if params['kernel'] == 'sigmoid':
            params['coef0'] = 0.0
        
        logger.info("开始训练最终SVM模型...")
        logger.info("注意: 如果训练时间超过5分钟，将自动终止")
        
        # 设置超时保护
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5分钟超时
        
        try:
            final_model = SVC(**params)
            
            # 显示一个大概的运行进度
            logger.info("正在训练最终模型，这可能需要几分钟时间...")
            start_time = time.time()
            
            final_model.fit(X_train_scaled, y_train_res_final, sample_weight=weights_train_res_final)
            
            end_time = time.time()
            logger.info(f"训练完成，用时 {end_time - start_time:.2f} 秒")
        except Exception as e:
            logger.warning(f"训练超时或出错: {str(e)}")
            logger.warning("将使用简化模型代替...")
            # 试用线性核心，速度更快
            params['kernel'] = 'linear'
            final_model = SVC(**params)
            final_model.fit(X_train_scaled, y_train_res_final, sample_weight=weights_train_res_final)
        finally:
            # 确保无论如何都要取消超时警报
            signal.alarm(0)
        
        logger.info("训练完成，进行测试集预测...")
        # 在test set评估
        y_prob_test = final_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = final_model.predict(X_test_scaled)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
        
        # 打印水平线分隔
        logger.info("─" * 60)
        logger.info("【最终模型参数】")
        for param_name, param_value in params.items():
            if param_name in ['C', 'kernel', 'gamma']:
                logger.info(f"  {param_name}: {param_value}")
        
        logger.info("─" * 60)
        logger.info("【最佳参数在测试集上的性能指标】")
        
        # 关键指标显示在前面
        priority_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'AUC-ROC']
        for metric in priority_metrics:
            logger.info(f"* {metric}: {test_metrics[metric]:.4f}")
        
        # 其他指标
        logger.info("─" * 30)
        logger.info("其他指标:")
        for metric, value in test_metrics.items():
            if metric not in priority_metrics:
                logger.info(f"  {metric}: {value:.4f}")
        logger.info("─" * 60)
        
        # 只保存模型和最佳参数
        # 保存模型
        model_path = model_dir / 'SVM_best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"模型已保存到 {model_path}")
        
        # 保存最佳参数
        param_path = model_dir / 'SVM_best_params.json'
        with open(param_path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"最佳参数已保存到 {param_path}")
        
        # 如果需要也可以保存标准化器供后续使用
        with open(model_dir / 'SVM_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_final, f)
        
        logger.info("运行完成!")
    else:
        logger.warning("未找到满足约束条件的有效模型。")
        logger.warning("请检查您的目标约束和数据。")

def reset_signals():
    """重置所有信号处理，确保程序能够正常退出"""
    # 恢复SIGALRM的默认行为
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    # 确保没有活跃的警报
    signal.alarm(0)

if __name__ == "__main__":
    try:
        main()
    finally:
        # 确保在程序退出前重置所有信号处理
        reset_signals()
        logger = logging.getLogger()
        logger.info("程序执行完毕，所有资源已释放")
