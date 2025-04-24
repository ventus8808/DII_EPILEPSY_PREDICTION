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
        missing_idx = X_raw[X_raw.isnull().any(axis=1)].index
        # 删除缺失行
        X_raw = X_raw.drop(missing_idx)
        y = y.drop(missing_idx)
        if weights is not None:
            weights = weights.drop(missing_idx)
        logger.info(f"删除了 {len(missing_idx)} 行，剩余 {len(X_raw)} 行")
    
    # 独热编码处理类别特征
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_raw[categorical_features])
    
    # 保存编码器
    with open(model_dir / 'encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    # 分别获取数值特征和类别特征
    numeric_data = X_raw[['DII_food'] + numeric_features].copy()
    categorical_data = X_raw[categorical_features].copy()
    
    # 应用独热编码
    encoded_cats = encoder.transform(categorical_data)
    
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
        
        # 获取SVM参数
        params = {
            'C': get_suggested_param(trial, 'C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.params.get('kernel', 'rbf') in ['rbf', 'poly', 'sigmoid'] else 'scale',
            'probability': True,  # 需要概率输出以计算AUC等指标
            'class_weight': 'balanced',
            'random_state': 42,
        }
        
        # 如果选择了'poly'核函数，则需要选择degree参数
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            
        # 交叉验证设置
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 存储每个折叠的指标
        fold_metrics = []
        
        # 训练函数
        def train_fold(fold, train_idx, val_idx):
            # 获取当前折叠的训练和验证数据
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            weights_fold_train = weights_train.iloc[train_idx] if weights_train is not None else None
            weights_fold_val = weights_train.iloc[val_idx] if weights_train is not None else None
            
            # 用SMOTE处理类别不平衡
            smote = SMOTE(random_state=42)
            if weights_fold_train is not None:
                # 带权重的SMOTE处理需要保留权重
                Xy_train = X_fold_train.copy()
                Xy_train['__label__'] = y_fold_train
                Xy_train['__weight__'] = weights_fold_train
                X_res, y_res = smote.fit_resample(Xy_train.drop(['__label__'], axis=1), Xy_train['__label__'])
                weights_fold_train_res = X_res['__weight__'].reset_index(drop=True)
                X_fold_train_res = X_res.drop(['__weight__'], axis=1)
                y_fold_train_res = y_res.reset_index(drop=True)
            else:
                X_fold_train_res, y_fold_train_res = smote.fit_resample(X_fold_train, y_fold_train)
                weights_fold_train_res = None
            
            # 标准化处理
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train_res)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            # 训练SVM模型
            model = SVC(**params)
            model.fit(X_fold_train_scaled, y_fold_train_res, sample_weight=weights_fold_train_res)
            
            # 预测
            y_pred = model.predict(X_fold_val_scaled)
            y_prob = model.predict_proba(X_fold_val_scaled)[:, 1]
            
            # 计算指标
            metrics = calculate_metrics(y_fold_val, y_pred, y_prob, weights_fold_val)
            return metrics
        
        try:
            # 执行交叉验证
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.info(f"[Trial {trial.number+1}] 训练折叠 {fold+1}/{n_splits}...")
                fold_metric = train_fold(fold, train_idx, val_idx)
                logger.info(f"[Trial {trial.number+1}] 折叠 {fold+1} 指标: AUC = {fold_metric['AUC-ROC']:.4f}, F1 = {fold_metric['F1 Score']:.4f}")
                fold_metrics.append(fold_metric)
            
            # 计算平均指标
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics]
                avg_metrics[key] = sum(values) / len(values)
            
            # 计算目标分数
            score = objective_function(avg_metrics)
            logger.info(f"[Trial {trial.number+1}] 平均指标: AUC = {avg_metrics['AUC-ROC']:.4f}, F1 = {avg_metrics['F1 Score']:.4f}, 分数 = {score:.4f}")
            
            return score
        except Exception as e:
            logger.error(f"[Trial {trial.number+1}] 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return float('-inf')
    
    # 优化超参数
    n_trials = config.get('n_trials', 10)
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    
    # 运行Optuna优化
    logger.info(f"开始Optuna优化 ({n_trials} 次试验)...")
    for trial in range(n_trials):
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(optuna_objective, n_trials=1)
            
            # 获取当前最佳结果
            if study.best_value > best_score:
                best_score = study.best_value
                best_params = study.best_params
                
                # 计算最佳参数的指标
                params = {
                    'C': best_params.get('C', 1.0),
                    'kernel': best_params.get('kernel', 'rbf'),
                    'gamma': best_params.get('gamma', 'scale'),
                    'probability': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                }
                
                if params['kernel'] == 'poly':
                    params['degree'] = best_params.get('degree', 3)
                
                # 保存最佳参数
                logger.info(f"[Trial {trial+1}] 更新最佳参数...")
                
                # 转换参数以便序列化
                params_serializable = {}
                for k, v in params.items():
                    if isinstance(v, np.ndarray):
                        params_serializable[k] = v.tolist()
                    else:
                        params_serializable[k] = v
                
                # 保存最佳参数
                param_path = model_dir / f'SVM_best_params.json'
                with open(param_path, 'w') as f:
                    json.dump(params_serializable, f, indent=4)
                
                # 保存特征信息
                feature_info = {
                    'features': features,
                    'categorical_features': categorical_features,
                    'numeric_features': numeric_features
                }
                with open(model_dir / 'SVM_feature_info.json', 'w') as f:
                    json.dump(feature_info, f, indent=4)
                
                logger.info(f"[Trial {trial+1}] 新的最佳分数: {best_score}")
                logger.info("最佳参数:")
                for k, v in best_params.items():
                    logger.info(f"{k}: {v}")
        except Exception as e:
            logger.error(f"[Trial {trial+1}] 错误: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info(f"训练完成。最佳分数: {best_score}")
    
    if best_params is not None:
        logger.info("最终最佳参数:")
        for k, v in best_params.items():
            logger.info(f"{k}: {v}")
        
        # 用最终最优参数在训练集做采样训练模型，并在test set评估
        logger.info("在保留测试集上评估(训练/调优期间从未见过)...")
        
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
        
        # 标准化处理
        scaler_final = StandardScaler()
        X_train_scaled = scaler_final.fit_transform(X_train_res_final)
        X_test_scaled = scaler_final.transform(X_test)
        
        # 用最优参数训练模型
        params = {
            'C': best_params.get('C', 1.0),
            'kernel': best_params.get('kernel', 'rbf'),
            'gamma': best_params.get('gamma', 'scale'),
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42,
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = best_params.get('degree', 3)
        
        final_model = SVC(**params)
        final_model.fit(X_train_scaled, y_train_res_final, sample_weight=weights_train_res_final)
        
        # 在test set评估
        y_prob_test = final_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = final_model.predict(X_test_scaled)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
        
        logger.info("测试集指标 (训练/调优期间从未见过):")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 保存模型、标准化器和参数
        # 保存模型
        with open(model_dir / 'SVM_best_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        
        # 保存标准化器
        with open(model_dir / 'SVM_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_final, f)
        
        # 保存最佳参数
        best_param_path = model_dir / 'SVM_best_params.json'
        with open(best_param_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        # 保存最终测试集指标
        test_metrics_path = output_dir / 'SVM_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        logger.warning("未找到满足约束条件的有效模型。")
        logger.warning("请检查您的目标约束和数据。")

if __name__ == "__main__":
    main()
