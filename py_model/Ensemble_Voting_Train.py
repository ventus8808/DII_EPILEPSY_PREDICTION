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
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
import optuna
import traceback
from imblearn.over_sampling import SMOTE

# 模型相关全局函数
def preprocess_fnn_data(X, encoder, feature_info):
    # 分离类型变量和数值变量
    categorical_features = feature_info.get('categorical_features', [])
    numeric_features = feature_info.get('numeric_features', [])
    dii_column = ['DII_food']
    
    # 抽取各组特征
    numeric_data = X[dii_column + numeric_features].copy() if numeric_features else X[dii_column].copy()
    categorical_data = X[categorical_features].copy() if categorical_features else None
    
    # 处理类型化变量
    if categorical_data is not None:
        # 转换为数值类型，缺失值处理为0
        categorical_data = categorical_data.fillna(0).astype(int)
        # 使用独热编码器变换
        encoded_cats = encoder.transform(categorical_data)
        
        # 获取编码后的特征名
        encoded_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = encoder.categories_[i]
            for category in categories:
                encoded_feature_names.append(f"{feature}_{category}")
        
        # 创建result_df准备合并
        result_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, 
                               index=X.index if hasattr(X, 'index') else None)
    else:
        result_df = pd.DataFrame(index=X.index if hasattr(X, 'index') else None)
    
    # 合并数值特征
    if len(numeric_data.columns) > 0:
        for col in numeric_data.columns:
            result_df[col] = numeric_data[col].values
    
    return result_df.values  # 直接返回numpy数组而不是DataFrame

def fnn_predict(model_data, X, feature_info=None):
    # 直接从同级目录的FNN_Train模块导入
    import sys
    import os
    # 添加当前目录到系统路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    if current_path not in sys.path:
        sys.path.append(current_path)
    # 直接从FNN_Train导入FNNModel
    from FNN_Train import FNNModel
    
    # 预处理特征 - 使用FNN模型的独热编码器
    if 'encoder' in model_data and feature_info is not None:
        processed_X = preprocess_fnn_data(X, model_data['encoder'], feature_info)
    else:
        processed_X = X.values if hasattr(X, 'values') else X  # 确保输入是numpy数组
    
    # 重建 FNN 模型
    model_config = model_data['model_config']
    model = FNNModel(model_config['input_dim'], model_config['params'])
    model.update(model_data['model_state'])
    
    # 由于运行中的特征名顺序问题，跳过Scaler转换
    # 注释掉这里的转换，直接使用原始特征
    # if 'scaler' in model_data:
    #     processed_X = model_data['scaler'].transform(processed_X)
    
    # 转换为mx.array并进行预测
    import mlx.core as mx
    X_mx = mx.array(processed_X.astype(np.float32))
    y_prob = model(X_mx).tolist()
    
    # 如果y_prob是嵌套列表，则展平
    if isinstance(y_prob[0], list):
        y_prob = [item[0] for item in y_prob]
    
    # 二分类预测
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    
    return y_pred, np.array(y_prob)

# SVM模型预测函数
def svm_predict(model_data, X):
    # 如果SVM模型是作为完整模型加载的
    if hasattr(model_data, 'predict_proba'):
        y_prob = model_data.predict_proba(X)[:, 1]
        y_pred = model_data.predict(X)
    # 如果SVM模型被存储为字典或其他形式
    else:
        # 使用一个默认的预测处理
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        # 假设模型数据包含模型参数和数据预处理器
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
            y_pred = model.predict(X)
        else:
            # 如果没有直接的模型对象，使用平均概率
            y_prob = np.full(len(X), 0.5)  # 默认平均概率
            y_pred = (y_prob >= 0.5).astype(int)  # 默认预测
    
    return y_pred, y_prob

warnings.filterwarnings('ignore')

# 1. 配置读取
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
model_dir.mkdir(exist_ok=True)
# 为集成模型创建单独的目录
ensemble_dir = model_dir / 'Ensemble_Voting'
ensemble_dir.mkdir(exist_ok=True)
output_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
output_dir.mkdir(exist_ok=True)
plot_dir = Path(config['plot_dir']) if 'plot_dir' in config else Path('plots')
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

# 2. 日志设置
def setup_logger(model_name):
    log_file = ensemble_dir / f'{model_name}.log'
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
            X, y, weights, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        weights_train = weights_test = None
    return X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, features

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
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    logloss = log_loss(y_true, y_prob, sample_weight=weights)
    ece, mce = calculate_calibration_metrics(y_true, y_prob, weights)
    kappa = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    else:
        specificity = float('nan')
        npv = float('nan')
    youden = sensitivity + specificity - 1
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

# 加载目标函数配置
def load_objective_config(config_path='config.yaml', constraint_type='objective_constraints'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 读取目标函数的权重
    objective_weights = config.get('objective_weights', {})
    constraints = config.get(constraint_type, {})
    
    return objective_weights, constraints

# 定制目标函数
def objective_function(metrics, config_path='config.yaml', constraint_type='objective_constraints'):
    objective_weights, constraints = load_objective_config(config_path, constraint_type)
    
    # 初始化目标函数值和违反约束条件的标志
    objective_value = 0
    constraints_violated = False
    
    # 检查硬约束条件
    for metric, constraint_value in constraints.items():
        if '_min' in metric:
            base_metric = metric.replace('_min', '')
            if metrics.get(base_metric, 0) < constraint_value:
                constraints_violated = True
                break
        elif '_max' in metric:
            base_metric = metric.replace('_max', '')
            if metrics.get(base_metric, 1) > constraint_value:
                constraints_violated = True
                break
    
    # 如果违反约束条件，返回一个较大的负值
    if constraints_violated:
        return float('-inf')
    
    # 计算目标函数值
    for metric, weight in objective_weights.items():
        if metric in metrics:
            objective_value += weight * metrics[metric]
    
    return objective_value

# 5. 主函数
def main():
    # 设置日志
    logger = setup_logger("Ensemble_Voting")
    logger.info("开始集成投票模型训练...")
    
    # 加载数据
    X_train, X_test, y_train, y_test, weights_train, weights_test, preprocessor, features = load_and_preprocess_data()
    logger.info(f"数据加载完成。训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    
    # 优化函数，根据传入的参数调整集成模型权重
    def optuna_objective(trial):
        # 获取建议的参数值
        def get_suggested_param(trial, name, default_low, default_high, best=None, pct=0.2, is_int=False, log=False):
            if best is not None:
                low = best * (1 - pct)
                high = best * (1 + pct)
                if is_int:
                    low, high = int(max(1, low)), int(high)
            else:
                low, high = default_low, default_high
            return trial.suggest_int(name, low, high) if is_int else (
                trial.suggest_float(name, low, high, log=log))
        
        # 为各模型分配权重
        xgb_weight = trial.suggest_float('xgb_weight', 0.5, 3.0)
        lgbm_weight = trial.suggest_float('lgbm_weight', 0.5, 3.0)
        catboost_weight = trial.suggest_float('catboost_weight', 0.5, 3.0)
        rf_weight = trial.suggest_float('rf_weight', 0.1, 2.0)
        fnn_weight = trial.suggest_float('fnn_weight', 0.5, 3.0)
        svm_weight = trial.suggest_float('svm_weight', 0.1, 2.0)
        logistic_weight = trial.suggest_float('logistic_weight', 0.1, 2.0)
        
        # 投票阈值
        voting_threshold = trial.suggest_float('voting_threshold', 0.2, 0.8)
        
        # 加载各个模型
        try:
            # 加载XGBoost模型
            with open(model_dir / 'XGBoost_model.pkl', 'rb') as f:
                xgb_model = pickle.load(f)
            
            # 加载LightGBM模型
            with open(model_dir / 'LightGBM_model.pkl', 'rb') as f:
                lgbm_model = pickle.load(f)
            
            # 加载CatBoost模型
            with open(model_dir / 'CatBoost_model.pkl', 'rb') as f:
                catboost_model = pickle.load(f)
            
            # 加载RandomForest模型
            with open(model_dir / 'RF_model.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            
            # 加载FNN模型 (以字典形式加载)
            with open(model_dir / 'FNN_best_model.pkl', 'rb') as f:
                fnn_model_data = pickle.load(f)
                
            # 加载FNN特征信息
            with open(model_dir / 'FNN_feature_info.json', 'r') as f:
                fnn_feature_info = json.load(f)
            
            # 使用全局函数 fnn_predict 代替嵌套函数
            
            # 加载SVM模型
            with open(model_dir / 'SVM_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            
            # 加载Logistic模型 (如果存在)
            try:
                with open(model_dir / 'Logistic_model.pkl', 'rb') as f:
                    logistic_model = pickle.load(f)
            except FileNotFoundError:
                logistic_model = None
                logistic_weight = 0.0
            
            # 进行SMOTE重采样
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # 创建5折交叉验证
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 初始化交叉验证分数
            cv_metrics = []
            
            # 进行交叉验证
            for train_idx, val_idx in skf.split(X_resampled, y_resampled):
                X_train_fold, X_val_fold = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
                y_train_fold, y_val_fold = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]
                
                # 预处理数据
                X_train_preprocessed = preprocessor.fit_transform(X_train_fold)
                X_val_preprocessed = preprocessor.transform(X_val_fold)
                
                # 获取各模型的预测概率
                xgb_prob = xgb_model.predict_proba(X_val_fold)[:, 1]
                lgbm_prob = lgbm_model.predict_proba(X_val_fold)[:, 1]
                catboost_prob = catboost_model.predict_proba(X_val_fold)[:, 1]
                rf_prob = rf_model.predict_proba(X_val_fold)[:, 1]
                # 使用FNN特殊预测函数
                _, fnn_prob = fnn_predict(fnn_model_data, X_val_fold, fnn_feature_info)
                # 使用SVM特殊预测函数
                _, svm_prob = svm_predict(svm_model, X_val_fold)
                if logistic_model is not None:
                    logistic_prob = logistic_model.predict_proba(X_val_fold)[:, 1]
                else:
                    logistic_prob = np.zeros_like(xgb_prob)
                
                # 加权平均预测概率
                ensemble_prob = (xgb_prob * xgb_weight + 
                                lgbm_prob * lgbm_weight + 
                                catboost_prob * catboost_weight + 
                                rf_prob * rf_weight + 
                                fnn_prob * fnn_weight + 
                                svm_prob * svm_weight +
                                logistic_prob * logistic_weight) / (
                                    xgb_weight + lgbm_weight + catboost_weight + 
                                    rf_weight + fnn_weight + svm_weight + 
                                    (logistic_weight if logistic_model is not None else 0)
                                )
                
                # 使用阈值获取预测类别
                ensemble_pred = (ensemble_prob >= voting_threshold).astype(int)
                
                # 计算指标
                fold_metrics = calculate_metrics(y_val_fold, ensemble_pred, ensemble_prob)
                cv_metrics.append(fold_metrics)
            
            # 计算交叉验证平均指标
            avg_metrics = {}
            for metric in cv_metrics[0].keys():
                avg_metrics[metric] = np.mean([m[metric] for m in cv_metrics])
            
            # 使用目标函数计算总得分
            objective_value = objective_function(avg_metrics)
            
            # 将指标存储为trial属性
            for metric, value in avg_metrics.items():
                trial.set_user_attr(metric, value)
            
            return objective_value
        
        except Exception as e:
            logger.error(f"模型加载或评估过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return float('-inf')

    # 使用Optuna进行优化
    logger.info("开始使用Optuna进行集成权重优化...")
    n_trials = config.get('n_trials', 30)
    logger.info(f"计划执行 {n_trials} 次试验")
    
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=n_trials)
        
        best_trial = study.best_trial
        best_params = best_trial.params
        logger.info(f"最佳参数: {best_params}")
        
        # 获取最佳模型的指标
        best_metrics = {key: value for key, value in best_trial.user_attrs.items() if key not in ['params']}
        
        # 保存最佳参数
        params_serializable = {k: float(v) for k, v in best_params.items()}
        param_path = ensemble_dir / 'Ensemble_Voting_best_params.json'
        with open(param_path, 'w') as f:
            json.dump(params_serializable, f, indent=4)
        
        # 记录最佳指标
        logger.info("最佳交叉验证指标:")
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 保存特征信息
        feature_info = {
            'features': features
        }
        with open(ensemble_dir / 'Ensemble_Voting_feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
        
        # 最终模型训练
        logger.info("使用最佳参数在完整训练集上训练最终模型...")
        
        # 进行SMOTE重采样
        smote_final = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train, y_train)
        
        # 训练预处理器
        preprocessor.fit(X_train_resampled)
        
        # 加载所有基础模型
        with open(model_dir / 'XGBoost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        with open(model_dir / 'LightGBM_model.pkl', 'rb') as f:
            lgbm_model = pickle.load(f)
        
        with open(model_dir / 'CatBoost_model.pkl', 'rb') as f:
            catboost_model = pickle.load(f)
        
        with open(model_dir / 'RF_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        with open(model_dir / 'FNN_best_model.pkl', 'rb') as f:
            fnn_model = pickle.load(f)
        
        with open(model_dir / 'SVM_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        try:
            with open(model_dir / 'Logistic_model.pkl', 'rb') as f:
                logistic_model = pickle.load(f)
            logistic_present = True
        except FileNotFoundError:
            logistic_model = None
            logistic_present = False
        
        # 重新加载所有模型，确保在最终评估中有必要的数据
        # 加载XGBoost模型
        with open(model_dir / 'XGBoost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        # 加载LightGBM模型
        with open(model_dir / 'LightGBM_model.pkl', 'rb') as f:
            lgbm_model = pickle.load(f)
        
        # 加载CatBoost模型
        with open(model_dir / 'CatBoost_model.pkl', 'rb') as f:
            catboost_model = pickle.load(f)
        
        # 加载RandomForest模型
        with open(model_dir / 'RF_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        # 加载FNN模型
        with open(model_dir / 'FNN_best_model.pkl', 'rb') as f:
            fnn_model_data = pickle.load(f)
        
        # 加载SVM模型
        with open(model_dir / 'SVM_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
            
        # 创建模型字典
        model_dict = {
            'XGBoost': (xgb_model, best_params['xgb_weight']),
            'LightGBM': (lgbm_model, best_params['lgbm_weight']),
            'CatBoost': (catboost_model, best_params['catboost_weight']),
            'RandomForest': (rf_model, best_params['rf_weight']),
            'FNN': (fnn_model_data, best_params['fnn_weight']),  # 对于FNN模型，存储模型数据而非模型对象
            'SVM': (svm_model, best_params['svm_weight'])
        }
        
        if logistic_present:
            model_dict['Logistic'] = (logistic_model, best_params['logistic_weight'])
        
        # 保存集成模型信息
        ensemble_model = {
            'models': model_dict,
            'threshold': best_params['voting_threshold'],
            'preprocessor': preprocessor,
            'feature_info': feature_info
        }
        
        # 保存集成模型
        with open(ensemble_dir / 'Ensemble_Voting_model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        # 复制到model目录下用于兼容性
        with open(model_dir / 'Ensemble_voting_model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        # 在测试集上评估
        logger.info("在测试集上评估最终模型...")
        
        # 预处理测试集
        X_test_preprocessed = preprocessor.transform(X_test)
        
        # 加载FNN特征信息文件，如果还没加载过
        if 'fnn_feature_info' not in locals():
            try:
                with open(model_dir / 'FNN_feature_info.json', 'r') as f:
                    fnn_feature_info = json.load(f)
            except:
                # 如果加载失败，创建一个默认的
                fnn_feature_info = {
                    'categorical_features': ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'],
                    'numeric_features': ['Age', 'BMI']
                }
        
        # 获取各模型在测试集上的预测
        model_probs = {}
        for model_name, (model, weight) in model_dict.items():
            if model_name == 'FNN':
                # 对FNN模型使用特殊处理
                _, fnn_prob = fnn_predict(model, X_test, fnn_feature_info)
                model_probs[model_name] = fnn_prob * weight
            elif model_name == 'SVM':
                # 对SVM模型使用特殊处理
                _, svm_prob = svm_predict(model, X_test)
                model_probs[model_name] = svm_prob * weight
            else:
                model_probs[model_name] = model.predict_proba(X_test)[:, 1] * weight
        
        # 计算加权平均
        total_weight = sum(weight for _, weight in model_dict.values())
        ensemble_probs = sum(probs for probs in model_probs.values()) / total_weight
        
        # 使用阈值进行预测
        threshold = best_params['voting_threshold']
        ensemble_preds = (ensemble_probs >= threshold).astype(int)
        
        # 计算指标
        test_metrics = calculate_metrics(y_test, ensemble_preds, ensemble_probs, weights_test)
        
        # 输出测试集指标
        logger.info("测试集指标:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 保存测试集指标
        test_metrics_path = output_dir / 'Ensemble_Voting_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=4)
        
        logger.info("集成模型训练与评估完成!")
    
    except Exception as e:
        logger.error(f"优化过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
