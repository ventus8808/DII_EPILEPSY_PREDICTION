import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import yaml
import warnings
import random
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
import traceback
from imblearn.over_sampling import SMOTE
from tqdm.auto import tqdm

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)
if hasattr(mx.random, 'seed'):
    mx.random.seed(42)

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

class FNNModel(nn.Module):
    """FNN模型使用MLX框架"""
    def __init__(self, input_dim, params):
        super().__init__()
        
        # 定义层
        layers = []
        
        # 第一个隐藏层
        layers.append(nn.Linear(input_dim, params['units_1']))
        if params['activation'] == 'relu':
            layers.append(nn.ReLU())
        elif params['activation'] == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())
        
        if params.get('use_batch_norm', False):
            layers.append(nn.BatchNorm(params['units_1']))
        
        layers.append(nn.Dropout(p=params['dropout_1']))
        
        # 第二个隐藏层
        layers.append(nn.Linear(params['units_1'], params['units_2']))
        if params['activation'] == 'relu':
            layers.append(nn.ReLU())
        elif params['activation'] == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())
        
        if params.get('use_batch_norm', False):
            layers.append(nn.BatchNorm(params['units_2']))
        
        layers.append(nn.Dropout(p=params['dropout_2']))
        
        # 输出层
        layers.append(nn.Linear(params['units_2'], 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.layers(x)

def binary_cross_entropy_loss(model, X, y):
    """二元交叉熵损失函数"""
    preds = model(X).reshape(-1)
    eps = 1e-7
    loss = -mx.mean(y * mx.log(preds + eps) + (1 - y) * mx.log(1 - preds + eps))
    return loss

def train_step(model, X, y, optimizer):
    """单步训练"""
    loss_and_grad_fn = nn.value_and_grad(model, binary_cross_entropy_loss)
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

def evaluate(model, X, y):
    """评估模型在验证数据上的表现"""
    # 转换为NumPy数组以便于使用sklearn指标
    preds = model(X).reshape(-1)
    preds_np = mx.array(preds).astype(mx.float32).tolist()
    y_np = mx.array(y).astype(mx.float32).tolist()
    
    # 计算损失值
    loss = binary_cross_entropy_loss(model, X, y)
    loss_val = float(loss.item())
    
    # 分类指标
    y_pred = (np.array(preds_np) > 0.5).astype(int)
    y_true = np.array(y_np)
    
    acc = accuracy_score(y_true, y_pred)
    
    # 计算AUC
    try:
        auc_val = roc_auc_score(y_true, preds_np)
    except Exception:
        auc_val = 0.5  # 默认值
    
    # 计算F1分数
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'loss': loss_val, 
        'accuracy': acc, 
        'auc': auc_val, 
        'f1': f1,
        'preds': np.array(preds_np),
        'y_pred': y_pred
    }

def train_model(model, X_train, y_train, X_val, y_val, params):
    """使用早停训练模型"""
    # 转换验证集为MLX格式
    X_val_mx = mx.array(X_val.astype(np.float32))
    y_val_mx = mx.array(y_val.astype(np.float32))
    # 训练集不提前转为mlx.array，batch循环内再转
    
    # 设置优化器
    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = optim.SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'adamw':
        # MLX的AdamW实现
        optimizer = optim.AdamW(learning_rate=params['learning_rate'], weight_decay=0.01)
    else:
        optimizer = optim.Adam(learning_rate=params['learning_rate'])
    
    batch_size = params['batch_size']
    n_epochs = params['epochs']
    early_stop_patience = params.get('early_stop_patience', 10)
    
    # 批处理函数
    def get_batches(X, y, batch_size):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    # 训练循环
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    train_history = []
    val_history = []
    
    for epoch in range(n_epochs):
        # 训练模式
        epoch_losses = []
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            X_batch_mx = mx.array(X_batch.astype(np.float32))
            y_batch_mx = mx.array(y_batch.astype(np.float32))
            model.train()
            loss = train_step(model, X_batch_mx, y_batch_mx, optimizer)
            epoch_losses.append(float(loss.item()))
        
        # 评估训练集
        X_train_eval_mx = mx.array(X_train.astype(np.float32))
        y_train_eval_mx = mx.array(y_train.astype(np.float32))
        train_metrics = evaluate(model, X_train_eval_mx, y_train_eval_mx)
        train_history.append(train_metrics)
        
        # 评估验证集
        val_metrics = evaluate(model, X_val_mx, y_val_mx)
        val_history.append(val_metrics)
        
        # 早停检查
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = {k: v for k, v in model.parameters().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logging.getLogger().info(f"早停于第 {epoch+1} 轮。最佳验证集AUC: {best_val_auc:.4f}")
                break
        
        # 输出进度 - 只在日志中记录，减少控制台输出
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger = logging.getLogger()
            logger.debug(f"轮次 {epoch+1}/{n_epochs}: 训练AUC={train_metrics['auc']:.4f}, 验证AUC={val_metrics['auc']:.4f}")
    
    # 恢复最佳模型
    if best_model_state is not None:
        for k, v in best_model_state.items():
            model.parameters()[k] = v
    
    return model, best_val_auc, train_history, val_history

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
    # 设置根记录器级别为INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    
    # 创建控制台处理器但只显示关键信息(WARNING及以上)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    # 获取记录器并添加处理器
    logger = logging.getLogger()
    logger.addHandler(console)
    
    return logger

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
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 保存标准化器
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_scaled, y.values, weights, encoder, scaler

# 4. 目标函数配置和指标
def load_objective_config(config, constraint_type='objective_constraints'):
    """加载目标函数配置，包括权重和约束
    
    Args:
        config: 配置字典
        constraint_type: 约束类型，可选 'cv_constraints', 'test_constraints' 或默认的 'objective_constraints'
    """
    weights = config.get('objective_weights', {})
    constraints = config.get(constraint_type, {})
    return weights, constraints

def objective_function(metrics, config, constraint_type='objective_constraints'):
    """定制目标函数，使用config中的硬约束及自定义权重
    
    Args:
        metrics: 评估指标字典
        config: 配置字典
        constraint_type: 约束类型，可选 'cv_constraints', 'test_constraints' 或默认的 'objective_constraints'
    """
    # 打印进度
    print(f"---> 计算目标函数，约束类型: {constraint_type}")
    print(f"---> 指标: AUC-ROC={metrics.get('AUC-ROC', 0):.4f}, 敏感度={metrics.get('Sensitivity', 0):.4f}, 特异度={metrics.get('Specificity', 0):.4f}")
    
    # 获取配置文件中的约束条件
    weights, constraints = load_objective_config(config, constraint_type)
    print(f"---> 加载到约束条件: {constraints}")
    
    # 关键指标
    auc_roc = metrics.get('AUC-ROC', 0)
    sensitivity = metrics.get('Sensitivity', 0)
    specificity = metrics.get('Specificity', 0)
    f1 = metrics.get('F1', 0)
    ece = metrics.get('ECE', 0)
    
    # 根据约束条件检查硬约束
    failed_constraints = []
    
    # AUC约束
    if 'AUC_min' in constraints and auc_roc < constraints['AUC_min']:
        failed_msg = f"AUC-ROC {auc_roc:.4f} < {constraints['AUC_min']}"
        failed_constraints.append(failed_msg)
        print(f"---> 约束失败: {failed_msg}")
        
    if 'AUC_max' in constraints and auc_roc > constraints['AUC_max']:
        failed_msg = f"AUC-ROC {auc_roc:.4f} > {constraints['AUC_max']}"
        failed_constraints.append(failed_msg)
        print(f"---> 约束失败: {failed_msg}")
    
    # 敏感度约束    
    if 'Sensitivity_min' in constraints and sensitivity < constraints['Sensitivity_min']:
        failed_msg = f"Sensitivity {sensitivity:.4f} < {constraints['Sensitivity_min']}"
        failed_constraints.append(failed_msg)
        print(f"---> 约束失败: {failed_msg}")
    
    # 特异度约束
    if 'Specificity_min' in constraints and specificity < constraints['Specificity_min']:
        failed_msg = f"Specificity {specificity:.4f} < {constraints['Specificity_min']}"
        failed_constraints.append(failed_msg)
        print(f"---> 约束失败: {failed_msg}")
    
    # 如果有失败的约束，返回 -inf
    if failed_constraints:
        return float('-inf'), failed_constraints
    
    # 计算加权得分
    score = (
        weights.get('AUC', 0.3) * auc_roc +
        weights.get('F1', 0.2) * f1 +
        weights.get('Precision', 0.2) * metrics.get('Precision', 0) +
        weights.get('Sensitivity', 0.2) * sensitivity +
        weights.get('Specificity', 0.2) * specificity +
        weights.get('ECE', -0.1) * ece
    )
    
    print(f"目标函数值: {score}")
    return score, []  # 成功时返回计算出的目标函数值和空的失败原因列表

# 5. 主流程
def main():
    # 读取配置文件
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    model_name = "FNN"
    logger = setup_logger(model_name)
    logger.info(f"====== 开始FNN模型训练 ======")
    print(f"====== 开始FNN模型训练 ======")
    logger.info(f"特征列表: {features}")
    logger.info(f"类别特征: {categorical_features}")
    logger.info(f"数值特征: {numeric_features}")
    
    # 从配置文件读取n_trials参数
    n_trials = config.get('n_trials', 10)  # 如果未设置，默认为10
    logger.info(f"计划运行{n_trials}次Optuna试验")
    
    # 加载数据
    X, y, weights, encoder, scaler = load_and_preprocess_data()
    
    # 训练测试分割
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"数据分割完成：训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    logger.info(f"训练集类别分布: {np.bincount(y_train.astype(int))}")
    logger.info(f"测试集类别分布: {np.bincount(y_test.astype(int))}")
    
    # 定义Optuna目标函数
    def optuna_objective(trial):
        # 获取超参数
        params = {
            # 网络结构 - 优化搜索空间
            'units_1': trial.suggest_int('units_1', 32, 512),  # 增大范围
            'units_2': trial.suggest_int('units_2', 16, 256),  # 增大范围
            'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.6),  # 调整范围
            'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.6),  # 调整范围
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'epochs': trial.suggest_int('epochs', 20, 200),  # 新增epochs参数
            
            # 训练设置
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),  # 添加AdamW
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-2, log=True),  # 扩大范围
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),  # 添加更大batch size
            'n_epochs': 150,  # 增加训练轮数
            'early_stop_patience': 25  # 增加早停耐心
        }
        
        # 交叉验证设置
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        n_splits = 5  # 与XGBoost和其他模型保持一致的折数
        
        # 训练函数
        def train_fold(fold, train_idx, val_idx):
            print(f"\n[Fold {fold+1}/5] 开始训练...")
            # 获取当前折叠的训练和验证数据
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            print(f"[Fold {fold+1}/5] 训练集样本数: {len(X_fold_train)}, 验证集样本数: {len(X_fold_val)}")
            
            fold_weights_train = None
            fold_weights_val = None
            if weights_train is not None:
                # 保证索引与train_idx一致，避免KeyError
                weights_train_reset = weights_train.reset_index(drop=True)
                fold_weights_train = weights_train_reset[train_idx]
                fold_weights_val = weights_train_reset[val_idx]
            
            # 用SMOTE处理类别不平衡
            try:
                # 检查正例样本数量，确保足够进行SMOTE过采样
                pos_samples = np.sum(y_fold_train == 1)
                print(f"[Fold {fold+1}/5] 训练集中正例样本数: {pos_samples}")
                
                # 如果正例样本太少，则不使用SMOTE
                if pos_samples >= 5:  # SMOTE至少需补5个少数类样本
                    print(f"[Fold {fold+1}/5] 开始SMOTE过采样...")
                    smote = SMOTE(random_state=42, k_neighbors=min(pos_samples-1, 5))
                    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
                    print(f"[Fold {fold+1}/5] SMOTE后样本数: {len(X_fold_train_resampled)}, 正例样本数: {np.sum(y_fold_train_resampled == 1)}")
                else:
                    # 样本量不足，直接使用原始数据
                    X_fold_train_resampled, y_fold_train_resampled = X_fold_train, y_fold_train
                    logger.warning(f"折叠{fold+1}中正例样本数量不足({pos_samples}个)，跳过SMOTE")
                
                # 创建FNN模型
                input_dim = X_fold_train_resampled.shape[1]
                model = FNNModel(input_dim, params)
                
                # 训练模型
                model, best_val_auc, _, _ = train_model(
                    model, X_fold_train_resampled, y_fold_train_resampled, X_fold_val, y_fold_val, params
                )
                
                # 验证集评估
                print(f"[Fold {fold+1}/5] 训练完成，开始验证集评估...")
                val_metrics = evaluate(model, mx.array(X_fold_val.astype(np.float32)), mx.array(y_fold_val.astype(np.float32)))
                y_pred = val_metrics['y_pred']
                y_prob = val_metrics['preds']
                print(f"[Fold {fold+1}/5] 验证集AUC: {val_metrics['auc']:.4f}")
                
                # 计算全部指标
                metrics_dict = calculate_metrics(y_fold_val, y_pred, y_prob, fold_weights_val)
                print(f"[Fold {fold+1}/5] 所有指标计算完成: AUC={metrics_dict['AUC-ROC']:.4f}, 敏感度={metrics_dict['Sensitivity']:.4f}, 特异度={metrics_dict['Specificity']:.4f}")
                return metrics_dict
            except Exception as e:
                logger.error(f"[Trial {trial.number+1}] 折叠 {fold+1} 错误: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        
        try:
            # 执行交叉验证
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.debug(f"[Trial {trial.number+1}] 训练折叠 {fold+1}/{n_splits}...")
                fold_metric = train_fold(fold, train_idx, val_idx)
                if fold_metric is not None:
                    logger.debug(f"[Trial {trial.number+1}] 折叠 {fold+1} 指标: AUC-ROC = {fold_metric['AUC-ROC']:.4f}")
                    fold_metrics.append(fold_metric)
            
            # 如果所有折叠都失败了
            if len(fold_metrics) == 0:
                logger.warning(f"[Trial {trial.number+1}] 所有折叠训练失败")
                return float('-inf')
            
            # 计算平均指标
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics if key in m]
                avg_metrics[key] = np.mean(values) if values else float('nan')
            
            # 计算目标分数
            print("\n====> 计算最终目标分数...")
            score, failed_constraints = objective_function(avg_metrics, config, constraint_type='cv_constraints')  # 使用交叉验证约束
        
            # 如果有失败的约束，记录到日志
            if failed_constraints:
                logger.info(f"[Trial {trial.number+1}] 约束失败: {', '.join(failed_constraints)}")
                return float('-inf')  # 确保optuna不会选择这些失败的试验
        
            logger.debug(f"[Trial {trial.number+1}] 平均指标: AUC-ROC = {avg_metrics['AUC-ROC']:.4f}, 分数 = {score:.4f}")
            
            return score
        except Exception as e:
            logger.error(f"[Trial {trial.number+1}] 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return float('-inf')
    
    # 优化超参数
    study = optuna.create_study(direction='maximize')
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    
    # 关闭Optuna自身的输出
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # 运行Optuna优化
    logger.info(f"开始Optuna优化 ({n_trials} 次试验)...")
    for trial in range(n_trials):
        try:
            trial_obj = study.ask()
            study.tell(trial_obj, optuna_objective(trial_obj))
            
            # 获取当前最佳结果
            if study.best_value > best_score:
                best_score = study.best_value
                best_params = study.best_params
                
                # 创建验证模型进行测试
                logger.info(f"[Trial {trial+1}] 更新最佳参数...")
                
                # 保存最佳参数
                param_path = model_dir / f'FNN_best_params.json'
                with open(param_path, 'w') as f:
                    json.dump(best_params, f, indent=4)
                
                # 保存特征信息
                feature_info = {
                    'features': features,
                    'categorical_features': categorical_features,
                    'numeric_features': numeric_features,
                    'input_dim': X_train.shape[1]
                }
                with open(model_dir / 'FNN_feature_info.json', 'w') as f:
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
        logger.info("使用最佳参数训练最终模型...")
        logger.info(f"最佳参数: {best_params}")
        
        # SMOTE处理整个训练集 - 约束与其他模型一致
        try:
            # 重新采样整个训练集
            logger.info("使用SMOTE对整个训练集进行采样...")
            
            # 处理缺失值
            from sklearn.impute import SimpleImputer
            logger.info("使用均值填充处理数据中的缺失值...")
            imputer = SimpleImputer(strategy='mean')
            
            # 创建SMOTE实例
            smote_final = SMOTE(random_state=42)
            
            # 处理权重
            if weights_train is not None:
                # 确保X_train是DataFrame
                if not isinstance(X_train, pd.DataFrame):
                    X_train_df = pd.DataFrame(X_train)
                else:
                    X_train_df = X_train.copy()
                
                # 确保所有列名都是字符串类型
                X_train_df.columns = X_train_df.columns.astype(str)
                    
                # 添加标签和权重
                X_train_df['__label__'] = y_train
                X_train_df['__weight__'] = weights_train
                
                # 处理缺失值
                X_train_df_features = X_train_df.drop(['__label__'], axis=1)
                X_train_df_imputed = pd.DataFrame(
                    imputer.fit_transform(X_train_df_features),
                    columns=X_train_df_features.columns
                )
                X_train_df_imputed['__weight__'] = X_train_df['__weight__']
                
                # 进行SMOTE过采样
                X_res_final, y_res_final = smote_final.fit_resample(
                    X_train_df_imputed.drop(['__weight__'], axis=1), 
                    X_train_df['__label__']
                )
                
                # SMOTE过采样后不会保留__weight__列，需要生成新的权重
                # 简单的处理方法：将过采样后的所有样本的权重设为1
                weights_train_res_final = np.ones(len(y_res_final))
                
                # 确保X_train_res_final是numpy数组，而不是DataFrame
                if isinstance(X_res_final, pd.DataFrame):
                    X_train_res_final = X_res_final.values
                else:
                    X_train_res_final = X_res_final
                    
                y_train_res_final = y_res_final
                logger.info(f"生成新的权重数组，长度：{len(weights_train_res_final)}")
            else:
                # 如果没有权重，直接进行SMOTE过采样
                # 如果是DataFrame
                if isinstance(X_train, pd.DataFrame):
                    X_train_copy = X_train.copy()
                    # 确保列名是字符串类型
                    X_train_copy.columns = X_train_copy.columns.astype(str)
                    # 处理缺失值
                    X_train_imputed = imputer.fit_transform(X_train_copy)
                    # 使用处理后的数据进行SMOTE
                    X_temp, y_train_res_final = smote_final.fit_resample(X_train_imputed, y_train)
                    # 确保返回的是numpy数组而不是DataFrame
                    X_train_res_final = np.array(X_temp)
                else:
                    # 如果是numpy数组，先处理缺失值
                    X_train_imputed = imputer.fit_transform(X_train)
                    X_temp, y_train_res_final = smote_final.fit_resample(X_train_imputed, y_train)
                    X_train_res_final = np.array(X_temp)
                weights_train_res_final = None
                
            logger.info(f"SMOTE后训练样本数: {len(X_train_res_final)}, 正例样本数: {np.sum(y_train_res_final == 1)}")
            
            # 创建最终模型
            best_params['epochs'] = 100  # 确保有足够的训练轮数
            best_params['early_stop_patience'] = 15  # 早停耐心
            input_dim = X_train.shape[1]
            final_model = FNNModel(input_dim, best_params)
            
            # 确保X_train_res_final和X_test都是numpy数组
            if isinstance(X_train_res_final, pd.DataFrame):
                X_train_res_final = X_train_res_final.values
            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
                
            # 训练最终模型
            logger.info(f"X_train_res_final类型: {type(X_train_res_final)}")
            final_model, _, _, _ = train_model(
                final_model, X_train_res_final, y_train_res_final, X_test, y_test, best_params
            )
            
            # 在测试集评估
            X_test_mx = mx.array(X_test.astype(np.float32))
            y_test_mx = mx.array(y_test.astype(np.float32))
            
            test_eval = evaluate(final_model, X_test_mx, y_test_mx)
            y_prob_test = test_eval['preds']
            y_pred_test = (y_prob_test >= 0.5).astype(int)
            
            # 计算测试集指标
            test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test, weights_test)
            
            logger.info("测试集指标 (训练/调优期间从未见过):")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # 检查测试集硬约束
            test_score, test_failed_constraints = objective_function(test_metrics, config, constraint_type='test_constraints')
            if test_failed_constraints:
                logger.warning(f"测试集约束失败: {', '.join(test_failed_constraints)}")
            else:
                logger.info(f"测试集通过所有硬约束，分数: {test_score:.4f}")
                
                # 保存模型和参数 - 统一与其他模型一致
                model_path = model_dir / 'FNN_best_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model_state': final_model.parameters(),
                        'model_config': {
                            'input_dim': int(X_train.shape[1]),
                            'params': best_params
                        },
                        'scaler': scaler,
                        'encoder': encoder
                    }, f)
                logger.info(f"模型保存到 {model_path}")
                
                # 保存参数
                param_path = model_dir / 'FNN_best_params.json'
                with open(param_path, 'w') as f:
                    json.dump(best_params, f, indent=4)
                logger.info(f"参数保存到 {param_path}")
                
                # 保存特征信息
                feature_info = {
                    'features': features,
                    'categorical_features': categorical_features,
                    'numeric_features': numeric_features
                }
                feature_path = model_dir / 'FNN_feature_info.json'
                with open(feature_path, 'w') as f:
                    json.dump(feature_info, f, indent=4)
                logger.info(f"特征信息保存到 {feature_path}")
                
                # 保存测试集指标
                metrics_path = output_dir / 'FNN_metrics.json'
                with open(metrics_path, 'w') as f:
                    if 'test_metrics' in locals():
                        json.dump(test_metrics, f, indent=4)
                    else:
                        logger.warning("test_metrics未定义，保存空字典")
                        json.dump({}, f, indent=4)
                logger.info(f"测试集指标保存到 {metrics_path}")
                
                # 保存最佳CV指标
                best_metrics_path = model_dir / 'FNN_best_metrics.json'
                with open(best_metrics_path, 'w') as f:
                    json.dump(best_metrics, f, indent=4)
                logger.info(f"最佳CV指标保存到 {best_metrics_path}")
        except Exception as e:
            logger.error(f"错误: {str(e)}")
            logger.error(traceback.format_exc())
            # 异常处理中确保 metrics_path 和 f 变量存在
            if 'metrics_path' not in locals():
                metrics_path = output_dir / 'FNN_metrics.json'
                with open(metrics_path, 'w') as f:
                    logger.warning("test_metrics未定义，保存空字典")
                    json.dump({}, f, indent=4)
    else:
        logger.warning("未找到满足约束条件的有效模型。")
        logger.warning("请检查您的目标约束和数据。")

if __name__ == "__main__":
    main()
