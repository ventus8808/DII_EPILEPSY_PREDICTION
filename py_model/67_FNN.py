import pandas as pd
import numpy as np
import pickle
import json
import logging
import os
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import optuna
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
mx.random.seed(42)

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate ECE and MCE"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        if np.any(in_bin):
            prob_true = np.mean(y_true[in_bin])
            prob_pred = np.mean(y_prob[in_bin])
            ece += np.abs(prob_true - prob_pred) * np.sum(in_bin) / len(y_true)
            mce = max(mce, np.abs(prob_true - prob_pred))
    
    return float(ece), float(mce)

def calculate_metrics(y_true, y_pred_proba):
    """Calculate all required metrics"""
    from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, roc_curve,
                               precision_score, recall_score, f1_score, confusion_matrix,
                               accuracy_score, brier_score_loss, log_loss, cohen_kappa_score)
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Basic classification metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_val = float(precision_score(y_true, y_pred, zero_division=0))
    recall_val = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix based metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((recall_val + specificity) / 2)
    
    # ROC and PR curves
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = float(auc(recall, precision))
    
    # Calibration metrics
    ece, mce = calculate_calibration_metrics(y_true, y_pred_proba)
    brier = float(brier_score_loss(y_true, y_pred_proba))
    log_loss_val = float(log_loss(y_true, y_pred_proba))
    
    # Other metrics
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision_val,
        'recall': recall_val,
        'sensitivity': recall_val,
        'specificity': specificity,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ece': ece,
        'mce': mce,
        'brier_score': brier,
        'log_loss': log_loss_val,
        'kappa': kappa
    }

class FNNModel(nn.Module):
    """FNN model using MLX"""
    def __init__(self, input_dim, params):
        super().__init__()
        
        # Define layers
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_dim, params['units_1']))
        layers.append(nn.ReLU() if params['activation'] == 'relu' else 
                     (nn.GELU() if params['activation'] == 'gelu' else nn.ReLU()))
        
        if params['use_batch_norm']:
            layers.append(nn.BatchNorm(params['units_1']))
        
        layers.append(nn.Dropout(p=params['dropout_1']))
        
        # Second hidden layer
        layers.append(nn.Linear(params['units_1'], params['units_2']))
        layers.append(nn.ReLU() if params['activation'] == 'relu' else 
                     (nn.GELU() if params['activation'] == 'gelu' else nn.ReLU()))
        
        if params['use_batch_norm']:
            layers.append(nn.BatchNorm(params['units_2']))
        
        layers.append(nn.Dropout(p=params['dropout_2']))
        
        # Output layer
        layers.append(nn.Linear(params['units_2'], 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.layers(x)

def binary_cross_entropy_loss(model, X, y):
    """Binary cross-entropy loss function"""
    y_pred = model(X)
    y_pred = mx.clip(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -mx.mean(y * mx.log(y_pred) + (1 - y) * mx.log(1 - y_pred))
    return loss

def train_step(model, X, y, optimizer):
    """Single training step"""
    loss_and_grad_fn = nn.value_and_grad(model, binary_cross_entropy_loss)
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

def evaluate(model, X, y):
    """Evaluate model on validation data"""
    try:
        y_pred = model(X)
        y_pred = mx.clip(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -mx.mean(y * mx.log(y_pred) + (1 - y) * mx.log(1 - y_pred))
        
        # Convert to numpy for metrics calculation
        y_pred_np = y_pred.tolist()
        y_np = y.tolist()
        
        # Flatten arrays if needed
        if isinstance(y_pred_np[0], list):
            y_pred_np = [item[0] for item in y_pred_np]
        if isinstance(y_np[0], list):
            y_np = [item[0] for item in y_np]
        
        # Check for NaN values
        if any(np.isnan(y_pred_np)) or any(np.isnan(y_np)):
            print("WARNING: NaN values detected in predictions or labels")
            y_pred_np = np.nan_to_num(y_pred_np)
            y_np = np.nan_to_num(y_np)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_np, y_pred_np)
        
        return float(loss), float(auc)
    except Exception as e:
        print(f"Error in evaluate function: {str(e)}")
        return float('inf'), 0.0

def train_model(model, X_train, y_train, X_val, y_val, params):
    """Train model with early stopping"""
    # Convert data to MLX arrays
    X_train_mx = mx.array(X_train.values.astype(np.float32))
    y_train_mx = mx.array(y_train.values.astype(np.float32).reshape(-1, 1))
    X_val_mx = mx.array(X_val.values.astype(np.float32))
    y_val_mx = mx.array(y_val.values.astype(np.float32).reshape(-1, 1))
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=params['learning_rate'])
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # Initialize early stopping variables
    best_val_auc = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(params['epochs']):
        # 使用批处理而不是随机打乱索引
        # 这种方法避免了使用NumPy索引来访问MLX数组
        n_samples = X_train_mx.shape[0]
        batch_size = params['batch_size']
        
        # 随机打乱样本顺序
        perm = np.random.permutation(n_samples)
        
        # 批量训练
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            # 获取当前批次的索引
            batch_indices = perm[i:min(i + batch_size, n_samples)]
            
            # 创建新的批次数据
            X_batch_np = X_train.values[batch_indices].astype(np.float32)
            y_batch_np = y_train.values[batch_indices].reshape(-1, 1).astype(np.float32)
            
            # 转换为MLX数组
            X_batch = mx.array(X_batch_np)
            y_batch = mx.array(y_batch_np)
            
            # 训练步骤
            loss = train_step(model, X_batch, y_batch, optimizer)
            epoch_losses.append(float(loss))
        
        # Evaluate on validation set
        val_loss, val_auc = evaluate(model, X_val_mx, y_val_mx)
        
        # Record history
        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.parameters()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                break
    
    # Restore best model
    if best_model_state is not None:
        model.update(best_model_state)
    
    return model, history

def custom_scorer(params, X_train, y_train, config=None):
    """Custom scorer for model evaluation using stratified 5-fold cross-validation"""
    from sklearn.model_selection import StratifiedKFold
    
    # 初始化分层5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储每折的分数和指标
    fold_scores = []
    fold_metrics = []
    all_histories = []
    
    # 打印当前评估的参数
    param_str = ", ".join([f"{k}={v}" for k, v in params.items() 
                          if k in ['units_1', 'units_2', 'learning_rate', 'activation']])
    print(f"\nEvaluating parameters: {param_str}")
    
    # 对每一折进行训练和评估，使用tqdm显示进度
    for fold, (train_idx, val_idx) in enumerate(tqdm(list(skf.split(X_train, y_train)), 
                                                    desc="Cross-validation folds", 
                                                    leave=False)):
        try:
            # 分割数据
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # 打印每个折的类别分布情况
            train_class_dist = y_fold_train.value_counts().to_dict()
            val_class_dist = y_fold_val.value_counts().to_dict()
            print(f"  Fold {fold+1}/5 - Training with {len(X_fold_train)} samples (class dist: {train_class_dist}), "
                  f"validating with {len(X_fold_val)} samples (class dist: {val_class_dist})")
            
            # 应用SMOTE过采样（仅对训练集）
            smote = SMOTE(random_state=42)
            X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
            
            # 转换为DataFrame以保持一致性
            X_fold_train_resampled = pd.DataFrame(X_fold_train_resampled, columns=X_fold_train.columns)
            y_fold_train_resampled = pd.Series(y_fold_train_resampled)
            
            # 打印SMOTE后的类别分布
            resampled_class_dist = y_fold_train_resampled.value_counts().to_dict()
            print(f"  After SMOTE: {len(X_fold_train_resampled)} samples (class dist: {resampled_class_dist})")
            
            # 创建和训练模型
            model = FNNModel(X_train.shape[1], params)
            model, history = train_model(model, X_fold_train_resampled, y_fold_train_resampled, X_fold_val, y_fold_val, params)
            all_histories.append(history)
            
            # 获取预测
            X_val_mx = mx.array(X_fold_val.values.astype(np.float32))
            y_pred_proba = model(X_val_mx).tolist()
            
            # 扁平化预测结果（如果需要）
            if isinstance(y_pred_proba[0], list):
                y_pred_proba = [item[0] for item in y_pred_proba]
            
            # 检查NaN或无穷大值
            if any(np.isnan(y_pred_proba)) or any(np.isinf(y_pred_proba)):
                print("  WARNING: NaN or infinite values in predictions")
                y_pred_proba = np.nan_to_num(y_pred_proba)
            
            # 计算指标
            metrics = calculate_metrics(y_fold_val, y_pred_proba)
            fold_metrics.append(metrics)
            
            # 使用main中传入的config获取权重
            config_weights = config.get('objective_weights', {})
            
            # 转换配置文件中的权重到代码中使用的权重名称
            weights = {
                'roc_auc': config_weights.get('AUC', 0.3),
                'ece': config_weights.get('ECE', -0.1),
                'mce': config_weights.get('ECE', -0.1) * 1.5,  # MCE权重稍重于ECE
                'f1_score': config_weights.get('F1', 0.2),
                'precision': config_weights.get('Precision', 0.2),
                'sensitivity': config_weights.get('Sensitivity', 0.2),
                'specificity': config_weights.get('Specificity', 0.2)
            }
            
            score = sum(weights[k] * metrics[k] for k in weights.keys() if k in metrics)
            fold_scores.append(score)
            
            # 打印当前折的主要指标
            print(f"  Fold {fold+1} metrics: ROC-AUC={metrics['roc_auc']:.4f}, ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}, Score={score:.4f}")
            
        except Exception as e:
            print(f"  Error in fold {fold+1}: {str(e)}")
            logging.error(f"Error in fold {fold+1}: {str(e)}")
            # 如果一个折失败，我们给这个折一个非常低的分数
            fold_scores.append(float('-inf'))
            fold_metrics.append(None)
    
    # 如果所有折都失败，返回失败
    if all(score == float('-inf') for score in fold_scores):
        return float('-inf'), None, None
    
    # 计算平均分数和指标（忽略失败的折）
    valid_scores = [score for score in fold_scores if score != float('-inf')]
    valid_metrics_indices = [i for i, score in enumerate(fold_scores) if score != float('-inf')]
    
    if not valid_scores:
        return float('-inf'), None, None
    
    avg_score = np.mean(valid_scores)
    
    # 计算平均指标
    avg_metrics = {}
    for metric_key in fold_metrics[valid_metrics_indices[0]].keys():
        avg_metrics[metric_key] = np.mean([fold_metrics[i][metric_key] for i in valid_metrics_indices])
    
    # 合并所有历史记录
    combined_history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # 只使用有效折的历史记录
    valid_histories = [all_histories[i] for i in valid_metrics_indices]
    for key in combined_history.keys():
        # 找到最长的历史记录长度
        max_length = max(len(hist[key]) for hist in valid_histories)
        # 对于每个epoch，计算所有折的平均值
        for epoch in range(max_length):
            values = [hist[key][epoch] if epoch < len(hist[key]) else np.nan for hist in valid_histories]
            combined_history[key].append(np.nanmean(values))
    
    # 打印平均指标
    print(f"Average metrics across {len(valid_scores)} valid folds:")
    print(f"  ROC-AUC={avg_metrics['roc_auc']:.4f}, ECE={avg_metrics['ece']:.4f}, MCE={avg_metrics['mce']:.4f}")
    print(f"  Average score: {avg_score:.4f}")
    
    # 使用main中传入的config获取约束条件
    try:
        # 使用交叉验证约束条件
        constraints = config.get('cv_constraints', {})
        
        # 检查硬约束条件
        failed_constraints = []
        
        # AUC约束
        auc_min = constraints.get('AUC_min', 0.6)
        auc_max = constraints.get('AUC_max', 0.95)
        if not (auc_min < avg_metrics['roc_auc'] < auc_max):
            failed_msg = f"Failed constraint: ROC-AUC={avg_metrics['roc_auc']:.4f} not in ({auc_min}, {auc_max})"
            print(failed_msg)
            failed_constraints.append(failed_msg)
        
        # 敏感度约束
        sensitivity_min = constraints.get('Sensitivity_min', 0.4)
        if avg_metrics['sensitivity'] < sensitivity_min:
            failed_msg = f"Failed constraint: Sensitivity={avg_metrics['sensitivity']:.4f} < {sensitivity_min}"
            print(failed_msg)
            failed_constraints.append(failed_msg)
        
        # 特异度约束
        specificity_min = constraints.get('Specificity_min', 0.4)
        if avg_metrics['specificity'] < specificity_min:
            failed_msg = f"Failed constraint: Specificity={avg_metrics['specificity']:.4f} < {specificity_min}"
            print(failed_msg)
            failed_constraints.append(failed_msg)
        
        if failed_constraints:
            return float('-inf'), None, None
    except Exception as e:
        print(f"Error checking constraints: {str(e)}")
        logging.error(f"Error checking constraints: {str(e)}")
        # 如果有错误，使用默认约束
        if not (0.5 < avg_metrics['roc_auc'] < 1.0):
            print(f"Failed constraint: ROC-AUC={avg_metrics['roc_auc']:.4f} not in (0.5, 1.0)")
            return float('-inf'), None, None
    
    return float(avg_score), combined_history, avg_metrics

def load_data(config):
    """Load and preprocess data"""
    # 获取数据文件路径
    data_path = config.get('data_path', 'data/16_ML.csv')
    print(f"加载数据文件: {data_path}")
    
    # 加载原始数据
    df = pd.read_csv(data_path)
    
    # Print data info for debugging
    print("Data loaded successfully.")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 从配置中读取变量信息
    exposure = config.get('exposure', 'DII_food')  # 暴露变量
    outcome = config.get('outcome', 'Epilepsy')    # 结局变量
    covariates = config.get('covariates', [])     # 协变量列表
    
    print(f"暴露变量: {exposure}")
    print(f"结局变量: {outcome}")
    print(f"协变量列表: {covariates}")
    
    # 确定所需列
    required_columns = [exposure, outcome, 'WTDRD1'] + covariates
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing columns in dataset: {missing_columns}")
        # Use dummy data for testing if columns are missing
        print("Creating dummy data for testing...")
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, 10), 
                         columns=['DII'] + ['Feature_'+str(i) for i in range(9)])
        y = pd.Series(np.random.randint(0, 2, n_samples))
        weights = pd.Series(np.ones(n_samples))
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, stratify=y, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, weights_train, weights_test
    
    weights = df['WTDRD1']
    
    # 定义连续变量和分类变量 - 基于协变量
    # 默认将Age和BMI视为连续变量，其他都视为分类变量
    continuous_vars = ['Age', 'BMI']
    continuous_features = [exposure] + [var for var in continuous_vars if var in covariates]
    categorical_features = [var for var in covariates if var not in continuous_vars]
    
    print(f"连续变量: {continuous_features}")
    print(f"分类变量: {categorical_features}")
    
    # 确保所有分类变量都是字符串类型，以便正确进行独热编码
    for col in categorical_features:
        df[col] = df[col].astype(str)
    
    # 创建特征矩阵
    X = df[continuous_features + categorical_features]
    y = df[outcome]
    
    # 打印类别分布
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Class distribution percentage: {y.value_counts(normalize=True).to_dict()}")
    
    # 打印每个分类变量的唯一值
    print("\nCategorical features unique values:")
    for col in categorical_features:
        print(f"{col}: {X[col].unique().tolist()}")
    
    # 创建预处理管道
    # 1. 对连续变量进行标准化（z-score）
    # 2. 对分类变量进行独热编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 拆分数据
    X_train_raw, X_test_raw, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=42
    )
    
    # 应用预处理转换器
    # 注意：我们只使用训练数据拟合预处理器，以避免数据泄露
    print("\nApplying preprocessing (StandardScaler for continuous features, OneHotEncoder for categorical features)...")
    preprocessor.fit(X_train_raw)
    
    # 转换训练集和测试集
    X_train = pd.DataFrame(
        preprocessor.transform(X_train_raw),
    )
    X_test = pd.DataFrame(
        preprocessor.transform(X_test_raw),
    )
    
    # 生成新的特征名称
    feature_names = []
    # 连续特征名称保持不变
    feature_names.extend(continuous_features)
    
    # 获取分类特征的独热编码名称
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names)
    
    # 设置DataFrame的列名
    X_train.columns = feature_names
    X_test.columns = feature_names
    
    # 打印预处理后的数据形状
    print(f"Preprocessed X_train shape: {X_train.shape}")
    print(f"Preprocessed X_test shape: {X_test.shape}")
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
    # 保存预处理器以便在预测时使用
    os.makedirs('model', exist_ok=True)
    with open('model/67_FNN_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to model/67_FNN_preprocessor.pkl")
    
    return X_train, X_test, y_train, y_test, weights_train, weights_test

def main():
    # 读取配置文件
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    os.makedirs('model', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    # Setup logging
    model_name = "67_FNN"
    log_file = f"model/{model_name}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log start time
    logging.info(f"Starting FNN model training with MLX at {datetime.now()}")
    
    # 加载数据
    X_train, X_test, y_train, y_test, weights_train, weights_test = load_data(config)
    logging.info(f"Data loaded: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    
    # Define Optuna objective function
    def objective(trial):
        # 将config传递给custom_scorer函数
        nonlocal config
        
        # Generate hyperparameters
        params = {
            'units_1': trial.suggest_categorical('units_1', [32, 64, 128, 256]),
            'units_2': trial.suggest_categorical('units_2', [16, 32, 64, 128]),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
            'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5),
            'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50,  # 固定值，使用早停
            'patience': trial.suggest_categorical('patience', [5, 10, 15, 20]),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
        }
        
        # 评估模型
        score, history, avg_metrics = custom_scorer(params, X_train.to_numpy(), y_train.to_numpy(), config)
        
        # 如果评估失败，返回最低分数
        if score == float('-inf'):
            return score
        
        # 记录指标
        if avg_metrics:
            for key, value in avg_metrics.items():
                trial.set_user_attr(f"metric_{key}", value)
            trial.set_user_attr("metrics", avg_metrics)
        
        # 记录历史
        if history:
            trial.set_user_attr("history", history)
        
        return score
    
    # 读取n_trials参数
    n_trials = config.get('n_trials', 10)  # 如果未设置，默认为10
    logging.info(f"Starting Optuna optimization with {n_trials} trials")
    print(f"Starting Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction="maximize")
    
    # 使用tqdm包装optimize函数以显示进度
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 获取最佳参数
    best_trial = study.best_trial
    best_params = best_trial.params if hasattr(best_trial, 'params') else None
    
    # 记录最佳参数
    logging.info(f"Best trial: {best_trial.number}")
    logging.info(f"Best value: {best_trial.value}")
    logging.info(f"Best parameters: {best_params}")
    
    # 记录最佳试验的指标
    if 'metrics' in best_trial.user_attrs:
        metrics_str = ', '.join([f"{k}={v:.4f}" for k, v in best_trial.user_attrs['metrics'].items()])
        logging.info(f"Best metrics: {metrics_str}")
        print(f"Best metrics: {metrics_str}")
    
    # 可视化参数重要性（可选）
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_str = ', '.join([f"{k}={v:.4f}" for k, v in importance.items()])
        logging.info(f"Parameter importance: {importance_str}")
        print(f"Parameter importance: {importance_str}")
    except:
        logging.warning("Could not compute parameter importance")
    
    # 训练最终模型
    if best_params is not None:
        try:
            print("\nTraining final model with best parameters...")
            logging.info("Training final model with best parameters")
            
            # 确保best_params包含所有必要的参数
            if 'epochs' not in best_params:
                best_params['epochs'] = 50  # 添加默认的epochs值
            
            print(f"Final model parameters: {best_params}")
            
            # Create final model
            final_model = FNNModel(X_train.shape[1], best_params)
            
            # Split training data for validation during final training
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
            
            # 应用SMOTE过采样到最终训练集
            print("Applying SMOTE oversampling to final training set...")
            smote = SMOTE(random_state=42)
            X_train_final_resampled, y_train_final_resampled = smote.fit_resample(X_train_final, y_train_final)
            
            # 转换为DataFrame以保持一致性
            X_train_final_resampled_df = pd.DataFrame(X_train_final_resampled, columns=X_train.columns)
            y_train_final_resampled_series = pd.Series(y_train_final_resampled)
            
            # 打印SMOTE后的类别分布
            original_class_dist = y_train_final.value_counts().to_dict()
            resampled_class_dist = y_train_final_resampled_series.value_counts().to_dict()
            print(f"Original class distribution: {original_class_dist}")
            print(f"After SMOTE: {resampled_class_dist}")
            
            # 创建最终模型
            print("Training final model with best parameters...")
            input_dim = X_train_final_resampled.shape[1]
            final_model = FNNModel(input_dim, best_params)
            
            # 训练最终模型
            final_model, best_val_auc, _, _ = train_model(
                final_model, 
                X_train_final_resampled, 
                y_train_final_resampled, 
                X_test, 
                y_test, 
                best_params
            )
            
            # Get predictions on test set
            X_test_mx = mx.array(X_test.values.astype(np.float32))
            y_pred_proba = final_model(X_test_mx).tolist()
            
            # 扁平化预测结果（如果需要）
            if isinstance(y_pred_proba[0], list):
                y_pred_proba = [float(item[0]) for item in y_pred_proba]
            else:
                y_pred_proba = [float(item) for item in y_pred_proba]
            
            # 阈值优化
            print("\nOptimizing decision threshold...")
            from sklearn.metrics import f1_score, roc_curve, precision_recall_curve
            
            # 计算不同阈值下的F1分数
            thresholds = np.arange(0.1, 0.9, 0.05)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (np.array(y_pred_proba) > threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                f1_scores.append(f1)
                print(f"Threshold: {threshold:.2f}, F1 Score: {f1:.4f}")
            
            # 找到最佳阈值
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            print(f"Best threshold: {best_threshold:.2f}, F1 Score: {f1_scores[best_threshold_idx]:.4f}")
            
            # 使用最佳阈值生成最终预测
            y_pred = (np.array(y_pred_proba) > best_threshold).astype(int)
            
            # 计算并保存最终指标
            from sklearn.metrics import classification_report
            print("\nFinal classification report:")
            print(classification_report(y_test, y_pred))
            
            # Save model
            with open(f'model/{model_name}_model.pkl', 'wb') as f:
                pickle.dump({
                    'model_state': final_model.parameters(),
                    'model_config': {
                        'input_dim': int(X_train.shape[1]),
                        'params': {k: (int(v) if isinstance(v, (np.int64, np.int32)) else
                                      float(v) if isinstance(v, (np.float64, np.float32)) else
                                      v) for k, v in best_params.items()}
                    },
                    'best_threshold': float(best_threshold)
                }, f)
            logging.info(f"Model saved to model/{model_name}_model.pkl")
            
            # Save parameters
            json_compatible_params = {}
            for k, v in best_params.items():
                if isinstance(v, (np.int64, np.int32)):
                    json_compatible_params[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    json_compatible_params[k] = float(v)
                else:
                    json_compatible_params[k] = v
            
            # 添加最佳阈值到参数
            json_compatible_params['best_threshold'] = float(best_threshold)
                    
            with open(f'model/{model_name}_param.json', 'w') as f:
                json.dump(json_compatible_params, f, indent=4)
            logging.info(f"Parameters saved to model/{model_name}_param.json")
            
            # Save test data and predictions for external evaluation
            pd.DataFrame({
                'y_true': y_test.astype(int).tolist(),
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred.tolist(),
                'weights': weights_test.astype(float).tolist()
            }).to_csv(f'result/{model_name}_predictions.csv', index=False)
            logging.info(f"Test predictions saved to result/{model_name}_predictions.csv")
            
            # 保存基本指标到JSON文件
            metrics = calculate_metrics(y_test, y_pred_proba)
            
            # 添加基于最佳阈值的指标
            y_pred_best = (np.array(y_pred_proba) > best_threshold).astype(int)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            threshold_metrics = {
                'best_threshold': float(best_threshold),
                'accuracy_at_best_threshold': float(accuracy_score(y_test, y_pred_best)),
                'precision_at_best_threshold': float(precision_score(y_test, y_pred_best, zero_division=0)),
                'recall_at_best_threshold': float(recall_score(y_test, y_pred_best, zero_division=0)),
                'f1_at_best_threshold': float(f1_score(y_test, y_pred_best, zero_division=0))
            }
            
            # 合并所有指标
            final_metrics = {**metrics, **threshold_metrics}
            
            # 保存指标（不包含ROC和PR曲线的原始数据）
            with open(f'result/{model_name}_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)
            logging.info(f"Metrics saved to result/{model_name}_metrics.json")
            
            # Save Optuna study for later analysis
            with open(f'model/{model_name}_optuna_study.pkl', 'wb') as f:
                pickle.dump(study, f)
            logging.info(f"Optuna study saved to model/{model_name}_optuna_study.pkl")
            
            print("Done! Model, parameters, and predictions have been saved.")
            
        except Exception as e:
            logging.error(f"Error training final model: {str(e)}")
            print(f"Error training final model: {str(e)}")
    else:
        logging.error("No valid parameters found during optimization")
        print("No valid parameters found during optimization")

if __name__ == "__main__":
    main()
