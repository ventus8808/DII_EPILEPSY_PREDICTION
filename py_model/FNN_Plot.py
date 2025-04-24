import pandas as pd
import pickle
import json
import yaml
import numpy as np
import mlx.core as mx
from pathlib import Path
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_calibration_curve, plot_decision_curve,
    plot_learning_curve, plot_confusion_matrix, plot_threshold_curve
)

# 读取配置文件
yaml_path = 'config.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(exist_ok=True)
plot_data_dir = Path('plot_original_data')
plot_data_dir.mkdir(exist_ok=True)

# 加载FNN模型
model_path = model_dir / 'FNN_best_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# FNN模型预测函数
def fnn_predict(model_data, X):
    from py_model.FNN_Train import FNNModel  # 导入FNN模型类
    
    # 重建FNN模型
    model_config = model_data['model_config']
    model = FNNModel(model_config['input_dim'], model_config['params'])
    
    # 加载模型参数
    model.update(model_data['model_state'])
    
    # 预处理数据
    if 'scaler' in model_data:
        X = model_data['scaler'].transform(X)
    
    # 转换为mx.array并进行预测
    X_mx = mx.array(X.astype(np.float32))
    y_prob = model(X_mx).tolist()
    
    # 如果y_prob是嵌套列表，则展平
    if isinstance(y_prob[0], list):
        y_prob = [item[0] for item in y_prob]
    
    # 二分类预测
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    
    return y_pred, np.array(y_prob)

# 加载数据
df = pd.read_csv(data_path)

# 加载特征信息
with open(model_dir / 'FNN_feature_info.json', 'r') as f:
    feature_info = json.load(f)

features = feature_info['features']
categorical_features = feature_info.get('categorical_features', [])
numeric_features = feature_info.get('numeric_features', [])

X = df[features]
y = df['Epilepsy']
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 数据分割
def split_data(X, y, weights=None, test_size=0.2, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split
    stratify_param = y if stratify else None
    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        weights_train = weights_test = None
    return X_train, X_test, y_train, y_test, weights_train, weights_test

X_train, X_test, y_train, y_test, weights_train, weights_test = split_data(X, y, weights)

# 使用FNN模型进行预测
y_pred, y_prob = fnn_predict(model_data, X_test)

model_name = "FNN"

# 绘图
print("绘制ROC曲线...")
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

print("绘制PR曲线...")
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

print("绘制校准曲线...")
plot_calibration_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

print("绘制决策曲线...")
plot_decision_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

print("绘制混淆矩阵...")
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)

# 为FNN模型实现自定义学习曲线功能
print("绘制学习曲线...")

def custom_fnn_learning_curve(model_data, X, y, weights=None, cv=5, train_sizes=None, random_state=42, plot_dir=None, plot_data_dir=None):
    """为FNN模型自定义学习曲线实现"""
    from sklearn.model_selection import StratifiedKFold
    from py_model.FNN_Train import FNNModel, train_model, evaluate
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    
    # 如果没有指定训练集大小比例，使用默认值
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    
    # 设置交叉验证
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # 初始化结果数组
    train_scores = []
    test_scores = []
    train_sizes_abs = []
    
    # 模型配置
    model_config = model_data['model_config']
    params = model_config['params']
    input_dim = model_config['input_dim']
    
    # 对每个指定的训练集大小
    for train_size in train_sizes:
        print(f"\n训练集大小: {train_size:.2f}...")
        fold_train_scores = []
        fold_test_scores = []
        fold_train_sizes = []
        
        # 对每个CV折叠
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            print(f"\t折叠 {fold+1}/{cv}...")
            # 切分数据
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            if weights is not None:
                weights_train_fold = weights.iloc[train_idx]
                weights_test_fold = weights.iloc[test_idx]
            else:
                weights_train_fold = weights_test_fold = None
            
            # 计算当前训练大小
            train_size_abs = int(len(X_train_fold) * train_size)
            train_size_abs = max(train_size_abs, 10)  # 至少使用10个样本
            train_size_abs = min(train_size_abs, len(X_train_fold))  # 不超过数据集大小
            
            # 子采样训练集
            train_subset_idx = np.random.choice(len(X_train_fold), train_size_abs, replace=False)
            X_train_subset = X_train_fold.iloc[train_subset_idx]
            y_train_subset = y_train_fold.iloc[train_subset_idx]
            
            if weights_train_fold is not None:
                weights_train_subset = weights_train_fold.iloc[train_subset_idx]
            else:
                weights_train_subset = None
            
            # 预处理数据
            if 'scaler' in model_data:
                X_train_subset_scaled = model_data['scaler'].transform(X_train_subset)
                X_test_fold_scaled = model_data['scaler'].transform(X_test_fold)
            else:
                X_train_subset_scaled = X_train_subset.values
                X_test_fold_scaled = X_test_fold.values
            
            # 创建并训练模型
            try:
                # 创建模型
                model = FNNModel(input_dim, params)
                
                # 设置简化的训练参数 - 避免冗长训练
                training_params = params.copy()
                training_params['epochs'] = 30  # 减少迭代次数加快学习曲线生成
                
                # 训练模型
                model, _, _, _ = train_model(
                    model, 
                    X_train_subset_scaled, 
                    y_train_subset.values, 
                    X_test_fold_scaled, 
                    y_test_fold.values, 
                    training_params
                )
                
                # 计算训练集分数
                X_train_mx = mx.array(X_train_subset_scaled.astype(np.float32))
                y_train_mx = mx.array(y_train_subset.values.astype(np.float32))
                train_eval = evaluate(model, X_train_mx, y_train_mx)
                train_score = train_eval['auc']
                
                # 计算测试集分数
                X_test_mx = mx.array(X_test_fold_scaled.astype(np.float32))
                y_test_mx = mx.array(y_test_fold.values.astype(np.float32))
                test_eval = evaluate(model, X_test_mx, y_test_mx)
                test_score = test_eval['auc']
                
                # 添加结果
                fold_train_scores.append(train_score)
                fold_test_scores.append(test_score)
                fold_train_sizes.append(train_size_abs)
                
                print(f"\t\t训练分数: {train_score:.4f}, 测试分数: {test_score:.4f}")
            except Exception as e:
                print(f"\t\t错误: {str(e)}")
                # 跳过失败的折叠
                continue
        
        # 平均每个折叠的结果
        if fold_train_scores:
            train_scores.append(np.mean(fold_train_scores))
            test_scores.append(np.mean(fold_test_scores))
            train_sizes_abs.append(np.mean(fold_train_sizes))
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.title(f"FNN Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score (AUC-ROC)")
    plt.grid()
    
    plt.plot(train_sizes_abs, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes_abs, test_scores, 'o-', color="g", label="Cross-validation score")
    
    plt.fill_between(
        train_sizes_abs, 
        train_scores,
        test_scores, 
        alpha=0.1, 
        color="g"
    )
    
    plt.legend(loc="best")
    plt.tight_layout()
    
    # 保存图表
    if plot_dir:
        plt.savefig(plot_dir / f"FNN_learning_curve.png", dpi=300, bbox_inches='tight')
        plt.savefig(plot_dir / f"FNN_learning_curve.svg", format='svg', bbox_inches='tight')
    
    # 保存原始数据
    if plot_data_dir:
        data_to_save = {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'test_scores': test_scores,
        }
        with open(plot_data_dir / f"FNN_learning_curve_data.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)
    
    plt.close()
    return train_sizes_abs, train_scores, test_scores

# 调用自定义学习曲线函数
train_sizes, train_scores, test_scores = custom_fnn_learning_curve(
    model_data, 
    X_train, 
    y_train, 
    weights_train, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42,
    plot_dir=plot_dir,
    plot_data_dir=plot_data_dir
)

print("绘制阈值曲线...")
plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)

print("所有图表已生成完毕。")
