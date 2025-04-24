import pandas as pd
import pickle
import json
import yaml
import numpy as np
import mlx.core as mx
from pathlib import Path
from model_metrics_utils import calculate_metrics

# 读取配置文件
yaml_path = 'config.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
result_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
result_dir.mkdir(exist_ok=True)

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

# 加载模型
print("加载FNN模型...")
model_path = model_dir / 'FNN_best_model.pkl'
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"成功加载模型: {model_path}")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 {model_path}")
    print("请先运行FNN_Train.py训练模型")
    exit(1)

# 加载数据
print("加载数据...")
df = pd.read_csv(data_path)
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

print("分割数据...")
X_train, X_test, y_train, y_test, weights_train, weights_test = split_data(X, y, weights)
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
print(f"测试集正例比例: {y_test.mean():.4f}")

# 预测
print("进行预测...")
y_pred, y_prob = fnn_predict(model_data, X_test)

# 计算指标
print("计算评估指标...")
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
print("\n测试集指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'FNN_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"指标已保存到: {metrics_path}")

# 检查硬约束
print("\n检查硬约束...")
with open('config.yaml', 'r') as f:
    constraints = yaml.safe_load(f).get('test_constraints', {})

failed_constraints = []
if 'AUC_min' in constraints and metrics['AUC-ROC'] < constraints['AUC_min']:
    failed_constraints.append(f"AUC-ROC {metrics['AUC-ROC']:.4f} < {constraints['AUC_min']}")
    
if 'AUC_max' in constraints and metrics['AUC-ROC'] > constraints['AUC_max']:
    failed_constraints.append(f"AUC-ROC {metrics['AUC-ROC']:.4f} > {constraints['AUC_max']}")
    
if 'Sensitivity_min' in constraints and metrics['Sensitivity'] < constraints['Sensitivity_min']:
    failed_constraints.append(f"Sensitivity {metrics['Sensitivity']:.4f} < {constraints['Sensitivity_min']}")
    
if 'Specificity_min' in constraints and metrics['Specificity'] < constraints['Specificity_min']:
    failed_constraints.append(f"Specificity {metrics['Specificity']:.4f} < {constraints['Specificity_min']}")

if failed_constraints:
    print("模型未通过以下硬约束:")
    for constraint in failed_constraints:
        print(f"- {constraint}")
else:
    print("模型通过所有硬约束！")
