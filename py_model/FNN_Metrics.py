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
    # 直接从同级目录的FNN_Train模块导入
    import sys
    import os
    # 添加当前目录到系统路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    if current_path not in sys.path:
        sys.path.append(current_path)
    # 直接从FNN_Train导入FNNModel
    from FNN_Train import FNNModel
    
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
model_path = model_dir / 'FNN_best_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# 加载数据
df = pd.read_csv(data_path)

# 加载特征信息和编码器
with open(model_dir / 'FNN_feature_info.json', 'r') as f:
    feature_info = json.load(f)

# 加载独热编码器
with open(model_dir / 'encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# 提取特征信息
categorical_features = feature_info.get('categorical_features', ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'])
numeric_features = feature_info.get('numeric_features', [col for col in ['Age', 'BMI'] if col in df.columns])
# DII_food也是数值特征
features = ['DII_food'] + numeric_features + categorical_features

# 检查特征是否存在
for feature in features:
    if feature not in df.columns:
        print(f"警告: 特征 '{feature}' 不在数据集中")

# 处理数据，应用独热编码
numeric_data = df[['DII_food'] + numeric_features].copy()
categorical_data = df[categorical_features].copy()



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

# 预测
y_pred, y_prob = fnn_predict(model_data, X_test)
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)

# 按照标准格式输出指标
print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'FNN_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)


