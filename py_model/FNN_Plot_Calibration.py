import pandas as pd
import pickle
import json
import yaml
import numpy as np
import mlx.core as mx
from pathlib import Path
from model_plot_calibration import plot_calibration_all_data

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

print(f"分类特征: {categorical_features}")
print(f"数值特征: {numeric_features}")

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

# 使用FNN模型进行预测
print("在全量数据集上绘制校准曲线...")
y_pred, y_prob = fnn_predict(model_data, X)

model_name = "FNN"

# 绘制校准曲线
plot_calibration_all_data(y, y_prob, weights, model_name, plot_dir, plot_data_dir, n_bins=30, use_smote=True)

print("FNN校准曲线绘制完成！")
