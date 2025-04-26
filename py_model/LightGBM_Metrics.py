import pandas as pd
import pickle
import json
import yaml
import numpy as np
from pathlib import Path
from model_metrics_utils import calculate_metrics
from sklearn.model_selection import train_test_split

# 读取配置文件
yaml_path = 'config.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
result_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
result_dir.mkdir(exist_ok=True)

# 检查模型和编码器文件
model_path = model_dir / 'LightGBM_best_model.pkl'
encoder_path = model_dir / 'encoder.pkl'

if not model_path.exists():
    print(f"错误: 未找到模型文件 {model_path}")
    print("请先运行LightGBM_Train.py")
    exit(1)

if not encoder_path.exists():
    print(f"错误: 未找到编码器文件 {encoder_path}")
    print("请先运行LightGBM_Train.py")
    exit(1)

# 加载模型和编码器
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# 加载原始数据
df = pd.read_csv(data_path)

print("注意：LightGBM模型使用Pipeline处理原始数据，与XGBoost保持一致...")

# 从特征信息文件加载特征列表
try:
    with open(model_dir / 'LightGBM_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    numeric_features = feature_info.get('numeric_features', ['DII_food', 'Age', 'BMI'])
    categorical_features = feature_info.get('categorical_features', ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'])
    features = numeric_features + categorical_features
    print(f"从特征信息文件读取到: {features}")
except FileNotFoundError:
    print("未找到特征信息文件，使用默认特征列表")
    numeric_features = ['DII_food', 'Age', 'BMI']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
    features = numeric_features + categorical_features

# 检查是否有outcome配置
outcome = config.get('outcome', 'Epilepsy')
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 使用保存的索引进行数据分割以保持一致性
def split_data(y, weights=None, test_size=0.3, random_state=42, stratify=True):
    """与XGBoost_Metrics.py保持一致的分割函数"""
    stratify_param = y if stratify else None
    
    # 创建索引数组
    indices = np.arange(len(y))
    
    if weights is not None:
        train_indices, test_indices, y_train, y_test, weights_train, weights_test = train_test_split(
            indices, y, weights, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    else:
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        weights_train = weights_test = None
    
    return train_indices, test_indices, y_train, y_test, weights_train, weights_test

# 尝试加载训练脚本保存的索引
try:
    # 使用保存的数据划分索引
    indices_path = model_dir / 'LightGBM_train_test_indices.json'
    if indices_path.exists():
        print("使用保存的训练/测试索引来得到与训练脚本完全相同的测试集...")
        with open(indices_path, 'r') as f:
            indices_data = json.load(f)
        test_indices = indices_data['test_indices']
        # 直接使用保存的测试集索引
        y_test = y.iloc[test_indices]
        weights_test = weights.iloc[test_indices] if weights is not None else None
    else:
        # 如果无法找到保存的索引，则在运行中分割
        print("未找到保存的索引，使用相同的随机种子进行数据分割...")
        train_indices, test_indices, y_train, y_test, weights_train, weights_test = split_data(y, weights)

    # 选取测试集
    df_test = df.iloc[test_indices]
except FileNotFoundError:
    # 回退到基本的分割方法
    print("回退到基本的分割方法...")
    train_indices, test_indices, y_train, y_test, weights_train, weights_test = split_data(y, weights)
    df_test = df.iloc[test_indices]



# 直接使用加载的模型进行预测
try:
    # 首先确定要使用的特征列表
    if 'feature_order' in feature_info:
        feature_cols = feature_info['feature_order']
    else:
        feature_cols = numeric_features + categorical_features
        
    # 直接使用测试集预测
    test_features = df_test[feature_cols]
    y_pred = model.predict(test_features)
    y_prob = model.predict_proba(test_features)[:, 1]
    print("成功使用模型直接预测")
except Exception as e:
    print(f"预测失败: {e}")
    print("运行失败。请先运行LightGBM_Train.py生成正确格式的模型")
    import sys
    sys.exit(1)

# 计算测试集指标
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)

# 输出指标
print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


# 保存指标
metrics_path = result_dir / 'LightGBM_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

