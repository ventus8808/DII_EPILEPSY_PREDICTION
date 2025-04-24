import pandas as pd
import pickle
import json
import yaml
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

# 加载模型
model_path = model_dir / 'LightGBM_best_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 加载编码器
encoder_path = model_dir / 'encoder.pkl'
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# 加载数据
df = pd.read_csv(data_path)

# 直接定义类别特征列，确保与训练时一致
categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
features = ['DII_food'] + numeric_features + categorical_features

# 打印特征信息以便调试
print(f"类别特征: {categorical_features}")
print(f"数值特征: {numeric_features}")
print(f"所有特征: {features}")

# 检查所有特征是否在数据集中
for feature in features:
    if feature not in df.columns:
        print(f"警告: 特征 '{feature}' 不在数据集中")

# 处理数据，应用独热编码
numeric_data = df[['DII_food'] + numeric_features].copy()
categorical_data = df[categorical_features].copy()

print(f"类别数据形状: {categorical_data.shape}")

# 应用独热编码
try:
    encoded_cats = encoder.transform(categorical_data)
    print(f"独热编码后形状: {encoded_cats.shape}")
except Exception as e:
    print(f"编码器转换错误: {e}")
    # 如果失败，尝试直接应用独热编码
    from sklearn.preprocessing import OneHotEncoder
    encoder_new = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder_new.fit_transform(categorical_data)

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
def split_data(X, y, weights=None, test_size=0.3, random_state=42, stratify=True):
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
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 计算指标
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
print("\nLightGBM 测试集评估指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'LightGBM_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\n指标已保存至 {metrics_path}")
