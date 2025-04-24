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
model_path = model_dir / 'CatBoost_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
# 加载数据
df = pd.read_csv(data_path)
with open(model_dir / 'CatBoost_feature_info.json', 'r') as f:
    feature_info = json.load(f)
features = feature_info['features']
cat_feature_indices = feature_info['cat_feature_indices']
X = df[features]
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
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 计算指标
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
# 保存指标
metrics_path = result_dir / 'CatBoost_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
