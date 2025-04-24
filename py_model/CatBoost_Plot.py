import pandas as pd
import pickle
import json
import yaml
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
model_name = "CatBoost"

# 绘图
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_calibration_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_decision_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir)
plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
