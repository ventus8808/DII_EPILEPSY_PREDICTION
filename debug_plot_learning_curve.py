# debug_plot_learning_curve.py
from py_model.CatBoost_Plot import plot_learning_curve
import pickle
import pandas as pd
import json
from pathlib import Path

# 路径与文件
with open('config.yaml', 'r') as f:
    config = json.load(f) if 'json' in f.name else None
    if config is None:
        f.seek(0)
        import yaml
        config = yaml.safe_load(f)
model_dir = Path(config['model_dir'])
data_path = Path(config['data_path'])

with open(model_dir / 'CatBoost_model.pkl', 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv(data_path)
with open(model_dir / 'CatBoost_feature_info.json', 'r') as f:
    feature_info = json.load(f)
features = feature_info['features']
X = df[features]
y = df['Epilepsy']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"n_trees: {getattr(model, 'tree_count_', None) or model.get_tree_count()}")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
plot_learning_curve(model, X_train, y_train, X_test, y_test)
print("plot_learning_curve finished")
