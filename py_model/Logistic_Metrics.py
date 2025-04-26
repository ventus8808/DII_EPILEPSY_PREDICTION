import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from model_metrics_utils import calculate_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 读取配置文件
project_root = Path(__file__).parent.parent
config_path = project_root / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = project_root / config['data_path']
model_dir = project_root / config['model_dir']
result_dir = project_root / (config['output_dir'] if 'output_dir' in config else 'result')
result_dir.mkdir(exist_ok=True, parents=True)

# 自动适配特征和标签
if 'covariates' in config:
    features = [config['exposure']] + config['covariates']
else:
    features = [config.get('exposure', 'DII_food')] + [
        'Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
        'Alcohol', 'Employment', 'ActivityLevel'
    ]
outcome = config.get('outcome', 'Epilepsy')

# 加载数据
if not data_path.exists():
    raise FileNotFoundError(f"未找到数据文件: {data_path}")
df = pd.read_csv(data_path)
for var in features + [outcome]:
    if var not in df.columns:
        raise ValueError(f"数据缺少列: {var}")

X = df[features]
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 与Logistic_Plot.py保持一致的特征处理：类别编码而非独热编码
# 所有变量转为类别编码（与61_logistic_E_M.py和Logistic_Plot.py一致）
for col in X.columns:
    X[col] = pd.Categorical(X[col]).codes

# 标签编码
if y.dtype.name == 'category' or y.dtype == object:
    y = pd.Categorical(y).codes



# 去除缺失
valid_idx = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X.loc[valid_idx]
y = y.loc[valid_idx]
if weights is not None:
    weights = weights.loc[valid_idx]

# 数据分割
def split_data(X, y, weights=None, test_size=0.3, random_state=42, stratify=True):
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

# SMOTE过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
if weights_train is not None:
    # 过采样后权重全部设为1（如需更复杂权重策略可自定义）
    weights_train_res = None
else:
    weights_train_res = None

# 训练模型
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_res, y_train_res)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 生成混淆矩阵以手动校验
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()



# 计算自动metrics
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()



# 标准指标输出
print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'Logistic_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
