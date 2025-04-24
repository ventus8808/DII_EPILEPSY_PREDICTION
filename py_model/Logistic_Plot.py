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
# Logistic_Plot.py
# 逻辑回归模型评估与绘图脚本
import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_calibration_curve, plot_decision_curve,
    plot_learning_curve, plot_confusion_matrix, plot_threshold_curve
)

# 读取配置文件（自动适配绝对路径）
project_root = Path(__file__).parent.parent
config_path = project_root / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = project_root / config['data_path']
model_dir = project_root / config['model_dir']
plot_dir = project_root / config['plot_dir']
plot_dir.mkdir(exist_ok=True, parents=True)
plot_data_dir = project_root / 'plot_original_data'
plot_data_dir.mkdir(exist_ok=True, parents=True)

# 加载数据和特征
# 自动适配config.yaml配置
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

# 特征与标签
X = df[features]
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 所有变量转为类别编码（与61_logistic_E_M.py一致）
for col in X.columns:
    X[col] = pd.Categorical(X[col]).codes
# 标签编码
y = pd.Categorical(y).codes if y.dtype.name == 'category' or y.dtype == object else y

# 去除缺失
valid_idx = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X.loc[valid_idx]
y = y.loc[valid_idx]
if weights is not None:
    weights = weights.loc[valid_idx]

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

# 训练并预测（sklearn）
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test, weights_train, weights_test = split_data(X, y, weights)

# 与Logistic_Metrics.py完全一致：添加SMOTE过采样
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 用于显示过采样前后的差异
print(f"SMOTE过采样 - 训练集原始分布: {pd.Series(y_train).value_counts().to_dict()}")
print(f"SMOTE过采样 - 训练集分布平衡后: {pd.Series(y_train_res).value_counts().to_dict()}")

# 误差权重处理与Logistic_Metrics.py保持一致
if weights_train is not None:
    weights_train_res = None  # 过采样后权重设为None
else:
    weights_train_res = None

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_res, y_train_res)

# 预测
try:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
except Exception as e:
    raise RuntimeError(f"模型预测出错: {e}")
model_name = "Logistic"

# 绘图
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_calibration_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_decision_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir)
plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)

print("逻辑回归评估与绘图完成！")
