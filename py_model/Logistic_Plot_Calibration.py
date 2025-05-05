import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from model_plot_calibration import plot_calibration_all_data

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

print(f"成功加载特征信息，模型使用的特征数：{len(features)}")

# 训练逻辑回归模型
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X, y)

# 获取预测概率
y_prob = model.predict_proba(X)[:, 1]

print("在全量数据集上绘制校准曲线...")
# 绘制校准曲线
plot_calibration_all_data(y, y_prob, weights, "Logistic", plot_dir, plot_data_dir, n_bins=30, use_smote=True)

print("逻辑回归校准曲线绘制完成！")
