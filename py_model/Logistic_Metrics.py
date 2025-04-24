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

print(f"\n特征列表: {list(X.columns)}")
print(f"标签分布: {pd.Series(y).value_counts().to_dict()}")

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

# 手动计算关键指标
manual_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
manual_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
manual_youden = manual_sensitivity + manual_specificity - 1

# 计算自动metrics
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)

# 打印对比信息
print("\n混淆矩阵:")
print(f"[{tn}, {fp}]\n[{fn}, {tp}]")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# 避免f-string中使用带引号的键名
youden_key = "Youden's Index"
f1_key = "F1 Score"

print("\n手算对比指标:")
print(f"Manual Sensitivity (Recall): {manual_sensitivity:.4f} | Metrics: {metrics['Sensitivity']:.4f}")
print(f"Manual Specificity: {manual_specificity:.4f} | Metrics: {metrics['Specificity']:.4f}")
print(f"Manual Precision: {manual_precision:.4f} | Metrics: {metrics['Precision']:.4f}")
print(f"Manual NPV: {manual_npv:.4f} | Metrics: {metrics['NPV']:.4f}")
print(f"Manual Accuracy: {manual_accuracy:.4f} | Metrics: {metrics['Accuracy']:.4f}")
print(f"Manual F1: {manual_f1:.4f} | Metrics: {metrics[f1_key]:.4f}")
print(f"Manual Youden Index: {manual_youden:.4f} | Metrics: {metrics[youden_key]:.4f}")

# 打印SMOTE影响
print("\n训练集大小和分布:")
print(f"X_train shape: {X_train.shape} | X_train_res shape: {X_train_res.shape}")
print(f"y_train distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"y_train_res distribution: {pd.Series(y_train_res).value_counts().to_dict()}")

# 打印测试集大小和分布
print("\n测试集大小和分布:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test distribution: {pd.Series(y_test).value_counts().to_dict()}")
print(f"y_pred distribution: {pd.Series(y_pred).value_counts().to_dict()}")

# 生成不同ROC阈值下的结果
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
print(f"\n最佳阈值 (Youden): {best_threshold:.4f}")

# 在最佳阈值下的预测
y_pred_best = (y_prob >= best_threshold).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best)
tn_best, fp_best, fn_best, tp_best = cm_best.ravel()

print(f"\n最佳阈值下的混淆矩阵:")
print(f"[{tn_best}, {fp_best}]\n[{fn_best}, {tp_best}]")

# 标准指标输出
print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'Logistic_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"\n指标已保存至: {metrics_path}")
