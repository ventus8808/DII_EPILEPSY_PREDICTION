import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# 加载模型、标准化器和编码器
model_path = model_dir / 'SVM_best_model.pkl'
scaler_path = model_dir / 'scaler.pkl'  # 使用通用标准化器
encoder_path = model_dir / 'encoder.pkl'  # 使用通用编码器

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"成功加载SVM模型")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"成功加载标准化器")
    
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print(f"成功加载特征编码器")
except FileNotFoundError as e:
    print(f"文件不存在: {str(e)}")
    print("请先运行SVM_Train.py生成模型和相关文件")
    import sys
    sys.exit(1)

# 加载数据
df = pd.read_csv(data_path)

# 尝试加载特征信息文件，如果不存在则使用默认特征
# 首先检查是否有exposure和outcome配置
outcome = config.get('outcome', 'Epilepsy')
exposure = config.get('exposure', 'DII_food')
categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]

# 尝试加载SVM特定的特征信息
try:
    with open(model_dir / 'SVM_feature_info.json', 'r') as f:
        feature_info = json.load(f)
        features = feature_info['features']
        categorical_features = feature_info.get('categorical_features', categorical_features)
        numeric_features = feature_info.get('numeric_features', numeric_features)
    print(f"成功加载特征信息文件，模型使用的特征数：{len(features)}")
except FileNotFoundError:
    print("未找到特征信息文件，使用默认特征...")
    features = [exposure] + numeric_features + categorical_features
    print(f"使用默认特征：{features}")

# 确保所有需要的特征都在数据集中
valid_features = [f for f in features if f in df.columns]
if len(valid_features) != len(features):
    print(f"警告：部分特征不在数据集中，仅使用有效特征。原始特征数：{len(features)}，有效特征数：{len(valid_features)}")

# 准备数据
X_raw = df[valid_features]
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 数据预处理：分离类别特征和数值特征
numeric_features = [col for col in ['DII_food', 'Age', 'BMI'] if col in X_raw.columns]
categorical_features = [col for col in valid_features if col not in numeric_features]

# 应用独热编码
numeric_data = X_raw[numeric_features]
if categorical_features:
    categorical_data = X_raw[categorical_features]
    # 使用加载的编码器进行转换
    encoded_array = encoder.transform(categorical_data)
    # 如果是稀疏矩阵则转换为数组
    if hasattr(encoded_array, 'toarray'):
        encoded_array = encoded_array.toarray()
    # 获取特征名称
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names)
    # 合并数值特征和独热编码特征
    X = pd.concat([numeric_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    print(f"特征处理完成。处理后特征数量：{X.shape[1]}")
else:
    X = numeric_data
    print("没有类别特征需要编码")

# 标准化所有特征
X_scaled = scaler.transform(X)

# 执行预测
try:
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    print("成功预测")
except Exception as e:
    # 如果模型对象不支持predict_proba，尝试使用decision_function
    print(f"使用predict_proba预测失败: {e}")
    print("尝试使用decision_function...")
    try:
        y_scores = model.decision_function(X_scaled)
        # 归一化decision scores到[0,1]区间
        y_scores_normalized = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        y_prob = y_scores_normalized
        print("成功使用decision_function预测")
    except Exception as e2:
        print(f"使用decision_function也失败: {e2}")
        print("无法获取预测概率，退出")
        import sys
        sys.exit(1)

model_name = "SVM"

print("在全量数据集上绘制校准曲线...")
# 绘制校准曲线
plot_calibration_all_data(y, y_prob, weights, model_name, plot_dir, plot_data_dir, n_bins=30, use_smote=True)

print("SVM校准曲线绘制完成！")
