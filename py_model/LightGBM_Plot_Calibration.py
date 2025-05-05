import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
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

# 加载模型
model_path = model_dir / 'LightGBM_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 加载数据
df = pd.read_csv(data_path)

# 检查是否有exposure和outcome配置
outcome = config.get('outcome', 'Epilepsy')
exposure = config.get('exposure', 'DII_food')
covariates = config.get('covariates', ["Gender", "Age", "BMI", "Education", "Marriage", "Smoke", "Alcohol", "Employment", "ActivityLevel"])

# 尝试加载特征信息
try:
    with open(model_dir / 'LightGBM_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    features = feature_info['features']
    print(f"成功加载特征信息文件，模型使用的特征数：{len(features)}")
except FileNotFoundError:
    print("未找到特征信息文件，使用默认特征...")
    features = [exposure] + covariates
    print(f"使用默认特征：{features}")

# 确保所有需要的特征都在数据集中
valid_features = [f for f in features if f in df.columns]
if len(valid_features) != len(features):
    print(f"警告：部分特征不在数据集中，仅使用有效特征。原始特征数：{len(features)}，有效特征数：{len(valid_features)}")

# LightGBM特别依赖特征名称和顺序，需要特别处理
print("注意：LightGBM模型特别依赖特征名称和顺序，直接使用原始数据集进行评估...")

# 准备特征和目标变量
X = df[valid_features]
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 使用模型预测全量数据的概率
y_prob = model.predict_proba(X)[:, 1]
model_name = "LightGBM"

print("在全量数据集上绘制校准曲线...")
print("=== 全量数据校准曲线计算进度 === ")
# 在内部使用SMOTE过采样平衡数据集
plot_calibration_all_data(y, y_prob, weights, model_name, plot_dir, plot_data_dir, use_smote=True)
print("校准曲线绘制完成！")
