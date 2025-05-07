import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve,
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
model_path = model_dir / 'XGBoost_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 加载数据
df = pd.read_csv(data_path)

# 尝试加载特征信息文件，如果不存在则使用默认特征
try:
    with open(model_dir / 'XGBoost_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    features = feature_info['features']
    print(f"成功加载特征信息文件，模型使用的特征数：{len(features)}")
except FileNotFoundError:
    print("未找到特征信息文件，使用默认特征...")
    # 检查是否有exposure和outcome配置
    outcome = config.get('outcome', 'Epilepsy')
    exposure = config.get('exposure', 'DII_food')
    covariates = config.get('covariates', ["Gender", "Age", "BMI", "Education", "Marriage", "Smoke", "Alcohol", "Employment", "ActivityLevel"])
    # 与XGBoost原始模型保持一致，添加数值特征
    numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
    features = [exposure] + numeric_features + covariates
    print(f"使用默认特征：{features}")

# 确保所有需要的特征都在数据集中
valid_features = [f for f in features if f in df.columns]
if len(valid_features) != len(features):
    print(f"警告：部分特征不在数据集中，仅使用有效特征。原始特征数：{len(features)}，有效特征数：{len(valid_features)}")

# 注意XGBoost模型特别依赖特征名称和顺序
print("注意：XGBoost模型特别依赖特征名称和顺序，直接使用原始数据集进行评估...")

outcome = config.get('outcome', 'Epilepsy')
y = df[outcome]
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

# 创建分割数据的函数
def split_data(y, weights=None, test_size=0.3, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split
    stratify_param = y if stratify else None
    
    # 创建索引数组
    indices = np.arange(len(y))
    
    if weights is not None:
        train_indices, test_indices, y_train, y_test, weights_train, weights_test = train_test_split(
            indices, y, weights, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    else:
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        weights_train = weights_test = None
    
    return train_indices, test_indices, y_train, y_test, weights_train, weights_test

# 分割数据
train_indices, test_indices, y_train, y_test, weights_train, weights_test = split_data(y, weights)

# 使用索引筛选测试集进行预测
# 对于XGBoost，我们需要使用原始全量数据，然后将模型应用于测试集索引
# 先获取测试集
df_test = df.iloc[test_indices]

try:
    # 直接预测
    y_pred = model.predict(df_test)
    y_prob = model.predict_proba(df_test)[:, 1]
    print("成功直接预测")
except Exception as e:
    print(f"直接预测失败: {e}")
    
    # 尝试获取模型的预处理器
    try:
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            print("使用模型预处理器...")
            preprocessor = model.named_steps['preprocessor']
            # 获取预处理器的特征名称
            if hasattr(preprocessor, 'feature_names_in_'):
                feature_names = preprocessor.feature_names_in_
                print(f"读取到模型的特征名称: {feature_names}")
                # 仅选择模型需要的列
                if all(f in df.columns for f in feature_names):
                    X_test = df_test[feature_names]
                    
                    # 使用预处理器转换
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]
                        print("使用模型原始特征预测成功")
                    else:
                        raise Exception("模型没有predict方法")
                else:
                    missing_features = [f for f in feature_names if f not in df.columns]
                    print(f"缺失特征: {missing_features}")
                    raise Exception(f"数据缺失模型需要的特征: {missing_features}")
            else:
                raise Exception("预处理器没有feature_names_in_")
        else:
            raise Exception("模型没有预处理器或预处理器不可访问")
    except Exception as e:
        print(f"预处理器方法失败: {e}")
        
        # 作为最后的尝试，使用所有可能的特征
        print("尝试使用所有可能的特征...")
        try:
            # 定义常见特征集
            potential_features = ["DII_food", "Age", "BMI", "Gender", "Education", "Marriage", "Smoke", "Alcohol", "Employment", "ActivityLevel"]
            available_features = [f for f in potential_features if f in df.columns]
            X_test = df_test[available_features]
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            print(f"使用以下特征成功预测: {available_features}")
        except Exception as e:
            print(f"所有尝试都失败，无法进行预测: {e}")
            print("运行失败。请先运行XGBoost_Train.py生成模型和特征信息文件")
            import sys
            sys.exit(1)

# 模型名称
model_name = "XGBoost"

# 绘图
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
# 校准曲线和决策曲线已移除
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
try:
    plot_learning_curve(model, df.iloc[train_indices], y_train, df.iloc[test_indices], y_test, model_name, plot_dir, plot_data_dir)
    print("学习曲线绘制成功")
except Exception as e:
    print(f"学习曲线绘制失败: {e}")
plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
