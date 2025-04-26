import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_calibration_curve, plot_decision_curve,
    plot_learning_curve, plot_confusion_matrix, plot_threshold_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    print(f"模型类型: {type(model)}")
    
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

# 对测试数据进行标准化处理
X_test_scaled = scaler.transform(X_test)

# 尝试从不同文件加载模型
if isinstance(model, np.ndarray) or not hasattr(model, 'predict'):
    print("加载的模型对象不能直接使用，尝试加载【RF_best_model.pkl】...")
    try:
        with open(model_dir / 'SVM_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"成功加载 SVM_model.pkl, 模型类型: {type(model)}")
    except FileNotFoundError:
        print("SVM_model.pkl 不存在，使用原模型")

# 预测
try:
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    print("成功预测")
except Exception as e:
    print(f"预测失败: {e}")
    print("预测特征形状:", X_test.shape)
    print("标准化后特征形状:", X_test_scaled.shape)
    print("模型类型和属性:")
    print(f"Type: {type(model)}")
    if hasattr(model, 'shape'):
        print(f"Shape: {model.shape}")
    print("尝试手动创建SVM模型...")
    
    # 如果上述尝试都失败，尝试创建一个新的SVM模型
    from sklearn.svm import SVC
    new_model = SVC(probability=True, random_state=42)
    # 加载有效参数
    try:
        with open(model_dir / 'SVM_best_parameter.json', 'r') as f:
            params_data = json.load(f)
        print("成功加载模型参数:", params_data)
        # 参数可能嵌套在best_params中
        if 'best_params' in params_data:
            params = params_data['best_params']
        else:
            params = params_data
            
        # 确保参数键值有效
        valid_params = {}
        for k, v in params.items():
            if k in ['C', 'kernel', 'gamma', 'class_weight', 'max_iter', 'tol', 'degree', 'coef0', 'shrinking']:
                valid_params[k] = v
        
        print("使用有效参数:", valid_params)
        # 创建一个新的SVM模型
        new_model.set_params(**valid_params)
    except Exception as param_e:
        print(f"加载参数失败: {param_e}")
    
    # 尝试使用一个小样本训练并预测
    print("使用小样本训练SVM模型...")
    X_train_small = X_train[:min(500, len(X_train))]
    y_train_small = y_train[:min(500, len(y_train))]
    try:
        new_model.fit(scaler.transform(X_train_small), y_train_small)
        y_pred = new_model.predict(X_test_scaled)
        y_prob = new_model.predict_proba(X_test_scaled)[:, 1]
        print("使用临时训练的模型进行预测")
        model = new_model  # 使用新模型继续
    except Exception as train_e:
        print(f"使用临时模型失败: {train_e}")
        print("运行失败。请先运行SVM_Train.py生成模型和特征信息文件")
        import sys
        sys.exit(1)

# 模型名称
model_name = "SVM"

# 绘图
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_calibration_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_decision_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)

# 尝试调用学习曲线函数
try:
    # 对于SVM，我们需要使用原始数据和标准化处理
    # 创建一个包含编码和标准化预处理步骤的Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # 由于我们已经对数据进行了预处理，这里简化为只包含标准化器和模型的管道
    pipeline_model = Pipeline([
        ('scaler', scaler),
        ('svm', model)
    ])
    plot_learning_curve(pipeline_model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir)
    print("学习曲线绘制成功")
except Exception as e:
    print(f"学习曲线绘制失败: {e}")

plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
