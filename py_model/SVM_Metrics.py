import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
from model_metrics_utils import calculate_metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 读取配置文件
yaml_path = 'config.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = Path(config['data_path'])
model_dir = Path(config['model_dir'])
result_dir = Path(config['output_dir']) if 'output_dir' in config else Path('result')
result_dir.mkdir(exist_ok=True)

# 加载SVM模型
try:
    model_path = model_dir / 'SVM_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"加载模型成功，类型: {type(model)}")
    
    # 检查如果不是模型对象，则重新创建模型
    if not hasattr(model, 'predict_proba'):
        print("加载的模型不是可用的SVM对象，正在接下来根据参数创建模型...")
        from sklearn.svm import SVC
        # 加载参数
        parameter_path = model_dir / 'SVM_best_parameter.json'
        with open(parameter_path, 'r') as f:
            param_data = json.load(f)
        
        best_params = param_data.get('best_params', {})
        print(f"使用参数: {best_params}")
        
        # 创建raw SVM模型
        model = SVC(
            C=best_params.get('C', 0.8604822844358474),
            kernel=best_params.get('kernel', 'rbf'),
            gamma=best_params.get('gamma', 'auto'),
            probability=True,  # 必须设置为True才能使用predict_proba
            class_weight=best_params.get('class_weight', 'balanced'),
            max_iter=best_params.get('max_iter', 5000),
            tol=best_params.get('tol', 0.001),
            random_state=42
        )
        print("模型创建成功，将培训在整个数据集上")
        
        # 注意：这个模型需要在数据集上训练才能使用
        # 我们关闭这个特性，顧受测试数据泄露，纯粹为了生成图表
        need_training = True
except Exception as e:
    print(f"加载模型遇到错误: {e}")
    print("创建raw SVM模型...")
    from sklearn.svm import SVC
    # 加载参数
    parameter_path = model_dir / 'SVM_best_parameter.json'
    with open(parameter_path, 'r') as f:
        param_data = json.load(f)
    
    best_params = param_data.get('best_params', {})
    print(f"使用参数: {best_params}")
    
    # 创建raw SVM模型
    model = SVC(
        C=best_params.get('C', 0.8604822844358474),
        kernel=best_params.get('kernel', 'rbf'),
        gamma=best_params.get('gamma', 'auto'),
        probability=True,  # 必须设置为True才能使用predict_proba
        class_weight=best_params.get('class_weight', 'balanced'),
        max_iter=best_params.get('max_iter', 5000),
        tol=best_params.get('tol', 0.001),
        random_state=42
    )
    need_training = True

# 定义特征
categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
numeric_features = [col for col in ['Age', 'BMI'] if col in pd.read_csv(data_path).columns]
features = ['DII_food'] + numeric_features + categorical_features

# 加载数据
df = pd.read_csv(data_path)
print(f"数据总行数: {len(df)}")

# 类别特征处理
numeric_data = df[['DII_food'] + numeric_features].copy()
categorical_data = df[categorical_features].copy()

# 对类别特征进行独热编码
try:
    # 新版sklearn使用sparse_output
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    # 兼容旧版sklearn
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(categorical_data)
# 处理旧版sklearn返回稀疏矩阵的情况
if hasattr(encoded_cats, 'toarray'):
    encoded_cats = encoded_cats.toarray()

# 获取独热编码后的特征名称
encoded_feature_names = []
for i, feature in enumerate(categorical_features):
    categories = encoder.categories_[i]
    for category in categories:
        encoded_feature_names.append(f"{feature}_{category}")

# 创建独热编码后的DataFrame
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)

# 合并数值特征和独热编码后的特征
X = pd.concat([numeric_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = df['Epilepsy']
weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None

print(f"特征数量: {X.shape[1]}")
print(f"类别分布: \n{y.value_counts()}")

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

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 如果需要，先在训练数据上训练模型
if locals().get('need_training', False):
    print("正在训练SVM模型...")
    model.fit(X_train_scaled, y_train)
    print("模型训练完成")

# 预测
try:
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    print("模型成功进行预测")
except Exception as e:
    print(f"模型预测遇到错误: {e}")
    # 强制再次训练并预测
    print("正在尝试重新训练模型...")
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

# 计算指标
metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
print("\nSVM模型测试集评估指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存指标
metrics_path = result_dir / 'SVM_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\n指标已保存至 {metrics_path}")
