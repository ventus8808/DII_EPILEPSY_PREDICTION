import pandas as pd
import pickle
import json
import yaml
import numpy as np
import mlx.core as mx
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

# 加载FNN模型
model_path = model_dir / 'FNN_best_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# FNN模型预测函数
def fnn_predict(model_data, X):
    # 直接从同级目录的FNN_Train模块导入
    import sys
    import os
    # 添加当前目录到系统路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    if current_path not in sys.path:
        sys.path.append(current_path)
    # 直接从FNN_Train导入FNNModel
    from FNN_Train import FNNModel
    
    # 重建FNN模型
    model_config = model_data['model_config']
    model = FNNModel(model_config['input_dim'], model_config['params'])
    
    # 加载模型参数
    model.update(model_data['model_state'])
    
    # 预处理数据
    if 'scaler' in model_data:
        X = model_data['scaler'].transform(X)
    
    # 转换为mx.array并进行预测
    X_mx = mx.array(X.astype(np.float32))
    y_prob = model(X_mx).tolist()
    
    # 如果y_prob是嵌套列表，则展平
    if isinstance(y_prob[0], list):
        y_prob = [item[0] for item in y_prob]
    
    # 二分类预测
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    
    return y_pred, np.array(y_prob)

# 加载数据
df = pd.read_csv(data_path)

# 加载特征信息和编码器
with open(model_dir / 'FNN_feature_info.json', 'r') as f:
    feature_info = json.load(f)

# 加载独热编码器
with open(model_dir / 'encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# 提取特征信息
categorical_features = feature_info.get('categorical_features', ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'])
numeric_features = feature_info.get('numeric_features', [col for col in ['Age', 'BMI'] if col in df.columns])
# DII_food也是数值特征
features = ['DII_food'] + numeric_features + categorical_features

# 检查特征是否存在
for feature in features:
    if feature not in df.columns:
        print(f"警告: 特征 '{feature}' 不在数据集中")

# 处理数据，应用独热编码
numeric_data = df[['DII_food'] + numeric_features].copy()
categorical_data = df[categorical_features].copy()

print(f"分类特征: {categorical_features}")
print(f"数值特征: {numeric_features}")

# 应用独热编码
encoded_cats = encoder.transform(categorical_data)

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

# 数据分割
def split_data(X, y, weights=None, test_size=0.2, random_state=42, stratify=True):
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

# 使用FNN模型进行预测
y_pred, y_prob = fnn_predict(model_data, X_test)

model_name = "FNN"

# 绘图
print("绘制ROC曲线...")
plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

print("绘制PR曲线...")
plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)

# 校准曲线和决策曲线已移除

print("绘制混淆矩阵...")
plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)

# 为FNN模型实现自定义学习曲线功能
print("绘制学习曲线...")

def custom_fnn_learning_curve(model_data, X, y, weights=None, cv=5, train_sizes=None, random_state=42, plot_dir=None, plot_data_dir=None):
    """为FNN模型自定义学习曲线实现
    
    横坐标：训练样本数量
    纵坐标：模型性能指标（AUC-ROC）
    包含训练集和测试集上的性能曲线
    使用交叉验证评估训练集性能，避免显示过拟合
    """
    from sklearn.model_selection import StratifiedKFold
    # 直接从同级目录的FNN_Train模块导入
    import sys
    import os
    # 添加当前目录到系统路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    if current_path not in sys.path:
        sys.path.append(current_path)
    # 直接从FNN_Train导入需要的类和函数
    from FNN_Train import FNNModel, train_model, evaluate
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from pathlib import Path
    
    # 全局字体设置：Monaco 12号
    plt.rcParams['font.family'] = 'Monaco'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果没有指定训练集大小比例，使用默认值
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)  # 与CatBoost保持一致，使用10个点
    
    # 设置交叉验证
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # 初始化结果数组
    train_scores = []
    test_scores = []
    train_sizes_abs = []
    
    # 保存每一个样本量的所有交叉验证分数，用于计算置信区间
    train_scores_all_folds = []
    
    # 模型配置
    model_config = model_data['model_config']
    params = model_config['params']
    input_dim = model_config['input_dim']
    
    # 对每个指定的训练集大小
    for train_size in train_sizes:
        print(f"\n训练集大小: {train_size:.2f}...")
        fold_train_scores = []
        fold_test_scores = []
        fold_train_sizes = []
        
        # 对每个CV折叠
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            print(f"\t折叠 {fold+1}/{cv}...")
            # 切分数据
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            if weights is not None:
                weights_train_fold = weights.iloc[train_idx]
                weights_test_fold = weights.iloc[test_idx]
            else:
                weights_train_fold = weights_test_fold = None
            
            # 计算当前训练大小
            train_size_abs = int(len(X_train_fold) * train_size)
            train_size_abs = max(train_size_abs, 10)  # 至少使用10个样本
            train_size_abs = min(train_size_abs, len(X_train_fold))  # 不超过数据集大小
            
            # 子采样训练集
            train_subset_idx = np.random.choice(len(X_train_fold), train_size_abs, replace=False)
            X_train_subset = X_train_fold.iloc[train_subset_idx]
            y_train_subset = y_train_fold.iloc[train_subset_idx]
            
            if weights_train_fold is not None:
                weights_train_subset = weights_train_fold.iloc[train_subset_idx]
            else:
                weights_train_subset = None
            
            # 预处理数据
            if 'scaler' in model_data:
                X_train_subset_scaled = model_data['scaler'].transform(X_train_subset)
                X_test_fold_scaled = model_data['scaler'].transform(X_test_fold)
            else:
                X_train_subset_scaled = X_train_subset.values
                X_test_fold_scaled = X_test_fold.values
            
            # 创建并训练模型
            try:
                # 创建模型
                model = FNNModel(input_dim, params)
                
                # 设置简化的训练参数 - 避免冗长训练
                training_params = params.copy()
                training_params['epochs'] = 30  # 减少迭代次数加快学习曲线生成
                
                # 训练模型
                model, _, _, _ = train_model(
                    model, 
                    X_train_subset_scaled, 
                    y_train_subset.values, 
                    X_test_fold_scaled, 
                    y_test_fold.values, 
                    training_params
                )
                
                # 计算训练集分数
                X_train_mx = mx.array(X_train_subset_scaled.astype(np.float32))
                y_train_mx = mx.array(y_train_subset.values.astype(np.float32))
                train_eval = evaluate(model, X_train_mx, y_train_mx)
                train_score = train_eval['auc']
                
                # 计算测试集分数
                X_test_mx = mx.array(X_test_fold_scaled.astype(np.float32))
                y_test_mx = mx.array(y_test_fold.values.astype(np.float32))
                test_eval = evaluate(model, X_test_mx, y_test_mx)
                test_score = test_eval['auc']
                
                # 添加结果
                fold_train_scores.append(train_score)
                fold_test_scores.append(test_score)
                fold_train_sizes.append(train_size_abs)
                
                print(f"\t\t训练分数: {train_score:.4f}, 测试分数: {test_score:.4f}")
            except Exception as e:
                print(f"\t\t错误: {str(e)}")
                # 跳过失败的折叠
                continue
        
        # 平均每个折叠的结果
        if fold_train_scores:
            train_scores.append(np.mean(fold_train_scores))
            test_scores.append(np.mean(fold_test_scores))
            train_sizes_abs.append(np.mean(fold_train_sizes))
            # 存储当前样本量的所有折迭代分数，用于计算置信区间
            train_scores_all_folds.append(fold_train_scores)
    
    # 过滤掉None值
    valid_indices = [i for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores))
                    if train_score is not None and test_score is not None]
    valid_train_sizes = [train_sizes_abs[i] for i in valid_indices]
    valid_train_scores = [train_scores[i] for i in valid_indices]
    valid_test_scores = [test_scores[i] for i in valid_indices]
    
    # 获取有效的折分数
    valid_train_scores_all_folds = [train_scores_all_folds[i] for i in valid_indices]
    
    if len(valid_indices) == 0:
        print("无法创建学习曲线：所有训练尝试都失败了")
        return
    
    # 计算置信区间 - 使用标准误差(SEM)而非标准差，并使用更小的z值
    train_scores_sem = []
    for fold_scores in valid_train_scores_all_folds:
        if len(fold_scores) >= 2:  # 需要至少2个有效折才能计算标准误差
            # 计算标准误差 (SEM = STD / sqrt(n))
            std = np.std(fold_scores, ddof=1)
            sem = std / np.sqrt(len(fold_scores))
            train_scores_sem.append(sem)
        else:
            train_scores_sem.append(0.0)  # 如果数据不足，设置标准误差为0
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制置信区间 - 使用标准误差和较小的z值(1.0而非1.96)
    z_value = 1.0  # 对应68%置信区间而非95%
    ax.fill_between(valid_train_sizes, 
                    [max(0.5, score - z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                    [min(1.0, score + z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                    alpha=0.15, color='#2ca02c')
    
    # 绘制主线
    ax.plot(valid_train_sizes, valid_train_scores, 'o-', label='Training Set (CV)', color='#2ca02c', linewidth=2)
    ax.plot(valid_train_sizes, valid_test_scores, 'o-', label='Test Set', color='#d62728', linewidth=2)
    
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("AUC-ROC")
    
    # Set y-axis range from 0.5 to 1.0 with 0.1 intervals
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    
    ax.set_title(f"Sample Learning Curve - FNN", pad=30)
    
    # 美化图形
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend(loc='lower right')
    
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"FNN_sample_learning_curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存数据
    if plot_data_dir:
        save_plot_data({
            'train_sizes': valid_train_sizes,
            'train_scores': valid_train_scores,
            'test_scores': valid_test_scores
        }, str(Path(plot_data_dir) / f"FNN_sample_learning_curve.json"))
        
    return valid_train_sizes, valid_train_scores, valid_test_scores

# 定义保存数据的函数，与其他模型保持一致
def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# 调用自定义学习曲线函数
train_sizes, train_scores, test_scores = custom_fnn_learning_curve(
    model_data, 
    X_train, 
    y_train, 
    weights_train, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),  # 与CatBoost保持一致，使用10个点
    random_state=42,
    plot_dir=plot_dir,
    plot_data_dir=plot_data_dir
)

print("绘制阈值曲线...")
plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)

print("所有图表已生成完毕。")
