import pandas as pd
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置Monaco字体
font_path = None
font_dirs = ['/System/Library/Fonts/', '/Library/Fonts/']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    if 'monaco' in font_file.lower():
        font_path = font_file
        break
if font_path:
    fm.fontManager.addfont(font_path)
    monaco_font = fm.FontProperties(family='Monaco', size=13)  # 使用13号字体

# 全局字体和样式设置
plt.rcParams['font.family'] = 'Monaco'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['figure.dpi'] = 300

# 导入评估指标和绘图函数
from model_metrics_utils import calculate_metrics
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_learning_curve, 
    plot_confusion_matrix, plot_threshold_curve, plot_roc_curve_comparison
)
from model_plot_calibration import plot_calibration_all_data
from model_plot_DCA import plot_dca_curve, plot_dca_curve_comparison

# 学习曲线相关配置
LEARNING_CURVE_CONFIG = {
    'n_cv': 5,                    # 交叉验证折数
    'n_resample': 5,              # 重采样次数
    'learning_curve_step_size': 500  # 学习曲线步长
}

def main():
    # 命令行参数处理，允许覆盖配置文件中的设置
    parser = argparse.ArgumentParser(description="神经网络模型评估与可视化")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--metrics", type=int, choices=[0, 1], help="是否计算评估指标(0:否, 1:是)")
    parser.add_argument("--roc", type=int, choices=[0, 1], help="是否绘制ROC曲线(0:否, 1:是)")
    parser.add_argument("--pr", type=int, choices=[0, 1], help="是否绘制PR曲线(0:否, 1:是)")
    parser.add_argument("--calibration", type=int, choices=[0, 1], help="是否绘制校准曲线(0:否, 1:是)")
    parser.add_argument("--confusion", type=int, choices=[0, 1], help="是否绘制混淆矩阵(0:否, 1:是)")
    parser.add_argument("--learning", type=int, choices=[0, 1], help="是否绘制学习曲线(0:否, 1:是)")
    parser.add_argument("--threshold", type=int, choices=[0, 1], help="是否绘制阈值曲线(0:否, 1:是)")
    parser.add_argument("--dca", type=int, choices=[0, 1], help="是否绘制决策曲线分析(DCA)(0:否, 1:是)")
    args = parser.parse_args()

    # 读取配置文件
    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 读取评估和绘图设置
    eval_settings = config.get('eval_settings', {})
    
    # 设置默认值，命令行参数优先
    calc_metrics = args.metrics if args.metrics is not None else eval_settings.get('calc_metrics', 1)
    draw_roc = args.roc if args.roc is not None else eval_settings.get('draw_roc', 1)
    draw_pr = args.pr if args.pr is not None else eval_settings.get('draw_pr', 1)
    draw_calibration = args.calibration if args.calibration is not None else eval_settings.get('draw_calibration', 1)
    draw_confusion = args.confusion if args.confusion is not None else eval_settings.get('draw_confusion', 1)
    draw_learning = args.learning if args.learning is not None else eval_settings.get('draw_learning', 1)
    draw_threshold = args.threshold if args.threshold is not None else eval_settings.get('draw_threshold', 1)
    draw_dca = args.dca if args.dca is not None else eval_settings.get('draw_dca', 1)
    
    # 打印评估与可视化设置
    print("\n===== 评估与可视化设置 =====")
    print(f"计算评估指标: {'是' if calc_metrics else '否'}")
    print(f"绘制ROC曲线: {'是' if draw_roc else '否'}")
    print(f"绘制PR曲线: {'是' if draw_pr else '否'}")
    print(f"绘制校准曲线: {'是' if draw_calibration else '否'}")
    print(f"绘制混淆矩阵: {'是' if draw_confusion else '否'}")
    print(f"绘制学习曲线: {'是' if draw_learning else '否'}")
    print(f"绘制阈值曲线: {'是' if draw_threshold else '否'}")
    print(f"绘制决策曲线分析(DCA): {'是' if draw_dca else '否'}")
    print("===========================\n")
    
    # 读取基本配置
    data_path = Path(config['data_path'])
    model_dir = Path(config['model_dir'])
    plot_dir = Path(config['plot_dir'])
    plot_dir.mkdir(exist_ok=True)
    plot_data_dir = Path('plot_original_data')
    plot_data_dir.mkdir(exist_ok=True)
    result_dir = Path(config.get('output_dir', 'result'))
    result_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model_path = model_dir / 'FNN_best_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
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
    
    print(f"分类特征: {categorical_features}")
    print(f"数值特征: {numeric_features}")
    
    # 检查特征是否存在
    for feature in features:
        if feature not in df.columns:
            print(f"警告: 特征 '{feature}' 不在数据集中")
    
    # 处理数据，应用独热编码
    numeric_data = df[['DII_food'] + numeric_features].copy()
    categorical_data = df[categorical_features].copy()
    
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
        
        # 重建 FNN 模型
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
        
    # 预测
    print(f"成功加载模型，使用独热编码后的特征数：{X.shape[1]}")
    y_pred, y_prob = fnn_predict(model, X_test)
    model_name = "FNN"
    
    # 打印配置设置
    print(f"\n===== 模型评估 =====")
    
    # 计算评估指标
    if calc_metrics:
        start_time = time.time()
        print("计算评估指标...")
        metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
        
        # 打印评估结果
        print("\nTest Set Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"评估指标计算完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    # 绘制各种图表
    if draw_roc:
        start_time = time.time()
        print("绘制ROC曲线...")
        plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
        print(f"ROC曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
        
        # 新增：ROC比较曲线（含DII vs 不含DII）
        start_time = time.time()
        print("绘制ROC比较曲线（评估DII贡献）...")
        
        # 创建无DII版本的测试数据
        X_test_no_dii = X_test.copy()
        # 找到DII_food列并替换为均值，而不是0
        if 'DII_food' in X_test.columns:
            # 计算原始数据中DII_food的均值
            dii_mean = df['DII_food'].mean()
            print(f"使用DII_food的均值 {dii_mean:.4f} 替代所有样本的DII值")
            X_test_no_dii['DII_food'] = dii_mean
        
        # 预测测试集的概率
        y_prob_with_dii = y_prob  # 已有的测试集预测结果
        _, y_prob_no_dii = fnn_predict(model, X_test_no_dii)  # 正确解包元组，获取第二个元素（概率）
        
        # 构建比较字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_with_dii,
            f"{model_name}(with mean DII)": y_prob_no_dii  # 更新标签以反映使用均值
        }
        
        # 调用比较函数，在测试集上进行比较，不使用SMOTE过采样
        plot_roc_curve_comparison(y_test, y_probs_dict, weights_test, model_name, plot_dir, plot_data_dir, use_smote=False)
        print(f"ROC比较曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_pr:
        start_time = time.time()
        print("绘制PR曲线...")
        plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
        print(f"PR曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_calibration:
        start_time = time.time()
        print("绘制校准曲线...")
        # 使用测试集数据绘制校准曲线，而不是全量数据
        print("使用测试集数据绘制校准曲线")
        plot_calibration_all_data(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
        print(f"校准曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_confusion:
        start_time = time.time()
        print("绘制混淆矩阵...")
        plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
        print(f"混淆矩阵绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_learning:
        start_time = time.time()
        print("绘制学习曲线...")
        
        # 自定义FNN学习曲线函数，与model_plot_utils.py中的实现保持一致
        def custom_fnn_learning_curve(model_data, X_train, y_train, X_test, y_test, weights=None, cv=2, n_resample=2, train_sizes=None, random_state=42, plot_dir=None, plot_data_dir=None):
            """
            绘制FNN模型的学习曲线，支持重采样和交叉验证
            
            参数:
            -----------
            model_data: 模型数据
            X: 特征数据
            y: 标签数据
            weights: 样本权重
            cv: 交叉验证折数，默认为2
            n_resample: 重采样次数，默认为2
            train_sizes: 训练集大小列表
            random_state: 随机种子
            plot_dir: 图表保存目录
            plot_data_dir: 数据保存目录
            """
            from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_auc_score
            import numpy as np
            from imblearn.over_sampling import SMOTE
            from pathlib import Path
            import json
            from scipy import stats
            
            # 验证参数
            n_cv = max(2, int(cv)) if cv and cv > 0 else 2
            n_resample = max(1, int(n_resample)) if n_resample and n_resample > 0 else 2
            
            print(f"开始学习曲线评估 - 重采样次数: {n_resample}, 交叉验证折数: {n_cv}")
            
            # 直接在当前脚本中定义FNN模型类
            class SimpleFFN:
                """FNN模型简化版，使用sklearn的MLPClassifier"""
                def __init__(self, input_dim, hidden_dim=64):
                    from sklearn.neural_network import MLPClassifier
                    self.model = MLPClassifier(
                        hidden_layer_sizes=(hidden_dim, hidden_dim//2),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='adaptive',  # 使用自适应学习率
                        learning_rate_init=0.001,
                        max_iter=200,  # 增加迭代次数以确保收敛
                        shuffle=True,
                        random_state=random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10  # 增加不变迭代次数的耐心
                    )
                    
                def fit(self, X, y):
                    # 使用SMOTE平衡数据
                    sm = SMOTE(random_state=42)
                    X_res, y_res = sm.fit_resample(X, y)
                    self.model.fit(X_res, y_res)
                    return self
                    
                def predict(self, X):
                    return self.model.predict(X)
                    
                def predict_proba(self, X):
                    return self.model.predict_proba(X)[:, 1]
                
            class MLXSimpleFNN:
                """MLX版本的简单前馈神经网络模型包装类《使用GPU加速》"""
                def __init__(self, input_dim, random_state=42):
                    from pathlib import Path
                    import mlx.nn as nn
                    import mlx.core as mx
                    import mlx.optimizers as optim
                    
                    # 确保使用GPU
                    print(f"MLX设备信息: {mx.default_device()}")
                    print(f"Metal是否可用: {mx.metal.is_available() if hasattr(mx, 'metal') else 'Unknown'}")
                    
                    # 定义简单的前向神经网络
                    self.input_dim = input_dim
                    hidden_dim = 64
                    
                    # 定义网络结构
                    self.model = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(hidden_dim//2, 1),
                        nn.Sigmoid()
                    )
                    
                    # 随机种子设置
                    if hasattr(mx.random, 'seed'):
                        mx.random.seed(random_state)
                    
                    self.optimizer = optim.Adam(learning_rate=0.001)
                    
                def binary_cross_entropy_loss(self, params, X, y):
                    """ Binary cross entropy loss """
                    preds = self.model.apply(params, X).reshape(-1)
                    eps = 1e-7
                    # 确保计算保持在MLX张量内，不尝试转换为标量
                    loss = -mx.mean(y * mx.log(preds + eps) + (1 - y) * mx.log(1 - preds + eps))
                    return loss
                    
                def fit(self, X, y):
                    # 首先确保输入数据是NumPy数组
                    if hasattr(X, 'values'):
                        X = X.values  # DataFrame 转换为 NumPy数组
                    if hasattr(y, 'values'):
                        y = y.values  # Series 转换为 NumPy数组
                    
                    # 使用SMOTE平衡数据 (在CPU上运行)
                    sm = SMOTE(random_state=42)
                    X_res, y_res = sm.fit_resample(X, y)
                    
                    # 转换为MLX格式
                    X_mx = mx.array(X_res.astype(np.float32))
                    y_mx = mx.array(y_res.astype(np.float32))
                    
                    # 定义梯度函数
                    loss_and_grad_fn = nn.value_and_grad(self.model, self.binary_cross_entropy_loss)
                    
                    # 训练参数 - 减少循环次数提高速度
                    n_epochs = 20
                    batch_size = 256  # 增大批大小充分利用GPU
                    n_samples = len(X_res)
                    
                    # 训练循环
                    params = self.model.parameters()
                    for epoch in range(n_epochs):
                        # 随机打乱数据
                        idxs = mx.array(np.random.permutation(n_samples))
                        
                        # 批处理
                        for i in range(0, n_samples, batch_size):
                            batch_idxs = idxs[i:i + batch_size]
                            X_batch = mx.take(X_mx, batch_idxs, axis=0)
                            y_batch = mx.take(y_mx, batch_idxs, axis=0)
                            
                            # 计算损失和梯度
                            loss, grads = loss_and_grad_fn(params, X_batch, y_batch)
                            
                            # 更新参数
                            self.optimizer.update(params, grads)
                    
                    return self
                    
                def predict(self, X):
                    # 确保输入数据是NumPy数组
                    if hasattr(X, 'values'):
                        X = X.values  # DataFrame 转换为 NumPy数组
                    
                    # 转换为MLX格式
                    X_mx = mx.array(X.astype(np.float32))
                    
                    # 预测
                    preds = self.model(X_mx).reshape(-1)
                    # 安全地转换为NumPy数组
                    if isinstance(preds, mx.array):
                        preds_np = np.array(preds.tolist())
                    else:
                        preds_np = np.array(preds)
                    
                    # 转换为二分类标签
                    return (preds_np > 0.5).astype(int)
                    
                def predict_proba(self, X):
                    # 确保输入数据是NumPy数组
                    if hasattr(X, 'values'):
                        X = X.values  # DataFrame 转换为 NumPy数组
                    
                    # 转换为MLX格式
                    X_mx = mx.array(X.astype(np.float32))
                    
                    # 预测概率
                    preds = self.model(X_mx).reshape(-1)
                    
                    # 安全地转换为NumPy数组
                    if isinstance(preds, mx.array):
                        # 使用to_numpy()方法(如果可用)或转换为列表再转为NumPy数组
                        if hasattr(preds, 'to_numpy'):
                            return preds.to_numpy()
                        else:
                            return np.array(preds.tolist())
                    else:
                        return np.array(preds)
            
            # 使用标准版本的SimpleFFN代替MLX版本的模型以兼顾稳定性和速度
            print("注意: 使用scikit-learn模型进行学习曲线评估 (速度和稳定性原因)")
            
            # 定义要测试的样本量
            start_size = 1000  # 起始样本量
            max_train_size = 8000  # 最大样本量
            
            # 设置步长
            step_size = LEARNING_CURVE_CONFIG['learning_curve_step_size']
            
            # 打印样本量信息用于调试
            print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            print(f"起始样本量: {start_size}, 最大样本量: {max_train_size}, 步长: {step_size}")
            
            # 生成样本量列表
            train_sizes_abs = list(range(start_size, max_train_size + 1, step_size))
            
            # 确保包含最大样本量
            if max_train_size not in train_sizes_abs:
                train_sizes_abs.append(max_train_size)
                
            # 确保样本量不超过训练集大小
            train_sizes_abs = [size for size in train_sizes_abs if size <= len(X_train)]
            train_sizes_abs = sorted(list(set(train_sizes_abs)))  # 去重并排序
            
            print(f"样本量评估点: {train_sizes_abs}")
            
            # 初始化存储结构
            all_train_scores = []  # 存储每次重采样的训练集分数（用于计算置信区间）
            all_val_scores = []    # 存储每次重采样的验证集分数（用于计算置信区间）
            
            print(f"\n===== 开始学习曲线评估 =====")
            print(f"重采样次数: {n_resample}, 交叉验证折数: {cv}")
            
            # 进行多次重采样
            for resample_idx in range(n_resample):
                print(f"\n--- 重采样 {resample_idx + 1}/{n_resample} ---")
                
                resample_seed = random_state + resample_idx  # 为每次重采样设置不同的随机种子
                train_scores = []
                val_scores = []
                
                for n_samples in train_sizes_abs:
                    print(f"\n处理样本量: {n_samples}")
                    
                    # 使用分层抽样选择n_samples个样本进行训练
                    if n_samples >= len(X_train):
                        X_train_subset = X_train
                        y_train_subset = y_train
                    else:
                        sss = StratifiedShuffleSplit(
                            n_splits=1, 
                            train_size=n_samples, 
                            random_state=resample_seed
                        )
                        for train_idx, _ in sss.split(X_train, y_train):
                            X_train_subset = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                            y_train_subset = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                    
                    try:
                        # 使用交叉验证评估训练集性能
                        cv_scores = []
                        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=resample_seed)
                        
                        for fold_idx, (train_idx, val_idx) in enumerate(stratified_cv.split(X_train_subset, y_train_subset)):
                            # 获取交叉验证数据
                            if hasattr(X_train_subset, 'iloc'):
                                X_train_cv = X_train_subset.iloc[train_idx]
                                y_train_cv = y_train_subset.iloc[train_idx]
                                X_val_cv = X_train_subset.iloc[val_idx]
                                y_val_cv = y_train_subset.iloc[val_idx]
                            else:
                                X_train_cv = X_train_subset[train_idx]
                                y_train_cv = y_train_subset[train_idx]
                                X_val_cv = X_train_subset[val_idx]
                                y_val_cv = y_train_subset[val_idx]
                            
                            try:
                                # 对训练集应用SMOTE过采样（仅在训练集上）
                                smote = SMOTE(random_state=resample_seed * 100 + fold_idx)
                                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_cv, y_train_cv)
                                
                                # 使用原始SimpleFFN模型以确保稳定性
                                model = SimpleFFN(input_dim=X_train_resampled.shape[1])
                                model.fit(X_train_resampled, y_train_resampled)
                                
                                # 在验证集上评估（使用原始未过采样的验证集）
                                try:
                                    val_pred = model.predict_proba(X_val_cv)
                                    # 确保val_pred是标准NumPy数组
                                    if isinstance(val_pred, mx.array):
                                        if hasattr(val_pred, 'to_numpy'):
                                            val_pred = val_pred.to_numpy()
                                        else:
                                            val_pred = np.array(val_pred.tolist())
                                    elif not isinstance(val_pred, np.ndarray):
                                        val_pred = np.array(val_pred)
                                    val_auc = roc_auc_score(y_val_cv, val_pred)
                                    
                                    # 只在验证集上评估，不评估训练集
                                    cv_scores.append(val_auc)
                                    print(f"  折 {fold_idx+1}/{cv}: 验证AUC = {val_auc:.4f}")
                                except Exception as e:
                                    print(f"  计算AUC失败: {str(e)}")
                                
                            except Exception as e:
                                print(f"  折 {fold_idx+1} 训练失败: {str(e)}")
                        
                        # 计算当前样本量的平均AUC
                        if cv_scores:
                            cv_mean = np.mean(cv_scores)
                            train_scores.append(cv_mean)  # 存储交叉验证平均分作为训练集分数
                            
                            # 对子集应用SMOTE过采样，然后在测试集上评估
                            smote_test = SMOTE(random_state=resample_seed)
                            X_train_subset_resampled, y_train_subset_resampled = smote_test.fit_resample(X_train_subset, y_train_subset)
                            
                            try:
                                    # 使用原始SimpleFFN模型以确保稳定性
                                test_model = SimpleFFN(input_dim=X_train_subset_resampled.shape[1])
                                test_model.fit(X_train_subset_resampled, y_train_subset_resampled)
                                test_pred = test_model.predict_proba(X_test)
                                
                                # 确保是标准NumPy数组
                                if isinstance(test_pred, mx.array):
                                    if hasattr(test_pred, 'to_numpy'):
                                        test_pred = test_pred.to_numpy()
                                    else:
                                        test_pred = np.array(test_pred.tolist())
                                elif not isinstance(test_pred, np.ndarray):
                                    test_pred = np.array(test_pred)
                                    
                                test_auc = roc_auc_score(y_test, test_pred)
                                val_scores.append(test_auc)
                                
                                print(f"  样本量 {n_samples}, 重采样 {resample_idx+1}, 训练集AUC: {cv_mean:.4f}, 测试集AUC: {test_auc:.4f}")
                            except Exception as e:
                                print(f"  测试集评估失败: {str(e)}")
                                val_scores.append(np.nan)
                        else:
                            train_scores.append(np.nan)
                            val_scores.append(np.nan)
                            print(f"  样本量 {n_samples}, 重采样 {resample_idx+1} 没有有效的交叉验证分数")
                        
                    except Exception as e:
                        print(f"  样本量 {n_samples} 处理失败: {str(e)}")
                        train_scores.append(np.nan)
                        val_scores.append(np.nan)
                
                # 存储当前重采样的结果
                all_train_scores.append(train_scores)
                all_val_scores.append(val_scores)
            
            # 转换存储结构：按样本量组织数据
            train_scores_by_size = []
            val_scores_by_size = []
            
            for size_idx in range(len(train_sizes_abs)):
                # 训练集：收集所有重采样的分数
                train_scores_for_size = []
                for resample_idx in range(n_resample):
                    if (resample_idx < len(all_train_scores) and 
                        size_idx < len(all_train_scores[resample_idx]) and 
                        not np.isnan(all_train_scores[resample_idx][size_idx])):
                        train_scores_for_size.append(all_train_scores[resample_idx][size_idx])
                
                # 验证集：收集所有重采样的分数
                val_scores_for_size = []
                for resample_idx in range(n_resample):
                    if (resample_idx < len(all_val_scores) and 
                        size_idx < len(all_val_scores[resample_idx]) and 
                        not np.isnan(all_val_scores[resample_idx][size_idx])):
                        val_scores_for_size.append(all_val_scores[resample_idx][size_idx])
                
                train_scores_by_size.append(train_scores_for_size)
                val_scores_by_size.append(val_scores_for_size)
            
            # 计算均值和置信区间
            valid_train_sizes = []
            train_means = []
            val_means = []
            train_cis = []
            
            for size_idx, (train_scores, val_scores) in enumerate(zip(train_scores_by_size, val_scores_by_size)):
                if not train_scores or not val_scores:
                    continue
                    
                # 计算均值和标准误差
                train_mean = np.mean(train_scores)
                train_sem = stats.sem(train_scores)
                val_mean = np.mean(val_scores)
                
                # 计算95%置信区间 (使用t分布)
                ci = stats.t.interval(0.95, len(train_scores)-1, loc=train_mean, scale=train_sem)
                
                valid_train_sizes.append(train_sizes_abs[size_idx])
                train_means.append(train_mean)
                val_means.append(val_mean)
                train_cis.append((train_mean - ci[0], ci[1] - train_mean))  # 存储上下界与均值的差
                
                # 打印当前样本量的统计信息
                print(f"\n训练集大小: {train_sizes_abs[size_idx]}")
                print(f"训练集分数: {np.round(train_scores, 4)}")
                print(f"验证集分数: {np.round(val_scores, 4)}")
                print(f"训练集均值: {train_mean:.4f}, 验证集均值: {val_mean:.4f}")
                print(f"95% 置信区间: [{ci[0]:.4f}, {ci[1]:.4f}] (范围: {ci[1]-ci[0]:.4f})")
                print(f"样本量: {len(train_scores)}")
            
            # 计算每个样本量的均值和标准误
            # 清除之前的数据结构
            valid_train_sizes = []  # 存储有效的样本量
            train_means = []  # 训练集AUC均值
            val_means = []    # 验证集AUC均值
            train_sems = []   # 训练集标准误
            val_sems = []     # 验证集标准误
            
            # 简化处理逻辑，确保所有样本点都被处理
            print("\n调试: 原始样本点:", train_sizes_abs)
            print("\n调试: all_train_scores长度:", len(all_train_scores))
            
            # 处理每个样本量点
            for size_idx, size in enumerate(train_sizes_abs):
                # 收集该样本量的所有重采样的分数
                train_scores_at_size = []
                val_scores_at_size = []
                
                for resample_idx in range(n_resample):
                    if (resample_idx < len(all_train_scores) and 
                        size_idx < len(all_train_scores[resample_idx]) and 
                        not np.isnan(all_train_scores[resample_idx][size_idx])):
                        train_scores_at_size.append(all_train_scores[resample_idx][size_idx])
                    
                    if (resample_idx < len(all_val_scores) and 
                        size_idx < len(all_val_scores[resample_idx]) and 
                        not np.isnan(all_val_scores[resample_idx][size_idx])):
                        val_scores_at_size.append(all_val_scores[resample_idx][size_idx])
                
                # 如果有有效数据，计算统计量
                if len(train_scores_at_size) > 0 and len(val_scores_at_size) > 0:
                    valid_train_sizes.append(size)
                    
                    # 训练集统计
                    train_mean = np.mean(train_scores_at_size)
                    train_sem = stats.sem(train_scores_at_size) if len(train_scores_at_size) > 1 else 0
                    
                    # 验证集统计
                    val_mean = np.mean(val_scores_at_size)
                    val_sem = stats.sem(val_scores_at_size) if len(val_scores_at_size) > 1 else 0
                    
                    # 存储结果
                    train_means.append(train_mean)
                    val_means.append(val_mean)
                    train_sems.append(train_sem)
                    val_sems.append(val_sem)
                    
                    print(f"\n样本量 {size} 有效数据: ")
                    print(f"  训练集有效数据点数: {len(train_scores_at_size)}")
                    print(f"  验证集有效数据点数: {len(val_scores_at_size)}")
                    print(f"  训练集均值: {train_mean:.4f}, 标准误: {train_sem:.4f}")
                    print(f"  验证集均值: {val_mean:.4f}, 标准误: {val_sem:.4f}")
            
            # 将结果转换为numpy数组以便后续处理
            valid_train_sizes = np.array(valid_train_sizes)
            valid_train_scores = np.array(train_means)  # 训练集AUC均值
            valid_val_scores = np.array(val_means)      # 验证集AUC均值
            train_scores_sem = np.array(train_sems)     # 训练集标准误
            val_scores_sem = np.array(val_sems)         # 验证集标准误
            
            # 打印汇总信息
            print("\n学习曲线评估完成:")
            
            # 调试信息
            print(f"valid_train_sizes的长度: {len(valid_train_sizes)}")
            print(f"valid_train_sizes的值: {valid_train_sizes}")
            
            for idx, (size, train_score, val_score, train_sem, val_sem) in enumerate(zip(
                valid_train_sizes, valid_train_scores, valid_val_scores, train_scores_sem, val_scores_sem
            )):
                print(f"样本量 {size}:")
                print(f"  训练集 AUC = {train_score:.4f} ± {train_sem:.4f}")
                print(f"  验证集 AUC = {val_score:.4f} ± {val_sem:.4f}")
                print()
            
            if len(valid_train_sizes) == 0:
                print("警告: 没有有效的评估结果，请检查输入数据或参数设置")
                return None, None, None
            
            # 计算置信区间 - 使用已经计算好的train_scores_sem
            
            # 绘图 - 与model_plot_utils中完全一致
            with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
                # 保持与其他图表一致的大小
                fig, ax = plt.subplots(figsize=(7, 7))
                
                # 只显示训练集置信区间
                z_value = 1.0  # 对应68%置信区间
                ax.fill_between(valid_train_sizes, 
                                np.maximum(0.5, valid_train_scores - z_value * train_scores_sem),
                                np.minimum(1.0, valid_train_scores + z_value * train_scores_sem),
                                alpha=0.15, color='#7fbf7f')  # 使用与训练集相同的浅绿色
                
                # 绘制主线 - 使用与model_plot_utils相同的颜色
                light_green = '#7fbf7f'  # 浅绿色
                pink = '#ff9eb5'  # 粉色
                
                ax.plot(valid_train_sizes, valid_train_scores, '-', label='Training Set (CV)', color=light_green, linewidth=2)
                ax.plot(valid_train_sizes, valid_val_scores, '-', label='Test Set', color=pink, linewidth=2)
                
                # 单独绘制数据点
                ax.plot(valid_train_sizes, valid_train_scores, 'o', color=light_green, markersize=5, alpha=0.8)
                ax.plot(valid_train_sizes, valid_val_scores, 'o', color=pink, markersize=5, alpha=0.8)
                
                # 设置固定的X轴范围为0-9000样本，确保数据点（1000-8000）完全显示
                ax.set_xlim(0, 9000)  
                
                # 创建0到9000的整数刻度
                fixed_tick_values = [i * 1000 for i in range(10)]  # 0, 1000, 2000, ..., 9000
                
                # 设置网格线
                ax.set_xticks(valid_train_sizes, minor=True)
                ax.set_xticklabels([""] * len(valid_train_sizes), minor=True)
                
                # 设置主要刻度标签
                ax.set_xticks(fixed_tick_values)
                ax.set_xticklabels([str(i) for i in range(10)])
                
                # 确保网格线显示
                ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
                
                # 设置轴标签
                ax.set_xlabel('Number of Training Samples (×1000)', fontproperties=monaco_font)
                ax.set_ylabel('AUC-ROC', fontproperties=monaco_font)
                
                # 设置y轴范围
                ax.set_ylim(0.5, 1.0)
                ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                
                # 设置刻度标签字体
                for label in ax.get_xticklabels():
                    label.set_fontproperties(monaco_font)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(monaco_font)
                
                # 设置标题
                ax.set_title(f'Sample Learning Curve - {model_name}', pad=30, fontproperties=monaco_font)
                
                # 添加网格
                ax.grid(linestyle='--', alpha=0.3, color='gray')
                
                # 设置图例
                ax.legend(loc='lower right', prop=monaco_font, frameon=True, framealpha=1.0, facecolor='white', edgecolor='lightgray')
                
                # 显示上边框和右边框
                for spine in ['top','right']:
                    ax.spines[spine].set_visible(True)
            
            fig.tight_layout()
            fig.savefig(str(Path(plot_dir) / f"{model_name}_Learning_Curve.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # 保存数据
            def save_plot_data(data, filename):
                # 将NumPy数组转换为Python原生类型
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                # 确保目录存在
                Path(plot_data_dir).mkdir(parents=True, exist_ok=True)
                
                # 转换数据并保存
                with open(filename, 'w') as f:
                    json.dump(convert_numpy(data), f, indent=4, ensure_ascii=False)
            
            # 准备保存的结果
            results = {
                'train_sizes': valid_train_sizes.tolist() if hasattr(valid_train_sizes, 'tolist') else valid_train_sizes,
                'train_scores_mean': valid_train_scores.tolist() if hasattr(valid_train_scores, 'tolist') else valid_train_scores,
                'train_scores_std': train_scores_sem.tolist() if hasattr(train_scores_sem, 'tolist') else train_scores_sem,
                'val_scores_mean': valid_val_scores.tolist() if hasattr(valid_val_scores, 'tolist') else valid_val_scores,
                'val_scores_std': val_scores_sem.tolist() if hasattr(val_scores_sem, 'tolist') else val_scores_sem,
                'n_cv': n_cv,
                'n_resample': n_resample
            }
            
            save_plot_data(results, str(Path(plot_data_dir) / f"{model_name}_Learning_Curve.json"))
            
            return valid_train_sizes, valid_train_scores, valid_val_scores
        
        # 调用自定义学习曲线函数
        # 从配置中获取学习曲线相关参数
        n_cv = config.get('n_cv', LEARNING_CURVE_CONFIG['n_cv'])
        n_resample = config.get('n_resample', LEARNING_CURVE_CONFIG['n_resample'])
        
        print(f"学习曲线配置 - 交叉验证折数: {n_cv}, 重采样次数: {n_resample}")
        
        # 使用完整的训练集进行学习曲线分析
        train_sizes, train_scores, test_scores = custom_fnn_learning_curve(
            model, 
            X_train, y_train,  # 使用完整的训练集
            X_test, y_test,    # 使用独立的测试集
            weights=weights_train, 
            cv=n_cv, 
            n_resample=n_resample,
            random_state=42, 
            plot_dir=plot_dir, 
            plot_data_dir=plot_data_dir
        )
        print(f"学习曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_threshold:
        start_time = time.time()
        print("绘制阈值曲线...")
        plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
        print(f"阈值曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_dca:
        start_time = time.time()
        print("绘制决策曲线分析(DCA)...")
        # 使用与校准曲线相同的处理方式：全量数据集+SMOTE过采样
        # 预测全量数据集的概率
        _, y_prob_all = fnn_predict(model, X)
        # 增加DII贡献评估
        print("\n评估DII对模型预测能力的贡献...")
        # 创建无DII版本的数据(将DII_food列设为0)
        X_no_dii = X.copy()
        X_no_dii['DII_food'] = X['DII_food'].median()  # 填充中位数而不是0
        
        # 预测无DII数据
        _, y_prob_no_dii = fnn_predict(model, X_no_dii)
        
        # 构建DII对比字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_all,
            f"{model_name}(without DII)": y_prob_no_dii
        }
        
        # 绘制DII贡献对比DCA曲线 - 现在直接返回数据而不是路径
        comparison_data = plot_dca_curve_comparison(y, y_probs_dict, weights, model_name, plot_dir, plot_data_dir, use_smote=True)
        
        # 创建单模型格式数据
        single_model_data = {
            "thresholds": comparison_data["thresholds"],
            "net_benefits_model": comparison_data["models"][f"{model_name}(all feature)"],
            "net_benefits_all": comparison_data["treat_all"],
            "net_benefits_none": comparison_data["treat_none"],
            "model_name": model_name,
            "prevalence": comparison_data.get("prevalence", np.mean(y))
        }
        
        # 保存单模型数据
        with open(plot_data_dir / f"{model_name}_DCA.json", 'w') as f:
            json.dump(single_model_data, f, indent=4)
            
        print(f"已从比较版本中提取并保存单模型DCA数据 -> {model_name}_DCA.json")
        print(f"决策曲线分析(DCA)绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
