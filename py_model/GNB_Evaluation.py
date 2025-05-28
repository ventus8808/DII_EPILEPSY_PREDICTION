import pandas as pd
import numpy as np
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time

# 定义与训练脚本中相同的ModelWrapper类，但添加更多兼容scikit-learn的方法
class ModelWrapper:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        
    def predict(self, X):
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)
        
    def predict_proba(self, X):
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)
    
    # 添加兼容scikit-learn的方法
    def fit(self, X, y, sample_weight=None):
        """sklearn估计器需要的fit方法"""
        # 复制预处理器，避免修改原始模型
        from sklearn.base import clone
        
        try:
            # 尝试深度克隆预处理器
            preprocessor_clone = clone(self.preprocessor)
        except:
            try:
                # 如果无法克隆，尝试创建ColumnTransformer
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
                from sklearn.pipeline import Pipeline
                
                # 假设我们的数据包含这些特征
                categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
                numeric_features = [col for col in ['Age', 'BMI', 'DII_food'] if col in X.columns]
                
                # 假设基本预处理流程
                numeric_transformer = Pipeline([
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, include_bias=False))
                ])
                categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
                
                preprocessor_clone = ColumnTransformer([
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            except:
                # 最后的尝试 - 返回原样的X
                print("\n警告: 无法正确克隆预处理器，将使用原始数据进行训练")
                if hasattr(self.model, 'fit'):
                    if sample_weight is not None and 'sample_weight' in self.model.fit.__code__.co_varnames:
                        self.model.fit(X, y, sample_weight=sample_weight)
                    else:
                        self.model.fit(X, y)
                return self
        
        # 使用克隆的预处理器进行拟合和转换
        try:
            X_transformed = preprocessor_clone.fit_transform(X)
            # 拟合模型
            if hasattr(self.model, 'fit'):
                if sample_weight is not None and 'sample_weight' in self.model.fit.__code__.co_varnames:
                    self.model.fit(X_transformed, y, sample_weight=sample_weight)
                else:
                    self.model.fit(X_transformed, y)
        except Exception as e:
            print(f"\n学习曲线训练失败: {e}")
            # 如果预处理失败，直接用原始数据
            if hasattr(self.model, 'fit'):
                if sample_weight is not None and 'sample_weight' in self.model.fit.__code__.co_varnames:
                    self.model.fit(X, y, sample_weight=sample_weight)
                else:
                    self.model.fit(X, y)
                    
        return self
    
    def get_params(self, deep=True):
        """sklearn估计器需要的参数获取方法"""
        params = {'preprocessor': self.preprocessor, 'model': self.model}
        if deep:
            # 如果内部模型有get_params方法，也返回其参数
            if hasattr(self.model, 'get_params'):
                model_params = self.model.get_params(deep=True)
                model_params = {'model__' + key: val for key, val in model_params.items()}
                params.update(model_params)
            if hasattr(self.preprocessor, 'get_params'):
                preproc_params = self.preprocessor.get_params(deep=True)
                preproc_params = {'preprocessor__' + key: val for key, val in preproc_params.items()}
                params.update(preproc_params)
        return params
    
    def set_params(self, **params):
        """sklearn估计器需要的参数设置方法"""
        # 处理顶层参数
        if 'preprocessor' in params:
            self.preprocessor = params.pop('preprocessor')
        if 'model' in params:
            self.model = params.pop('model')
            
        # 处理嵌套参数
        model_params = {}
        preproc_params = {}
        
        # 分组参数
        for key, val in params.items():
            if key.startswith('model__'):
                model_params[key[7:]] = val  # 去除'model__'前缀
            elif key.startswith('preprocessor__'):
                preproc_params[key[14:]] = val  # 去除'preprocessor__'前缀
        
        # 设置内部模型参数
        if model_params and hasattr(self.model, 'set_params'):
            self.model.set_params(**model_params)
        if preproc_params and hasattr(self.preprocessor, 'set_params'):
            self.preprocessor.set_params(**preproc_params)
            
        return self
    
    def score(self, X, y, sample_weight=None):
        """sklearn估计器需要的评分方法"""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        if sample_weight is not None:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        return accuracy_score(y, y_pred)

# 导入评估指标和绘图函数
from model_metrics_utils import calculate_metrics
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_learning_curve, 
    plot_confusion_matrix, plot_threshold_curve
)
from model_plot_calibration import plot_calibration_all_data
from model_plot_DCA import plot_dca_curve, plot_dca_curve_comparison

def main():
    # 命令行参数处理，允许覆盖配置文件中的设置
    parser = argparse.ArgumentParser(description="GNB模型评估与可视化")
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
    draw_learning = args.learning if args.learning is not None else eval_settings.get('draw_learning', 0)  # 默认不绘制学习曲线
    draw_threshold = args.threshold if args.threshold is not None else eval_settings.get('draw_threshold', 1)
    draw_dca = args.dca if args.dca is not None else eval_settings.get('draw_dca', 1)
    
    # 打印评估与可视化设置
    print("\n===== 评估与可视化设置 =====")
    print(f"计算评估指标: {'是' if calc_metrics else '否'}")
    print(f"绘制ROC曲线: {'是' if draw_roc else '否'}")
    print(f"绘制PR曲线: {'是' if draw_pr else '否'}")
    print(f"绘制校准曲线: {'是' if draw_calibration else '否'}")
    print(f"绘制混淆矩阵: {'是' if draw_confusion else '否'}")
    # 启用学习曲线绘制，使用自定义的GNB学习曲线函数
    draw_learning = eval_settings.get('draw_learning', 1)
    print(f"绘制学习曲线: {'是' if draw_learning else '否'}")
    print(f"绘制阈值曲线: {'是' if draw_threshold else '否'}")
    print(f"绘制决策曲线分析(DCA): {'是' if draw_dca else '否'}")
    print("===========================\n")
    
    # 读取基本配置
    data_path = Path(config['data_path'])
    model_dir = Path('model')  # 使用默认模型目录
    plot_dir = Path(config.get('plot_dir', 'plot'))
    plot_dir.mkdir(exist_ok=True)
    plot_data_dir = Path('plot_original_data')
    plot_data_dir.mkdir(exist_ok=True)
    result_dir = Path(config.get('output_dir', 'result'))
    result_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model_path = model_dir / 'GNB_model.pkl'
    print(f"正在从 {model_path} 加载模型...")
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 不存在")
        print(f"当前工作目录: {Path.cwd()}")
        print(f"模型目录内容: {[f.name for f in model_dir.glob('*')] if model_dir.exists() else '目录不存在'}")
        return
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        features = model_data.get('features', None)
        last_updated = model_data.get('last_updated', 'unknown')
        print(f"已加载GNB(高斯朴素贝叶斯)模型")
        print(f"最后更新时间: {last_updated}")
        
        # 打印模型参数
        print("\n===== 模型参数 =====")
        print(f"模型类型: {model_data.get('model_type', '未知')}")
        print("\n最佳参数:")
        for param, value in model_data.get('best_params', {}).items():
            print(f"  {param}: {value}")
    else:
        # 兼容旧模型格式
        model = model_data
        features = None
        print("已加载GNB模型（旧格式）")
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 确定特征列
    if features is not None:
        # 使用保存的特征列表
        available_features = [f for f in features if f in df.columns]
        missing_features = set(features) - set(available_features)
        if missing_features:
            print(f"警告：以下特征在数据中不可用: {missing_features}")
    else:
        # 使用所有非目标列作为特征
        available_features = [col for col in df.columns if col not in ['Epilepsy', 'WTDRD1']]
    
    print(f"已加载数据，样本数：{df.shape[0]}，特征数：{len(available_features)}")
    
    # 准备数据
    X = df[available_features]
    y = df['Epilepsy']
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
    # 确认模型特征数量是否匹配
    try:
        model.predict(X.iloc[0:1])
        print(f"成功加载模型，使用特征数：{len(available_features)}")
    except Exception as e:
        print(f"错误：模型特征不匹配 - {e}")
        return
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test, weights_train, weights_test = split_data(X, y, weights)
    
    # 预测
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 使用较低的阈值提高敏感度
    threshold = 0.05  # 从默认的0.5改为0.05以大幅提高敏感度
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n决策阈值: {threshold} (降低阈值以提高敏感度)")
    
    model_name = "GNB"  # 用于文件命名
    
    # 计算评估指标
    if calc_metrics:
        start_time = time.time()
        print("\n===== 模型评估 =====")
        print("计算评估指标...")
        
        try:
            # 使用阈值计算评估指标
            metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
            
            print("\n===== 测试集评估指标 =====")
            # 打印主要指标
            print("\n主要指标:")
            main_metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "AUC-ROC", "AUC-PR"]
            for metric in main_metrics:
                if metric in metrics:
                    print(f"{metric}: {metrics[metric]:.4f}")
            
            # 打印其他指标
            print("\n其他指标:")
            other_metrics = [m for m in metrics.keys() if m not in main_metrics]
            for metric in other_metrics:
                print(f"{metric}: {metrics[metric]:.4f}")
            
            print(f"\n评估指标计算完成 (耗时 {time.time() - start_time:.2f}秒)")
            
            # 保存评估指标
            metrics_path = result_dir / f"{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"指标已保存到: {metrics_path}")
            
        except Exception as e:
            print(f"\n计算评估指标时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("根据配置，跳过评估指标计算")
    
    # 绘图
    if draw_roc:
        start_time = time.time()
        print("绘制ROC曲线...")
        plot_roc_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
        print(f"ROC曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_pr:
        start_time = time.time()
        print("绘制PR曲线...")
        plot_pr_curve(y_test, y_prob, weights_test, model_name, plot_dir, plot_data_dir)
        print(f"PR曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_calibration:
        start_time = time.time()
        print("绘制校准曲线...")
        # 预测全量数据集的概率
        y_prob_all = model.predict_proba(X)[:, 1]
        # 使用全量数据集绘制校准曲线
        plot_calibration_all_data(y, y_prob_all, weights, model_name, plot_dir, plot_data_dir)
        print(f"校准曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_confusion:
        start_time = time.time()
        print("绘制混淆矩阵...")
        plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
        print(f"混淆矩阵绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_learning:
        start_time = time.time()
        print("绘制学习曲线...")
        plot_gnb_learning_curve(X_train, y_train, X_test, y_test, model_data, model_name, plot_dir, plot_data_dir)
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
        y_prob_all = model.predict_proba(X)[:, 1]
        # 增加DII贡献评估
        print("\n评估DII对模型预测能力的贡献...")
        # 创建无DII版本的数据(将DII_food列设为0)
        X_no_dii = X.copy()
        X_no_dii['DII_food'] = 0  # 将DII列设为0
        
        # 预测无DII数据
        y_prob_no_dii = model.predict_proba(X_no_dii)[:, 1]
        
        # 构建DII对比字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_all,
            f"{model_name}(without DII)": y_prob_no_dii
        }
        
        # 绘制DII贡献对比DCA曲线
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

def plot_gnb_learning_curve(X_train, y_train, X_test, y_test, model_data, model_name, plot_dir, plot_data_dir, cv=5, n_resamples=3):
    """为高斯朴素贝叶斯模型绘制学习曲线。
    这是一个定制版函数，不使用通用的plot_learning_curve函数。
    
    参数：
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    model_data: 包含模型参数的字典
    model_name: 模型名称
    plot_dir: 图表保存目录
    plot_data_dir: 原始数据保存目录
    cv: 交叉验证折数
    n_resamples: 重采样次数
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import json
    
    # 获取模型参数
    if isinstance(model_data, dict) and 'best_params' in model_data:
        best_params = model_data['best_params']
    else:
        # 使用默认参数
        best_params = {'var_smoothing': 1e-9}
    
    # 获取特征信息
    if isinstance(model_data, dict) and 'features' in model_data:
        features = model_data['features']
    else:
        # 使用默认特征列表
        features = X_train.columns.tolist()
    
    # 定义特征类型
    categorical_features = [col for col in ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'] if col in features]
    numeric_features = [col for col in features if col not in categorical_features]
    
    # 创建预处理器
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 定义要评估的样本量 - 使用1000开始，每500递增，最大到8000
    start_size = 1000
    step_size = 500
    max_train_size = min(8000, len(X_train))  # 限制最大样本量为8000
    train_sizes_abs = list(range(start_size, max_train_size, step_size))
    # 确保包含最后一个点（全部训练数据）
    if train_sizes_abs[-1] != max_train_size and max_train_size - train_sizes_abs[-1] > 100:
        train_sizes_abs.append(max_train_size)
    
    # 初始化存储结构
    train_scores_mean = []
    train_scores_std = []
    test_scores_mean = []
    test_scores_std = []
    
    print(f"\n===== 开始学习曲线评估 =====")
    print(f"重采样次数: {n_resamples}, 交叉验证折数: {cv}")
    
    # 对每个训练集大小进行评估
    for n_samples in train_sizes_abs:
        print(f"\n处理训练集大小: {n_samples}")
        
        # 存储不同重采样的分数
        train_scores_resamples = []
        test_scores_resamples = []
        
        # 进行多次重采样
        for resample_idx in range(n_resamples):
            resample_seed = 42 + resample_idx
            
            # 分层采样选择训练集子集
            if n_samples >= len(X_train):
                X_train_subset = X_train.copy()
                y_train_subset = y_train.copy()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=resample_seed)
                for train_idx, _ in sss.split(X_train, y_train):
                    X_train_subset = X_train.iloc[train_idx].copy()
                    y_train_subset = y_train.iloc[train_idx].copy()
            
            # 存储交叉验证分数
            cv_train_scores = []
            cv_test_scores = []
            
            # 交叉验证
            stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=resample_seed)
            for fold_idx, (train_idx, val_idx) in enumerate(stratified_cv.split(X_train_subset, y_train_subset)):
                # 获取交叉验证集
                X_train_cv = X_train_subset.iloc[train_idx].copy()
                y_train_cv = y_train_subset.iloc[train_idx].copy()
                X_val_cv = X_train_subset.iloc[val_idx].copy()
                y_val_cv = y_train_subset.iloc[val_idx].copy()
                
                try:
                    # 对训练集应用SMOTE过采样
                    smote = SMOTE(random_state=resample_seed * 100 + fold_idx)
                    X_train_cv_resampled, y_train_cv_resampled = smote.fit_resample(X_train_cv, y_train_cv)
                    
                    # 预处理特征
                    preprocessor_clone = preprocessor.fit(X_train_cv_resampled)
                    X_train_cv_transformed = preprocessor_clone.transform(X_train_cv_resampled)
                    X_val_cv_transformed = preprocessor_clone.transform(X_val_cv)
                    X_test_transformed = preprocessor_clone.transform(X_test)
                    
                    # 创建和训练模型
                    gnb_model = GaussianNB(**best_params)
                    gnb_model.fit(X_train_cv_transformed, y_train_cv_resampled)
                    
                    # 预测并计算AUC
                    train_prob = gnb_model.predict_proba(X_train_cv_transformed)[:, 1]
                    val_prob = gnb_model.predict_proba(X_val_cv_transformed)[:, 1]
                    test_prob = gnb_model.predict_proba(X_test_transformed)[:, 1]
                    
                    train_auc = roc_auc_score(y_train_cv_resampled, train_prob)
                    val_auc = roc_auc_score(y_val_cv, val_prob)
                    test_auc = roc_auc_score(y_test, test_prob)
                    
                    print(f"  样本量 {n_samples}, 重采样 {resample_idx+1}, 折 {fold_idx+1}/{cv}: AUC = {val_auc:.4f}")
                    
                    cv_train_scores.append(train_auc)
                    cv_test_scores.append(test_auc)
                except Exception as e:
                    print(f"  训练失败: {e}")
            
            # 汇总交叉验证结果
            if cv_train_scores and cv_test_scores:
                train_scores_resamples.append(np.mean(cv_train_scores))
                test_scores_resamples.append(np.mean(cv_test_scores))
        
        # 计算多次重采样的平均值和标准差
        if train_scores_resamples and test_scores_resamples:
            train_scores_mean.append(np.mean(train_scores_resamples))
            train_scores_std.append(np.std(train_scores_resamples))
            test_scores_mean.append(np.mean(test_scores_resamples))
            test_scores_std.append(np.std(test_scores_resamples))
        else:
            print(f"  警告: 样本量 {n_samples} 没有有效的评估结果")
    
    # 检查是否有足够的数据点来绘制曲线
    if len(train_scores_mean) < 2:
        print("\n警告: 没有足够的数据点来绘制学习曲线")
        return
    
    # 尝试加载Monaco字体
    try:
        # 获取Monaco字体
        import matplotlib.font_manager as fm
        from pathlib import Path
        
        monaco_font_path = "/System/Library/Fonts/Monaco.ttf"
        if Path(monaco_font_path).exists():
            monaco_font = fm.FontProperties(fname=monaco_font_path)
            print(f"找到Monaco字体: {monaco_font_path}")
        else:
            # 如果找不到Monaco字体，使用默认字体
            monaco_font = fm.FontProperties()
            print("未找到Monaco字体，使用默认字体")
    except Exception as e:
        print(f"加载字体时出错: {e}")
        monaco_font = fm.FontProperties()
    
    # 定义颜色
    light_green = '#7fbf7f'  # 浅绿色
    pink = '#ff9eb5'  # 粉色
    
    # 与其他图表保持一致的大小
    with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # 训练集曲线和置信区间
        ax.plot(train_sizes_abs, train_scores_mean, '-', label='Training Set (CV)', color=light_green, linewidth=2)
        ax.plot(train_sizes_abs, train_scores_mean, 'o', color=light_green, markersize=5, alpha=0.8)
        
        # 测试集曲线
        ax.plot(train_sizes_abs, test_scores_mean, '-', label='Test Set', color=pink, linewidth=2)
        ax.plot(train_sizes_abs, test_scores_mean, 'o', color=pink, markersize=5, alpha=0.8)
        
        # 训练集的置信区间
        ax.fill_between(train_sizes_abs, 
                       np.array(train_scores_mean) - np.array(train_scores_std),
                       np.array(train_scores_mean) + np.array(train_scores_std), 
                       color=light_green, alpha=0.2)
        
        # 设置固定的X轴范围为0-9000样本，确保数据点（1000-8000）完全显示
        ax.set_xlim(0, 9000)
        
        # 创建0到9000的整数刻度，每1000一个刻度
        fixed_tick_values = [i * 1000 for i in range(10)]  # 0, 1000, 2000, ..., 9000
        
        # 设置主要刻度标签 (0-9)
        ax.set_xticks(fixed_tick_values)
        ax.set_xticklabels([str(i) for i in range(10)])  # 0-9 对应 0-9000
        
        # 设置Y轴范围从0.5到1.0，间隔0.1
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        
        # 设置轴标签和标题
        ax.set_xlabel('Number of Training Samples (\u00d71000)', fontproperties=monaco_font)
        ax.set_ylabel('AUC-ROC', fontproperties=monaco_font)
        ax.set_title(f'Sample Learning Curve - {model_name}', pad=20, fontproperties=monaco_font)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.tick_params(axis='x', which='both', length=4)
        
        # 设置刻度标签的字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(monaco_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(monaco_font)
        
        # 设置图例样式与ROC曲线完全相同
        legend = ax.legend(loc='lower right', prop=monaco_font, frameon=True, framealpha=1.0, facecolor='white', edgecolor='lightgray')
        
        # 调整布局
        plt.tight_layout()
        
        # 设置边框可见性
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)
    
    # 保存图表
    plot_path = Path(plot_dir) / f"{model_name}_Learning_Curve.png"
    plt.savefig(str(plot_path), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"学习曲线已保存至: {plot_path}")
    
    # 保存原始数据
    curve_data = {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'test_scores_mean': test_scores_mean,
        'test_scores_std': test_scores_std
    }
    
    with open(str(Path(plot_data_dir) / f"{model_name}_Learning_Curve.json"), 'w') as f:
        json.dump(curve_data, f, indent=4)

def split_data(X, y, weights=None, test_size=0.2, random_state=42, stratify=True):
    """将数据分割为训练集和测试集"""
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

if __name__ == "__main__":
    main()
