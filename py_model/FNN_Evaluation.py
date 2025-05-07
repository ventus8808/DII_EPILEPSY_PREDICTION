import pandas as pd
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
import mlx.core as mx

# 导入评估指标和绘图函数
from model_metrics_utils import calculate_metrics
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_learning_curve, 
    plot_confusion_matrix, plot_threshold_curve
)
from model_plot_calibration import plot_calibration_all_data

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
    
    # 打印评估与可视化设置
    print("\n===== 评估与可视化设置 =====")
    print(f"计算评估指标: {'是' if calc_metrics else '否'}")
    print(f"绘制ROC曲线: {'是' if draw_roc else '否'}")
    print(f"绘制PR曲线: {'是' if draw_pr else '否'}")
    print(f"绘制校准曲线: {'是' if draw_calibration else '否'}")
    print(f"绘制混淆矩阵: {'是' if draw_confusion else '否'}")
    print(f"绘制学习曲线: {'是' if draw_learning else '否'}")
    print(f"绘制阈值曲线: {'是' if draw_threshold else '否'}")
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
        def custom_fnn_learning_curve(model_data, X, y, weights=None, cv=2, train_sizes=None, random_state=42, plot_dir=None, plot_data_dir=None):
            from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_auc_score
            import numpy as np
            from imblearn.over_sampling import SMOTE
            from pathlib import Path
            import json
            
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
            
            # 定义要测试的样本量 - 使用1000开始，每500递增，与model_plot_utils.py一致
            start_size = 1000
            step_size = 500
            max_train_size = len(X)
            train_sizes_abs = list(range(start_size, max_train_size, step_size))
            # 确保包含最后一个点（全部训练数据）
            if train_sizes_abs[-1] != max_train_size and max_train_size - train_sizes_abs[-1] > 100:
                train_sizes_abs.append(max_train_size)
            
            train_scores = []
            test_scores = []
            
            # 保存每一个样本量的所有交叉验证分数，用于计算置信区间
            train_scores_all_folds = []
            
            print(f"正在计算学习曲线，与其他模型保持一致的实现方式...")
            
            for n_samples in train_sizes_abs:
                # 使用分层抽样选择n_samples个样本进行训练
                # 处理边界情况：如果n_samples等于训练集大小，直接使用全部样本
                if n_samples >= len(X):
                    X_train_subset = X
                    y_train_subset = y
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
                    for train_idx, _ in sss.split(X, y):
                        X_train_subset = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                        y_train_subset = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                
                try:
                    # 使用传入的cv参数决定折数
                    cv_splits = cv if isinstance(cv, int) else 5  # 默认与函数参数一致
                    stratified_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                    train_cv_scores = []
                    
                    print(f"  计算样本量 {n_samples}...")
                    
                    # 使用交叉验证评估训练集性能
                    fold_scores = []  # 收集当前样本量的每折分数
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
                        
                        # 对训练集应用SMOTE过采样
                        smote = SMOTE(random_state=42)
                        X_train_cv_resampled, y_train_cv_resampled = smote.fit_resample(X_train_cv, y_train_cv)
                        
                        # 训练模型并预测
                        cv_model = SimpleFFN(input_dim=X_train_cv.shape[1])
                        cv_model.fit(X_train_cv_resampled, y_train_cv_resampled)
                        val_pred = cv_model.predict_proba(X_val_cv)
                        
                        # 计算验证集上AUC
                        try:
                            val_score = roc_auc_score(y_val_cv, val_pred)
                            train_cv_scores.append(val_score)
                            fold_scores.append(val_score)
                            print(f"  - 折 {fold_idx+1}/{cv_splits}: AUC = {val_score:.4f}")
                        except Exception as e:
                            print(f"交叉验证失败: {e}")
                    
                    # 如果有效的交叉验证分数，则平均
                    if train_cv_scores:
                        train_score = np.mean(train_cv_scores)
                        # 存储每一个样本量的所有折分数，用于计算置信区间
                        train_scores_all_folds.append(fold_scores)
                    else:
                        # 如果交叉验证失败，回退到直接训练
                        model = SimpleFFN(input_dim=X_train_subset.shape[1])
                        model.fit(X_train_subset, y_train_subset)
                        train_pred = model.predict_proba(X_train_subset)
                        train_score = roc_auc_score(y_train_subset, train_pred)
                        print("交叉验证失败，使用直接评估")
                        # 当交叉验证失败时，添加空列表
                        train_scores_all_folds.append([])
                    
                    # 对子集应用SMOTE过采样，然后在测试集上评估
                    smote_test = SMOTE(random_state=42+n_samples)  # 使用不同的随机种子
                    X_train_subset_resampled, y_train_subset_resampled = smote_test.fit_resample(X_train_subset, y_train_subset)
                    
                    # 在过采样后的训练集上训练模型并在测试集上评估
                    model = SimpleFFN(input_dim=X_train_subset.shape[1])
                    model.fit(X_train_subset_resampled, y_train_subset_resampled)
                    test_pred = model.predict_proba(X_test)
                    test_score = roc_auc_score(y_test, test_pred)
                    
                    train_scores.append(train_score)
                    test_scores.append(test_score)
                    
                    print(f"样本数量: {n_samples}, 训练集AUC: {train_score:.4f}, 测试集AUC: {test_score:.4f}")
                except Exception as e:
                    print(f"样本数量 {n_samples} 训练失败: {e}")
                    # 如果失败，添加None或上一个值
                    if len(train_scores) > 0:
                        train_scores.append(train_scores[-1])
                        test_scores.append(test_scores[-1])
                    else:
                        train_scores.append(None)
                        test_scores.append(None)
            
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
            
            # 计算置信区间
            train_scores_sem = []
            for fold_scores in valid_train_scores_all_folds:
                if len(fold_scores) >= 2:  # 需要至少2个有效折才能计算标准误差
                    # 计算标准误差 (SEM = STD / sqrt(n))
                    std = np.std(fold_scores, ddof=1)
                    sem = std / np.sqrt(len(fold_scores))
                    train_scores_sem.append(sem)
                else:
                    train_scores_sem.append(0.0)  # 如果数据不足，设置标准误差为0
            
            # 绘图 - 与model_plot_utils中完全一致
            with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
                # 保持与其他图表一致的大小
                fig, ax = plt.subplots(figsize=(7, 7))
                
                # 用浅色填充置信区间
                z_value = 1.0  # 对应68%置信区间
                ax.fill_between(valid_train_sizes, 
                                [max(0.5, score - z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                                [min(1.0, score + z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                                alpha=0.15, color='green')
                
                # 绘制主线 - 使用与model_plot_utils相同的颜色
                light_green = '#7fbf7f'  # 浅绿色
                pink = '#ff9eb5'  # 粉色
                
                ax.plot(valid_train_sizes, valid_train_scores, '-', label='Training Set (CV)', color=light_green, linewidth=2)
                ax.plot(valid_train_sizes, valid_test_scores, '-', label='Test Set', color=pink, linewidth=2)
                
                # 单独绘制数据点
                ax.plot(valid_train_sizes, valid_train_scores, 'o', color=light_green, markersize=5, alpha=0.8)
                ax.plot(valid_train_sizes, valid_test_scores, 'o', color=pink, markersize=5, alpha=0.8)
                
                # 设置固定的X轴范围
                ax.set_xlim(0, 9000)  
                
                # 创建刻度
                fixed_tick_values = [i * 1000 for i in range(10)]
                
                # 设置网格线
                ax.set_xticks(valid_train_sizes, minor=True)
                ax.set_xticklabels([""] * len(valid_train_sizes), minor=True)
                
                # 设置主要刻度标签
                ax.set_xticks(fixed_tick_values)
                ax.set_xticklabels([str(i) for i in range(10)])
                
                # 确保网格线显示
                ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
                
                # 设置轴标签
                ax.set_xlabel("Number of Training Samples (×1000)")
                ax.set_ylabel("AUC-ROC")
                
                # 设置y轴范围
                ax.set_ylim(0.5, 1.0)
                ax.set_yticks(np.arange(0.5, 1.01, 0.1))
                
                # 设置标题
                ax.set_title(f"Sample Learning Curve - {model_name}", pad=20)
                
                # 添加网格
                ax.grid(linestyle='--', alpha=0.3, color='gray')
                
                # 设置图例
                ax.legend(loc='lower right', frameon=True, framealpha=1.0, facecolor='white', edgecolor='lightgray')
                
                # 显示上边框和右边框
                for spine in ['top','right']:
                    ax.spines[spine].set_visible(True)
            
            fig.tight_layout()
            fig.savefig(str(Path(plot_dir) / f"{model_name}_Sample_Learning_Curve.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # 保存数据
            def save_plot_data(data, filename):
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
            
            save_plot_data({
                'train_sizes': valid_train_sizes,
                'train_scores': valid_train_scores,
                'test_scores': valid_test_scores
            }, str(Path(plot_data_dir) / f"{model_name}_Sample_Learning_Curve.json"))
            
            return valid_train_sizes, valid_train_scores, valid_test_scores
        
        # 调用自定义学习曲线函数
        train_sizes, train_scores, test_scores = custom_fnn_learning_curve(
            model, X_train, y_train, weights_train, cv=5, 
            random_state=42, plot_dir=plot_dir, plot_data_dir=plot_data_dir
        )
        print(f"学习曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_threshold:
        start_time = time.time()
        print("绘制阈值曲线...")
        plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
        print(f"阈值曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
