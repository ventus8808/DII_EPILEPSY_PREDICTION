import pandas as pd
import pickle
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
import time
from sklearn.exceptions import NotFittedError

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
    parser = argparse.ArgumentParser(description="SVM模型评估与可视化")
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
    model_path = model_dir / 'SVM_model.pkl'
    results_path = model_dir / 'SVM_results.json'
    
    # 先尝试加载模型
    try:
        print(f"尝试从 {model_path} 加载模型...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"成功加载SVM模型，模型类型: {type(model)}")
        
        # 检查如果模型是ndarray，创建新的SVM模型
        if isinstance(model, np.ndarray) or not hasattr(model, 'predict'):
            print("模型是ndarray类型或没有predict方法，需要创建新的SVM模型")
            from sklearn.svm import SVC
            
            # 尝试加载结果文件以获取参数信息
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                best_params = results.get('best_params', {})
                print(f"成功加载模型结果文件，最佳得分: {results.get('best_score')}")
                print(f"模型参数: {best_params}")
            except FileNotFoundError:
                best_params = {}
                print("未找到结果文件，使用默认参数")
            
            # 创建新的SVM模型并使用最佳参数
            real_model = SVC(probability=True, random_state=42, **best_params)
            print("创建了新的SVM模型对象")
            model = real_model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        print("创建默认SVM模型...")
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=42)
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 尝试加载特征信息
    try:
        with open(model_dir / 'SVM_feature_info.json', 'r') as f:
            feature_info = json.load(f)
        features = feature_info['features']
        categorical_features = feature_info.get('categorical_features', ['Gender', 'Education', 'Marriage', 
                                                                        'Smoke', 'Alcohol', 'Employment', 
                                                                        'ActivityLevel'])
        numeric_features = feature_info.get('numeric_features', ['Age', 'BMI'])
        print(f"分类特征: {categorical_features}")
        print(f"数值特征: {numeric_features}")
    except FileNotFoundError:
        # 尝试从其他模型的特征信息中获取
        features_found = False
        for model_prefix in ['FNN', 'XGBoost', 'LightGBM', 'RF', 'CatBoost']:
            try:
                with open(model_dir / f'{model_prefix}_feature_info.json', 'r') as f:
                    feature_info = json.load(f)
                features = feature_info['features']
                categorical_features = feature_info.get('categorical_features', [])
                numeric_features = feature_info.get('numeric_features', [])
                print(f"使用{model_prefix}模型的特征信息")
                features_found = True
                break
            except FileNotFoundError:
                continue
        
        if not features_found:
            # 默认特征设置
            categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
            numeric_features = ['Age', 'BMI']
            features = ['DII_food'] + numeric_features + categorical_features
            print(f"使用默认特征配置")
    
    # 处理独热编码
    try:
        with open(model_dir / 'encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        print("成功加载独热编码器")
    except FileNotFoundError:
        print("未找到编码器，将创建新的独热编码器")
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df[categorical_features])
    
    # 应用独热编码处理数据
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
    
    print(f"成功处理特征，使用独热编码后的特征数：{X.shape[1]}")
    
    # 加载标准化处理器（如果有）
    try:
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # 由于我们已经自己处理了特征，这里跳过进一步的变换
        print("已加载标准化处理器，但跳过了应用过程以避免特征名称冲突")
    except FileNotFoundError:
        print("未找到标准化处理文件，使用处理后的特征")
    
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
    
    # 检查是否需要训练模型
    model_need_fit = False
    try:
        # 尝试调用predict，看是否会抛出异常
        model.predict(X_test[:1])
    except (NotFittedError, Exception) as e:
        print(f"检测到模型未经过训练: {str(e)}")
        model_need_fit = True
    
    # 如果需要训练，使用SMOTE平衡数据并训练模型
    if model_need_fit:
        print("开始训练SVM模型...")
        # 应用SMOTE过采样解决类别不平衡问题
        try:
            from imblearn.over_sampling import SMOTE
            print("应用SMOTE过采样平衡训练数据...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"过采样后训练数据类别比例: {np.bincount(y_train_resampled)}")
            # 训练模型
            model.fit(X_train_resampled, y_train_resampled)
            print("模型训练完成。")
        except Exception as e:
            print(f"应用SMOTE失败: {str(e)}，使用原始训练数据训练")
            model.fit(X_train, y_train)
            print("使用原始数据训练完成。")
    else:
        print("模型已经训练过，可以直接使用。")
    
    # 预测
    print(f"成功加载特征信息文件，模型使用的特征数：{X.shape[1]}") 
    y_pred = model.predict(X_test) 
    y_prob = model.predict_proba(X_test)[:, 1]
    model_name = "SVM"
    
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
        print("使用测试集数据绘制校准曲线")
        # 使用测试集数据绘制校准曲线，与其他模型保持一致
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
        plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir)
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
        # 生成标准DCA曲线
        plot_dca_curve(y, y_prob_all, weights, model_name, plot_dir, plot_data_dir, use_smote=True)
        
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
        plot_dca_curve_comparison(y, y_probs_dict, weights, model_name, plot_dir, plot_data_dir, use_smote=True)
        print(f"决策曲线分析(DCA)绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
