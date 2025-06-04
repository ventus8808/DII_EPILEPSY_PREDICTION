import pandas as pd
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
import sys
import os

# 导入评估指标和绘图函数
from model_metrics_utils import calculate_metrics
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_learning_curve, 
    plot_confusion_matrix, plot_threshold_curve, plot_roc_curve_comparison
)
from model_plot_calibration import plot_calibration_all_data
from model_plot_DCA import plot_dca_curve, plot_dca_curve_comparison

# FNN模型预测函数
def fnn_predict(model_data, X, feature_info):
    # 加载FNN模型
    current_path = os.path.dirname(os.path.abspath(__file__))
    if current_path not in sys.path:
        sys.path.append(current_path)
    # 从FNN_Train导入FNNModel
    from FNN_Train import FNNModel
    
    # 预处理特征 - 使用FNN模型的独热编码器
    if 'encoder' in model_data:
        processed_X = preprocess_fnn_data(X, model_data['encoder'], feature_info)
    else:
        processed_X = X.values if hasattr(X, 'values') else X  # 确保输入是numpy数组
    
    # 重建 FNN 模型
    model_config = model_data['model_config']
    model = FNNModel(model_config['input_dim'], model_config['params'])
    model.update(model_data['model_state'])
    
    # 转换为mx.array并进行预测
    import mlx.core as mx
    X_mx = mx.array(processed_X.astype(np.float32))
    y_prob = model(X_mx).tolist()
    
    # 如果y_prob是嵌套列表，则展平
    if isinstance(y_prob[0], list):
        y_prob = [item[0] for item in y_prob]
    
    # 二分类预测
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    
    return y_pred, np.array(y_prob)

# SVM模型预测函数
def svm_predict(model_data, X):
    # 如果SVM模型是作为完整模型加载的
    if hasattr(model_data, 'predict_proba'):
        y_prob = model_data.predict_proba(X)[:, 1]
        y_pred = model_data.predict(X)
    # 如果SVM模型被存储为字典或其他形式
    else:
        # 使用一个默认的预测处理
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
            y_pred = model.predict(X)
        else:
            # 如果没有直接的模型对象，使用平均概率
            y_prob = np.full(len(X), 0.5)  # 默认平均概率
            y_pred = (y_prob >= 0.5).astype(int)  # 默认预测
    
    return y_pred, y_prob

# 特征预处理函数
def preprocess_fnn_data(X, encoder, feature_info):
    # 分离类型变量和数值变量
    categorical_features = feature_info.get('categorical_features', [])
    numeric_features = feature_info.get('numeric_features', [])
    dii_column = ['DII_food']
    
    # 抽取各组特征
    numeric_data = X[dii_column + numeric_features].copy() if numeric_features else X[dii_column].copy()
    categorical_data = X[categorical_features].copy() if categorical_features else None
    
    # 处理类型化变量
    if categorical_data is not None:
        # 转换为数值类型，缺失值处理为0
        categorical_data = categorical_data.fillna(0).astype(int)
        # 使用独热编码器变换
        encoded_cats = encoder.transform(categorical_data)
        
        # 获取编码后的特征名
        encoded_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = encoder.categories_[i]
            for category in categories:
                encoded_feature_names.append(f"{feature}_{category}")
        
        # 创建result_df准备合并
        result_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, 
                               index=X.index if hasattr(X, 'index') else None)
    else:
        result_df = pd.DataFrame(index=X.index if hasattr(X, 'index') else None)
    
    # 合并数值特征
    if len(numeric_data.columns) > 0:
        for col in numeric_data.columns:
            result_df[col] = numeric_data[col].values
    
    return result_df.values  # 直接返回numpy数组而不是DataFrame

# 数据分割函数
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

def main():
    # 命令行参数处理，允许覆盖配置文件中的设置
    parser = argparse.ArgumentParser(description="集成投票模型评估与可视化")
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
    ensemble_dir = model_dir / 'Ensemble_Voting'
    plot_dir = Path(config['plot_dir'])
    plot_dir.mkdir(exist_ok=True)
    plot_data_dir = Path('plot_original_data')
    plot_data_dir.mkdir(exist_ok=True)
    result_dir = Path(config.get('output_dir', 'result'))
    result_dir.mkdir(exist_ok=True)
    
    # 加载模型
    try:
        # 首先尝试从专用目录加载
        model_path = ensemble_dir / 'Ensemble_Voting_model.pkl'
        with open(model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
    except FileNotFoundError:
        # 如果不存在，尝试从主目录加载
        model_path = model_dir / 'Ensemble_voting_model.pkl'
        with open(model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
    
    # 解析集成模型信息
    models_dict = ensemble_model['models']
    threshold = ensemble_model['threshold']
    
    print(f"成功加载模型，集成了{len(models_dict)}个底层模型")
    preprocessor = ensemble_model['preprocessor']
    feature_info = ensemble_model.get('feature_info', {})
    
    # 加载数据
    df = pd.read_csv(data_path)
    features = feature_info.get('features', [])
    
    # 如果feature_info中没有features信息，尝试获取
    if not features:
        print("Warning: 未在模型中找到特征信息，使用默认特征列表")
        categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel']
        numeric_features = [col for col in ['Age', 'BMI'] if col in df.columns]
        features = ['DII_food'] + numeric_features + categorical_features
    
    X = df[features]
    y = df['Epilepsy']
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
    # 使用外部定义的分割函数
    
    X_train, X_test, y_train, y_test, weights_train, weights_test = split_data(X, y, weights)
    
    # 预测
    print(f"加载了{len(models_dict)}个底层模型，进行集成预测")
    
    # 加载FNN特征信息
    feature_info = {}
    try:
        with open(model_dir / 'FNN_feature_info.json', 'r') as f:
            feature_info = json.load(f)
    except:
        feature_info = {
            'categorical_features': ['Gender', 'Education', 'Marriage', 'Smoke', 'Alcohol', 'Employment', 'ActivityLevel'],
            'numeric_features': ['Age', 'BMI']
        }
    
    # 使用外部定义的预处理函数
    
    # 使用外部定义的FNN预测函数
        
    # 使用外部定义的SVM预测函数
    
    # 获取各个模型预测
    model_probs = {}
    for model_name, (model, weight) in models_dict.items():
        print(f"获取{model_name}模型预测...")
        if model_name == 'FNN':
            # 对FNN模型使用特殊处理
            _, fnn_prob = fnn_predict(model, X_test, feature_info)
            model_probs[model_name] = fnn_prob * weight
        elif model_name == 'SVM':
            # 对SVM模型使用特殊处理
            _, svm_prob = svm_predict(model, X_test)
            model_probs[model_name] = svm_prob * weight
        else:
            model_probs[model_name] = model.predict_proba(X_test)[:, 1] * weight
    
    # 计算加权平均
    total_weight = sum(weight for _, weight in models_dict.values())
    y_prob = sum(probs for probs in model_probs.values()) / total_weight
    
    # 使用阈值进行最终预测
    y_pred = (y_prob >= threshold).astype(int)
    
    model_name = "Ensemble_Voting"
    
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
        
        # 保存指标到结果目录
        metrics_json = {k: float(v) for k, v in metrics.items()}
        with open(result_dir / f'{model_name}_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=4)
    
    # 绘图
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
        X_test_no_dii['DII_food'] = X['DII_food'].median()  # 填充中位数而不是0
        
        # 预测测试集的概率
        y_prob_with_dii = y_prob  # 已有的测试集预测结果
        
        # 使用与原始预测相同的方式，手动集成各模型的预测
        model_probs_no_dii = {}
        for base_model_name, (model, weight) in models_dict.items():
            if base_model_name == 'FNN':
                # 对FNN模型使用特殊处理
                _, fnn_prob = fnn_predict(model, X_test_no_dii, feature_info)
                model_probs_no_dii[base_model_name] = fnn_prob * weight
            elif base_model_name == 'SVM':
                # 对SVM模型使用特殊处理
                _, svm_prob = svm_predict(model, X_test_no_dii)
                model_probs_no_dii[base_model_name] = svm_prob * weight
            else:
                model_probs_no_dii[base_model_name] = model.predict_proba(X_test_no_dii)[:, 1] * weight
        
        # 计算所有模型的加权平均
        y_prob_no_dii = sum(probs for probs in model_probs_no_dii.values()) / total_weight
        
        # 构建比较字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_with_dii,
            f"{model_name}(without DII)": y_prob_no_dii
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
        # 预测全量数据集的概率
        model_probs_all = {}
        for model_name_local, (model, weight) in models_dict.items():
            if model_name_local == 'FNN':
                # 对FNN模型使用特殊处理
                _, fnn_prob = fnn_predict(model, X, feature_info)
                model_probs_all[model_name_local] = fnn_prob * weight
            elif model_name_local == 'SVM':
                # 对SVM模型使用特殊处理
                _, svm_prob = svm_predict(model, X)
                model_probs_all[model_name_local] = svm_prob * weight
            else:
                model_probs_all[model_name_local] = model.predict_proba(X)[:, 1] * weight
        
        total_weight = sum(weight for _, weight in models_dict.values())
        y_prob_all = sum(probs for probs in model_probs_all.values()) / total_weight
        
        # 使用全量数据集绘制校准曲线
        try:
            # 捕获校准曲线绘制过程中可能的异常，防止程序中断
            plot_calibration_all_data(y, y_prob_all, weights, model_name, plot_dir, plot_data_dir)
            # 检查文件是否正确保存，确保大小写一致
            calibration_file = Path(plot_dir) / f"{model_name}_Calibration_Curve.png"
            if not calibration_file.exists():
                print(f"警告：校准曲线文件{calibration_file}未找到，可能使用了不同的命名格式")
        except Exception as e:
            print(f"校准曲线绘制过程中出错: {e}")
        print(f"校准曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_confusion:
        start_time = time.time()
        print("绘制混淆矩阵...")
        # 确保使用模型名称的一致大小写形式
        # 在保存文件名中明确使用大写和下划线以匹配其他图像命名格式
        plot_confusion_matrix(y_test, y_pred, model_name, plot_dir, plot_data_dir, normalize=False)
        # 如果保存的文件使用的是小写文件名，手动复制一份为大写文件名
        confusion_file_lower = Path(plot_dir) / f"{model_name.lower()}_confusion_matrix.png"
        confusion_file_upper = Path(plot_dir) / f"{model_name}_Confusion_Matrix.png"
        if confusion_file_lower.exists() and not confusion_file_upper.exists():
            import shutil
            shutil.copy(str(confusion_file_lower), str(confusion_file_upper))
        print(f"混淆矩阵绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    # 对于集成投票模型，学习曲线不适用于评估阶段，因为需要重新训练底层模型
    if draw_learning:
        start_time = time.time()
        print("学习曲线对于预训练的集成投票模型不适用")
        print("因为该模型已经训练好，无法展示学习曲线过程")
        print(f"学习曲线处理跳过 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_threshold:
        start_time = time.time()
        print("绘制阈值曲线...")
        # 确保使用一致的大小写形式
        plot_threshold_curve(y_test, y_prob, model_name, plot_dir, plot_data_dir)
        # 如果保存的文件使用的是小写文件名，手动复制一份为大写文件名
        threshold_file_lower = Path(plot_dir) / f"{model_name.lower()}_threshold_curve.png"
        threshold_file_upper = Path(plot_dir) / f"{model_name}_Threshold_Curve.png"
        if threshold_file_lower.exists() and not threshold_file_upper.exists():
            import shutil
            shutil.copy(str(threshold_file_lower), str(threshold_file_upper))
        print(f"阈值曲线绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    if draw_dca:
        start_time = time.time()
        print("绘制决策曲线分析(DCA)...")
        # 使用与校准曲线相同的处理方式：全量数据集+SMOTE过采样
        # 预测全量数据集的概率
        X_full = df[features]
        y_full = df['Epilepsy']
        weights_full = df['WTDRD1'] if 'WTDRD1' in df.columns else None
        
        # 使用与测试集相同的手动集成预测方式
        model_probs_full = {}
        for base_model_name, (model, weight) in models_dict.items():
            if base_model_name == 'FNN':
                # 对FNN模型使用特殊处理
                _, fnn_prob = fnn_predict(model, X_full, feature_info)
                model_probs_full[base_model_name] = fnn_prob * weight
            elif base_model_name == 'SVM':
                # 对SVM模型使用特殊处理
                _, svm_prob = svm_predict(model, X_full)
                model_probs_full[base_model_name] = svm_prob * weight
            else:
                model_probs_full[base_model_name] = model.predict_proba(X_full)[:, 1] * weight
        
        # 计算所有模型的加权平均
        total_weight = sum(weight for _, weight in models_dict.values())
        y_prob_all = sum(probs for probs in model_probs_full.values()) / total_weight
        
        # 增加DII贡献评估
        print("\n评估DII对模型预测能力的贡献...")
        # 创建无DII版本的数据(将DII_food列设为0)
        X_no_dii = X_full.copy()
        X_no_dii['DII_food'] = X['DII_food'].median()  # 填充中位数而不是0
        
        # 预测无DII数据
        model_probs_no_dii = {}
        for base_model_name, (model, weight) in models_dict.items():
            if base_model_name == 'FNN':
                # 对FNN模型使用特殊处理
                _, fnn_prob = fnn_predict(model, X_no_dii, feature_info)
                model_probs_no_dii[base_model_name] = fnn_prob * weight
            elif base_model_name in ['SVM', 'PSGD']:
                # 对SVM或PSGD模型使用特殊处理
                try:
                    _, svm_prob = svm_predict(model, X_no_dii)
                    model_probs_no_dii[base_model_name] = svm_prob * weight
                except Exception as e:
                    print(f"预测{base_model_name}遇到错误: {e}")
                    model_probs_no_dii[base_model_name] = model.predict_proba(X_no_dii)[:, 1] * weight
            else:
                model_probs_no_dii[base_model_name] = model.predict_proba(X_no_dii)[:, 1] * weight
        
        # 计算所有模型的加权平均
        y_prob_no_dii = sum(probs for probs in model_probs_no_dii.values()) / total_weight
        
        # 构建DII对比字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_all,
            f"{model_name}(without DII)": y_prob_no_dii
        }
        
        # 绘制DII贡献对比DCA曲线 - 现在直接返回数据而不是路径
        comparison_data = plot_dca_curve_comparison(y_full, y_probs_dict, weights_full, model_name, plot_dir, plot_data_dir, use_smote=True)
        
        # 创建单模型格式数据
        single_model_data = {
            "thresholds": comparison_data["thresholds"],
            "net_benefits_model": comparison_data["models"][f"{model_name}(all feature)"],
            "net_benefits_all": comparison_data["treat_all"],
            "net_benefits_none": comparison_data["treat_none"],
            "model_name": model_name,
            "prevalence": comparison_data.get("prevalence", np.mean(y_full))
        }
        
        # 保存单模型数据
        with open(plot_data_dir / f"{model_name}_DCA.json", 'w') as f:
            json.dump(single_model_data, f, indent=4)
            
        print(f"已从比较版本中提取并保存单模型DCA数据 -> {model_name}_DCA.json")
        print(f"决策曲线分析(DCA)绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
