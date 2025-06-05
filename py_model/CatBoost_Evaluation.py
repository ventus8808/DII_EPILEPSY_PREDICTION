import pandas as pd
import numpy as np
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time

# 导入评估指标和绘图函数
from model_metrics_utils import calculate_metrics
from model_plot_utils import (
    plot_roc_curve, plot_pr_curve, plot_learning_curve, 
    plot_confusion_matrix, plot_threshold_curve, plot_roc_curve_comparison
)
from model_plot_calibration import plot_calibration_all_data
from model_plot_DCA import plot_dca_curve, plot_dca_curve_comparison

def main():
    # 命令行参数处理，允许覆盖配置文件中的设置
    parser = argparse.ArgumentParser(description="CatBoost模型评估与可视化")
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
    model_path = model_dir / 'CatBoost_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(data_path)
    with open(model_dir / 'CatBoost_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    features = feature_info['features']
    cat_feature_indices = feature_info['cat_feature_indices']
    X = df[features]
    y = df['Epilepsy']
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
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
    
    # 预测
    print("成功加载特征信息文件，模型使用的特征数：{}".format(len(features)))
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    model_name = "CatBoost"
    
    # 输出评估设置信息
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
    
    # 计算评估指标
    if calc_metrics:
        start_time = time.time()
        print("计算评估指标...")
        metrics = calculate_metrics(y_test, y_pred, y_prob, weights_test)
        metrics_path = result_dir / f'{model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nTest Set Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
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
        
        # 获取DII_food的中位数
        dii_median = X['DII_food'].median()
        print(f"DII_food的中位数为: {dii_median:.4f}")
        
        # 创建填充DII中位数版本的测试数据
        X_test_median_dii = X_test.copy()
        X_test_median_dii['DII_food'] = dii_median  # 填充中位数而不是0
        
        # 预测测试集的概率
        y_prob_with_dii = y_prob  # 已有的测试集预测结果
        y_prob_median_dii = model.predict_proba(X_test_median_dii)[:, 1]
        
        # 构建比较字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_with_dii,
            f"{model_name}(without DII)": y_prob_median_dii  # 标签保持一致，便于后续处理
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
        y_prob_all = model.predict_proba(X)[:, 1]
        # 使用全量数据集绘制校准曲线，与CatBoost_Plot_Calibration.py保持一致
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
        # 增加DII贡献评估
        print("\n评估DII对模型预测能力的贡献...")
        
        # 获取DII_food的中位数
        dii_median = X['DII_food'].median()
        print(f"DII_food的中位数为: {dii_median:.4f}")
        
        # 创建填充DII中位数版本的数据
        X_median_dii = X.copy()
        X_median_dii['DII_food'] = dii_median  # 填充中位数而不是0
        
        # 预测填充中位数的数据
        y_prob_median_dii = model.predict_proba(X_median_dii)[:, 1]
        
        # 构建DII对比字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_all,
            f"{model_name}(without DII)": y_prob_median_dii  # 标签保持一致，便于后续处理
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
        
        # 直接调用plot_dca_curve函数绘制单模型DCA图表
        plot_dca_curve(y, y_prob_all, weights, model_name, plot_dir, plot_data_dir, use_smote=True)
        
        print(f"决策曲线分析(DCA)绘制完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
