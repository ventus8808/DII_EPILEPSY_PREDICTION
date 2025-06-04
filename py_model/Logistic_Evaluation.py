import pandas as pd
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

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
    parser = argparse.ArgumentParser(description="逻辑回归模型评估与可视化")
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
    
    # 模型名称
    model_name = "Logistic"
    
    # 加载数据
    print(f"加载数据：{data_path}")
    df = pd.read_csv(data_path)
    
    # 尝试加载特征信息
    try:
        with open(model_dir / 'Logistic_feature_info.json', 'r') as f:
            feature_info = json.load(f)
        features = feature_info['features']
        print(f"使用Logistic模型特征信息文件中的特征")
    except FileNotFoundError:
        # 尝试从其他模型的特征信息中获取
        features_found = False
        for model_prefix in ['CatBoost', 'XGBoost', 'LightGBM', 'SVM', 'FNN']:
            try:
                with open(model_dir / f'{model_prefix}_feature_info.json', 'r') as f:
                    feature_info = json.load(f)
                features = feature_info['features']
                print(f"使用{model_prefix}模型的特征信息")
                features_found = True
                break
            except FileNotFoundError:
                continue
        
        if not features_found:
            # 如果是config.yaml中的配置
            if 'covariates' in config:
                features = [config['exposure']] + config['covariates']
            else:
                # 默认特征集
                features = [config.get('exposure', 'DII_food')] + [
                    'Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
                    'Alcohol', 'Employment', 'ActivityLevel'
                ]
            print(f"未找到特征信息文件，使用配置文件中的特征: {features}")
    
    # 获取特征和标签数据
    outcome = config.get('outcome', 'Epilepsy')
    X = df[features]
    y = df[outcome]
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    
    # 所有变量转为类别编码（参考Logistic_Plot.py）
    for col in X.columns:
        if X[col].dtype.name == 'category' or X[col].dtype == object:
            X[col] = pd.Categorical(X[col]).codes
    
    # 标签编码
    if y.dtype.name == 'category' or y.dtype == object:
        y = pd.Categorical(y).codes
    
    # 去除缺失
    valid_idx = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    if weights is not None:
        weights = weights.loc[valid_idx]
    
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
    
    # 使用SMOTE过采样平衡训练数据（参考Logistic_Plot.py）
    print("应用SMOTE过采样处理不平衡数据...")
    print(f"SMOTE过采样 - 训练集原始分布: {pd.Series(y_train).value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE过采样 - 训练集分布平衡后: {pd.Series(y_train_res).value_counts().to_dict()}")
    
    # 训练逻辑回归模型
    print("开始训练逻辑回归模型...")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_res, y_train_res)
    print(f"模型训练完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    # 预测
    print(f"开始模型预测，使用特征数：{len(features)}")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
        
        # 新增：ROC比较曲线（含DII vs 不含DII）
        start_time = time.time()
        print("绘制ROC比较曲线（评估DII贡献）...")
        
        # 创建无DII版本的测试数据
        X_test_no_dii = X_test.copy()
        X_test_no_dii['DII_food'] = 0
        
        # 预测测试集的概率
        y_prob_with_dii = y_prob  # 已有的测试集预测结果
        y_prob_no_dii = model.predict_proba(X_test_no_dii)[:, 1]
        
        # 构建比较字典
        y_probs_dict = {
            f"{model_name}(all feature)": y_prob_with_dii,
            f"{model_name}(without DII)": y_prob_no_dii
        }
        
        # 调用比较函数，在测试集上进行比较，不使用SMOTE过采样
        plot_roc_curve_comparison(y_test, y_probs_dict, weights_test, model_name, plot_dir, plot_data_dir, use_smote=False)
        # 打印提示，明确说明ROC比较中不使用SMOTE过采样
        print("注意: ROC比较曲线保持与单独ROC曲线相同的数据处理方式，不进行SMOTE过采样")
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
    
    # 打印逻辑回归模型系数（β值）
    print("\n===== 逻辑回归模型系数 (β值) =====")
    coef = model.coef_[0]  # 获取系数，对于二分类只有一组系数
    intercept = model.intercept_[0]  # 获取截距
    
    # 创建特征名称和系数的对应关系
    feature_names = X.columns.tolist()
    coefficients = list(coef)
    
    # 打印截距(Intercept)
    print(f"Intercept: {intercept:.4f}")
    
    # 打印各个特征的系数
    print("\n特征系数（从高到低排序）:")
    
    # 创建特征-系数对，并按系数绝对值降序排序
    feature_coef_pairs = sorted(zip(feature_names, coefficients), 
                               key=lambda x: abs(x[1]), reverse=True)
    
    # 创建简洁的JSON格式
    coef_dict = {"Intercept": float(intercept)}
    for feature, coefficient in feature_coef_pairs:
        print(f"{feature}: {coefficient:.4f}")
        coef_dict[feature] = float(coefficient)
    
    # 将系数保存到model目录，文件名为LR_best_params.json
    model_dir_path = Path(config['model_dir'])
    coef_path = model_dir_path / "LR_best_params.json"
    with open(coef_path, 'w') as f:
        json.dump(coef_dict, f, indent=4)
    
    print(f"\n模型系数已保存到: {coef_path}")
    
    # 计算和打印优势比(Odds Ratio)，但不保存
    print("\n===== 优势比 (Odds Ratio) =====")
    print("特征优势比（从高到低排序）:")
    
    # 计算优势比并按降序排序
    for feature, coefficient in feature_coef_pairs:
        odds_ratio = np.exp(coefficient)
        ci_lower = np.exp(coefficient - 1.96 * np.sqrt(1/len(y_train_res)))  # 简化的95%置信区间下限
        ci_upper = np.exp(coefficient + 1.96 * np.sqrt(1/len(y_train_res)))  # 简化的95%置信区间上限
        
        print(f"{feature}: {odds_ratio:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    # 添加截距的优势比
    intercept_odds = np.exp(intercept)
    ci_lower = np.exp(intercept - 1.96 * np.sqrt(1/len(y_train_res)))
    ci_upper = np.exp(intercept + 1.96 * np.sqrt(1/len(y_train_res)))
    print(f"Intercept: {intercept_odds:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    print("\n所有评估与可视化任务完成！")

if __name__ == "__main__":
    main()
