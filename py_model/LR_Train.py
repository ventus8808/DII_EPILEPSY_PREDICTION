import pandas as pd
import pickle
import json
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
# 恢复SMOTE导入
from imblearn.over_sampling import SMOTE

def main():
    # 命令行参数处理
    parser = argparse.ArgumentParser(description="逻辑回归模型训练")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 读取配置文件
    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 读取基本配置
    data_path = Path(config['data_path'])
    model_dir = Path(config['model_dir'])
    model_dir.mkdir(exist_ok=True)
    
    # 模型名称
    model_name = "LR"
    
    # 加载数据
    print(f"加载数据：{data_path}")
    df = pd.read_csv(data_path)
    
    # 尝试加载特征信息
    try:
        with open(model_dir / 'LR_feature_info.json', 'r') as f:
            feature_info = json.load(f)
        features = feature_info['features']
        print(f"使用LR模型特征信息文件中的特征")
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
    
    # 所有变量转为类别编码
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
    
    # 使用SMOTE过采样平衡训练数据，设置正例比例为30%
    print("应用SMOTE过采样处理不平衡数据...")
    print(f"SMOTE过采样 - 训练集原始分布: {pd.Series(y_train).value_counts().to_dict()}")
    
    # 计算采样策略，使正例比例为30%
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    target_pos_ratio = 0.58 # 目标正例比例
    target_pos_count = int(neg_count * target_pos_ratio / (1 - target_pos_ratio))
    
    # 设置采样策略
    sampling_strategy = {1: target_pos_count}
    print(f"目标正例数量: {target_pos_count}，目标正例比例: {target_pos_ratio:.1%}")
    
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 打印过采样后的分布
    y_train_res_dist = pd.Series(y_train_res).value_counts().to_dict()
    pos_ratio = y_train_res_dist.get(1, 0) / len(y_train_res)
    print(f"SMOTE过采样 - 训练集分布平衡后: {y_train_res_dist}")
    print(f"过采样后正例比例: {pos_ratio:.1%}")
    
    # 训练逻辑回归模型
    print("开始训练逻辑回归模型...")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_res, y_train_res)
    print(f"模型训练完成 (耗时 {time.time() - start_time:.2f}秒)")
    
    # 保存模型
    model_path = model_dir / f"{model_name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")
    
    # 保存特征信息
    feature_info = {
        "features": features,
        "feature_count": len(features),
        "training_samples": len(X_train_res),
        "original_training_samples": len(X_train),
        "test_samples": len(X_test),
        "use_smote": True,  # 标记使用SMOTE
        "smote_pos_ratio": pos_ratio
    }
    
    with open(model_dir / f'{model_name}_feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=4)
    print(f"特征信息已保存到: {model_dir / f'{model_name}_feature_info.json'}")
    
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
    coef_path = model_dir / "LR_best_params.json"
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
    
    print("\n模型训练与保存完成！")

if __name__ == "__main__":
    main() 