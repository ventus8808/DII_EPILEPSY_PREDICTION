import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, accuracy_score,
                           balanced_accuracy_score, cohen_kappa_score, log_loss)
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10):
    """Calculate calibration metrics with sample weights."""
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.array(weights)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize metrics
    bin_metrics = []
    ece = 0.0
    mce = 0.0
    
    # Calculate calibration metrics for each bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples in this bin
        bin_mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        if np.any(bin_mask):
            bin_weights = weights[bin_mask]
            bin_y_true = y_true[bin_mask]
            bin_y_prob = y_prob[bin_mask]
            
            # Calculate weighted statistics
            bin_total = np.sum(bin_weights)
            bin_actual = np.sum(bin_y_true * bin_weights) / bin_total
            bin_predicted = np.sum(bin_y_prob * bin_weights) / bin_total
            bin_count = np.sum(bin_mask)
            
            # Update ECE and MCE
            bin_error = np.abs(bin_actual - bin_predicted)
            ece += bin_error * (bin_total / np.sum(weights))
            mce = max(mce, bin_error)
            
            # Store bin metrics
            bin_metrics.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_count': int(bin_count),
                'bin_actual': float(bin_actual),
                'bin_predicted': float(bin_predicted),
                'bin_error': float(bin_error)
            })
    
    # Calculate Brier score
    brier = np.sum(weights * (y_prob - y_true) ** 2) / np.sum(weights)
    
    return ece, mce, bin_metrics, brier

def calculate_classification_metrics(y_true, y_pred, y_prob, weights=None):
    """
    计算分类模型的各种性能指标
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    y_prob: 预测概率
    weights: 样本权重
    
    返回:
    包含所有指标的字典
    """
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.array(weights)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    tn, fp, fn, tp = cm.ravel()
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred, sample_weight=weights)
    
    # 分类性能指标
    sensitivity = recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    youdens_index = sensitivity + specificity - 1
    kappa = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    
    # 概率指标
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    log_loss_val = log_loss(y_true, y_prob, sample_weight=weights)
    
    # 校准指标
    ece, mce, bin_metrics, brier = calculate_calibration_metrics(y_true, y_prob, weights)
    
    # Hosmer-Lemeshow测试
    def hosmer_lemeshow_test(y_true, y_prob, weights=None, n_bins=10):
        """执行Hosmer-Lemeshow测试"""
        # 确保所有输入都是numpy数组
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        if weights is None:
            weights = np.ones_like(y_true, dtype=float)
        else:
            weights = np.array(weights)
        
        # 按预测概率排序
        sorted_indices = np.argsort(y_prob)
        sorted_y_true = y_true[sorted_indices]
        sorted_y_prob = y_prob[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 创建等大小的分组
        total_weight = np.sum(sorted_weights)
        target_bin_weight = total_weight / n_bins
        
        bins = []
        current_bin = {'indices': [], 'weight': 0}
        
        for i, w in enumerate(sorted_weights):
            if current_bin['weight'] + w > target_bin_weight and len(current_bin['indices']) > 0:
                bins.append(current_bin)
                current_bin = {'indices': [i], 'weight': w}
            else:
                current_bin['indices'].append(i)
                current_bin['weight'] += w
        
        # 添加最后一个bin
        if current_bin['indices']:
            bins.append(current_bin)
        
        # 计算每个bin的观察值和期望值
        observed = []
        expected = []
        
        for bin_data in bins:
            indices = bin_data['indices']
            bin_y_true = sorted_y_true[indices]
            bin_y_prob = sorted_y_prob[indices]
            bin_weights = sorted_weights[indices]
            
            # 观察到的加权事件数
            o = np.sum(bin_y_true * bin_weights)
            observed.append(o)
            
            # 期望的加权事件数
            e = np.sum(bin_y_prob * bin_weights)
            expected.append(e)
        
        # 计算卡方统计量
        chi_square = np.sum([(o - e) ** 2 / (e * (1 - e / np.sum(bin_weights))) for o, e, bin_data in zip(observed, expected, bins)])
        
        # 自由度 = 分组数 - 2
        df = len(bins) - 2
        
        # 计算p值
        p_value = 1 - chi2_dist.cdf(chi_square, df)
        
        return chi_square, p_value
    
    # 执行Hosmer-Lemeshow测试
    hl_chi2, hl_pvalue = hosmer_lemeshow_test(y_true, y_prob, weights)
    
    # 按类别组织指标
    metrics = {
        # 基本指标
        'Accuracy': float(accuracy),
        
        # 分类性能指标
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'Precision': float(precision_val),
        'NPV': float(npv),
        'F1_Score': float(f1),
        'Youdens_Index': float(youdens_index),
        'Cohens_Kappa': float(kappa),
        
        # 概率指标
        'AUC-ROC': float(roc_auc),
        'AUC-PR': float(pr_auc),
        'Log_Loss': float(log_loss_val),
        'Brier': float(brier),
        'ECE': float(ece),
        'MCE': float(mce),
        'HL_Chi2': float(hl_chi2),
        'HL_pvalue': float(hl_pvalue)
    }
    
    return metrics

def plot_roc_curve(fpr, tpr, auc_value, model_name):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_ROC.png', dpi=300)
    
    # 保存数据
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/75_{model_name}_ROC.json', 'w') as f:
        json.dump(roc_data, f, indent=4)

def plot_pr_curve(precision, recall, auc_value, model_name):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {auc_value:.4f})')
    plt.axhline(y=np.mean(precision), color='navy', linestyle='--', label=f'No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_PR.png', dpi=300)
    
    # 保存数据
    pr_data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/75_{model_name}_PR.json', 'w') as f:
        json.dump(pr_data, f, indent=4)

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
    # 计算校准指标
    ece, mce, bin_metrics, brier = calculate_calibration_metrics(y_true, y_prob, weights)
    
    # 提取数据用于绘图
    bin_lowers = [bin_info['bin_lower'] for bin_info in bin_metrics]
    bin_uppers = [bin_info['bin_upper'] for bin_info in bin_metrics]
    bin_midpoints = [(lower + upper) / 2 for lower, upper in zip(bin_lowers, bin_uppers)]
    bin_actuals = [bin_info['bin_actual'] for bin_info in bin_metrics]
    bin_predicteds = [bin_info['bin_predicted'] for bin_info in bin_metrics]
    bin_counts = [bin_info['bin_count'] for bin_info in bin_metrics]
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制完美校准线
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # 绘制校准曲线
    plt.plot(bin_midpoints, bin_actuals, 'o-', color='red', label=f'Calibration curve (ECE = {ece:.4f}, MCE = {mce:.4f})')
    
    # 绘制每个bin的预测概率
    for i, (midpoint, predicted) in enumerate(zip(bin_midpoints, bin_predicteds)):
        plt.plot([midpoint, midpoint], [midpoint, predicted], 'k-', alpha=0.3)
    
    # 添加bin的大小作为气泡图
    max_size = 500
    sizes = [count / max(bin_counts) * max_size for count in bin_counts]
    plt.scatter(bin_midpoints, bin_predicteds, s=sizes, alpha=0.5, color='blue', label='Bin size (samples)')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_Calibration.png', dpi=300)
    
    # 保存数据
    calibration_data = {
        'bin_midpoints': bin_midpoints,
        'bin_actuals': bin_actuals,
        'bin_predicteds': bin_predicteds,
        'bin_counts': bin_counts,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/75_{model_name}_Calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)

def plot_decision_curve(y_true, y_prob, weights, model_name):
    """Plot and save Decision Curve Analysis."""
    # 设置阈值范围
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    # 计算模型的净收益
    net_benefits = [calculate_net_benefit(y_true, y_prob, t, weights) for t in thresholds]
    
    # 计算"全部治疗"策略的净收益
    all_net_benefits = [calculate_net_benefit(y_true, np.ones_like(y_prob), t, weights) for t in thresholds]
    
    # 计算"无人治疗"策略的净收益（总是为0）
    none_net_benefits = np.zeros_like(thresholds)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制净收益曲线
    plt.plot(thresholds, net_benefits, 'r-', linewidth=2, label='Model')
    plt.plot(thresholds, all_net_benefits, 'b-', linewidth=1, label='Treat all')
    plt.plot(thresholds, none_net_benefits, 'k-', linewidth=1, label='Treat none')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, max(max(net_benefits), max(all_net_benefits)) + 0.05])
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_DCA.png', dpi=300)
    
    # 保存数据
    dca_data = {
        'thresholds': thresholds.tolist(),
        'model_net_benefits': [float(x) for x in net_benefits],
        'treat_all_net_benefits': [float(x) for x in all_net_benefits],
        'treat_none_net_benefits': [float(x) for x in none_net_benefits]
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/75_{model_name}_DCA.json', 'w') as f:
        json.dump(dca_data, f, indent=4)

def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
    """
    计算给定阈值下的净收益
    
    参数:
    y_true: 真实标签
    y_prob: 预测概率
    threshold: 决策阈值
    weights: 样本权重
    
    返回:
    净收益值
    """
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    
    # 根据阈值进行分类
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算真阳性和假阳性
    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    
    # 计算加权的真阳性和假阳性数量
    tp_count = np.sum(weights[tp_mask])
    fp_count = np.sum(weights[fp_mask])
    total_weight = np.sum(weights)
    
    # 计算净收益
    net_benefit = (tp_count / total_weight) - (fp_count / total_weight) * (threshold / (1 - threshold))
    
    return net_benefit

def load_super_learner_model(model_name="SuperLearner"):
    """加载Super Learner模型、元学习器和模型信息"""
    base_models = {}
    meta_learners = {}
    super_learner = None
    scaler = None
    model_info = {}
    
    # 加载基础模型
    base_model_files = {
        'XGB': '62_XGB_model.pkl',
        'RF': '63_RF_model.pkl',
        'CatBoost': '64_CatBoost_model.pkl',
        'LightGBM': '65_LightGBM_model.pkl'
    }
    
    for name, file in base_model_files.items():
        try:
            with open(f'/Users/maguoli/Documents/Development/Predictive/Models/{file}', 'rb') as f:
                base_models[name] = pickle.load(f)
            print(f"Successfully loaded {name} model")
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
    
    # 加载元学习器模型
    try:
        meta_learner_types = ['LogisticRegression', 'XGBoost', 'RandomForest', 'ElasticNet']
        for meta_type in meta_learner_types:
            try:
                with open(f'/Users/maguoli/Documents/Development/Predictive/Models/75_{model_name}_{meta_type}_metamodel.pkl', 'rb') as f:
                    meta_learners[meta_type] = pickle.load(f)
                print(f"Successfully loaded {meta_type} meta-learner")
            except Exception as e:
                print(f"Error loading {meta_type} meta-learner: {str(e)}")
    except Exception as e:
        print(f"Error loading meta-learners: {str(e)}")
    
    # 加载Super Learner模型
    try:
        with open(f'/Users/maguoli/Documents/Development/Predictive/Models/75_{model_name}_supermodel.pkl', 'rb') as f:
            super_learner = pickle.load(f)
        print(f"Successfully loaded Super Learner model")
    except Exception as e:
        print(f"Error loading Super Learner model: {str(e)}")
    
    # 加载标准化器
    try:
        with open(f'/Users/maguoli/Documents/Development/Predictive/Models/75_{model_name}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"Successfully loaded scaler")
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
    
    # 加载模型信息
    try:
        with open(f'/Users/maguoli/Documents/Development/Predictive/Models/75_{model_name}_info.json', 'r') as f:
            model_info = json.load(f)
        print(f"Successfully loaded model info")
    except Exception as e:
        print(f"Error loading model info: {str(e)}")
    
    return base_models, meta_learners, super_learner, scaler, model_info

def generate_meta_features(models, X):
    """生成元特征"""
    meta_features = np.zeros((X.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        try:
            meta_features[:, i] = model.predict_proba(X)[:, 1]
        except:
            try:
                meta_features[:, i] = model.predict(X)
            except Exception as e:
                print(f"Error generating meta-features for {name}: {str(e)}")
                meta_features[:, i] = 0.5  # 默认值
    
    return meta_features

def generate_meta_learner_predictions(meta_learners, meta_features):
    """使用元学习器生成预测"""
    meta_learner_preds = np.zeros((meta_features.shape[0], len(meta_learners)))
    
    for i, (name, model) in enumerate(meta_learners.items()):
        try:
            if hasattr(model, 'predict_proba'):
                meta_learner_preds[:, i] = model.predict_proba(meta_features)[:, 1]
            else:
                # 对于回归模型，将预测值限制在[0, 1]范围内
                meta_learner_preds[:, i] = np.clip(model.predict(meta_features), 0, 1)
        except Exception as e:
            print(f"Error generating predictions for {name} meta-learner: {str(e)}")
            meta_learner_preds[:, i] = 0.5  # 默认值
    
    return meta_learner_preds

def super_learner_predict(base_models, meta_learners, super_learner, scaler, X):
    """使用Super Learner模型进行预测"""
    # 生成元特征
    meta_features = generate_meta_features(base_models, X)
    
    # 使用元学习器生成预测
    meta_learner_preds = generate_meta_learner_predictions(meta_learners, meta_features)
    
    # 标准化元学习器预测
    meta_learner_preds_scaled = scaler.transform(meta_learner_preds)
    
    # 使用Super Learner进行最终预测
    if hasattr(super_learner, 'predict_proba'):
        y_prob = super_learner.predict_proba(meta_learner_preds_scaled)[:, 1]
    else:
        # 对于回归模型，将预测值限制在[0, 1]范围内
        y_prob = np.clip(super_learner.predict(meta_learner_preds_scaled), 0, 1)
    
    return y_prob

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    
    # 加载模型
    model_name = "SuperLearner"
    base_models, meta_learners, super_learner, scaler, model_info = load_super_learner_model(model_name)
    
    # 加载数据
    print("Loading data...")
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    weights_data = df['WTDRD1']
    covariables = ['Gender', 'Age', 'BMI', 'Education', 'Marriage', 'Smoke',
                   'Alcohol', 'Employment', 'ActivityLevel']
    
    X = pd.concat([
        df['DII_food'],
        df[covariables]
    ], axis=1)
    y = df['Epilepsy']
    
    # 分割数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights_data, test_size=0.2, stratify=y, random_state=42
    )
    
    # 使用Super Learner模型进行预测
    print("Making predictions with Super Learner...")
    y_prob = super_learner_predict(base_models, meta_learners, super_learner, scaler, X_test)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 计算性能指标
    print("Calculating performance metrics...")
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob, weights_test)
    
    # 打印指标
    print("\nBasic Metrics:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print()
    
    print("Classification Performance Metrics:")
    print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"Specificity: {metrics['Specificity']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"NPV: {metrics['NPV']:.4f}")
    print(f"F1_Score: {metrics['F1_Score']:.4f}")
    print(f"Youdens_Index: {metrics['Youdens_Index']:.4f}")
    print(f"Cohens_Kappa: {metrics['Cohens_Kappa']:.4f}")
    print()
    
    print("Probabilistic Metrics:")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print(f"AUC-PR: {metrics['AUC-PR']:.4f}")
    print(f"Log_Loss: {metrics['Log_Loss']:.4f}")
    print(f"Brier: {metrics['Brier']:.4f}")
    print(f"ECE: {metrics['ECE']:.4f}")
    print(f"MCE: {metrics['MCE']:.4f}")
    print(f"HL_Chi2: {metrics['HL_Chi2']:.4f}")
    print(f"HL_pvalue: {metrics['HL_pvalue']:.4f}")
    print()
    
    # 计算指标总数
    print(f"Total number of metrics calculated: {len(metrics)}")
    
    # 保存指标到JSON文件
    metrics_file_path = f'/Users/maguoli/Documents/Development/Predictive/Model metrics/75_{model_name}_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file_path}")
    
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=weights_test)
    roc_auc = metrics['AUC-ROC']
    
    # 计算PR曲线数据
    precision, recall, _ = precision_recall_curve(y_test, y_prob, sample_weight=weights_test)
    pr_auc = metrics['AUC-PR']
    
    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, model_name)
    print(f"ROC curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_ROC.png")
    
    # 绘制PR曲线
    plot_pr_curve(precision, recall, pr_auc, model_name)
    print(f"PR curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_PR.png")
    
    # 绘制校准曲线
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    print(f"Calibration curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_Calibration.png")
    
    # 绘制决策曲线
    plot_decision_curve(y_test, y_prob, weights_test, model_name)
    print(f"Decision curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/75_{model_name}_DCA.png")
    
    # 打印Super Learner信息
    print(f"\nSuper Learner type: {model_info.get('super_learner_type', 'Unknown')}")
    print(f"Meta-learners: {', '.join(model_info.get('meta_learners', ['Unknown']))}")
    
    print("\nSuper Learner evaluation completed successfully!")

if __name__ == "__main__":
    main()
