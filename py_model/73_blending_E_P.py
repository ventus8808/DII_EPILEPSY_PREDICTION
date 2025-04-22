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
    
    # Create bins for Hosmer-Lemeshow test
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    
    # Calculate observed and expected frequencies for each bin
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    total = np.zeros(n_bins)
    
    for i in range(len(y_true)):
        bin_idx = bin_indices[i]
        weight = weights[i]
        total[bin_idx] += weight
        observed[bin_idx] += y_true[i] * weight
        expected[bin_idx] += y_prob[i] * weight
    
    # Calculate chi-square statistic
    chi2_stat = 0
    valid_bins = []
    
    for i in range(n_bins):
        if total[i] > 0:
            observed_bin = observed[i]
            expected_bin = expected[i]
            
            # For each bin, calculate observed and expected counts
            o_pos = observed_bin
            o_neg = total[i] - observed_bin
            e_pos = expected_bin
            e_neg = total[i] - expected_bin
            
            # Only include bins with at least 5 expected positive and negative cases
            if e_pos >= 5 and e_neg >= 5:
                valid_bins.append([o_pos, o_neg, e_pos, e_neg])
    
    # Calculate chi-square and p-value if we have valid bins
    if len(valid_bins) >= 2:
        # Convert to array for chi2_contingency
        valid_bins = np.array(valid_bins)
        observed_table = valid_bins[:, 0:2]
        expected_table = valid_bins[:, 2:4]
        
        # Calculate chi-square statistic manually
        chi2_stat = np.sum((observed_table - expected_table) ** 2 / expected_table)
        # 使用scipy.stats.chi2_dist的cdf函数
        p_value = 1 - chi2_dist.cdf(chi2_stat, df=len(valid_bins) - 2)
    else:
        chi2_stat = np.nan
        p_value = np.nan
    
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
            
            # Calculate calibration error for this bin
            bin_error = np.abs(bin_predicted - bin_actual)
            bin_size = np.sum(bin_mask)
            
            # Update ECE and MCE
            bin_weight = np.sum(bin_weights) / np.sum(weights)
            ece += bin_weight * bin_error
            mce = max(mce, bin_error)
            
            # Store bin metrics
            bin_metrics.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_size': int(bin_size),
                'actual_prob': float(bin_actual),
                'predicted_prob': float(bin_predicted),
                'bin_error': float(bin_error),
                'bin_weight': float(bin_weight)
            })
    
    # Calculate Brier score
    brier = np.average((y_prob - y_true) ** 2, weights=weights)
    
    return ece, mce, bin_metrics, brier, chi2_stat, p_value

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
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    tn, fp, fn, tp = cm.ravel()
    
    # 基本指标
    metrics = {}
    
    # 1. 准确率 (Accuracy) - most basic metric
    metrics['Accuracy'] = accuracy_score(y_true, y_pred, sample_weight=weights)
    
    # 2. 灵敏度/召回率 (Sensitivity/Recall)
    metrics['Sensitivity'] = recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 3. 特异度 (Specificity)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 4. 阳性预测值/精确率 (PPV/Precision)
    metrics['Precision'] = precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 5. 阴性预测值 (NPV)
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # 6. F1 Score - harmonic mean of precision and recall
    metrics['F1_Score'] = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
    
    # 7. Youden's Index - simple combination of sensitivity and specificity
    metrics['Youdens_Index'] = metrics['Sensitivity'] + metrics['Specificity'] - 1
    
    # 8. Cohen's Kappa - agreement beyond chance
    metrics['Cohens_Kappa'] = cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    
    # 9. Log Loss - probabilistic metric
    metrics['Log_Loss'] = log_loss(y_true, y_prob, sample_weight=weights)
    
    # 10. AUC-ROC - area under curve metric
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob, sample_weight=weights)
    
    # 11. AUC-PR - area under curve metric for imbalanced data
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    metrics['AUC-PR'] = auc(recall_curve, precision_curve)
    
    return metrics

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve."""
    # 计算校准指标
    ece, mce, bin_metrics, brier, hl_stat, p_value = calculate_calibration_metrics(y_true, y_prob, weights)
    
    # 提取数据用于绘图
    bin_lowers = [bin_info['bin_lower'] for bin_info in bin_metrics]
    bin_uppers = [bin_info['bin_upper'] for bin_info in bin_metrics]
    bin_midpoints = [(lower + upper) / 2 for lower, upper in zip(bin_lowers, bin_uppers)]
    bin_actuals = [bin_info['actual_prob'] for bin_info in bin_metrics]
    bin_predicteds = [bin_info['predicted_prob'] for bin_info in bin_metrics]
    bin_sizes = [bin_info['bin_size'] for bin_info in bin_metrics]
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制完美校准线
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # 绘制校准曲线
    plt.plot(bin_midpoints, bin_actuals, 'o-', color='red', label='Calibration curve')
    
    # 添加置信区间（可选）
    # ...
    
    # 设置图形属性
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve\nECE: {ece:.3f}, MCE: {mce:.3f}, Brier: {brier:.3f}, HL Chi2: {hl_stat:.3f}, HL p-value: {p_value:.3f}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_Calibration.png', dpi=300)
    plt.close()
    
    # 保存数据
    calibration_data = {
        'bin_midpoints': bin_midpoints,
        'bin_actuals': bin_actuals,
        'bin_predicteds': bin_predicteds,
        'bin_sizes': bin_sizes,
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'hl_stat': float(hl_stat),
        'p_value': float(p_value)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/73_{model_name}_Calibration_data.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)

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
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_ROC.png', dpi=300)
    
    # 保存数据
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/73_{model_name}_ROC.json', 'w') as f:
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
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_PR.png', dpi=300)
    
    # 保存数据
    pr_data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/73_{model_name}_PR.json', 'w') as f:
        json.dump(pr_data, f, indent=4)

def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
    """Calculate net benefit for a given threshold."""
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.array(weights)
    
    # 根据阈值创建预测
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算真阳性和假阳性
    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    
    # 计算加权的真阳性和假阳性数量
    tp_count = np.sum(weights[tp_mask])
    fp_count = np.sum(weights[fp_mask])
    
    # 计算总样本权重
    total_weight = np.sum(weights)
    
    # 计算净收益
    if tp_count + fp_count > 0:
        net_benefit = (tp_count - (threshold / (1 - threshold)) * fp_count) / total_weight
    else:
        net_benefit = 0
    
    return net_benefit

def plot_decision_curve(y_true, y_prob, weights, model_name):
    """Plot Decision Curve Analysis."""
    # 设置阈值范围
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 计算每个阈值的净收益
    net_benefits = [calculate_net_benefit(y_true, y_prob, t, weights) for t in thresholds]
    
    # 计算"全部治疗"策略的净收益
    prevalence = np.sum(y_true * weights) / np.sum(weights)
    treat_all_net_benefits = [prevalence - (t / (1 - t)) * (1 - prevalence) for t in thresholds]
    
    # 计算"全不治疗"策略的净收益（总是0）
    treat_none_net_benefits = [0] * len(thresholds)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制决策曲线
    plt.plot(thresholds, net_benefits, 'r-', linewidth=2, label='Model')
    plt.plot(thresholds, treat_all_net_benefits, 'b--', linewidth=2, label='Treat all')
    plt.plot(thresholds, treat_none_net_benefits, 'g--', linewidth=2, label='Treat none')
    
    # 添加标签和标题
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylim([-0.05, max(max(net_benefits), max(treat_all_net_benefits)) + 0.05])
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_DCA.png', dpi=300)
    
    # 保存数据
    dca_data = {
        'thresholds': thresholds.tolist(),
        'model_net_benefits': net_benefits,
        'treat_all_net_benefits': treat_all_net_benefits,
        'treat_none_net_benefits': treat_none_net_benefits,
        'prevalence': float(prevalence)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/73_{model_name}_DCA.json', 'w') as f:
        json.dump(dca_data, f, indent=4)
    
    # 输出一些关键信息
    print(f"Disease prevalence: {prevalence:.4f}")
    print(f"Prediction range: {np.min(y_prob):.4f} to {np.max(y_prob):.4f}")
    print(f"Mean prediction: {np.mean(y_prob):.4f}")
    print()
    
    # 输出特定阈值的详细信息
    threshold_of_interest = 0.01  # 可以根据需要调整
    idx = np.abs(thresholds - threshold_of_interest).argmin()
    
    # 根据阈值创建预测
    y_pred_at_threshold = (y_prob >= thresholds[idx]).astype(int)
    
    # 计算真阳性和假阳性
    tp_mask = (y_pred_at_threshold == 1) & (y_true == 1)
    fp_mask = (y_pred_at_threshold == 1) & (y_true == 0)
    
    # 计算阳性预测数量和真阳性数量
    positive_preds = np.sum(y_pred_at_threshold)
    true_positives = np.sum(tp_mask)
    false_positives = np.sum(fp_mask)
    
    print(f"At threshold {thresholds[idx]:.2f}:")
    print(f"Model net benefit: {net_benefits[idx]:.4f}")
    print(f"Treat-all net benefit: {treat_all_net_benefits[idx]:.4f}")
    print(f"Number of positive predictions: {positive_preds}")
    print(f"Number of true positives in dataset: {np.sum(y_true)}")
    print(f"True positives at this threshold: {true_positives}")
    print(f"False positives at this threshold: {false_positives}")
    print(f"True positive rate: {true_positives / np.sum(weights):.4f}")
    print(f"False positive rate: {false_positives / np.sum(weights):.4f}")

def load_blending_ensemble_model(model_name="BlendingEnsemble"):
    """加载混合集成模型及其元学习器"""
    # 基础模型信息
    base_models_info = {
        'XGB': {'file': '62_XGB_model.pkl', 'metrics': '62_XGB_metrics.json'},
        'RF': {'file': '63_RF_model.pkl', 'metrics': '63_RF_metrics.json'},
        'CatBoost': {'file': '64_CatBoost_model.pkl', 'metrics': '64_CatBoost_metrics.json'},
        'LightGBM': {'file': '65_LightGBM_model.pkl', 'metrics': '65_LightGBM_metrics.json'}
    }
    
    # 加载基础模型
    base_models = {}
    model_metrics = {}
    
    for model_key, files in base_models_info.items():
        model_path = f"/Users/maguoli/Documents/Development/Predictive/Models/{files['file']}"
        metrics_path = f"/Users/maguoli/Documents/Development/Predictive/Model metrics/{files['metrics']}"
        
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                base_models[model_key] = pickle.load(f)
            
            # 加载指标
            with open(metrics_path, 'r') as f:
                model_metrics[model_key] = json.load(f)
            
            print(f"Successfully loaded {model_key} model and metrics")
        except Exception as e:
            print(f"Error loading {model_key} model or metrics: {e}")
    
    # 加载元学习器
    meta_learner_path = f'/Users/maguoli/Documents/Development/Predictive/Models/73_{model_name}_metamodel.pkl'
    try:
        with open(meta_learner_path, 'rb') as f:
            meta_learner = pickle.load(f)
        print(f"Successfully loaded meta-learner")
    except Exception as e:
        print(f"Error loading meta-learner: {e}")
        meta_learner = None
    
    # 加载元学习器信息
    info_path = f'/Users/maguoli/Documents/Development/Predictive/Models/73_{model_name}_info.json'
    try:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        print(f"Successfully loaded model info")
    except Exception as e:
        print(f"Error loading model info: {e}")
        model_info = {'meta_learner_type': 'Unknown', 'base_models': list(base_models.keys())}
    
    return base_models, meta_learner, model_info

def generate_meta_features(models, X):
    """生成元特征"""
    meta_features = np.zeros((X.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        if name == 'LightGBM':
            # LightGBM的Booster对象使用不同的预测方法
            meta_features[:, i] = model.predict(X)
        else:
            try:
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            except:
                meta_features[:, i] = model.predict(X)
    
    return meta_features

def blending_ensemble_predict(base_models, meta_learner, X):
    """使用混合集成模型进行预测"""
    # 生成元特征
    meta_features = generate_meta_features(base_models, X)
    
    # 使用元学习器进行预测
    y_prob = meta_learner.predict_proba(meta_features)[:, 1]
    
    return y_prob

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    
    # 加载模型
    model_name = "BlendingEnsemble"
    base_models, meta_learner, model_info = load_blending_ensemble_model(model_name)
    
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
    
    # 使用混合集成模型进行预测
    print("Making predictions with blending ensemble...")
    y_prob = blending_ensemble_predict(base_models, meta_learner, X_test)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 计算性能指标
    print("Calculating performance metrics...")
    classification_metrics = calculate_classification_metrics(y_test, y_pred, y_prob, weights_test)
    
    # 计算校准指标
    ece, mce, bin_metrics, brier, hl_stat, p_value = calculate_calibration_metrics(y_test, y_prob, weights_test)
    
    # 组织指标按类别
    basic_metrics = {
        'Accuracy': classification_metrics['Accuracy']
    }
    
    classification_performance = {
        'Sensitivity': classification_metrics['Sensitivity'],
        'Specificity': classification_metrics['Specificity'],
        'Precision': classification_metrics['Precision'],
        'NPV': classification_metrics['NPV'],
        'F1_Score': classification_metrics['F1_Score'],
        'Youdens_Index': classification_metrics['Youdens_Index'],
        'Cohens_Kappa': classification_metrics['Cohens_Kappa']
    }
    
    probabilistic_metrics = {
        'AUC-ROC': classification_metrics['AUC-ROC'],
        'AUC-PR': classification_metrics['AUC-PR'],
        'Log_Loss': classification_metrics['Log_Loss'],
        'Brier': brier,
        'ECE': ece,
        'MCE': mce,
        'HL_Chi2': hl_stat,
        'HL_pvalue': p_value
    }
    
    # 合并所有指标
    all_metrics = {**basic_metrics, **classification_performance, **probabilistic_metrics}
    
    # 按类别打印指标
    print("\nBasic Metrics:")
    for metric in ['Accuracy']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    print("\nClassification Performance Metrics:")
    for metric in ['Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_Score', 'Youdens_Index', 'Cohens_Kappa']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    print("\nProbabilistic Metrics:")
    for metric in ['AUC-ROC', 'AUC-PR', 'Log_Loss', 'Brier', 'ECE', 'MCE', 'HL_Chi2', 'HL_pvalue']:
        print(f"{metric}: {all_metrics[metric]:.4f}")
    
    # 计算指标总数
    total_metrics = len(all_metrics)
    print(f"\nTotal number of metrics calculated: {total_metrics}")
    
    # 保存指标到JSON文件
    metrics_file_path = f'/Users/maguoli/Documents/Development/Predictive/Model metrics/73_{model_name}_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file_path}")
    
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=weights_test)
    roc_auc = all_metrics['AUC-ROC']
    
    # 计算PR曲线数据
    precision, recall, _ = precision_recall_curve(y_test, y_prob, sample_weight=weights_test)
    pr_auc = all_metrics['AUC-PR']
    
    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, model_name)
    print(f"ROC curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_ROC.png")
    
    # 绘制PR曲线
    plot_pr_curve(precision, recall, pr_auc, model_name)
    print(f"PR curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_PR.png")
    
    # 绘制校准曲线
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    print(f"Calibration curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_Calibration.png")
    
    # 绘制决策曲线
    plot_decision_curve(y_test, y_prob, weights_test, model_name)
    print(f"Decision curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/73_{model_name}_DCA.png")
    
    print("\nBlending ensemble evaluation completed successfully!")

if __name__ == "__main__":
    main()
