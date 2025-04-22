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

def calculate_calibration_metrics(y_true, y_prob, weights=None, n_bins=10, equal_freq=False):
    """Calculate calibration metrics with sample weights."""
    # 确保所有输入都是numpy数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = np.array(weights)
    
    if not equal_freq:
        # 等宽分箱（原始方法）
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
    else:
        # 等频分箱
        # 按预测概率排序
        sorted_indices = np.argsort(y_prob)
        sorted_y_true = y_true[sorted_indices]
        sorted_y_prob = y_prob[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 创建等频分箱
        total_weight = np.sum(sorted_weights)
        target_bin_weight = total_weight / n_bins
        
        bin_metrics = []
        ece = 0.0
        mce = 0.0
        
        current_bin = {'indices': [], 'weight': 0}
        for i, (prob, true, w) in enumerate(zip(sorted_y_prob, sorted_y_true, sorted_weights)):
            if current_bin['weight'] + w > target_bin_weight and len(current_bin['indices']) > 0:
                # 计算当前箱的指标
                indices = current_bin['indices']
                bin_y_true = sorted_y_true[indices]
                bin_y_prob = sorted_y_prob[indices]
                bin_weights = sorted_weights[indices]
                
                bin_total = np.sum(bin_weights)
                bin_actual = np.sum(bin_y_true * bin_weights) / bin_total
                bin_predicted = np.sum(bin_y_prob * bin_weights) / bin_total
                bin_count = len(indices)
                
                bin_error = np.abs(bin_actual - bin_predicted)
                ece += bin_error * (bin_total / total_weight)
                mce = max(mce, bin_error)
                
                bin_metrics.append({
                    'bin_lower': np.min(bin_y_prob),
                    'bin_upper': np.max(bin_y_prob),
                    'bin_count': int(bin_count),
                    'bin_actual': float(bin_actual),
                    'bin_predicted': float(bin_predicted),
                    'bin_error': float(bin_error)
                })
                
                # 开始新的箱
                current_bin = {'indices': [i], 'weight': w}
            else:
                current_bin['indices'].append(i)
                current_bin['weight'] += w
        
        # 添加最后一个箱
        if current_bin['indices']:
            indices = current_bin['indices']
            bin_y_true = sorted_y_true[indices]
            bin_y_prob = sorted_y_prob[indices]
            bin_weights = sorted_weights[indices]
            
            bin_total = np.sum(bin_weights)
            bin_actual = np.sum(bin_y_true * bin_weights) / bin_total
            bin_predicted = np.sum(bin_y_prob * bin_weights) / bin_total
            bin_count = len(indices)
            
            bin_error = np.abs(bin_actual - bin_predicted)
            ece += bin_error * (bin_total / total_weight)
            mce = max(mce, bin_error)
            
            bin_metrics.append({
                'bin_lower': np.min(bin_y_prob),
                'bin_upper': np.max(bin_y_prob),
                'bin_count': int(bin_count),
                'bin_actual': float(bin_actual),
                'bin_predicted': float(bin_predicted),
                'bin_error': float(bin_error)
            })
    
    # Calculate Brier score
    brier = np.sum(weights * (y_prob - y_true) ** 2) / np.sum(weights)
    
    return ece, mce, bin_metrics, brier

def plot_calibration_curve(y_true, y_prob, weights, model_name):
    """Plot and save calibration curve using sklearn's implementation."""
    from sklearn.calibration import calibration_curve
    
    # 使用sklearn的calibration_curve函数计算校准曲线
    # 使用更多的分箱以获得更平滑的曲线
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    
    # 计算Brier分数
    brier = np.mean((y_prob - y_true) ** 2)
    
    # 计算ECE (Expected Calibration Error)
    # 使用等频分箱 (quantile binning)
    ece, mce, bin_metrics, _ = calculate_calibration_metrics(y_true, y_prob, weights, n_bins=10, equal_freq=True)
    
    # 计算Hosmer-Lemeshow检验
    from scipy import stats
    
    # 使用10个分箱进行Hosmer-Lemeshow检验
    n_bins = 10
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_prob, quantiles)
    bins[0] = 0  # 确保第一个bin从0开始
    bins[-1] = 1  # 确保最后一个bin到1结束
    
    # 将预测概率分到不同的bin中
    binned_y_prob = np.digitize(y_prob, bins) - 1
    binned_y_prob[binned_y_prob == n_bins] = n_bins - 1  # 处理边界情况
    
    # 计算每个bin中的观察值和预期值
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (binned_y_prob == i)
        if np.any(bin_mask):
            counts[i] = np.sum(bin_mask)
            observed[i] = np.mean(y_true[bin_mask])
            expected[i] = np.mean(y_prob[bin_mask])
    
    # 计算Hosmer-Lemeshow统计量
    non_empty_bins = counts > 0
    if np.sum(non_empty_bins) > 1:  # 至少需要2个非空bin
        valid_counts = counts[non_empty_bins]
        valid_observed = observed[non_empty_bins]
        valid_expected = expected[non_empty_bins]
        
        # 计算每个bin的贡献
        hl_contributions = valid_counts * ((valid_observed - valid_expected) ** 2) / (valid_expected * (1 - valid_expected))
        hl_stat = np.sum(hl_contributions)
        df = np.sum(non_empty_bins) - 2  # 自由度 = 非空分组数 - 2
        if df > 0:  # 确保自由度为正
            p_value = 1 - stats.chi2.cdf(hl_stat, df)
        else:
            hl_stat = np.nan
            p_value = np.nan
    else:
        hl_stat = np.nan
        p_value = np.nan
        df = 0
    
    # 创建图形
    plt.figure(figsize=(8, 8), dpi=300)
    
    # 绘制完美校准线
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # 绘制校准曲线
    plt.plot(prob_pred, prob_true, 'ro-', markersize=8, label='Calibration curve')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    
    # 标题中包含校准指标
    title = f'Calibration Curve\nBrier: {brier:.4f}, ECE: {ece:.4f}'
    if not np.isnan(hl_stat):
        title += f'\nH-L χ²: {hl_stat:.3f}, df: {df}, p: {p_value:.4f}'
    
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_Calibration.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存数据
    calibration_data = {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
        'brier': float(brier),
        'ece': float(ece),
        'mce': float(mce),
        'hosmer_lemeshow_chi2': float(hl_stat) if not np.isnan(hl_stat) else None,
        'hosmer_lemeshow_df': int(df),
        'hosmer_lemeshow_pvalue': float(p_value) if not np.isnan(p_value) else None
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/74_{model_name}_Calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)

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
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_ROC.png', dpi=300)
    
    # 保存数据
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/74_{model_name}_ROC.json', 'w') as f:
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
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_PR.png', dpi=300)
    
    # 保存数据
    pr_data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(auc_value)}
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/74_{model_name}_PR.json', 'w') as f:
        json.dump(pr_data, f, indent=4)

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
    plt.savefig(f'/Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_DCA.png', dpi=300)
    
    # 保存数据
    dca_data = {
        'thresholds': thresholds.tolist(),
        'model_net_benefits': net_benefits,
        'treat_all_net_benefits': treat_all_net_benefits,
        'treat_none_net_benefits': treat_none_net_benefits,
        'prevalence': float(prevalence)
    }
    with open(f'/Users/maguoli/Documents/Development/Predictive/plot/original_data/74_{model_name}_DCA.json', 'w') as f:
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

def load_stacking_ensemble_model(model_name="StackingEnsemble"):
    """加载堆叠集成模型及其元学习器"""
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
    meta_learner_path = f'/Users/maguoli/Documents/Development/Predictive/Models/74_{model_name}_metamodel.pkl'
    try:
        with open(meta_learner_path, 'rb') as f:
            meta_learner = pickle.load(f)
        print(f"Successfully loaded meta-learner")
    except Exception as e:
        print(f"Error loading meta-learner: {e}")
        meta_learner = None
    
    # 加载元学习器信息
    info_path = f'/Users/maguoli/Documents/Development/Predictive/Models/74_{model_name}_info.json'
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

def stacking_ensemble_predict(base_models, meta_learner, X):
    """使用堆叠集成模型进行预测"""
    # 生成元特征
    meta_features = generate_meta_features(base_models, X)
    
    # 使用元学习器进行预测
    y_prob = meta_learner.predict_proba(meta_features)[:, 1]
    
    return y_prob

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

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/plot/original_data', exist_ok=True)
    os.makedirs('/Users/maguoli/Documents/Development/Predictive/Model metrics', exist_ok=True)
    
    # 加载模型
    model_name = "StackingEnsemble"
    base_models, meta_learner, model_info = load_stacking_ensemble_model(model_name)
    
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
    
    # 使用堆叠集成模型进行预测
    print("Making predictions with stacking ensemble...")
    y_prob = stacking_ensemble_predict(base_models, meta_learner, X_test)
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
    metrics_file_path = f'/Users/maguoli/Documents/Development/Predictive/Model metrics/74_{model_name}_metrics.json'
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
    print(f"ROC curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_ROC.png")
    
    # 绘制PR曲线
    plot_pr_curve(precision, recall, pr_auc, model_name)
    print(f"PR curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_PR.png")
    
    # 绘制校准曲线
    plot_calibration_curve(y_test, y_prob, weights_test, model_name)
    print(f"Calibration curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_Calibration.png")
    
    # 绘制决策曲线
    plot_decision_curve(y_test, y_prob, weights_test, model_name)
    print(f"Decision curve saved to: /Users/maguoli/Documents/Development/Predictive/plot/74_{model_name}_DCA.png")
    
    # 打印元学习器信息
    print(f"\nMeta-learner type: {model_info['meta_learner_type']}")
    print(f"Base models: {', '.join(model_info['base_models'])}")
    
    print("\nStacking ensemble evaluation completed successfully!")

if __name__ == "__main__":
    main()
