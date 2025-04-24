import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
# 全局字体设置：MONACO 12号
plt.rcParams['font.family'] = 'Monaco'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='red', linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Receiver Operating Characteristic - {model_name}', pad=30)
    # 只显示左下角一个0
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower right')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_ROC.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': float(roc_auc)}, str(Path(plot_data_dir) / f"{model_name}_ROC_data.json"))

def plot_pr_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})', color='green', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Precision-Recall Curve - {model_name}', pad=30)
    # 只显示左下角一个0
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower left')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_PR.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({'recall': recall.tolist(), 'precision': precision.tolist(), 'pr_auc': float(pr_auc)}, str(Path(plot_data_dir) / f"{model_name}_PR_data.json"))

def plot_calibration_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir, n_bins=10):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = np.array(weights)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    actuals = []
    predicteds = []
    for i in range(n_bins):
        in_bin = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i+1])
        if np.any(in_bin):
            bin_weight = weights[in_bin]
            bin_true = y_true[in_bin]
            bin_prob = y_prob[in_bin]
            actual = np.sum(bin_true * bin_weight) / np.sum(bin_weight)
            predicted = np.sum(bin_prob * bin_weight) / np.sum(bin_weight)
            actuals.append(actual)
            predicteds.append(predicted)
        else:
            actuals.append(np.nan)
            predicteds.append(np.nan)
    # 过滤掉NaN值，确保数据点有效
    valid_indices = ~(np.isnan(predicteds) | np.isnan(actuals))
    valid_predicteds = np.array(predicteds)[valid_indices]
    valid_actuals = np.array(actuals)[valid_indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated', linewidth=2)
    ax.plot(valid_predicteds, valid_actuals, marker='o', color='red', label='Calibration curve', linewidth=2)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Calibration Curve - {model_name}', pad=30)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower right')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_calibration.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({'prob_pred': predicteds, 'prob_true': actuals}, str(Path(plot_data_dir) / f"{model_name}_calibration_data.json"))


def plot_decision_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir):
    thresholds = np.linspace(0, 1, 100)
    prevalence = np.sum(weights[y_true == 1]) / np.sum(weights) if weights is not None else np.mean(y_true)
    net_benefits_model = []
    net_benefits_all = []
    net_benefit_none = [0] * len(thresholds)
    for threshold in thresholds:
        def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            if weights is None:
                weights = np.ones_like(y_true)
            else:
                weights = np.array(weights)
            pred_1 = (y_prob >= threshold)
            tp = np.sum(weights[(pred_1) & (y_true == 1)])
            fp = np.sum(weights[(pred_1) & (y_true == 0)])
            N = np.sum(weights)
            if threshold == 1.0 or N == 0:
                return 0.0
            net_benefit = (tp / N) - (fp / N) * (threshold / (1 - threshold))
            return net_benefit
        nb_model = calculate_net_benefit(y_true, y_prob, threshold, weights)
        # Treat All 也用权重
        N = np.sum(weights) if weights is not None else len(y_true)
        n_pos = np.sum(weights[y_true == 1]) if weights is not None else np.sum(y_true == 1)
        prevalence = n_pos / N if N > 0 else 0
        nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold)) if threshold < 1.0 else 0.0
        net_benefits_model.append(nb_model)
        net_benefits_all.append(nb_all)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(thresholds, net_benefits_model, label='Model', color='red', linewidth=2)
    ax.plot(thresholds, net_benefits_all, label='Treat All', color='blue', linewidth=2)
    ax.plot(thresholds, net_benefit_none, label='Treat None', color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.set_title('Decision Curve Analysis', pad=30)
    ax.legend(loc='lower right')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_DCA.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({
        'thresholds': [float(t) for t in thresholds],
        'net_benefit_model': [float(nb) for nb in net_benefits_model],
        'net_benefit_all': [float(nb) for nb in net_benefits_all],
        'net_benefit_none': [float(nb) for nb in net_benefit_none]
    }, str(Path(plot_data_dir) / f"{model_name}_DCA_data.json"))

def plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir):
    """绘制样本量学习曲线，展示随着训练样本数量的增加，模型性能的变化
    
    横坐标：训练样本数量
    纵坐标：模型性能指标（AUC-ROC）
    包含训练集和测试集上的性能曲线
    使用交叉验证评估训练集性能，避免显示过拟合
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    
    # 定义要测试的样本量比例
    train_sizes = np.linspace(0.1, 1.0, 10)  # 从10%到90%的训练数据
    train_sizes_abs = [int(train_size * len(X_train)) for train_size in train_sizes]
    
    train_scores = []
    test_scores = []
    
    # 保存每一个样本量的所有交叉验证分数，用于计算置信区间
    train_scores_all_folds = []
    
    for n_samples in train_sizes_abs:
        # 随机选择n_samples个样本进行训练
        indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train_subset = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
        y_train_subset = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        try:
            # 克隆模型并使用子集训练
            model_clone = clone(model)
            
            # 使用5折交叉验证评估训练集性能，与训练代码中使用的5折保持一致
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_cv_scores = []
            
            # 使用交叉验证评估训练集性能
            fold_scores = []  # 收集当前样本量的每折分数
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_subset, y_train_subset)):
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
                
                # 训练模型并预测
                cv_model = clone(model)
                cv_model.fit(X_train_cv, y_train_cv)
                val_pred = cv_model.predict_proba(X_val_cv)[:, 1]
                
                # 计算验证集上AUC
                try:
                    val_score = roc_auc_score(y_val_cv, val_pred)
                    train_cv_scores.append(val_score)
                    fold_scores.append(val_score)
                    print(f"  - 折 {fold_idx+1}/5: AUC = {val_score:.4f}")
                except Exception as e:
                    print(f"交叉验证失败: {e}")
            
            # 如果有效的交叉验证分数，则平均
            if train_cv_scores:
                train_score = np.mean(train_cv_scores)
                # 存储每一个样本量的所有折分数，用于计算置信区间
                train_scores_all_folds.append(fold_scores)
            else:
                # 如果交叉验证失败，回退到直接训练
                model_clone.fit(X_train_subset, y_train_subset)
                train_pred = model_clone.predict_proba(X_train_subset)[:, 1]
                train_score = roc_auc_score(y_train_subset, train_pred)
                print("交叉验证失败，使用直接评估")
                # 当交叉验证失败时，添加空列表
                train_scores_all_folds.append([])
                
            # 在全量训练集上训练模型并在测试集上评估
            model_clone = clone(model)
            model_clone.fit(X_train_subset, y_train_subset)
            test_pred = model_clone.predict_proba(X_test)[:, 1]
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
    
    # 计算置信区间 - 使用标准误差(SEM)而非标准差，并使用更小的z值
    train_scores_sem = []
    for fold_scores in valid_train_scores_all_folds:
        if len(fold_scores) >= 2:  # 需要至少2个有效折才能计算标准误差
            # 计算标准误差 (SEM = STD / sqrt(n))
            std = np.std(fold_scores, ddof=1)
            sem = std / np.sqrt(len(fold_scores))
            train_scores_sem.append(sem)
        else:
            train_scores_sem.append(0.0)  # 如果数据不足，设置标准误差为0
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制置信区间 - 使用标准误差和较小的z值(1.0而非1.96)
    z_value = 1.0  # 对应约68%置信区间而非95%
    ax.fill_between(valid_train_sizes, 
                    [max(0.5, score - z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                    [min(1.0, score + z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                    alpha=0.15, color='#2ca02c')
    
    # 绘制主线
    ax.plot(valid_train_sizes, valid_train_scores, 'o-', label='Training Set (CV)', color='#2ca02c', linewidth=2)
    ax.plot(valid_train_sizes, valid_test_scores, 'o-', label='Test Set', color='#d62728', linewidth=2)
    
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("AUC-ROC")
    
    # Set y-axis range from 0.5 to 1.0 with 0.1 intervals
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    
    ax.set_title(f"Sample Learning Curve - {model_name}", pad=30)
    
    # 美化图形
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend(loc='lower right')
    
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_sample_learning_curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存数据
    save_plot_data({
        'train_sizes': valid_train_sizes,
        'train_scores': valid_train_scores,
        'test_scores': valid_test_scores
    }, str(Path(plot_data_dir) / f"{model_name}_sample_learning_curve.json"))

def plot_confusion_matrix(y_true, y_pred, model_name, plot_dir, plot_data_dir, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f"Normalized Confusion Matrix - {model_name}"
        fname = f"{model_name}_confusion_matrix_normalized.png"
        data_fname = f"{model_name}_confusion_matrix_normalized.json"
    else:
        cm_display = cm
        fmt = 'd'
        title = f"Confusion Matrix - {model_name}"
        fname = f"{model_name}_confusion_matrix.png"
        data_fname = f"{model_name}_confusion_matrix.json"
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=['0', '1'], yticklabels=['0', '1'],
                square=True, ax=ax)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_title(title, pad=30)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    save_plot_data({'confusion_matrix': cm.tolist()}, str(Path(plot_data_dir) / data_fname))

def plot_threshold_curve(y_true, y_prob, model_name, plot_dir, plot_data_dir):
    thresholds = np.linspace(0, 1, 101)
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    f1_list = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        f1_list.append(f1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.plot(thresholds, sensitivity_list, label="Sensitivity", color='red', linewidth=2)
    ax.plot(thresholds, specificity_list, label="Specificity", color='blue', linewidth=2)
    ax.plot(thresholds, precision_list, label="Precision", color='green', linewidth=2)
    ax.plot(thresholds, f1_list, label="F1 Score", color='orange', linewidth=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Threshold Curve - {model_name}", pad=30)
    # 只显示左下角一个0
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower left')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_threshold_curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({
        'thresholds': thresholds.tolist(),
        'sensitivity': sensitivity_list,
        'specificity': specificity_list,
        'precision': precision_list,
        'f1': f1_list
    }, str(Path(plot_data_dir) / f"{model_name}_threshold_curve.json"))
