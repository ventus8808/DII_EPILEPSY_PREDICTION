import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import matplotlib.font_manager as fm

# 确保所有文本都使用Monaco字体
font_path = None
font_dirs = ['/System/Library/Fonts/', '/Library/Fonts/']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    if 'monaco' in font_file.lower():
        font_path = font_file
        break
if font_path:
    fm.fontManager.addfont(font_path)
    monaco_font = fm.FontProperties(family='Monaco', size=13)  # 调整为13号字体

# 全局字体和样式设置 - 确保一致性
sns.set(style='white')  # 使用纯白色背景无网格

# 使用完全相同的字体大小设置
plt.rcParams['font.family'] = 'Monaco'
plt.rcParams['font.size'] = 13  # 调整为13号字体
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'  # 设置图表区域背景为白色
plt.rcParams['figure.facecolor'] = 'white'  # 设置整个图片背景为白色
plt.rcParams['axes.grid'] = False  # 关闭网格线
plt.rcParams['lines.linewidth'] = 2  # 设置线条宽度
plt.rcParams['lines.markersize'] = 4  # 设置标记大小
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  # 一致的颜色循环

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_roc_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir):
    # 使用与阈值曲线相同的上下文设置
    with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
        fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
        # 使用与calculate_metrics函数完全相同的方法计算AUC
        roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)
        # 四舍五入到3位小数，用于图表显示
        roc_auc_rounded = round(roc_auc, 3)
        # 设置图的大小与阈值曲线相同
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal', adjustable='box')
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_rounded:.3f})', color='red', linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
        
        # 设置标签和字体
        ax.set_xlabel('False Positive Rate', fontproperties=monaco_font)
        ax.set_ylabel('True Positive Rate', fontproperties=monaco_font)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'ROC - {model_name}', pad=30, fontproperties=monaco_font)
        
        # 只显示x轴的零点，不显示y轴的零点
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 明确设置刻度标签的字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(monaco_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(monaco_font)
        
        # 设置图例字体
        legend = ax.legend(loc='lower right', prop=monaco_font)
        
        # 显示上边框和右边框
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_ROC.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': float(roc_auc)}, str(Path(plot_data_dir) / f"{model_name}_ROC_data.json"))

def plot_pr_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir):
    # 使用与阈值曲线和ROC曲线相同的上下文设置
    with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
        precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=weights)
        # 使用与calculate_metrics函数完全相同的方法计算PR-AUC
        pr_auc = auc(recall, precision)
        # 四舍五入到3位小数，用于图表显示
        pr_auc_rounded = round(pr_auc, 3)
        # 设置图的大小与阈值曲线相同
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal', adjustable='box')
        ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc_rounded:.3f})', color='green', linewidth=2)
        
        # 设置标签和字体
        ax.set_xlabel('Recall', fontproperties=monaco_font)
        ax.set_ylabel('Precision', fontproperties=monaco_font)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'PR - {model_name}', pad=30, fontproperties=monaco_font)
        
        # 只显示x轴的零点，不显示y轴的零点
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 明确设置刻度标签的字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(monaco_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(monaco_font)
        
        # 设置图例字体
        legend = ax.legend(loc='lower left', prop=monaco_font)
        
        # 显示上边框和右边框
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_PR.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    save_plot_data({'recall': recall.tolist(), 'precision': precision.tolist(), 'pr_auc': float(pr_auc)}, str(Path(plot_data_dir) / f"{model_name}_PR_data.json"))

def plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir, cv=5, scoring='roc_auc'):
    """绘制样本量学习曲线，展示随着训练样本数量的增加，模型性能的变化
    
    横坐标：训练样本数量
    纵坐标：模型性能指标（AUC-ROC）
    包含训练集和测试集上的性能曲线
    使用交叉验证评估训练集性能，避免显示过拟合
    
    Parameters:
    -----------
    cv : int, 交叉验证折数, 默认5
    scoring : str, 评分标准, 默认'roc_auc'
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from imblearn.over_sampling import SMOTE
    
    # 定义要测试的样本量 - 使用1000开始，每500递增
    start_size = 1000
    step_size = 500
    max_train_size = len(X_train)
    train_sizes_abs = list(range(start_size, max_train_size, step_size))
    # 确保包含最后一个点（全部训练数据）
    if train_sizes_abs[-1] != max_train_size and max_train_size - train_sizes_abs[-1] > 100:
        train_sizes_abs.append(max_train_size)
    
    train_scores = []
    test_scores = []
    
    # 保存每一个样本量的所有交叉验证分数，用于计算置信区间
    train_scores_all_folds = []
    
    for n_samples in train_sizes_abs:
        # 使用分层抽样选择n_samples个样本进行训练
        # 处理边界情况：如果n_samples等于训练集大小，直接使用全部样本
        if n_samples >= len(X_train):
            X_train_subset = X_train
            y_train_subset = y_train
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
            for train_idx, _ in sss.split(X_train, y_train):
                X_train_subset = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_train_subset = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        
        try:
            # 克隆模型并使用子集训练
            model_clone = clone(model)
            
            # 使用传入的cv参数决定折数
            cv_splits = cv if isinstance(cv, int) else 5  # 默认与函数参数一致
            stratified_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            train_cv_scores = []
            
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
                cv_model = clone(model)
                cv_model.fit(X_train_cv_resampled, y_train_cv_resampled)
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
                
            # 对子集应用SMOTE过采样，然后在测试集上评估
            smote_test = SMOTE(random_state=42)
            X_train_subset_resampled, y_train_subset_resampled = smote_test.fit_resample(X_train_subset, y_train_subset)
            
            # 在过采样后的训练集上训练模型并在测试集上评估
            model_clone = clone(model)
            model_clone.fit(X_train_subset_resampled, y_train_subset_resampled)
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
    
    # 绘图 - 使用与阈值曲线和ROC曲线相同的上下文设置
    with plt.rc_context({'font.family': 'Monaco', 'font.size': 13}):
        # 与其他图表保持一致的大小
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # 去掉equal aspect以防止图表变形
        # ax.set_aspect('equal', adjustable='box')
        
        # 绘制置信区间 - 使用标准误差和较小的z值(1.0而非1.96)
        z_value = 1.0  # 对应68%置信区间而非95%
        ax.fill_between(valid_train_sizes, 
                        [max(0.5, score - z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                        [min(1.0, score + z_value * sem) for score, sem in zip(valid_train_scores, train_scores_sem)],
                        alpha=0.15, color='green')
        
        # 绘制主线 - 线条上不显示点 - 使用浅绿色和粉色
        light_green = '#7fbf7f'  # 浅绿色
        pink = '#ff9eb5'  # 粉色
        
        ax.plot(valid_train_sizes, valid_train_scores, '-', label='Training Set (CV)', color=light_green, linewidth=2)
        ax.plot(valid_train_sizes, valid_test_scores, '-', label='Test Set', color=pink, linewidth=2)
        
        # 单独绘制数据点(不加到图例中) - 同样使用新颜色
        ax.plot(valid_train_sizes, valid_train_scores, 'o', color=light_green, markersize=5, alpha=0.8)
        ax.plot(valid_train_sizes, valid_test_scores, 'o', color=pink, markersize=5, alpha=0.8)
        
        # 设置固定的X轴范围为0-9
        ax.set_xlim(0, 9000)  # 从0到9000个样本
        
        # 创建介于0到9000的整数刻度
        fixed_tick_values = [i * 1000 for i in range(10)]  # 0, 1000, 2000, ..., 9000
        
        # 设置所有数据点的网格线
        ax.set_xticks(valid_train_sizes, minor=True)
        ax.set_xticklabels([""] * len(valid_train_sizes), minor=True)
        
        # 设置主要刻度标签 (0-9)
        ax.set_xticks(fixed_tick_values)
        ax.set_xticklabels([str(i) for i in range(10)])
        
        # 确保网格线显示
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', which='both', length=4)
        
        # 设置轴标签
        ax.set_xlabel("Number of Training Samples (×1000)", fontproperties=monaco_font)
        ax.set_ylabel("AUC-ROC", fontproperties=monaco_font)
        
        # Set y-axis range from 0.5 to 1.0 with 0.1 intervals
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        
        # 设置标题 - 将标题移到图内
        ax.set_title(f"Sample Learning Curve - {model_name}", pad=20, fontproperties=monaco_font)
        
        # 明确设置刻度标签的字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(monaco_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(monaco_font)
        
        # 添加更明显的背景网格
        ax.grid(linestyle='--', alpha=0.3, color='gray')
        
        # 设置图例样式与ROC曲线完全相同
        legend = ax.legend(loc='lower right', prop=monaco_font, frameon=True, framealpha=1.0, facecolor='white', edgecolor='lightgray')
    
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_Sample_Learning_Curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存数据
    save_plot_data({
        'train_sizes': valid_train_sizes,
        'train_scores': valid_train_scores,
        'test_scores': valid_test_scores
    }, str(Path(plot_data_dir) / f"{model_name}_Sample_Learning_Curve.json"))

def plot_confusion_matrix(y_true, y_pred, model_name, plot_dir, plot_data_dir, normalize=False):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f"Confusion Matrix - {model_name}"
        fname = f"{model_name}_Normalized_Confusion_Matrix.png"
        data_fname = f"{model_name}_Normalized_Confusion_Matrix.json"
    else:
        cm_display = cm
        fmt = 'd'
        title = f"Confusion Matrix - {model_name}"
        fname = f"{model_name}_Confusion_Matrix.png"
        data_fname = f"{model_name}_Confusion_Matrix.json"
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # 绘制热图
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', cbar=False,
               xticklabels=['0', '1'], yticklabels=['0', '1'],
               square=True, ax=ax, annot_kws={"fontproperties": monaco_font})
    
    # 设置轴标签和标题
    ax.set_ylabel('True label', fontproperties=monaco_font)
    ax.set_xlabel('Predicted label', fontproperties=monaco_font)
    ax.set_title(title, pad=40, fontproperties=monaco_font)  # 增加pad值，将标题往上移动
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # 保存图片
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 保存数据
    save_plot_data({'confusion_matrix': cm.tolist()}, str(Path(plot_data_dir) / data_fname))

def plot_threshold_curve(y_true, y_prob, model_name, plot_dir, plot_data_dir):
    # 计算各个阈值下的指标
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
    
    # 创建图形并确保使用全局设置
    with plt.rc_context({'font.family': 'Monaco'}):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal', adjustable='box')
        
        # 绘制线条
        ax.plot(thresholds, sensitivity_list, label="Sensitivity", color='red', linewidth=2)
        ax.plot(thresholds, specificity_list, label="Specificity", color='blue', linewidth=2)
        ax.plot(thresholds, precision_list, label="Precision", color='green', linewidth=2)
        ax.plot(thresholds, f1_list, label="F1 Score", color='orange', linewidth=2)
        
        # 设置轴和标题
        ax.set_xlabel("Threshold", fontproperties=monaco_font)
        ax.set_ylabel("Score", fontproperties=monaco_font)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Threshold Curve - {model_name}", pad=30, fontproperties=monaco_font)
        
        # 设置刻度 - 只显示x轴的零点，不显示y轴的零点
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 明确设置刻度标签的字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(monaco_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(monaco_font)
        
        # 设置图例字体
        legend = ax.legend(loc='lower left', prop=monaco_font)
        
        # 显示顶部和右侧边框
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)
    
    # 保存图片
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_Threshold_Curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存数据
    save_plot_data({
        'thresholds': thresholds.tolist(),
        'sensitivity': sensitivity_list,
        'specificity': specificity_list,
        'precision': precision_list,
        'f1': f1_list
    }, str(Path(plot_data_dir) / f"{model_name}_Threshold_Curve.json"))
