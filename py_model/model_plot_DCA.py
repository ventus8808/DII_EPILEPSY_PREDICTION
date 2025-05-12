import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
from imblearn.over_sampling import SMOTE

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
plt.rcParams['font.size'] = 13
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

def save_plot_data(data, filename):
    """保存绘图数据为JSON文件，便于后续分析"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def plot_dca_curve_comparison(y_true, y_probs_dict, weights, model_name, plot_dir, plot_data_dir, use_smote=True):
    """
    绘制多模型决策曲线分析(DCA)曲线对比，特别用于评估DII对模型性能的贡献
    
    Parameters:
    -----------
    y_true : array-like
        真实标签，0或1
    y_probs_dict : dict
        预测为阳性的概率字典，格式为 {model_name: y_prob}
    weights : array-like, optional
        样本权重
    model_name : str
        基础模型名称，用于图表标题和文件名
    plot_dir : Path
        图表保存目录
    plot_data_dir : Path
        原始数据保存目录
    use_smote : bool
        是否使用SMOTE过采样
    """
    # 原始数据统计信息
    total_samples = len(y_true)
    positive_samples = np.sum(y_true)
    positive_rate = positive_samples / total_samples * 100
    
    print(f"\n==== DCA对比曲线信息 =====")
    print(f"原始数据集统计信息:")
    print(f"- 总样本数: {total_samples}")
    print(f"- 正例数量: {int(positive_samples)} ({positive_rate:.2f}%)")
    
    # 如果启用SMOTE过采样，对数据进行过采样处理
    X = np.column_stack([list(y_probs_dict.values())[0]])  # 使用第一个模型的预测概率作为特征
    y = y_true
    
    if use_smote:
        print("\n应用SMOTE过采样使正负样本比例平衡...")
        try:
            # 创建SMOTE过采样器，使用适当比例
            smote = SMOTE(random_state=42, sampling_strategy=0.3)  # 0.3表示少数类:多数类 = 0.3:1
            X_res, y_res = smote.fit_resample(X, y)
            
            # 提取过采样后的数据
            y_true = y_res
            
            # 过采样后的统计信息
            total_samples_bal = len(y_true)
            positive_samples_bal = np.sum(y_true)
            positive_rate_bal = positive_samples_bal / total_samples_bal * 100
            
            print(f"\n过采样后数据集统计信息:")
            print(f"- 总样本数: {total_samples_bal}")
            print(f"- 正例数量: {int(positive_samples_bal)} ({positive_rate_bal:.2f}%)")
            print(f"- 负例数量: {total_samples_bal - int(positive_samples_bal)} ({100 - positive_rate_bal:.2f}%)")
            
            # 过采样后权重都设为1
            weights = np.ones_like(y_true)
            
            # 对每个模型的概率进行过采样处理
            # 解决方案：将原始数据按标签重新组合，然后重新提取概率
            y_probs_resampled = {}
            
            # 第一个模型的过采样概率从X_res直接获取
            first_model_name = list(y_probs_dict.keys())[0]
            y_probs_resampled[first_model_name] = X_res[:, 0]
            
            # 其他模型需要单独进行过采样 
            for idx, (model_name, y_prob) in enumerate(y_probs_dict.items()):
                if idx == 0:  # 第一个模型已处理
                    continue
                    
                # 为其他模型创建特征并过采样
                X_model = np.column_stack([y_prob])
                X_model_res, _ = smote.fit_resample(X_model, y)
                y_probs_resampled[model_name] = X_model_res[:, 0]
            
            # 使用过采样后的概率
            y_probs_dict = y_probs_resampled
        except Exception as e:
            print(f"SMOTE过采样失败: {e}")
            print("使用原始不平衡数据集继续...")
    
    # 确保完整的阈值范围，从0到1
    thresholds = np.linspace(0.001, 0.999, 100)  # 使用100个点均匀覆盖从接近0到接近1
    
    # 计算患病率
    if weights is None:
        prevalence = np.mean(y_true)
    else:
        prevalence = np.average(y_true, weights=weights)
    
    print(f"当前数据患病率: {prevalence:.4f}")
    
    # 创建正方形图形，使用与model_plot_utils一致的尺寸
    plt.figure(figsize=(7, 7))
    
    # 遵循用户要求，使用Monaco字体
    plt.rcParams['font.family'] = 'Monaco'
    
    # 计算"全部治疗"策略的净收益
    net_benefits_all = []
    for threshold in thresholds:
        if threshold >= 1:
            threshold = 0.999  # 防止除以零
        net_benefit_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        # 保留原始值，包括负值
        net_benefits_all.append(net_benefit_all)
        
    # "无人治疗"策略(视所有人为阴性)的净收益总是0
    net_benefits_none = [0] * len(thresholds)
    
    # 保存所有模型的净收益，用于后续分析
    comparison_data = {
        'thresholds': thresholds.tolist(),
        'treat_all': net_benefits_all,
        'treat_none': net_benefits_none,
        'models': {}
    }
    
    # 计算并绘制每个模型的DCA曲线
    for i, (model_name, y_prob) in enumerate(y_probs_dict.items()):
        # 计算该模型的净收益
        net_benefits_model = []
        for threshold in thresholds:
            net_benefit = calculate_net_benefit(y_true, y_prob, threshold, weights)
            net_benefits_model.append(net_benefit)
                
        # 保存数据
        comparison_data['models'][model_name] = net_benefits_model
                
        # 绘制曲线 - 使用model_plot_utils的颜色方案
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][i % 4]
        plt.plot(thresholds, net_benefits_model, label=model_name, color=color, linewidth=2)
        
    # 绘制"Treat All"策略的曲线 - 绿色长虚线
    plt.plot(
        thresholds, net_benefits_all, 
        label='Treat All', 
        color='green', 
        linewidth=1.5, 
        linestyle='--'
    )
    
    # 绘制"Treat None"策略的曲线 - 红色长虚线
    plt.plot(
        thresholds, net_benefits_none, 
        label='Treat None', 
        color='red', 
        linewidth=1.5, 
        linestyle='--'
    )
    
    # 设置图表样式
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    # 从model_name中提取基础名称（不包含括号）
    base_model_name = model_name.split('(')[0].strip()
    plt.title(f'Decision Curve Analysis - {base_model_name}')
    # 图例移动到右下角并缩小，添加边框并增加虹线长度
    plt.legend(loc='lower right', frameon=True, fontsize=10, fancybox=True, framealpha=0.8, handlelength=3.0)
    plt.grid(False)
    plt.xlim(0.0, 1.0)  # 设置横轴范围
    plt.ylim(-0.5, 0.5)  # 设置纵轴范围
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)  # 留出空间显示完整曲线
    
    # 提取基础模型名称，不考虑括号和特殊字符
    base_name = model_name.split('(')[0].strip()
    # 防止空白基础名称
    if not base_name:
        base_name = "Model"
    
    # 保存图表
    dca_plot_path = plot_dir / f'{base_name}_DCA_DII.png'
    plt.savefig(dca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据以便后续分析
    dca_data_path = plot_data_dir / f'{base_name}_dca_dii_data.json'
    save_plot_data(comparison_data, dca_data_path)
    
    print(f"DII contribution DCA curve saved to: {dca_plot_path}")
    print(f"DII contribution data saved to: {dca_data_path}")
    
    # 计算并打印DII的贡献
    dii_contribution = {}
    for threshold_idx, threshold in enumerate(thresholds):
        with_dii = comparison_data['models'][list(y_probs_dict.keys())[0]][threshold_idx]
        without_dii = comparison_data['models'][list(y_probs_dict.keys())[1]][threshold_idx]
        contribution = with_dii - without_dii
        if threshold in [0.05, 0.1, 0.2, 0.3]:
            dii_contribution[threshold] = contribution
            print(f"DII net benefit at threshold {threshold:.2f}: {contribution:.4f}")
    
    return dca_plot_path

def calculate_net_benefit(y_true, y_prob, threshold, weights=None):
    """
    计算给定阈值下的净收益(Net Benefit)
    
    Parameters:
    -----------
    y_true : array-like
        真实标签，0或1
    y_prob : array-like
        预测为阳性的概率
    threshold : float
        决策阈值，范围[0,1]
    weights : array-like, optional
        样本权重
        
    Returns:
    --------
    net_benefit : float
        净收益值
    """
    if weights is None:
        weights = np.ones_like(y_true)
    
    # 基于阈值将概率转换为预测标签
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算被视为阳性(预测为1)的样本数和总样本数
    n_total = np.sum(weights)
    
    # 确保y_true和y_pred是布尔数组以进行逻辑运算
    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred, dtype=bool)
    
    # 计算真阳性和假阳性的权重和
    w_tp = np.sum(weights[np.logical_and(y_pred_bool, y_true_bool)])
    w_fp = np.sum(weights[np.logical_and(y_pred_bool, np.logical_not(y_true_bool))])
    
    # 防止除以零
    if threshold >= 1:
        threshold = 0.999
    
    # 计算净收益: TP-率 - (FP-率 * 阈值/(1-阈值))
    net_benefit = (w_tp / n_total) - (w_fp / n_total) * (threshold / (1 - threshold))
    
    return net_benefit

def plot_dca_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir, use_smote=True):
    """绘制DCA曲线，用于评估模型预测的净收益
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        weights: 样本权重
        model_name: 模型名称
        plot_dir: 图像保存路径
        plot_data_dir: 数据保存路径
        use_smote: 是否使用SMOTE过采样
        原始数据保存目录
    """
    # 原始数据统计信息
    total_samples = len(y_true)
    positive_samples = np.sum(y_true)
    positive_rate = positive_samples / total_samples * 100
    avg_pred_prob = np.mean(y_prob)
    
    print(f"\n==== DCA曲线信息 =====")
    print(f"原始数据集统计信息:")
    print(f"- 总样本数: {total_samples}")
    print(f"- 正例数量: {int(positive_samples)} ({positive_rate:.2f}%)")
    print(f"- 平均预测概率: {avg_pred_prob:.4f}")
    
    # 如果启用SMOTE过采样，对数据进行过采样处理
    X = np.column_stack([y_prob])
    y = y_true
    
    if use_smote:
        print("\n应用SMOTE过采样使正负样本比例平衡...")
        try:
            # 创建 SMOTE 过采样器，使用适当比例
            smote = SMOTE(random_state=42, sampling_strategy=0.3)  # 0.3表示少数类:多数类 = 0.3:1
            X_res, y_res = smote.fit_resample(X, y)
            
            # 提取过采样后的预测概率和真实标签
            y_prob = X_res[:, 0]
            y_true = y_res
            
            # 输出过采样后的数据集信息
            total_samples_bal = len(y_true)
            positive_samples_bal = np.sum(y_true)
            positive_rate_bal = positive_samples_bal / total_samples_bal * 100
            avg_pred_prob_bal = np.mean(y_prob)
            
            print(f"\n过采样后数据集统计信息:")
            print(f"- 总样本数: {total_samples_bal}")
            print(f"- 正例数量: {int(positive_samples_bal)} ({positive_rate_bal:.2f}%)")
            print(f"- 负例数量: {total_samples_bal - int(positive_samples_bal)} ({100 - positive_rate_bal:.2f}%)")
            print(f"- 平均预测概率: {avg_pred_prob_bal:.4f}")
            
            # 过采样后权重都设为1
            weights = np.ones_like(y_true)
        except Exception as e:
            print(f"SMOTE过采样失败: {e}")
            print("使用原始不平衡数据集继续...")
    
    # 创建阈值序列 - 避免极端值
    thresholds = np.linspace(0.01, 0.99, 50)  # 重点关注低阈值区间，因为疾病为稀有事件
    
    # 计算模型在各阈值下的净收益
    net_benefits_model = []
    for threshold in thresholds:
        net_benefit = calculate_net_benefit(y_true, y_prob, threshold, weights)
        net_benefits_model.append(net_benefit)
    
    # 计算"全部治疗"策略(视所有人为阳性)的净收益
    # 全部治疗的净收益 = 患病率 - (1-患病率) * pt/(1-pt)
    if weights is None:
        prevalence = np.mean(y_true)
    else:
        prevalence = np.average(y_true, weights=weights)
    
    print(f"当前数据患病率: {prevalence:.4f}")
    
    net_benefits_all = []
    for threshold in thresholds:
        if threshold >= 1:
            threshold = 0.999  # 防止除以零
        net_benefit_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        # 保留原始值，包括负值
        net_benefits_all.append(net_benefit_all)
    
    # "无人治疗"策略(视所有人为阴性)的净收益总是0
    net_benefits_none = [0] * len(thresholds)
    
    # 不应用限制，保留原始净收益值，包括负值
    net_benefits_model_adj = net_benefits_model
    
    # 创建正方形图形，使用与model_plot_utils一致的尺寸
    plt.figure(figsize=(7, 7))
    
    # 遵循用户要求，使用Monaco字体
    plt.rcParams['font.family'] = 'Monaco'
    
    # 绘制模型的DCA曲线
    plt.plot(
        thresholds, net_benefits_model_adj, 
        label=f'{model_name}', 
        color='#1f77b4', 
        linewidth=2
    )
    
    # 绘制"Treat All"策略的曲线 - 绿色长虚线
    plt.plot(
        thresholds, net_benefits_all, 
        label='Treat All', 
        color='green', 
        linewidth=1.5, 
        linestyle='--'
    )
    
    # 绘制"Treat None"策略的曲线 - 红色长虚线
    plt.plot(
        thresholds, net_benefits_none, 
        label='Treat None', 
        color='red', 
        linewidth=1.5, 
        linestyle='--'
    )
    
    # 设置图表样式
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    # 从model_name中提取基础名称（不包含括号）
    base_model_name = model_name.split('(')[0].strip()
    # 简化标题，只保留模型名
    plt.title(f'Decision Curve Analysis - {base_model_name}')
    # 图例移动到右下角并缩小，添加边框并增加虹线长度
    plt.legend(loc='lower right', frameon=True, fontsize=10, fancybox=True, framealpha=0.8, handlelength=3.0)
    plt.grid(False)
    plt.xlim(0.0, 1.0)  # 设置横轴范围
    plt.ylim(-0.5, 0.5)  # 设置纵轴范围
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)  # 留出空间显示完整曲线
    
    # 保存图表
    dca_plot_path = plot_dir / f'{model_name}_DCA.png'
    plt.savefig(dca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据以便后续分析
    dca_data = {
        'thresholds': thresholds.tolist(),
        'net_benefits_model': net_benefits_model,
        'net_benefits_all': net_benefits_all,
        'net_benefits_none': net_benefits_none,
        'model_name': model_name,
        'prevalence': float(prevalence)
    }
    
    dca_data_path = plot_data_dir / f'{model_name}_dca_data.json'
    save_plot_data(dca_data, dca_data_path)
    
    print(f"决策曲线分析(DCA)图表已保存至: {dca_plot_path}")
    print(f"决策曲线分析(DCA)数据已保存至: {dca_data_path}")
    
    return dca_plot_path
