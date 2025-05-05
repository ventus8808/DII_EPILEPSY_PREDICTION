import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import json
import signal
import warnings
from scipy import interpolate, stats
from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from sklearn.metrics import (
    brier_score_loss, precision_recall_curve, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# 设置matplotlib简约风格
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Monaco']  # Monaco字体
plt.rcParams['font.size'] = 12  # 12号字体
plt.rcParams['axes.linewidth'] = 1.0  # 细边框
plt.rcParams['axes.edgecolor'] = '#333333'  # 深灰色边框
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 1.5  # 更细的线条
plt.rcParams['lines.markersize'] = 4  # 更小的标记点
plt.rcParams['lines.antialiased'] = True  # 抗锯齿
plt.rcParams['figure.dpi'] = 300  # 提高DPI到300

# 保存图表数据到JSON文件
def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def plot_calibration_all_data(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir, n_bins=30, use_smote=True):
    """在全量数据集上绘制校准曲线，包含多种校准方法
    
    参数：
    y_true : 真实标签
    y_prob : 预测概率
    weights : 样本权重
    model_name : 模型名称
    plot_dir : 图表保存目录
    plot_data_dir : 图表数据保存目录
    n_bins : 分箱数量
    use_smote : 是否使用SMOTE过采样
    """
    # 设置简约风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 装饰器用于设置超时
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # 进度的下标
    method_index = 0
    total_methods = 10  # 选定的校准方法数
    
    # 忽略警告
    warnings.filterwarnings('ignore')
    
    print(f"=== 全量数据校准曲线计算进度 === ")
    
    # 初始化校准方法字典
    calibration_methods = {}
    
    # 1. Platt缩放（逻辑回归）
    method_index += 1
    print(f"[{method_index}/{total_methods}] Platt缩放(逻辑回归)...")
    try:
        signal.alarm(10)  # 设置10秒超时
        # 使用逻辑回归做校准
        lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        X_train = y_prob.reshape(-1, 1)  # 转成二维数组
        lr.fit(X_train, y_true)
        platt_probs = lr.predict_proba(X_train)[:, 1]
        calibration_methods['platt'] = {
            'prob': platt_probs,
            'brier': brier_score_loss(y_true, platt_probs, sample_weight=weights)
        }
        signal.alarm(0)  # 取消超时
    except Exception as e:
        print(f"Platt缩放校准失败: {e}")
    
    # 2. 等温回归
    method_index += 1
    print(f"[{method_index}/{total_methods}] 等温回归...")
    try:
        signal.alarm(10)  # 设置10秒超时
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_prob, y_true, sample_weight=weights)
        isotonic_probs = ir.predict(y_prob)
        calibration_methods['isotonic'] = {
            'prob': isotonic_probs,
            'brier': brier_score_loss(y_true, isotonic_probs, sample_weight=weights)
        }
        signal.alarm(0)  # 取消超时
    except Exception as e:
        print(f"等温回归校准失败: {e}")
    
    # 3. 自适应分箱
    method_index += 1
    print(f"[{method_index}/{total_methods}] 自适应分箱...")
    try:
        signal.alarm(10)
        # 使用分位数进行自适应分箱
        quantiles = np.linspace(0, 1, n_bins+1)
        bins = np.quantile(y_prob, quantiles)
        # 去除重复的bin
        bins = np.unique(bins)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_sums = np.bincount(bin_indices, weights=y_true if weights is None else y_true * weights, minlength=len(bins))
        bin_weights = np.bincount(bin_indices, weights=weights, minlength=len(bins))
        # 防止除零
        bin_weights[bin_weights == 0] = 1
        bin_means = bin_sums / bin_weights
        # 映射概率
        adaptive_probs = bin_means[bin_indices]
        calibration_methods['adaptive'] = {
            'prob': adaptive_probs,
            'brier': brier_score_loss(y_true, adaptive_probs, sample_weight=weights)
        }
        signal.alarm(0)
    except Exception as e:
        print(f"自适应分箱校准失败: {e}")
        
    # 4. 直方图分箱
    method_index += 1
    print(f"[{method_index}/{total_methods}] 直方图分箱...")
    try:
        signal.alarm(10)
        # 使用等宽分箱
        bins = np.linspace(0, 1, n_bins+1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_sums = np.bincount(bin_indices, weights=y_true if weights is None else y_true * weights, minlength=len(bins))
        bin_weights = np.bincount(bin_indices, weights=weights, minlength=len(bins))
        # 防止除零
        bin_weights[bin_weights == 0] = 1
        bin_means = bin_sums / bin_weights
        # 映射概率
        histogram_probs = bin_means[bin_indices]
        calibration_methods['histogram'] = {
            'prob': histogram_probs,
            'brier': brier_score_loss(y_true, histogram_probs, sample_weight=weights)
        }
        signal.alarm(0)
    except Exception as e:
        print(f"直方图分箱校准失败: {e}")
        
    # 5. 贝叶斯分箱
    method_index += 1
    print(f"[{method_index}/{total_methods}] 贝叶斯分箱...")
    try:
        signal.alarm(10)
        # 实现简化版贝叶斯分箱量化(BBQ)
        # 使用分位数确保每个箱有足够的样本
        n_bbq_bins = max(10, n_bins // 5)  # 贝叶斯分箱使用较少的箱
        quantiles = np.linspace(0, 1, n_bbq_bins+1)
        bins = np.quantile(y_prob, quantiles)
        bins = np.unique(bins)  # 去除重复的bin
        
        # 计算每个箱中的先验分布
        prior_alpha = 1  # 先验分布的超参数
        prior_beta = 1   # 使用Beta(1,1)作为均匀先验
        
        bin_indices = np.digitize(y_prob, bins) - 1
        
        # 求每个箱中的正例数量和总样本数
        bin_positive_counts = np.bincount(bin_indices, weights=y_true, minlength=len(bins))
        bin_total_counts = np.bincount(bin_indices, minlength=len(bins))
        
        # 应用贝叶斯公式计算后验分布
        posterior_alpha = prior_alpha + bin_positive_counts
        posterior_beta = prior_beta + bin_total_counts - bin_positive_counts
        
        # 使用后验平均值作为校准概率
        bayesian_probs = posterior_alpha / (posterior_alpha + posterior_beta)
        bayesian_probs = bayesian_probs[bin_indices]
        
        calibration_methods['bbq'] = {
            'prob': bayesian_probs,
            'brier': brier_score_loss(y_true, bayesian_probs, sample_weight=weights)
        }
        signal.alarm(0)
    except Exception as e:
        print(f"贝叶斯分箱校准失败: {e}")
    
    # 6. Sigmoid拟合校准
    method_index += 1
    print(f"[{method_index}/{total_methods}] Sigmoid拟合校准...")
    try:
        signal.alarm(10)
        
        # 定义sigmoid函数
        def sigmoid(x, a, b):
            return 1 / (1 + np.exp(-(a*x + b)))
        
        # 定义损失函数
        def sigmoid_loss(params, x, y, weights=None):
            a, b = params
            pred = sigmoid(x, a, b)
            # 使用交叉熵损失
            if weights is None:
                weights = np.ones_like(y)
            loss = -np.sum(weights * (y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10)))
            return loss
        
        # 使用优化算法找到最佳参数
        initial_guess = [1.0, 0.0]  # 初始猜测
        result = minimize(sigmoid_loss, initial_guess, args=(y_prob, y_true, weights), method='Nelder-Mead')
        a_opt, b_opt = result.x
        
        # 计算校准后的概率
        sigmoid_probs = sigmoid(y_prob, a_opt, b_opt)
        
        calibration_methods['sigmoid'] = {
            'prob': sigmoid_probs,
            'brier': brier_score_loss(y_true, sigmoid_probs, sample_weight=weights)
        }
        signal.alarm(0)
    except Exception as e:
        print(f"Sigmoid拟合校准失败: {e}")
        
    # 7. 集成校准 (结合多种方法)
    method_index += 1
    print(f"[{method_index}/{total_methods}] 集成校准...")
    try:
        signal.alarm(10)
        # 将所有校准方法的结果进行简单平均作为集成结果
        if len(calibration_methods) >= 2:
            ensemble_probs = np.mean([m['prob'] for m in calibration_methods.values()], axis=0)
            
            calibration_methods['ensemble'] = {
                'prob': ensemble_probs,
                'brier': brier_score_loss(y_true, ensemble_probs, sample_weight=weights)
            }
        signal.alarm(0)
    except Exception as e:
        print(f"集成校准失败: {e}")
        
    print(f"所有校准方法计算完成，开始绘制图表...")
    
    # 创建正方形图形，并启用高分辨率
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    # 确保图表区域为正方形
    ax.set_aspect('equal')
    
    # 设置简约风格的网格线和背景
    ax.grid(True, linestyle='-', alpha=0.2, color='#dddddd')
    ax.set_axisbelow(True)  # 网格线在数据点之下
    
    # 启用双对数坐标轴
    use_transform = True
    transform_type = 'loglog'  # 使用双对数坐标，使完美校准曲线显示为直线
    
    # 绘制适合变换后的理想直线
    if use_transform:
        if transform_type == 'loglog':
            # 设置双对数坐标
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # 生成参考线的点
            x_ref = np.logspace(-3, 0, 100)  # 从0.001到1.0的对数空间
            y_ref = x_ref.copy()  # 在双对数图中，完美校准是直线 y=x
            
            # 绘制直线
            ax.plot(x_ref, y_ref, 'k:', label='Perfectly calibrated', alpha=0.7)
            
            # 设置对数坐标的标签和格式
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            # 为对数轴添加简约风格的网格线
            ax.grid(True, which='both', linestyle='-', alpha=0.2, color='#dddddd')
            ax.set_axisbelow(True)  # 网格线在数据点之下
            
            # 设置刻度格式
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)
        
        elif transform_type == 'log':
            # 生成参考线的点
            x_ref = np.linspace(0.001, 1.0, 100)
            y_ref = x_ref.copy()  # 理想直线 y=x
            
            # 对数变换: log(1+9x) 放大小概率区域
            x_ref_trans = np.log1p(x_ref * 9) / np.log(10)
            
            # 绘制理想直线
            ax.plot(x_ref_trans, y_ref, 'k:', label='Perfectly calibrated', alpha=0.7)
            
            # 设置分析刻度
            tick_positions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            tick_labels = [f'{x:.2f}' if x < 1 else '1.0' for x in tick_positions]
            tick_trans = np.log1p(np.array(tick_positions) * 9) / np.log(10)
            
            ax.set_xticks(tick_trans)
            ax.set_xticklabels(tick_labels)
    else:
        # 不使用变换，直接绘制普通直线
        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated', alpha=0.7)
    
    # 定义颜色映射和标记样式
    colors = {
        'platt': '#1f77b4',       # 蓝色
        'isotonic': '#2ca02c',    # 绿色
        'adaptive': '#ff7f0e',    # 橙色
        'histogram': '#2ca02c',   # 绿色
        'bbq': '#e377c2',         # 粉色
        'sigmoid': '#ffff33',     # 黄色
        'ensemble': '#17becf'     # 青色
    }
    
    # 方法名称映射
    method_labels = {
        'platt': 'Platt Scaling',
        'isotonic': 'Isotonic Regression',
        'adaptive': 'Adaptive Binning',
        'histogram': 'Histogram Binning',
        'bbq': 'Bayesian Binning',
        'sigmoid': 'Sigmoid Fitting',
        'ensemble': 'Ensemble Calibration'
    }
    
    print(f"开始绘制校准曲线...")
    
    # 创建校准曲线
    save_data = {}
    
    # 按Brier分数排序方法
    sorted_methods = sorted([(name, data) for name, data in calibration_methods.items()],
                           key=lambda x: x[1]['brier'])
    
    # 定义更小的marker size提高清晰度
    marker_size = 3
    
    # 绘制每个校准方法的曲线
    for method_name, method_data in sorted_methods:
        try:
            signal.alarm(3)  # 设置3秒超时
            
            y_cal = method_data['prob']
            brier = method_data['brier']
            
            # 使用更稳定的分箱方法，考虑样本数量
            # 减少分箱数量，降低平滑度
            additional_bins = min(n_bins, 8)  # 使用更少的分箱
            
            # 首先使用sklearn的原生分箱方法
            prob_true, prob_pred = calibration_curve(
                y_true, y_cal, n_bins=additional_bins, strategy='quantile'
            )
            
            # 去除异常值
            valid_idx = ~np.isnan(prob_true) & ~np.isnan(prob_pred)
            if np.sum(valid_idx) >= 3:  # 至少需要3个有效点
                prob_true = prob_true[valid_idx]
                prob_pred = prob_pred[valid_idx]
                
                # 保存结果数据
                method_key = method_labels.get(method_name, method_name.capitalize())
                save_data[method_key] = {
                    'fraction_of_positives': prob_true.tolist(),
                    'mean_predicted_value': prob_pred.tolist(),
                    'brier_score': float(brier)
                }
                
                # 绘制校准曲线
                color = colors.get(method_name, 'gray')
                
                # 保留数据点的原始分布特征，不过度平滑
                if len(prob_pred) >= 3:
                    # 排序数据点，确保 x 值升序
                    sort_idx = np.argsort(prob_pred)
                    sorted_x = prob_pred[sort_idx]
                    sorted_y = prob_true[sort_idx]
                    
                    if use_transform and transform_type == 'loglog':
                        # 双对数坐标下处理数据点
                        valid_mask = (sorted_x > 0) & (sorted_y > 0)
                        if np.sum(valid_mask) >= 2:  # 至少需要2个有效点
                            sorted_x_valid = sorted_x[valid_mask]
                            sorted_y_valid = sorted_y[valid_mask]
                            
                            # 绘制简约风格的数据点
                            ax.scatter(sorted_x_valid, sorted_y_valid, s=marker_size*1.5, 
                                      alpha=0.8, color=color, edgecolor='none')
                            
                            # 绘制简约风格的折线
                            ax.plot(sorted_x_valid, sorted_y_valid, '-o', 
                                  linewidth=1.0, markersize=marker_size, 
                                  markerfacecolor=color, markeredgecolor='none',
                                  alpha=0.9, color=color,
                                  label=f'{method_labels.get(method_name, method_name.capitalize())} (Brier: {brier:.4f})')
                    
                    elif use_transform and transform_type == 'log':
                        # 对数坐标处理
                        sorted_x_min = np.maximum(sorted_x, 0.001)  # 避免负数和零
                        sorted_x_trans = np.log1p(sorted_x_min * 9) / np.log(10)
                        
                        ax.scatter(sorted_x_trans, sorted_y, s=marker_size*1.5, 
                                 alpha=0.8, color=color, edgecolor='none')
                        ax.plot(sorted_x_trans, sorted_y, '-o', 
                              linewidth=1.0, markersize=marker_size, 
                              markerfacecolor=color, markeredgecolor='none',
                              alpha=0.9, color=color,
                              label=f'{method_labels.get(method_name, method_name.capitalize())} (Brier: {brier:.4f})')
                    
                    else:
                        # 不使用坐标变换
                        ax.scatter(sorted_x, sorted_y, s=marker_size*1.5, 
                                 alpha=0.8, color=color, edgecolor='none')
                        ax.plot(sorted_x, sorted_y, '-o', 
                              linewidth=1.0, markersize=marker_size, 
                              markerfacecolor=color, markeredgecolor='none',
                              alpha=0.9, color=color,
                              label=f'{method_labels.get(method_name, method_name.capitalize())} (Brier: {brier:.4f})')
                
                else:
                    # 数据点太少，直接绘制
                    ax.plot(prob_pred, prob_true, '-o', markersize=marker_size, linewidth=1.0, color=color,
                           label=f'{method_labels.get(method_name, method_name.capitalize())} (Brier: {brier:.4f})')
            
            signal.alarm(0)  # 取消超时
        except Exception as e:
            print(f"处理方法 {method_name} 时出错: {e}")
            signal.alarm(0)  # 确保取消超时
    
    # 调整图形尺寸、标题和标签
    if use_transform:
        if transform_type == 'loglog':
            # 双对数坐标轴范围
            plt.xlim([0.001, 1.05])
            plt.ylim([0.001, 1.05])
            plt.title(f"Calibration Curve (Log-Log Scale) - {model_name}", fontsize=14, fontweight='normal')
        elif transform_type == 'log':
            # 单对数变换，范围需要计算
            plt.xlim([-0.01, np.log1p(1.02 * 9) / np.log(10)])
            plt.ylim([-0.01, 1.02])
            plt.title(f"Calibration Curve (Log Scale) - {model_name}", fontsize=14, fontweight='normal')
    else:
        plt.xlim([-0.01, 1.02])
        plt.ylim([-0.01, 1.02])
        plt.title(f"Calibration Curve - {model_name}", fontsize=14, fontweight='normal')
    
    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives", fontsize=12)
    
    # 添加简约风格的坐标轴变换注释
    if use_transform and transform_type == 'loglog':
        transform_note = "Log-Log scale: perfect calibration appears as a straight line"
        plt.figtext(0.5, 0.01, transform_note, ha="center", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.9, "pad":3, "edgecolor":"#cccccc", "boxstyle":"round,pad=0.3"})
    elif use_transform and transform_type == 'log':
        transform_note = "X-axis uses log$_{10}$(1+9x) transform to emphasize small probability region"
        plt.figtext(0.5, 0.01, transform_note, ha="center", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.9, "pad":3, "edgecolor":"#cccccc", "boxstyle":"round,pad=0.3"})
    
    # 设置简约风格的图例到图外右侧
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
              frameon=True, fancybox=False, shadow=False, edgecolor='#cccccc',
              fontsize=10)
    
    # 调整图表边距，确保正方形区域不变形
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # 左右下上边距比例
    
    # 获取原始数据集信息
    total_samples = len(y_true)
    positive_samples = np.sum(y_true)
    positive_rate = positive_samples / total_samples * 100
    avg_pred_prob = np.mean(y_prob)
    
    print(f"\n原始数据集统计信息:")
    print(f"- 总样本数: {total_samples}")
    print(f"- 正例数量: {int(positive_samples)} ({positive_rate:.2f}%)")
    print(f"- 平均预测概率: {avg_pred_prob:.4f}")
    
    # 如果启用SMOTE过采样，对数据进行过采样处理
    X = np.column_stack([y_prob]) # 使用预测概率作为特征
    y = y_true
    
    if use_smote:
        print("\n应用SMOTE过采样使正负样本比例平衡...")
        try:
            # 创建SMOTE过采样器，使用适当比例
            smote = SMOTE(random_state=42, sampling_strategy=0.3)  # 0.3表示少数类:多数类 = 0.3:1
            X_res, y_res = smote.fit_resample(X, y)
            
            # 提取过采样后的预测概率和真实标签
            y_prob_bal = X_res[:, 0]
            y_true_bal = y_res
            
            # 输出过采样后的数据集信息
            total_samples_bal = len(y_true_bal)
            positive_samples_bal = np.sum(y_true_bal)
            positive_rate_bal = positive_samples_bal / total_samples_bal * 100
            avg_pred_prob_bal = np.mean(y_prob_bal)
            
            print(f"\n过采样后数据集统计信息:")
            print(f"- 总样本数: {total_samples_bal}")
            print(f"- 正例数量: {int(positive_samples_bal)} ({positive_rate_bal:.2f}%)")
            print(f"- 负例数量: {total_samples_bal - int(positive_samples_bal)} ({100 - positive_rate_bal:.2f}%)")
            print(f"- 平均预测概率: {avg_pred_prob_bal:.4f}")
            
            # 使用过采样后的数据进行后续校准
            y_true = y_true_bal
            y_prob = y_prob_bal
            
            # 权重在过采样后都设为1
            weights = np.ones_like(y_true)
        except Exception as e:
            print(f"SMOTE过采样失败: {e}")
            print("使用原始不平衡数据集继续...")
    
    # 保存高分辨率图表
    os.makedirs(plot_dir, exist_ok=True)
    plt_path = str(Path(plot_dir) / f"{model_name}_All_Calibration.png")
    plt.savefig(plt_path, dpi=300, bbox_inches='tight', transparent=False)
    
    # 保存数据
    os.makedirs(plot_data_dir, exist_ok=True)
    save_plot_data(save_data, str(Path(plot_data_dir) / f"{model_name}_All_Calibration_data.json"))
    
    # 找出最佳方法
    if sorted_methods:
        best_method, best_data = sorted_methods[0]
        best_brier = best_data['brier']
        print(f"最佳校准方法: {method_labels.get(best_method, best_method.capitalize())} (Brier: {best_brier:.4f})")
    
    plt.close(fig)
    return calibration_methods
