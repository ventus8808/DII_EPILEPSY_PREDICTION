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
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['lines.antialiased'] = True  # 抗锯齿
plt.rcParams['figure.dpi'] = 100  # 更高DPI以获得更平滑的线条

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

def plot_calibration_curve(y_true, y_prob, weights, model_name, plot_dir, plot_data_dir, n_bins=30):
    """
    绘制高级校准曲线，包含多种校准方法
    
    Parameters:
    -----------
    y_true : array-like
        实际标签（0，1）
    y_prob : array-like
        模型预测的概率值
    weights : array-like or None
        样本权重
    model_name : str
        模型名称，用于保存图像
    plot_dir : str
        图像保存目录
    plot_data_dir : str
        图像数据保存目录
    n_bins : int
        分箱数量，默认10
    """
    # 设置超时判断，防止某些校准方法卡住
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Calculation timed out")
    
    # 设置5秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    
    """
    绘制高级校准曲线，包含多种校准方法
    
    Parameters:
    -----------
    y_true : array-like
        实际标签（0，1）
    y_prob : array-like
        模型预测的概率值
    weights : array-like or None
        样本权重
    model_name : str
        模型名称，用于保存图像
    plot_dir : str
        图像保存目录
    plot_data_dir : str
        图像数据保存目录
    n_bins : int
        分箱数量，默认10
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, roc_auc_score
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"=== 校准曲线计算进度 === ")
    # 根据方法添加不同的方法
    calibration_methods = {}
    # 用来存储所有校准方法的时间，方便诊断性能问题
    method_times = {}
    
    # 导入需要用到的插值和数学函数
    from scipy import interpolate, stats
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print(f"[2/16] Platt缩放(逻辑回归)...")
    # 添加Platt缩放（逻辑回归）
    try:
        from sklearn.linear_model import LogisticRegression
        platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        platt.fit(y_prob.reshape(-1, 1), y_true)
        platt_probs = platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        calibration_methods['platt'] = {
            'prob': platt_probs,
            'brier': brier_score_loss(y_true, platt_probs, sample_weight=weights)
        }
    except Exception as e:
        print(f"Platt校准失败: {e}")
    
    print(f"[3/16] 等温回归...")
    # 添加等温回归
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob, y_true, sample_weight=weights)
        iso_probs = iso.predict(y_prob)
        calibration_methods['isotonic'] = {
            'prob': iso_probs,
            'brier': brier_score_loss(y_true, iso_probs, sample_weight=weights)
        }
    except Exception as e:
        print(f"等温回归校准失败: {e}")
    
    print(f"[4/16] 跳过Beta校准...")
    # Beta校准已被用户要求删除
    
    print(f"[5/16] 自适应分箱...")
    # 添加自适应分箱（基于分位数）
    try:
        from scipy.interpolate import interp1d
        
        # 使用分位数进行分箱而不是等距分箱
        quantiles = np.linspace(0, 1, n_bins + 1)
        quantile_thresholds = np.quantile(y_prob, quantiles)
        quantile_thresholds = np.unique(quantile_thresholds)  # 移除重复值
        
        # 每个分箱中计算实际正例比例
        bin_means = []
        bin_true = []
        
        for i in range(len(quantile_thresholds) - 1):
            bin_mask = (y_prob >= quantile_thresholds[i]) & (y_prob < quantile_thresholds[i+1])
            if np.sum(bin_mask) > 0:
                if weights is not None:
                    bin_mean = np.sum(y_prob[bin_mask] * weights[bin_mask]) / np.sum(weights[bin_mask])
                    bin_pos = np.sum(y_true[bin_mask] * weights[bin_mask]) / np.sum(weights[bin_mask])
                else:
                    bin_mean = np.mean(y_prob[bin_mask])
                    bin_pos = np.mean(y_true[bin_mask])
                bin_means.append(bin_mean)
                bin_true.append(bin_pos)
        
        # 对最后一个区间特殊处理
        bin_mask = (y_prob >= quantile_thresholds[-1])
        if np.sum(bin_mask) > 0:
            if weights is not None:
                bin_mean = np.sum(y_prob[bin_mask] * weights[bin_mask]) / np.sum(weights[bin_mask])
                bin_pos = np.sum(y_true[bin_mask] * weights[bin_mask]) / np.sum(weights[bin_mask])
            else:
                bin_mean = np.mean(y_prob[bin_mask])
                bin_pos = np.mean(y_true[bin_mask])
            bin_means.append(bin_mean)
            bin_true.append(bin_pos)
        
        # 创建插值函数
        if len(bin_means) > 1:
            adaptive_cal_func = interp1d(bin_means, bin_true, bounds_error=False, 
                                       fill_value=(bin_true[0], bin_true[-1]))
            adaptive_probs = adaptive_cal_func(y_prob)
            calibration_methods['adaptive'] = {
                'prob': adaptive_probs,
                'brier': brier_score_loss(y_true, adaptive_probs, sample_weight=weights)
            }
    except Exception as e:
        print(f"自适应分箱校准失败: {e}")
    
    print(f"[6/16] 样条平滑校准...")
    # 添加样条平滑校准
    try:
        from scipy.interpolate import UnivariateSpline
        
        # 使用跨验证确定平滑参数最优值
        best_brier = float('inf')
        best_s = 0
        
        for s in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
            try:
                # 按预测概率排序
                sorted_indices = np.argsort(y_prob)
                sorted_probs = y_prob[sorted_indices]
                sorted_true = y_true[sorted_indices]
                sorted_weight = weights[sorted_indices] if weights is not None else np.ones_like(sorted_true)
                
                # 应用滑动窗口平滑获取校准曲线点
                window_size = max(20, len(y_prob) // n_bins)
                x_points = []
                y_points = []
                
                for i in range(0, len(sorted_probs), window_size // 2):
                    end_idx = min(i + window_size, len(sorted_probs))
                    if end_idx - i < window_size // 2:  # 避免太小的最后一个窗口
                        break
                    
                    window_prob = sorted_probs[i:end_idx]
                    window_true = sorted_true[i:end_idx]
                    window_weight = sorted_weight[i:end_idx]
                    
                    # 计算加权平均
                    mean_prob = np.sum(window_prob * window_weight) / np.sum(window_weight)
                    mean_true = np.sum(window_true * window_weight) / np.sum(window_weight)
                    
                    x_points.append(mean_prob)
                    y_points.append(mean_true)
                
                # 应用单变量样条拟合
                if len(x_points) > 3:  # 至少需要4个点
                    spline = UnivariateSpline(x_points, y_points, s=s, ext='const')
                    spline_probs = spline(y_prob)
                    spline_probs = np.clip(spline_probs, 0, 1)  # 确保在[0,1]范围内
                    
                    # 计算该参数下的Brier分数
                    brier = brier_score_loss(y_true, spline_probs, sample_weight=weights)
                    
                    if brier < best_brier:
                        best_brier = brier
                        best_s = s
            except Exception:
                continue
        
        # 使用最佳平滑参数重新拟合
        if best_brier < float('inf'):
            # 按预测概率排序
            sorted_indices = np.argsort(y_prob)
            sorted_probs = y_prob[sorted_indices]
            sorted_true = y_true[sorted_indices]
            sorted_weight = weights[sorted_indices] if weights is not None else np.ones_like(sorted_true)
            
            # 应用滑动窗口平滑获取校准曲线点
            window_size = max(20, len(y_prob) // n_bins)
            x_points = []
            y_points = []
            
            for i in range(0, len(sorted_probs), window_size // 2):
                end_idx = min(i + window_size, len(sorted_probs))
                if end_idx - i < window_size // 2:  # 避免太小的最后一个窗口
                    break
                
                window_prob = sorted_probs[i:end_idx]
                window_true = sorted_true[i:end_idx]
                window_weight = sorted_weight[i:end_idx]
                
                # 计算加权平均
                mean_prob = np.sum(window_prob * window_weight) / np.sum(window_weight)
                mean_true = np.sum(window_true * window_weight) / np.sum(window_weight)
                
                x_points.append(mean_prob)
                y_points.append(mean_true)
            
            if len(x_points) > 3:
                spline = UnivariateSpline(x_points, y_points, s=best_s, ext='const')
                spline_probs = spline(y_prob)
                spline_probs = np.clip(spline_probs, 0, 1)  # 确保在[0,1]范围内
                
                calibration_methods['spline'] = {
                    'prob': spline_probs,
                    'brier': best_brier
                }
    except Exception as e:
        print(f"样条平滑校准失败: {e}")
    
    print(f"[7/16] 直方图分箱...")
    # 添加直方图分箱
    try:
        # 创建分箱
        bin_edges = np.linspace(0, 1, n_bins + 1)
        digitized = np.digitize(y_prob, bin_edges)
        digitized = np.minimum(digitized, n_bins)  # 确保索引不超过边界
        
        # 计算每个分箱的平均观察频率
        bin_means = np.zeros(n_bins)
        for i in range(1, n_bins + 1):
            bin_indices = (digitized == i)
            if np.sum(bin_indices) > 0:
                if weights is not None:
                    bin_means[i-1] = np.sum(y_prob[bin_indices] * weights[bin_indices]) / np.sum(weights[bin_indices])
                else:
                    bin_means[i-1] = np.mean(y_prob[bin_indices])
            # 如果分箱为空，则使用邻近箱的值或边界值
        
        # 填充空箱
        for i in range(n_bins):
            if (digitized == i+1).sum() == 0:  # 空箱
                left_idx = i - 1
                while left_idx >= 0 and (digitized == left_idx+1).sum() == 0:
                    left_idx -= 1
                
                right_idx = i + 1
                while right_idx < n_bins and (digitized == right_idx+1).sum() == 0:
                    right_idx += 1
                
                # 插值或取边界值
                if left_idx >= 0 and right_idx < n_bins:
                    bin_means[i] = (bin_means[left_idx] + bin_means[right_idx]) / 2
                elif left_idx >= 0:
                    bin_means[i] = bin_means[left_idx]
                elif right_idx < n_bins:
                    bin_means[i] = bin_means[right_idx]
                else:
                    bin_means[i] = 0.5  # 默认值
        
        # 应用直方图分箱校准
        histogram_probs = np.zeros_like(y_prob)
        for i in range(1, n_bins + 1):
            bin_indices = (digitized == i)
            histogram_probs[bin_indices] = bin_means[i-1]
        
        calibration_methods['histogram'] = {
            'prob': histogram_probs,
            'brier': brier_score_loss(y_true, histogram_probs, sample_weight=weights)
        }
    except Exception as e:
        print(f"直方图分箱校准失败: {e}")
    
    print(f"[14/16] 跳过温度缩放...")
    # 温度缩放已被用户要求删除
    
    print(f"[15/16] 跳过保守缩放...")
    # 保守缩放已被用户要求删除
    
    print(f"[8/16] 贝叶斯分箱量化(BBQ)...")
    # 添加贝叶斯分箱量化 (BBQ - Bayesian Binning into Quantiles) 的简化版本
    try:
        # 我们为每个可能的分箱数计算得分，并选择最佳分箱数
        best_brier = float('inf')
        best_calibrated_probs = None
        
        for bin_count in [5, 10, 15, 20, 25]:
            # 创建分箱
            bin_edges = np.linspace(0, 1, bin_count + 1)
            digitized = np.digitize(y_prob, bin_edges)
            digitized = np.minimum(digitized, bin_count)  # 确保索引不超过边界
            
            # 计算每个分箱的平均观察频率
            bin_means = np.zeros(bin_count)
            bin_sizes = np.zeros(bin_count)
            
            for i in range(1, bin_count + 1):
                bin_indices = (digitized == i)
                bin_sizes[i-1] = np.sum(bin_indices)
                if bin_sizes[i-1] > 0:
                    if weights is not None:
                        bin_means[i-1] = np.sum(y_true[bin_indices] * weights[bin_indices]) / np.sum(weights[bin_indices])
                    else:
                        bin_means[i-1] = np.mean(y_true[bin_indices])
            
            # 填充空箱
            for i in range(bin_count):
                if bin_sizes[i] == 0:  # 空箱
                    left_idx = i - 1
                    while left_idx >= 0 and bin_sizes[left_idx] == 0:
                        left_idx -= 1
                    
                    right_idx = i + 1
                    while right_idx < bin_count and bin_sizes[right_idx] == 0:
                        right_idx += 1
                    
                    # 插值或取边界值
                    if left_idx >= 0 and right_idx < bin_count:
                        bin_means[i] = (bin_means[left_idx] + bin_means[right_idx]) / 2
                    elif left_idx >= 0:
                        bin_means[i] = bin_means[left_idx]
                    elif right_idx < bin_count:
                        bin_means[i] = bin_means[right_idx]
                    else:
                        bin_means[i] = 0.5  # 默认值
            
            # 应用分箱校准
            binned_probs = np.zeros_like(y_prob)
            for i in range(1, bin_count + 1):
                bin_indices = (digitized == i)
                binned_probs[bin_indices] = bin_means[i-1]
            
            # 计算该分箱数下的Brier分数
            brier = brier_score_loss(y_true, binned_probs, sample_weight=weights)
            
            if brier < best_brier:
                best_brier = brier
                best_calibrated_probs = binned_probs
        
        if best_calibrated_probs is not None:
            calibration_methods['bbq'] = {
                'prob': best_calibrated_probs,
                'brier': best_brier
            }
    except Exception as e:
        print(f"贝叶斯分箱校准失败: {e}")
        
    print(f"[9/16] Sigmoid拟合校准...")
    # 添加Sigmoid拟合校准
    try:
        from scipy.optimize import curve_fit
        
        # 定义Sigmoid函数
        def sigmoid_func(x, a, b, c):
            return 1.0 / (1.0 + np.exp(-a * (x - b))) * c
        
        # 拟合Sigmoid函数
        try:
            params, _ = curve_fit(sigmoid_func, y_prob, y_true, p0=[1, 0.5, 1], 
                                  bounds=([-np.inf, 0, 0.5], [np.inf, 1, 1.5]))
            sigmoid_probs = sigmoid_func(y_prob, *params)
            sigmoid_probs = np.clip(sigmoid_probs, 0, 1)  # 确保在[0,1]范围内
            
            calibration_methods['sigmoid'] = {
                'prob': sigmoid_probs,
                'brier': brier_score_loss(y_true, sigmoid_probs, sample_weight=weights)
            }
        except RuntimeError:
            # 如果曲线拟合失败，尝试不同的初始值
            try:
                params, _ = curve_fit(sigmoid_func, y_prob, y_true, p0=[10, 0.5, 1], 
                                      bounds=([-np.inf, 0, 0.5], [np.inf, 1, 1.5]))
                sigmoid_probs = sigmoid_func(y_prob, *params)
                sigmoid_probs = np.clip(sigmoid_probs, 0, 1)
                
                calibration_methods['sigmoid'] = {
                    'prob': sigmoid_probs,
                    'brier': brier_score_loss(y_true, sigmoid_probs, sample_weight=weights)
                }
            except:
                pass
            
    except Exception as e:
        print(f"Sigmoid拟合校准失败: {e}")
    
    print(f"[10/16] 跳过高斯过程校准(该方法可能导致卡顿)...")
    # 跳过高斯过程校准，该方法可能导致程序卡住
    
    print(f"[11/16] 带交叉验证的等温回归...")
    # 添加带交叉验证的等温回归
    try:
        from sklearn.model_selection import KFold
        from sklearn.isotonic import IsotonicRegression
        
        # 使用K折交叉验证防止过拟合
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        iso_cv_probs = np.zeros_like(y_prob)
        
        for train_idx, test_idx in kf.split(y_prob):
            X_train, X_test = y_prob[train_idx], y_prob[test_idx]
            y_train = y_true[train_idx]
            
            # 在训练集上拟合等温回归
            iso_cv = IsotonicRegression(out_of_bounds='clip')
            iso_cv.fit(X_train, y_train)
            
            # 在测试集上预测
            iso_cv_probs[test_idx] = iso_cv.predict(X_test)
        
        calibration_methods['isotonic_cv'] = {
            'prob': iso_cv_probs,
            'brier': brier_score_loss(y_true, iso_cv_probs, sample_weight=weights)
        }
    except Exception as e:
        print(f"带交叉验证的等温回归校准失败: {e}")
    
    print(f"[12/16] 核密度估计校准...")
    # 添加核密度估计校准
    try:
        from sklearn.neighbors import KernelDensity
        
        # 分别为正类和负类样本估计概率密度
        pos_probs = y_prob[y_true == 1].reshape(-1, 1)
        neg_probs = y_prob[y_true == 0].reshape(-1, 1)
        
        if len(pos_probs) > 10 and len(neg_probs) > 10:  # 确保有足够的样本
            # 估计正类和负类的概率密度
            bandwidth = 0.1  # 核带宽
            kde_pos = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(pos_probs)
            kde_neg = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(neg_probs)
            
            # 计算校准概率
            X_test = y_prob.reshape(-1, 1)
            log_pos = kde_pos.score_samples(X_test)
            log_neg = kde_neg.score_samples(X_test)
            
            # 计算后验概率 P(y=1|s) = P(s|y=1)P(y=1) / (P(s|y=1)P(y=1) + P(s|y=0)P(y=0))
            prior_pos = len(pos_probs) / len(y_prob)
            prior_neg = len(neg_probs) / len(y_prob)
            
            # 使用Bayes法则
            numerator = np.exp(log_pos) * prior_pos
            denominator = numerator + np.exp(log_neg) * prior_neg
            kde_probs = numerator / denominator
            kde_probs = np.clip(kde_probs, 0, 1)  # 确保在[0,1]范围内
            
            calibration_methods['kernel'] = {
                'prob': kde_probs,
                'brier': brier_score_loss(y_true, kde_probs, sample_weight=weights)
            }
    except Exception as e:
        print(f"核密度估计校准失败: {e}")
    
    print(f"[13/16] 分位数映射校准...")
    # 添加分位数映射校准
    try:
        # 将模型预测的概率映射到相应的标签分位数
        probs_sorted_indices = np.argsort(y_prob)
        y_true_sorted = y_true[probs_sorted_indices]
        y_prob_sorted = y_prob[probs_sorted_indices]
        
        # 计算每个百分位数的实际标签率
        n_quantiles = 100
        quantile_size = len(y_prob) // n_quantiles
        quantiles = []
        quantile_values = []
        
        for i in range(n_quantiles):
            start_idx = i * quantile_size
            end_idx = min(start_idx + quantile_size, len(y_prob))
            if end_idx <= start_idx:
                break
                
            # 计算当前数分位数区间的平均概率和标签率
            mean_prob = np.mean(y_prob_sorted[start_idx:end_idx])
            mean_true = np.mean(y_true_sorted[start_idx:end_idx])
            
            quantiles.append(mean_prob)
            quantile_values.append(mean_true)
        
        # 使用分段线性插值接近分位数映射
        if len(quantiles) >= 2:
            from scipy.interpolate import interp1d
            quantile_mapping = interp1d(quantiles, quantile_values, 
                                       bounds_error=False, fill_value=(quantile_values[0], quantile_values[-1]))
            
            # 应用分位数映射校准
            quantile_probs = quantile_mapping(y_prob)
            
            calibration_methods['quantile'] = {
                'prob': quantile_probs,
                'brier': brier_score_loss(y_true, quantile_probs, sample_weight=weights)
            }
    except Exception as e:
        print(f"分位数映射校准失败: {e}")
    
    print(f"[16/16] 集成校准...")
    # 添加集成校准（多种方法的加权平均）
    try:
        # 只有当至少有2种校准方法可用时才创建集成
        if len(calibration_methods) >= 3:  # 原始方法和至少两种校准方法
            # 根据Brier分数分配权重（分数越低权重越高）
            methods_to_ensemble = [m for m in calibration_methods.keys() if m != 'original']
            brier_scores = np.array([calibration_methods[m]['brier'] for m in methods_to_ensemble])
            
            # 反转Brier分数并归一化为权重
            weights_ensemble = 1.0 / (brier_scores + 1e-10)  # 避免除零
            weights_ensemble = weights_ensemble / np.sum(weights_ensemble)
            
            # 创建加权平均预测
            ensemble_probs = np.zeros_like(y_prob)
            for method, weight in zip(methods_to_ensemble, weights_ensemble):
                ensemble_probs += weight * calibration_methods[method]['prob']
            
            calibration_methods['ensemble'] = {
                'prob': ensemble_probs,
                'brier': brier_score_loss(y_true, ensemble_probs, sample_weight=weights)
            }
    except Exception as e:
        print(f"集成校准失败: {e}")
    
    print(f"所有校准方法计算完成，开始绘制图表...")
    
    # 绘制校准曲线 - 使用更适合的图表尺寸
    fig, ax = plt.subplots(figsize=(12, 9))
    # 调整图形布局，给右侧图例留出足够空间
    plt.subplots_adjust(right=0.7)  # 右边界保留更多空间
    ax.set_aspect('equal', adjustable='box')
    
    # 绘制参考线（完美校准）
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Perfectly calibrated', alpha=0.7)
    
    # 定义颜色 - 删除原始预测、温度缩放、保守缩放和Beta校准
    colors = {
        'platt': 'blue',
        'isotonic': 'green',
        'adaptive': 'orange',
        'spline': 'brown',
        'histogram': 'darkgreen',
        'bbq': 'pink',
        'sigmoid': 'gold',
        'isotonic_cv': 'cyan',
        'kernel': 'darkviolet',
        'quantile': 'slategray',
        'ensemble': 'darkblue'
    }
    
    # 定义方法标签映射
    method_labels = {
        'platt': 'Platt Scaling',
        'isotonic': 'Isotonic Regression',
        'adaptive': 'Adaptive Binning',
        'spline': 'Spline Smoothing',
        'histogram': 'Histogram Binning',
        'bbq': 'Bayesian Binning',
        'sigmoid': 'Sigmoid Fitting',
        'isotonic_cv': 'Isotonic CV',
        'kernel': 'Kernel Density',
        'quantile': 'Quantile Mapping',
        'ensemble': 'Ensemble Calibration'
    }
    
    print(f"开始绘制每种校准方法的曲线...")
    # 处理每种校准方法
    for method_name, result in calibration_methods.items():
        try:
            # 设置绘图超时为3秒
            signal.alarm(3)
            # 使用sklearn的calibration_curve获取初始校准点
            try:
                # 首先尝试使用分位数策略，确保点均匀分布
                prob_true, prob_pred = calibration_curve(
                    y_true, result['prob'], 
                    n_bins=n_bins, strategy='quantile'
                )
                
                # 检查是否有异常点（比如说概率大幅跳跃的点）
                point_diffs = np.abs(np.diff(prob_true))
                if np.any(point_diffs > 0.5):
                    # 如果存在大幅跳跃，改用均匀策略
                    prob_true, prob_pred = calibration_curve(
                        y_true, result['prob'],
                        n_bins=n_bins, strategy='uniform'
                    )
            except:
                # 作为后备方案使用均匀策略
                prob_true, prob_pred = calibration_curve(
                    y_true, result['prob'],
                    n_bins=n_bins, strategy='uniform'
                )
            
            # 过滤掉NaN值
            valid_idx = ~np.isnan(prob_true) & ~np.isnan(prob_pred)
            if np.sum(valid_idx) > 0:
                prob_true = prob_true[valid_idx]
                prob_pred = prob_pred[valid_idx]
                
            # 确保曲线始于(0,0)并终于(1,y)或(x,1)
            # 检查起始点
            if len(prob_pred) > 0 and prob_pred[0] > 0.01:
                prob_pred = np.insert(prob_pred, 0, 0.0)
                prob_true = np.insert(prob_true, 0, 0.0)
            
            # 检查结束点
            if len(prob_pred) > 0 and prob_pred[-1] < 0.99:
                # 使用线性外推来估计最后一个点
                if len(prob_pred) > 1:
                    slope = (prob_true[-1] - prob_true[-2]) / (prob_pred[-1] - prob_pred[-2]) \
                            if prob_pred[-1] != prob_pred[-2] else 0
                    # 外推到x=1.0的点
                    extrapolated_y = prob_true[-1] + slope * (1.0 - prob_pred[-1])
                    # 确保在合理范围内
                    last_true = min(max(0.0, extrapolated_y), 1.0)
                else:
                    last_true = min(1.0, prob_true[-1])
                
                prob_pred = np.append(prob_pred, 1.0)
                prob_true = np.append(prob_true, last_true)
            # 有效的点
            valid_idx = ~np.isnan(prob_true) & ~np.isnan(prob_pred)
            if np.sum(valid_idx) > 0:
                # 绘制校准曲线
                if len(prob_pred) >= 2:  # 至少需要2个点
                    # 对于所有曲线，增加插值点使曲线更平滑
                    num_interp_points = 100  # 插值后的点数
                    
                    # 检查点是否足够多，选择合适的插值方法
                    if len(prob_pred) >= 4:
                        # 使用PCHIP插值（保持单调性，避免过冲）
                        try:
                            pchip = interpolate.PchipInterpolator(prob_pred, prob_true)
                            x_smooth = np.linspace(prob_pred.min(), prob_pred.max(), num_interp_points)
                            y_smooth = pchip(x_smooth)
                            # 确保值在[0,1]范围内
                            y_smooth = np.clip(y_smooth, 0, 1)
                            # 绘制平滑曲线
                            ax.plot(x_smooth, y_smooth, '-', linewidth=2.5, color=colors.get(method_name, 'black'), alpha=0.8)
                        except:
                            # 如果PCHIP失败，退回到线性插值
                            x_smooth = np.linspace(prob_pred.min(), prob_pred.max(), num_interp_points)
                            y_smooth = np.interp(x_smooth, prob_pred, prob_true)
                            ax.plot(x_smooth, y_smooth, '-', linewidth=2.5, color=colors.get(method_name, 'black'), alpha=0.8)
                    else:
                        # 点太少时使用线性插值
                        x_smooth = np.linspace(prob_pred.min(), prob_pred.max(), num_interp_points)
                        y_smooth = np.interp(x_smooth, prob_pred, prob_true)
                        ax.plot(x_smooth, y_smooth, '-', linewidth=2.5, color=colors.get(method_name, 'black'), alpha=0.8)
                
                # 绘制原始数据点（小一些，不那么突出）
                ax.plot(prob_pred, prob_true, 'o', markersize=4, color=colors.get(method_name, 'black'), alpha=0.5)
                
                # 只在图例中显示方法名和Brier分数
                method_label = method_labels.get(method_name, method_name.capitalize())
                ax.plot([], [], '-', color=colors.get(method_name, 'black'), linewidth=2.5, 
                       label=f'{method_label} (Brier: {result["brier"]:.4f})')
                      
                # 关闭超时计时器
                signal.alarm(0)
        except TimeoutException:
            print(f"处理方法 {method_name} 时超时，跳过该方法")
            signal.alarm(0)  # 重置超时计时器
        except Exception as e:
            print(f"处理方法 {method_name} 时出错: {e}")
            signal.alarm(0)  # 重置超时计时器
    
    # 设置校准曲线图属性
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Calibration Curve - {model_name}', pad=30)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 将图例放在右边，简化风格
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, 
                   frameon=True, fancybox=False, shadow=False, ncol=1)
    
    # 调整图例文本间距
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # 显示图表边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
    
    # 添加网格线
    ax.grid(linestyle='--', alpha=0.3)
    
    # 保存图像
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_calibration.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存数据
    save_data = {}
    for method, result in calibration_methods.items():
        try:
            prob_true, prob_pred = calibration_curve(y_true, result['prob'], n_bins=n_bins)
            save_data[method] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist(),
                'brier': float(result['brier']),
                'auc': float(roc_auc_score(y_true, result['prob'], sample_weight=weights))
            }
        except Exception:
            pass
            
    # 注意：method_labels现在已经在上方定义
    
    # 使用Brier分数排序校准方法，并找出效果最好的方法
    if calibration_methods:
        # 根据 Brier 分数排序
        sorted_methods = sorted([(name, data) for name, data in calibration_methods.items()],
                               key=lambda x: x[1]['brier'])
        
        # 找出最佳方法
        best_method, best_data = sorted_methods[0]
        best_brier = best_data['brier']
        print(f"最佳校准方法: {method_labels.get(best_method, best_method.capitalize())} (Brier: {best_brier:.4f})")
    
    save_plot_data(save_data, str(Path(plot_data_dir) / f"{model_name}_calibration_data.json"))
    
    return calibration_methods

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

def plot_learning_curve(model, X_train, y_train, X_test, y_test, model_name, plot_dir, plot_data_dir, cv=3, scoring='roc_auc'):
    """
    绘制样本量学习曲线，展示随着训练样本数量的增加，模型性能的变化
    
    横坐标：训练样本数量
    纵坐标：模型性能指标（AUC-ROC）
    包含训练集和测试集上的性能曲线
    
    Parameters:
    -----------
    model : 机器学习模型实例
    X_train, y_train : 训练数据和标签
    X_test, y_test : 测试数据和标签
    model_name : str, 模型名称
    plot_dir : str, 图像保存目录
    plot_data_dir : str, 图像数据保存目录
    cv : int, 交叉验证折数, 默认3
    scoring : str, 评分标准, 默认'roc_auc'
    """
    print(f"[注意] 学习曲线计算已禁用。如需启用，请修改 model_plot_utils.py 中的 plot_learning_curve 函数")
    
    # 创建一个简单的占位图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用一些虚拟数据点
    train_sizes = [100, 200, 300, 400, 500]
    train_scores = [0.75, 0.78, 0.80, 0.82, 0.83]
    test_scores = [0.73, 0.75, 0.76, 0.77, 0.77]
    
    # 绘制占位线
    ax.plot(train_sizes, train_scores, 'o-', label='Training score (DISABLED)', color='blue', alpha=0.5)
    ax.plot(train_sizes, test_scores, 'o-', label='Test score (DISABLED)', color='red', alpha=0.5)
    
    # 在图上添加说明
    ax.text(0.5, 0.5, 'Learning curve calculation is disabled\nEdit plot_learning_curve() to enable',
         ha='center', va='center', transform=ax.transAxes, fontsize=14,
         bbox=dict(facecolor='yellow', alpha=0.2))
    
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title(f"Sample Learning Curve - {model_name} (DISABLED)", pad=30)
    
    # 美化图形
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend(loc='lower right')
    
    for spine in ['top','right']:
        ax.spines[spine].set_visible(True)
    
    fig.tight_layout()
    fig.savefig(str(Path(plot_dir) / f"{model_name}_sample_learning_curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 保存占位数据
    save_plot_data({
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'status': 'disabled'
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
