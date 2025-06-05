import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import yaml
import argparse
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec

# 设置高级风格函数
def set_premium_style():
    # 使用简洁风格作为基础
    plt.style.use('default')
    
    # 单色系深蓝色调色板 - 优雅专业
    primary_color = '#1f77b4'  # 主色：经典蓝色
    accent_color = '#ff7f0e'   # 强调色：橙色
    light_blue = '#c9daf8'     # 淡蓝色：用于直方图填充
    
    # 单色系颜色组
    colors = [primary_color, '#4682b4', '#0072b2', '#009ade', accent_color]
    sns.set_palette(colors)
    
    # 设置全局字体和样式参数
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 18,
        'font.weight': 'normal',
        'axes.unicode_minus': True,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'medium',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.color': '#cccccc',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'legend.fontsize': 18,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#333333',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })
    
    return colors, light_blue

def create_combined_plot(var_data, display_name, output_dir, light_blue, primary_color, accent_color):
    """创建合并图表：直方图在顶部，PP图和QQ图并排在底部"""
    
    # 创建一个大型图表和网格布局
    fig = plt.figure(figsize=(20, 18), dpi=150)
    
    # 创建网格：2行2列，直方图占据整个第一行
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])
    
    # 创建三个子图区域
    ax_hist = fig.add_subplot(gs[0, :])  # 直方图跨越第一行的两列
    ax_qq = fig.add_subplot(gs[1, 0])    # QQ图在左下
    ax_pp = fig.add_subplot(gs[1, 1])    # PP图在右下
    
    # ==================== 绘制频数分布直方图 ====================
    plt.sca(ax_hist)  # 设置当前活动轴为直方图区域
    
    # 绘制直方图
    plt.hist(var_data, bins=30, alpha=0.7, color=light_blue, 
            edgecolor='white', linewidth=1.0, density=True)
    
    # 绘制密度曲线
    sns.kdeplot(var_data, color=accent_color, linewidth=3, 
                label='Density', bw_adjust=0.5, ax=ax_hist)
    
    # 添加参考线
    mean_line = ax_hist.axvline(var_data.mean(), color='red', linestyle='--', 
                linewidth=2.5, label=f'Mean: {var_data.mean():.2f}')
    median_line = ax_hist.axvline(var_data.median(), color='green', linestyle='-.', 
                linewidth=2.5, label=f'Median: {var_data.median():.2f}')
    zero_line = ax_hist.axvline(0, color='darkblue', linestyle=':', 
                linewidth=2.5, label='Zero')
    
    # 基本统计量
    median = var_data.median()
    skewness = var_data.skew()
    kurtosis = var_data.kurtosis()
    
    # 创建统计信息文本框
    stats_text = (
        f"Count: {len(var_data)}\n"
        f"Mean: {var_data.mean():.4f}\n"
        f"Std Dev: {var_data.std():.4f}\n"
        f"Min: {var_data.min():.4f}\n"
        f"25%: {var_data.quantile(0.25):.4f}\n"
        f"Median: {median:.4f}\n"
        f"75%: {var_data.quantile(0.75):.4f}\n"
        f"Max: {var_data.max():.4f}\n"
        f"Skewness: {skewness:.4f}\n"
        f"Kurtosis: {kurtosis:.4f}"
    )
    
    ax_hist.annotate(stats_text, xy=(0.02, 0.96), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                          edgecolor='#333333', linewidth=1.5, alpha=0.95),
                 va='top', ha='left', fontsize=16)
    
    # 设置标题和标签
    ax_hist.set_title(f'Distribution of {display_name}', pad=20)
    ax_hist.set_xlabel(display_name)
    ax_hist.set_ylabel('Density')
    
    # 创建图例
    ax_hist.legend(fancybox=True, loc='upper right')
    
    # 调整边框
    for spine in ax_hist.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # ==================== 绘制QQ图 ====================
    plt.sca(ax_qq)  # 设置当前活动轴为QQ图区域
    
    # 创建QQ图
    res = stats.probplot(var_data, dist='norm', plot=None)
    theoretical_quantiles = res[0][0]
    sample_quantiles = res[0][1]
    
    # 创建散点图
    ax_qq.scatter(theoretical_quantiles, sample_quantiles, 
              color=primary_color, s=100, alpha=0.8, 
              edgecolor='black', linewidth=1.0)
    
    # 添加参考线
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    ax_qq.plot([min_val, max_val], [min_val, max_val], 
              color=accent_color, linewidth=3, linestyle='-')
    
    # 设置标题和标签
    ax_qq.set_title(f'Q-Q Plot of {display_name}', pad=20)
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')
    
    # 添加网格线
    ax_qq.grid(True, alpha=0.2, linestyle='--')
    
    # 确保坐标轴标签显示负号
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax_qq.xaxis.set_major_formatter(formatter)
    ax_qq.yaxis.set_major_formatter(formatter)
    
    # 调整边框
    for spine in ax_qq.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # ==================== 绘制PP图 ====================
    plt.sca(ax_pp)  # 设置当前活动轴为PP图区域
    
    # 计算经验累积分布函数
    ecdf = np.arange(1, len(var_data) + 1) / len(var_data)
    # 排序数据
    x = np.sort(var_data)
    # 计算理论累积分布函数
    mean, std = var_data.mean(), var_data.std()
    tcdf = stats.norm.cdf(x, mean, std)
    
    # 创建散点图
    ax_pp.scatter(tcdf, ecdf, color=primary_color, s=100, alpha=0.8,
              edgecolor='black', linewidth=1.0)
    
    # 添加参考线
    ax_pp.plot([0, 1], [0, 1], color=accent_color, 
                        linestyle='-', linewidth=3)
    
    # 设置标题和标签
    ax_pp.set_title(f'P-P Plot of {display_name}', pad=20)
    ax_pp.set_xlabel('Theoretical Cumulative Probability')
    ax_pp.set_ylabel('Empirical Cumulative Probability')
    
    # 添加网格线
    ax_pp.grid(True, alpha=0.2, linestyle='--')
    
    # 确保坐标轴标签显示负号
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax_pp.xaxis.set_major_formatter(formatter)
    ax_pp.yaxis.set_major_formatter(formatter)
    
    # 调整边框
    for spine in ax_pp.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # 调整整体布局
    plt.tight_layout()
    
    # 保存合并图表
    combined_path = output_dir / f"{display_name}_combined.png"
    plt.savefig(combined_path, dpi=300, facecolor='white')
    print(f"Saved combined plot to: {combined_path}")
    
    # 关闭图表
    plt.close(fig)

def main():
    # 设置高级风格
    colors, light_blue = set_premium_style()
    primary_color = colors[0]
    accent_color = colors[4]
    
    # 命令行参数处理
    parser = argparse.ArgumentParser(description="Analysis of variable distribution")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--column", default="DII_food", help="Name of column to analyze")
    parser.add_argument("--display_name", default="DII", help="Display name for the variable")
    args = parser.parse_args()

    # 读取配置文件
    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 读取数据路径
    data_path = Path(config['data_path'])
    
    # 创建输出目录
    output_dir = Path("plot_general")
    output_dir.mkdir(exist_ok=True)
    
    # 读取数据
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    
    # 获取变量
    column_name = args.column
    display_name = args.display_name  # 用于显示的变量名
    
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in dataset. Available columns: {df.columns.tolist()}")
        return
    
    # 提取数据，排除缺失值
    var_data = df[column_name].dropna()
    
    print(f"\n===== {display_name} Distribution Analysis =====")
    
    # 基本统计量
    stats_data = var_data.describe()
    median = var_data.median()
    mode = var_data.mode().iloc[0]
    skewness = var_data.skew()
    kurtosis = var_data.kurtosis()  # Fisher's definition (normal = 0)
    
    # 打印整洁的统计信息
    print("\nBasic Statistics:")
    print(f"Count:      {len(var_data)}")
    print(f"Mean:       {var_data.mean():.4f}")
    print(f"Std Dev:    {var_data.std():.4f}")
    print(f"Min:        {var_data.min():.4f}")
    print(f"25%:        {var_data.quantile(0.25):.4f}")
    print(f"Median:     {median:.4f}")
    print(f"75%:        {var_data.quantile(0.75):.4f}")
    print(f"Max:        {var_data.max():.4f}")
    print(f"Mode:       {mode:.4f}")
    print(f"Skewness:   {skewness:.4f}")
    print(f"Kurtosis:   {kurtosis:.4f}")
    
    # 正态性检验
    shapiro_test = stats.shapiro(var_data)
    ks_test = stats.kstest(var_data, 'norm', args=(var_data.mean(), var_data.std()))
    
    print("\nNormality Tests:")
    print(f"Shapiro-Wilk test:          W={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.8f}")
    print(f"Kolmogorov-Smirnov test:    D={ks_test[0]:.4f}, p-value={ks_test[1]:.8f}")
    
    # ==================== 绘制频数分布直方图 ====================
    plt.figure(figsize=(20, 10), dpi=150)
    ax = plt.gca()
    
    # 使用淡蓝色填充直方图
    plt.hist(var_data, bins=30, alpha=0.7, color=light_blue, 
            edgecolor='white', linewidth=1.0, density=True)
    
    # 单独绘制密度曲线，确保可见性
    kde = sns.kdeplot(var_data, color=accent_color, linewidth=3, 
                    label='Density', bw_adjust=0.5)
    
    # 添加参考线
    mean_line = plt.axvline(var_data.mean(), color='red', linestyle='--', 
                linewidth=2.5, label=f'Mean: {var_data.mean():.2f}')
    median_line = plt.axvline(median, color='green', linestyle='-.', 
                linewidth=2.5, label=f'Median: {median:.2f}')
    zero_line = plt.axvline(0, color='darkblue', linestyle=':', 
                linewidth=2.5, label='Zero')
    
    # 创建精美的统计信息文本框 - 向下移动位置
    stats_text = (
        f"Count: {len(var_data)}\n"
        f"Mean: {var_data.mean():.4f}\n"
        f"Std Dev: {var_data.std():.4f}\n"
        f"Min: {var_data.min():.4f}\n"
        f"25%: {var_data.quantile(0.25):.4f}\n"
        f"Median: {median:.4f}\n"
        f"75%: {var_data.quantile(0.75):.4f}\n"
        f"Max: {var_data.max():.4f}\n"
        f"Skewness: {skewness:.4f}\n"
        f"Kurtosis: {kurtosis:.4f}"
    )
    
    text_box = plt.annotate(stats_text, xy=(0.02, 0.96), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                          edgecolor='#333333', linewidth=1.5, alpha=0.95),
                 va='top', ha='left', fontsize=18)
    
    # 设置标题和标签 - 调整标题位置
    plt.title(f'Distribution of {display_name}', pad=20)  # 增加pad值使标题向上移动
    plt.xlabel(display_name)
    plt.ylabel('Density')
    
    # 创建精美的图例
    leg = plt.legend(fancybox=True, loc='upper right')
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor('#333333')
    
    # 调整轴和边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    plt.tight_layout()
    hist_path = output_dir / f"{display_name}_histogram.png"
    plt.savefig(hist_path, dpi=300, facecolor='white')
    print(f"\nSaved histogram to: {hist_path}")
    
    # ==================== 绘制QQ图 ====================
    plt.figure(figsize=(14, 14), dpi=150)
    ax = plt.gca()
    
    # 手动创建QQ图以更好地控制样式
    res = stats.probplot(var_data, dist='norm', plot=None)
    theoretical_quantiles = res[0][0]
    sample_quantiles = res[0][1]
    
    # 创建带深色边框的散点图，确保在白色背景上清晰可见
    plt.scatter(theoretical_quantiles, sample_quantiles, 
              color=primary_color, s=120, alpha=0.8, 
              edgecolor='black', linewidth=1.0)
    
    # 添加参考线
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    ref_line = plt.plot([min_val, max_val], [min_val, max_val], 
              color=accent_color, linewidth=3, linestyle='-', 
              label='Reference Line')
    
    # 设置标题和标签 - 调整标题位置
    plt.title(f'Q-Q Plot of {display_name}', pad=20)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    
    # 添加网格线但降低其显著性
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # 确保坐标轴标签显示负号
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # 调整轴和边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    plt.tight_layout()
    qq_path = output_dir / f"{display_name}_qqplot.png"
    plt.savefig(qq_path, dpi=300, facecolor='white')
    print(f"Saved QQ plot to: {qq_path}")
    
    # ==================== 绘制PP图 ====================
    plt.figure(figsize=(14, 14), dpi=150)
    ax = plt.gca()
    
    # 计算经验累积分布函数
    ecdf = np.arange(1, len(var_data) + 1) / len(var_data)
    # 排序数据
    x = np.sort(var_data)
    # 计算理论累积分布函数
    mean, std = var_data.mean(), var_data.std()
    tcdf = stats.norm.cdf(x, mean, std)
    
    # 创建带深色边框的散点图，确保在白色背景上清晰可见
    plt.scatter(tcdf, ecdf, color=primary_color, s=120, alpha=0.8,
              edgecolor='black', linewidth=1.0)
    
    # 添加参考线
    ref_line = plt.plot([0, 1], [0, 1], color=accent_color, 
                        linestyle='-', linewidth=3, label='Reference Line')
    
    # 设置标题和标签 - 调整标题位置
    plt.title(f'P-P Plot of {display_name}', pad=20)
    plt.xlabel('Theoretical Cumulative Probability')
    plt.ylabel('Empirical Cumulative Probability')
    
    # 添加网格线但降低其显著性
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # 确保坐标轴标签显示负号
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # 调整轴和边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    plt.tight_layout()
    pp_path = output_dir / f"{display_name}_ppplot.png"
    plt.savefig(pp_path, dpi=300, facecolor='white')
    print(f"Saved PP plot to: {pp_path}")
    
    # ==================== 创建合并图表 ====================
    create_combined_plot(var_data, display_name, output_dir, light_blue, primary_color, accent_color)
    
    # 将统计结果保存到文件
    stats_dict = {
        "count": int(len(var_data)),
        "mean": float(var_data.mean()),
        "std": float(var_data.std()),
        "min": float(var_data.min()),
        "25%": float(var_data.quantile(0.25)),
        "median": float(median),
        "75%": float(var_data.quantile(0.75)),
        "max": float(var_data.max()),
        "mode": float(mode),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "shapiro_test": {
            "statistic": float(shapiro_test[0]),
            "p_value": float(shapiro_test[1])
        },
        "ks_test": {
            "statistic": float(ks_test[0]),
            "p_value": float(ks_test[1])
        }
    }
    
    import json
    stats_path = output_dir / f"{display_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"\nSaved statistics to: {stats_path}")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 