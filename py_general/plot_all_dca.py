"""合成所有模型DCA曲线的对比图
读取各个模型的标准DCA原始数据，仅使用全特征版本（不含无DII版本）
保存为一张风格一致的合成图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体和风格
plt.rcParams['font.family'] = 'Monaco'
plt.rcParams['font.size'] = 12

# 模型列表和对应的颜色 - 按首字母排序
models = sorted([
    "CatBoost", 
    "Ensemble_Voting",
    "FNN",
    "LightGBM", 
    "Logistic", 
    "RF", 
    "SVM", 
    "XGBoost"
])

# 模型显示名称（可以根据需要修改）
model_display_names = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "RF": "Random Forest",
    "FNN": "FNN",
    "SVM": "SVM",
    "Logistic": "Logistic Regression",
    "Ensemble_Voting": "Voting"
}

# 颜色映射（保持与单独图表一致的颜色方案）
colors = {
    "XGBoost": "#1f77b4",        # 蓝色
    "LightGBM": "#ff7f0e",       # 橙色
    "CatBoost": "#2ca02c",       # 绿色
    "RF": "#d62728",             # 红色
    "FNN": "#9467bd",            # 紫色
    "SVM": "#8c564b",            # 棕色
    "Logistic": "#e377c2",       # 粉色
    "Ensemble_Voting": "#17becf", # 青色
    "Treat All": "green",        # 绿色虚线
    "Treat None": "red"          # 红色虚线
}

def load_dca_data(model_name):
    """加载模型的DCA数据"""
    data_path = Path("plot_original_data") / f"{model_name}_dca_data.json"
    if not data_path.exists():
        print(f"警告: {data_path} 不存在，跳过该模型")
        return None
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def plot_all_dca():
    """创建包含所有模型DCA曲线的合成图"""
    # 创建正方形图形，确保图形为正方形
    plt.figure(figsize=(7, 7))
    
    # 存储所有模型的数据
    all_models_data = {}
    thresholds = None
    treat_all = None
    treat_none = None
    prevalence = None
    
    # 加载所有模型数据
    for model_name in models:
        data = load_dca_data(model_name)
        if data is None:
            continue
        
        # 按照正确的JSON结构获取数据
        if "thresholds" in data and "net_benefits_model" in data:
            all_models_data[model_name] = data["net_benefits_model"]
            
            # 记录参考数据（从任何模型取一次即可）
            if thresholds is None:
                thresholds = data["thresholds"]
            if treat_all is None and "net_benefits_all" in data:
                treat_all = data["net_benefits_all"]
            if treat_none is None and "net_benefits_none" in data:
                treat_none = data["net_benefits_none"]
            if prevalence is None and "prevalence" in data:
                prevalence = data["prevalence"]
    
    # 如果没有加载到数据，退出
    if not all_models_data or thresholds is None:
        print("错误: 未能加载任何模型数据")
        return
    
    # 绘制各模型的DCA曲线
    for model_name, net_benefits in all_models_data.items():
        plt.plot(
            thresholds, 
            net_benefits,
            label=model_display_names.get(model_name, model_name),
            color=colors.get(model_name, "#000000"),
            linewidth=2.0
        )
    
    # 绘制"Treat All"策略的曲线 - 绿色长虚线
    plt.plot(
        thresholds, 
        treat_all,
        label="Treat All",
        color=colors["Treat All"],
        linewidth=1.5,
        linestyle="--"
    )
    
    # 绘制"Treat None"策略的曲线 - 红色长虚线
    plt.plot(
        thresholds, 
        treat_none,
        label="Treat None",
        color=colors["Treat None"],
        linewidth=1.5,
        linestyle="--"
    )
    
    # 设置图表样式
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis - All Models")
    
    # 将图例移动到图片下方
    # 左侧放虚线图例
    dashed_handles = [
        plt.Line2D([0], [0], color=colors["Treat All"], linewidth=1.5, linestyle="--"),
        plt.Line2D([0], [0], color=colors["Treat None"], linewidth=1.5, linestyle="--")
    ]
    
    dashed_labels = ["Treat All", "Treat None"]
    
    # 模型图例，按首字母排序
    model_handles = []
    model_labels = []
    
    for model_name in models:
        model_handles.append(plt.Line2D([0], [0], color=colors.get(model_name, "#000000"), linewidth=2.0))
        model_labels.append(model_display_names.get(model_name, model_name))
    
    # 创建图例组合（左侧虚线，右侧模型分两列）
    fig = plt.gcf()
    fig.legend(
        handles=dashed_handles,
        labels=dashed_labels,
        loc='lower left',
        bbox_to_anchor=(0.1, -0.05),
        frameon=True,
        fontsize=10,
        framealpha=0.8,
        handlelength=2.0
    )
    
    fig.legend(
        handles=model_handles,
        labels=model_labels,
        loc='lower right',
        bbox_to_anchor=(0.95, -0.08),
        frameon=True,
        fontsize=10,
        ncol=2,  # 分两列显示
        framealpha=0.8,
        handlelength=2.0
    )
    
    # 添加网格和设置范围
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0.0, 1.0)  # 设置横轴范围从0到1
    
    # 动态调整Y轴范围，确保所有曲线可见
    all_benefits = []
    for benefits in all_models_data.values():
        all_benefits.extend(benefits)
    all_benefits.extend(treat_all)
    all_benefits.extend(treat_none)
    
    # 手动设置合适的Y轴范围，更好地显示数据差异
    y_min = -0.1  # 固定下限以显示负值区域
    y_max = 0.25   # 固定上限保证能看到所有峰值
    plt.ylim(y_min, y_max)  
    
    # 紧凑布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例留出空间
    
    # 保存图表
    plot_dir = Path("plot")
    plot_dir.mkdir(exist_ok=True)
    output_path = plot_dir / "All_DCA.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"已生成所有模型的DCA对比图: {output_path}")
    if prevalence is not None:
        print(f"数据集患病率: {prevalence:.4f}")

if __name__ == "__main__":
    plot_all_dca()
