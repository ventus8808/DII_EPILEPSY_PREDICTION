"""
合成所有模型PR曲线的对比图
读取各个模型的PR原始数据，生成一张完整的对比图
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
    "Baseline": "gray",          # 基准线参考线
}

def load_pr_data(model_name):
    """加载模型的PR数据"""
    data_path = Path("plot_original_data") / f"{model_name}_PR_data.json"
    # SVM文件名可能不同
    if not data_path.exists() and model_name == "SVM":
        data_path = Path("plot_original_data") / f"{model_name}_PR_data.json".lower()
    
    if not data_path.exists():
        print(f"警告: {data_path} 不存在，跳过该模型")
        return None
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def plot_all_pr():
    """创建包含所有模型PR曲线的合成图"""
    # 创建正方形图形
    plt.figure(figsize=(7, 7))
    
    # 存储所有模型的PR数据
    all_models_data = {}
    auc_scores = {}
    baseline = None
    
    # 加载所有模型数据
    for model_name in models:
        data = load_pr_data(model_name)
        if data is None:
            continue
        
        # 保存Precision和Recall数据
        if "precision" in data and "recall" in data:
            all_models_data[model_name] = (data["recall"], data["precision"])
            if "auc_pr" in data:
                auc_scores[model_name] = data["auc_pr"]
            elif "average_precision" in data:
                auc_scores[model_name] = data["average_precision"]
            
            # 获取基准线（所有模型应该是相同的，所以取任意一个）
            if baseline is None and "baseline" in data:
                baseline = data["baseline"]
    
    # 如果没有加载到数据，退出
    if not all_models_data:
        print("错误: 未能加载任何模型PR数据")
        return
    
    # 绘制各模型的PR曲线
    for model_name, (recall, precision) in all_models_data.items():
        auc_text = f" (AP: {auc_scores.get(model_name, 0):.3f})" if model_name in auc_scores else ""
        plt.plot(
            recall, 
            precision,
            label=f"{model_display_names.get(model_name, model_name)}{auc_text}",
            color=colors.get(model_name, "#000000"),
            linewidth=2.0
        )
    
    # 绘制基准线参考线
    if baseline is not None:
        plt.axhline(
            y=baseline,
            label=f"Baseline ({baseline:.3f})",
            color=colors["Baseline"],
            linewidth=1.5,
            linestyle="--"
        )
    
    # 设置图表样式
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - All Models")
    
    # 将图例移动到图片下方
    # 左侧放基准线图例
    ref_handles = []
    ref_labels = []
    
    if baseline is not None:
        ref_handles.append(plt.Line2D([0], [0], color=colors["Baseline"], linewidth=1.5, linestyle="--"))
        ref_labels.append(f"Baseline ({baseline:.3f})")
    
    # 模型图例，按首字母排序
    model_handles = []
    model_labels = []
    
    for model_name in models:
        if model_name in all_models_data:
            model_handles.append(plt.Line2D([0], [0], color=colors.get(model_name, "#000000"), linewidth=2.0))
            auc_text = f" (AP: {auc_scores.get(model_name, 0):.3f})" if model_name in auc_scores else ""
            model_labels.append(f"{model_display_names.get(model_name, model_name)}{auc_text}")
    
    # 创建图例组合（左侧参考线，右侧模型分两列）
    fig = plt.gcf()
    
    if ref_handles:
        fig.legend(
            handles=ref_handles,
            labels=ref_labels,
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
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    
    # 紧凑布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例留出空间
    
    # 创建输出目录
    output_dir = Path("plot_combined")
    output_dir.mkdir(exist_ok=True)
    
    # 保存图表
    output_path = output_dir / "All_PR.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"已生成所有模型的PR对比图: {output_path}")

if __name__ == "__main__":
    plot_all_pr()
