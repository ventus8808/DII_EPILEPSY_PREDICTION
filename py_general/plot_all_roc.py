"""
合成所有模型ROC曲线的对比图
读取各个模型的ROC原始数据，生成一张完整的对比图
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 加载配置文件
with open(Path(__file__).parent.parent / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 设置字体和风格
plt.rcParams['font.family'] = 'Monaco'
plt.rcParams['font.size'] = 12

# 从配置中获取模型相关设置
models = config['models']['order']
model_display_names = config['models']['display_names']
colors = config['models']['colors']

# ROC配置
ROC_CONFIG = {
    'figure_size': (7, 8),
    'dpi': 300,
    'line_width': 2.0,
    'reference_line_style': '--',
    'reference_line_width': 1.5,
    'x_lim': [0.0, 1.0],
    'y_lim': [0.0, 1.0],
    'legend_bbox_to_anchor': [0.98, 0.02],
    'output_dir': 'plot_combined'
}

def sort_models(models):
    """按照指定的顺序对模型进行排序"""
    # 分离出Logistic、Ensemble和其他模型
    logistic_models = [m for m in models if m.lower().startswith('logistic')]
    ensemble_models = [m for m in models if m.lower().startswith('ensemble')]
    other_models = [m for m in models if not (m.lower().startswith('logistic') or m.lower().startswith('ensemble'))]
    
    # 对其他模型按字母顺序排序（不区分大小写）
    other_models_sorted = sorted(other_models, key=str.lower)
    
    # 合并列表：Logistic + 其他模型 + Ensemble
    return logistic_models + other_models_sorted + ensemble_models

# 按照指定顺序对模型进行排序
models = sort_models(models)

def load_metrics_from_csv():
    """从 metrics_comparison.csv 加载模型的AUC-ROC值"""
    metrics_comparison_path = Path("Table&Figure/metrics_comparison.csv")
    if metrics_comparison_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_comparison_path, index_col=0)
            if 'AUC-ROC' in metrics_df.index:
                # 返回所有模型的AUC-ROC值字典
                return {col: metrics_df.loc['AUC-ROC', col] for col in metrics_df.columns}
        except Exception as e:
            print(f"警告: 读取metrics_comparison.csv出错 - {e}")
    return {}

def load_roc_data(model_name):
    """加载模型的ROC数据"""
    data_path = Path("plot_original_data") / f"{model_name}_ROC.json"
    if not data_path.exists():
        print(f"警告: {data_path} 不存在，跳过该模型")
        return None
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 加载 {data_path} 时出错: {e}")
        return None

def plot_all_roc():
    """创建包含所有模型ROC曲线的合成图"""
    # 创建图形
    plt.figure(figsize=ROC_CONFIG['figure_size'], dpi=ROC_CONFIG['dpi'])
    
    # 存储所有模型的ROC数据
    all_models_data = {}
    auc_scores = {}
    
    # 从 metrics_comparison.csv 加载模型的AUC-ROC值
    csv_metrics = load_metrics_from_csv()
    print(f"从csv加载的指标: {csv_metrics}")
    
    # 加载所有模型数据
    for model_name in models:
        data = load_roc_data(model_name)
        if data is None:
            continue
        
        # 保存FPR和TPR数据
        if "fpr" in data and "tpr" in data:
            all_models_data[model_name] = (data["fpr"], data["tpr"])
            
            # 优先使用metrics_comparison.csv中的AUC值
            if model_name in csv_metrics:
                auc_scores[model_name] = csv_metrics[model_name]
                print(f"  使用CSV中的AUC值: {model_name} = {auc_scores[model_name]:.3f}")
            # 其次使用JSON文件中的AUC值
            elif "auc" in data:
                auc_scores[model_name] = data["auc"]
                print(f"  使用JSON中的auc值: {model_name} = {auc_scores[model_name]:.3f}")
            elif "roc_auc" in data:
                auc_scores[model_name] = data["roc_auc"]
                print(f"  使用JSON中的roc_auc值: {model_name} = {auc_scores[model_name]:.3f}")
    
    # 如果没有加载到数据，退出
    if not all_models_data:
        print("错误: 未能加载任何模型ROC数据")
        return
    
    # 绘制各模型的ROC曲线
    for model_name, (fpr, tpr) in all_models_data.items():
        auc_text = f" (AUC: {auc_scores.get(model_name, 0):.3f})" if model_name in auc_scores else ""
        plt.plot(
            fpr, 
            tpr,
            label=f"{model_display_names.get(model_name, model_name)}{auc_text}",
            color=colors.get(model_name, "#000000"),
            linewidth=ROC_CONFIG['line_width']
        )
    
    # 绘制随机分类器参考线
    plt.plot(
        [0, 1], 
        [0, 1],
        label="Random",
        color=colors["Random"],
        linewidth=ROC_CONFIG['reference_line_width'],
        linestyle=ROC_CONFIG['reference_line_style']
    )
    
    # 设置图表样式
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    
    # 模型图例，按首字母排序
    model_handles = []
    model_labels = []
    
    for model_name in models:
        if model_name in all_models_data:
            model_handles.append(plt.Line2D(
                [0], [0], 
                color=colors.get(model_name, "#000000"), 
                linewidth=ROC_CONFIG['line_width']
            ))
            auc_text = f" (AUC: {auc_scores.get(model_name, 0):.3f})" if model_name in auc_scores else ""
            model_labels.append(f"{model_display_names.get(model_name, model_name)}{auc_text}")
    
    # 创建Random参考线图例项
    random_line = plt.Line2D(
        [0], [0], 
        color=colors["Random"], 
        linestyle=ROC_CONFIG['reference_line_style'], 
        linewidth=ROC_CONFIG['reference_line_width']
    )
    
    # 将Random参考线添加到图例最前面
    all_handles = [random_line] + model_handles
    all_labels = ['Random'] + model_labels
    
    # 创建图例 - 包含所有模型和Random参考线
    plt.legend(
        handles=all_handles,
        labels=all_labels,
        loc='lower right',
        bbox_to_anchor=ROC_CONFIG['legend_bbox_to_anchor'],
        frameon=True,
        fontsize=10,
        ncol=1,  # 单列显示
        framealpha=0.8,
        handlelength=1.2,
        borderpad=0.3,
        borderaxespad=0.3,
        handletextpad=0.4,
        columnspacing=0.8
    )
    
    # 调整布局
    plt.tight_layout()
    
    # 添加网格和设置范围
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(ROC_CONFIG['x_lim'])
    plt.ylim(ROC_CONFIG['y_lim'])
    
    # 紧凑布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例留出空间
    
    # 创建输出目录
    output_dir = Path(ROC_CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 保存图表
    output_path = output_dir / "All_ROC.png"
    plt.savefig(output_path, dpi=ROC_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    
    print(f"ROC曲线图已保存至 {output_path}")

if __name__ == "__main__":
    plot_all_roc()