import json
import yaml
import numpy as np
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

# DCA配置
DCA_CONFIG = {
    'figure_size': (9.5, 7),
    'dpi': 300,
    'line_width': 2.0,
    'reference_line_style': '--',
    'reference_line_width': 1.5,
    'x_lim': [0.0, 1.0],
    'y_lim': [-0.1, 0.25],
    'legend_bbox_to_anchor': [1.05, 0.7],
    'output_dir': 'plot_combined'
}

def load_dca_data(model_name):
    """加载模型的DCA数据"""
    data_path = Path("plot_original_data") / f"{model_name}_DCA.json"
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

def plot_all_dca():
    """创建包含所有模型DCA曲线的合成图"""
    # 创建图形
    plt.figure(figsize=DCA_CONFIG['figure_size'], dpi=DCA_CONFIG['dpi'])
    
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
        # 确保数据长度一致
        if len(net_benefits) != len(thresholds):
            # 插值处理
            old_indices = np.linspace(0, 1, len(net_benefits))
            new_indices = np.linspace(0, 1, len(thresholds))
            net_benefits = np.interp(new_indices, old_indices, net_benefits)
        
        plt.plot(
            thresholds, 
            net_benefits,
            label=model_display_names.get(model_name, model_name),
            color=colors.get(model_name, "#000000"),
            linewidth=DCA_CONFIG['line_width']
        )
    
    # 绘制参考线
    plt.plot(
        thresholds, 
        treat_all,
        label="Treat All",
        color=colors["Treat_All"],
        linewidth=DCA_CONFIG['reference_line_width'],
        linestyle=DCA_CONFIG['reference_line_style']
    )
    
    plt.plot(
        thresholds, 
        treat_none,
        label="Treat None",
        color=colors["Treat_None"],
        linewidth=DCA_CONFIG['reference_line_width'],
        linestyle=DCA_CONFIG['reference_line_style']
    )
    
    # 设置图表样式
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA)")
    
    # 创建图例
    reference_handles = [
        plt.Line2D([0], [0], color=colors["Treat_All"], 
                  linewidth=DCA_CONFIG['reference_line_width'], 
                  linestyle=DCA_CONFIG['reference_line_style']),
        plt.Line2D([0], [0], color=colors["Treat_None"], 
                  linewidth=DCA_CONFIG['reference_line_width'], 
                  linestyle=DCA_CONFIG['reference_line_style'])
    ]
    
    model_handles = [
        plt.Line2D([0], [0], color=colors.get(model_name, "#000000"), 
                  linewidth=DCA_CONFIG['line_width'])
        for model_name in models if model_name in all_models_data
    ]
    
    model_labels = [
        model_display_names.get(model_name, model_name)
        for model_name in models if model_name in all_models_data
    ]
    
    # 添加参考线图例
    plt.legend(
        handles=reference_handles,
        labels=["Treat All", "Treat None"],
        loc='upper left',
        bbox_to_anchor=DCA_CONFIG['legend_bbox_to_anchor'],
        frameon=True,
        fontsize=11,
        title='Reference',
        title_fontsize=10
    )
    
    # 添加模型图例
    plt.legend(
        handles=model_handles,
        labels=model_labels,
        loc='center left',
        bbox_to_anchor=(DCA_CONFIG['legend_bbox_to_anchor'][0], 0.35),
        frameon=True,
        fontsize=11,
        title='Models',
        title_fontsize=10
    )
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.7 if DCA_CONFIG['legend_bbox_to_anchor'][0] > 1 else 0.9)
    
    # 设置坐标轴范围
    plt.xlim(DCA_CONFIG['x_lim'])
    plt.ylim(DCA_CONFIG['y_lim'])
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 创建输出目录
    output_dir = Path(DCA_CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 保存图表
    output_path = output_dir / "All_DCA.png"
    plt.savefig(output_path, dpi=DCA_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"DCA曲线图已保存至 {output_path} 和 {output_dir}/All_DCA.pdf")
    if prevalence is not None:
        print(f"数据集患病率: {prevalence:.4f}")

if __name__ == "__main__":
    plot_all_dca()
