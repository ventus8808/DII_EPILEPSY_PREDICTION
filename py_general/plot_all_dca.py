import os
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

# 添加参考线颜色
colors.update({
    "Treat_All": "#2ca02c",  # 绿色
    "Treat_None": "#7f7f7f"  # 灰色
})

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
    # 创建更接近正方形的图形，并增加宽度为图例留出空间
    plt.figure(figsize=(10, 7), dpi=300)  # 调整为更接近正方形的比例
    
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
        # 确保数据长度一致，有可能thresholds长度为100而数据长度为50
        if len(net_benefits) != len(thresholds):
            print(f"警告: 模型 {model_name} 的数据长度 ({len(net_benefits)}) 与阈值长度 ({len(thresholds)}) 不一致")
            # 如果数据长度小于阈值长度，则需要进行插值
            if len(net_benefits) < len(thresholds):
                import numpy as np
                old_indices = np.linspace(0, 1, len(net_benefits))
                new_indices = np.linspace(0, 1, len(thresholds))
                net_benefits = np.interp(new_indices, old_indices, net_benefits)
                print(f"    已对模型 {model_name} 的数据进行插值处理，使数据长度与阈值长度一致")
            # 如果数据长度大于阈值长度，则进行降采样
            else:
                step = len(net_benefits) // len(thresholds)
                net_benefits = net_benefits[::step][:len(thresholds)]
                print(f"    已对模型 {model_name} 的数据进行降采样处理，使数据长度与阈值长度一致")
        
        plt.plot(
            thresholds, 
            net_benefits,
            label=model_display_names.get(model_name, model_name),
            color=colors.get(model_name, "#000000"),
            linewidth=2.0
        )
    
    # 绘制"Treat All"策略的曲线
    plt.plot(
        thresholds, 
        treat_all,
        label="Treat All",
        color=colors["Treat_All"],
        linewidth=1.5,
        linestyle="--"
    )
    
    # 绘制"Treat None"策略的曲线
    plt.plot(
        thresholds, 
        treat_none,
        label="Treat None",
        color=colors["Treat_None"],
        linewidth=1.5,
        linestyle="--"
    )
    
    # 设置图表样式
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis - All Models")
    
    # 创建参考线图例
    dashed_handles = [
        plt.Line2D([0], [0], color=colors["Treat_All"], linewidth=1.5, linestyle="--"),
        plt.Line2D([0], [0], color=colors["Treat_None"], linewidth=1.5, linestyle="--")
    ]
    
    dashed_labels = ["Treat All", "Treat None"]
    
    # 创建模型图例
    model_handles = []
    model_labels = []
    
    for model_name in models:
        if model_name in all_models_data:  # 只添加有数据的模型
            model_handles.append(plt.Line2D([0], [0], color=colors.get(model_name, "#000000"), linewidth=2.0))
            model_labels.append(model_display_names.get(model_name, model_name))
    
    # 创建右侧图例（模型）
    legend_model = plt.legend(
        handles=model_handles,
        labels=model_labels,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),  # 垂直居中
        frameon=True,
        fontsize=10,  # 稍微减小字体
        title='Models',
        title_fontsize=10,
        labelspacing=0.5  # 减小行距
    )
    plt.gca().add_artist(legend_model)
    
    # 添加参考线图例
    plt.legend(
        handles=dashed_handles,
        labels=dashed_labels,
        loc='upper left',
        bbox_to_anchor=(1.05, 0.9),  # 靠近顶部
        frameon=True,
        fontsize=10,  # 统一字体大小
        title='Reference',
        title_fontsize=10,
        labelspacing=0.5  # 减小行距
    )
    
    # 调整布局，为右侧图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(
        right=0.7,  # 右侧留出30%空间给图例
        left=0.1,   # 左侧边距
        top=0.95,   # 顶部边距
        bottom=0.1  # 底部边距
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
    
    # 创建输出目录
    output_dir = Path("plot_combined")
    output_dir.mkdir(exist_ok=True)
    
    # 保存图表
    output_path = output_dir / "All_DCA.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"已生成所有模型的DCA对比图: {output_path}")
    if prevalence is not None:
        print(f"数据集患病率: {prevalence:.4f}")

if __name__ == "__main__":
    plot_all_dca()
