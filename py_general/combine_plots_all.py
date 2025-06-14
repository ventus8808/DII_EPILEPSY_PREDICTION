#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import re
import yaml

def load_config():
    """加载配置文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def combine_plots():
    # 加载配置
    config = load_config()
    
    # 输入和输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, '..', 'plot')
    output_dir = os.path.join(script_dir, '..', 'plot_combined')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图片文件（不区分大小写）
    files = [f for f in os.listdir(plot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 提取模型名称并组织文件
    model_file_map = {}
    files_by_model = {}
    
    for f in files:
        # 使用正则表达式分割文件名（不区分大小写）
        parts = re.split(r'[_\-]', f)
        if len(parts) < 2:
            continue
        
        # 获取模型名称（不区分大小写）
        first_part = parts[0].lower()
        if first_part == 'all':
            continue  # 跳过All模型
        elif first_part == 'ensemble':
            if len(parts) >= 3:
                # 处理 Ensemble_Voting_XXX 模式，统一使用Voting
                model_key = "Voting"
                # 从配置中获取显示名称
                if 'models' in config and 'display_names' in config['models'] and 'Voting' in config['models']['display_names']:
                    model_display = config['models']['display_names']['Voting']
                else:
                    model_display = "Voting"
                img_type = '_'.join(parts[2:])
            else:
                continue  # 格式不正确则跳过
        elif first_part == 'voting':
            # 直接处理Voting开头的文件
            model_key = "Voting"
            # 从配置中获取显示名称
            if 'models' in config and 'display_names' in config['models'] and 'Voting' in config['models']['display_names']:
                model_display = config['models']['display_names']['Voting']
            else:
                model_display = "Voting"
            img_type = '_'.join(parts[1:])
        else:
            model_key = parts[0]  # 键名称
            # 从配置中获取显示名称
            if 'models' in config and 'display_names' in config['models'] and model_key in config['models']['display_names']:
                model_display = config['models']['display_names'][model_key]
            else:
                model_display = model_key  # 默认使用键名称作为显示名称
            img_type = '_'.join(parts[1:])
        
        # 将文件添加到模型映射中
        if model_key not in model_file_map:
            model_file_map[model_key] = model_display
            files_by_model[model_key] = {}
        
        # 将文件添加到相应图片类型中
        img_type_no_ext = os.path.splitext(img_type)[0]  # 去除扩展名
        files_by_model[model_key][img_type_no_ext] = f
    
    # 从配置中获取模型顺序
    model_keys = []
    if 'models' in config and 'order' in config['models']:
        # 按照配置的顺序排列
        for model in config['models']['order']:
            # 如果模型在找到的文件中存在，添加到排序列表
            if model in model_file_map:
                model_keys.append(model)
    
    # 添加任何在配置中未指定但在files_by_model中存在的模型
    for model in files_by_model:
        if model not in model_keys:
            model_keys.append(model)
    
    print(f"找到的模型: {[model_file_map[key] for key in model_keys]}")
    print(f"模型显示顺序: {[model_file_map[key] for key in model_keys]}")
    
    # 定义两种不同的图片类型顺序（使用小写键）
    image_types_all = [
        "Confusion_Matrix", 
        "ROC", 
        "PR", 
        "Threshold", 
        "DCA",
        "DCA_DII",
        "Calibration_Curve", 
        "Learning_Curve"
    ]
    
    image_types_selected = [
        "Threshold", 
        "DCA_DII",
        "Calibration_Curve", 
        "Learning_Curve"
    ]
    
    # 获取图片文件名称（不区分大小写）
    def get_image_path(model_key, img_type):
        model_key = model_key.lower()
        img_type_lower = img_type.lower()
        
        # 查找匹配的模型和图片类型
        for model, img_dict in files_by_model.items():
            if model.lower() == model_key:
                for img_key, img_file in img_dict.items():
                    if img_type_lower in img_key.lower():
                        return os.path.join(plot_dir, img_file)
        return None
    
    # 创建并保存图表的函数
    def create_and_save_plot(image_types, output_filename):
        n_rows = len(image_types)
        n_cols = len(model_keys)
        
        plt.figure(figsize=(5*n_cols, 5*n_rows))
        gs = GridSpec(n_rows, n_cols)
        
        # 添加行标签和列标签
        for i, img_type in enumerate(image_types):
            plt.figtext(0.01, 1 - (i + 0.5) / n_rows, img_type.replace('_', ' '), 
                      va='center', ha='left', fontsize=12, rotation=90)
        
        for j, model_key in enumerate(model_keys):
            plt.figtext((j + 0.5) / n_cols, 0.98, model_file_map[model_key], 
                      va='top', ha='center', fontsize=14)
        
        # 填充图像
        for i, img_type in enumerate(image_types):
            for j, model_key in enumerate(model_keys):
                img_path = get_image_path(model_key, img_type)
                
                ax = plt.subplot(gs[i, j])
                if img_path and os.path.exists(img_path):
                    # 读取图像
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    if img_type in ["ROC", "PR", "Confusion_Matrix"]:  # 只为必要图表显示警告
                        print(f"警告: 未找到模型 {model_file_map[model_key]} 的 {img_type} 图片")
                    ax.text(0.5, 0.5, 'Missing Image', ha='center', va='center')
                    ax.axis('off')
        
        plt.tight_layout(rect=[0.03, 0.01, 1, 0.97])  # 留出标签空间
        
        # 保存组合后的图片
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存组合图片到: {output_path}")
        
        plt.close()
    
    # 生成包含所有图表类型的组合图
    print("正在生成包含所有图表类型的组合图...")
    create_and_save_plot(image_types_all, 'Combined_plots1.png')
    
    # 生成仅包含指定四种图表类型的组合图
    print("正在生成仅包含指定图表类型的组合图...")
    create_and_save_plot(image_types_selected, 'Combined_plots2.png')

if __name__ == "__main__":
    combine_plots()
