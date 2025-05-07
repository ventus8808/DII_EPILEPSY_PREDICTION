#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import re

def combine_plots():
    # 输入和输出路径
    plot_dir = '/Users/ventus/Repository/DII_EPILEPSY_PREDICTION/plot'
    output_dir = '/Users/ventus/Repository/DII_EPILEPSY_PREDICTION/Table&Figure'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图片文件
    files = [f for f in os.listdir(plot_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 提取模型名称并按字母排序
    model_names = set()
    for f in files:
        model_name = f.split('_')[0]
        model_names.add(model_name)
    
    model_names = sorted(list(model_names))
    print(f"找到的模型: {model_names}")
    
    # 定义图片类型顺序
    image_types = [
        "Confusion_Matrix", 
        "ROC", 
        "PR", 
        "Threshold_Curve", 
        "Calibration_Curve", 
        "Sample_Learning_Curve"
    ]
    
    # 修正FNN的sample_learning_curve名称不一致的问题
    def get_filename(model, img_type):
        if model == "FNN" and img_type == "Sample_Learning_Curve":
            return f"{model}_sample_learning_curve.png"
        return f"{model}_{img_type}.png"
    
    # 创建大图
    n_rows = len(image_types)
    n_cols = len(model_names)
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    gs = GridSpec(n_rows, n_cols)
    
    # 添加行标签和列标签
    for i, img_type in enumerate(image_types):
        plt.figtext(0.01, 1 - (i + 0.5) / n_rows, img_type.replace('_', ' '), 
                   va='center', ha='left', fontsize=12, rotation=90)
    
    for j, model in enumerate(model_names):
        plt.figtext((j + 0.5) / n_cols, 0.98, model, 
                   va='top', ha='center', fontsize=14)
    
    # 填充图像
    for i, img_type in enumerate(image_types):
        for j, model in enumerate(model_names):
            filename = get_filename(model, img_type)
            img_path = os.path.join(plot_dir, filename)
            
            if os.path.exists(img_path):
                # 读取图像
                img = plt.imread(img_path)
                
                # 添加到网格
                ax = plt.subplot(gs[i, j])
                ax.imshow(img)
                ax.axis('off')
            else:
                print(f"警告: 未找到图片 {img_path}")
                ax = plt.subplot(gs[i, j])
                ax.text(0.5, 0.5, 'Missing Image', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout(rect=[0.03, 0.01, 1, 0.97])  # 留出标签空间
    
    # 保存组合后的图片
    output_path = os.path.join(output_dir, 'combined_model_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存组合图片到: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    combine_plots()
