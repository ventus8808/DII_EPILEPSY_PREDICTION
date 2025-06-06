#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image, ImageFile
import numpy as np
from pathlib import Path
import yaml
import re

# 禁用PIL的DecompressionBomb保护
Image.MAX_IMAGE_PIXELS = None
# 允许加载大图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_config():
    """加载配置文件"""
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def sort_image_files(files, config):
    """按照config.yaml中的模型顺序对图片文件进行排序"""
    # 从配置中获取模型顺序
    if 'models' in config and 'order' in config['models']:
        model_order = config['models']['order']
    else:
        # 如果没有配置，使用文件名排序
        return sorted(files, key=str.lower)
    
    # 创建映射，将文件名映射到模型名和排序索引
    file_info = {}
    for file in files:
        basename = os.path.basename(file)
        # 提取文件名的第一部分作为模型名
        for model_name in model_order:
            if basename.lower().startswith(model_name.lower()):
                # 找到匹配的模型名
                file_info[file] = {
                    'model': model_name,
                    'order': model_order.index(model_name)
                }
                break
        else:
            # 特殊处理Voting/Ensemble
            if basename.lower().startswith("voting") or basename.lower().startswith("ensemble"):
                voting_idx = model_order.index("Voting") if "Voting" in model_order else len(model_order)
                file_info[file] = {
                    'model': "Voting",
                    'order': voting_idx
                }
            else:
                # 未找到匹配的模型名
                file_info[file] = {
                    'model': "Unknown",
                    'order': len(model_order)
                }
    
    # 根据模型顺序排序
    sorted_files = sorted(files, key=lambda f: file_info[f]['order'])
    
    # 调试输出，显示排序结果
    print("\n排序结果:")
    for file in sorted_files:
        model_name = file_info[file]['model']
        order = file_info[file]['order']
        print(f"  {os.path.basename(file)} -> {model_name} (顺序: {order})")
    
    return sorted_files

def combine_images(image_paths, output_path, spacing=20):
    """
    将多张图片合并成一个2xN的网格
    
    参数:
        image_paths: 图片路径列表
        output_path: 输出图片路径
        spacing: 图片之间的间距（像素），默认为10
    """
    if not image_paths:
        print("没有找到图片")
        return
    
    # 打开所有图片
    images = [Image.open(img_path) for img_path in image_paths]
    
    # 计算网格大小
    num_images = len(images)
    cols = (num_images + 1) // 2  # 向上取整
    rows = 2
    
    # 计算每张图片的宽度和高度
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)
    
    # 计算新图片的尺寸（考虑间距）
    total_width = (max_width * cols) + (spacing * (cols - 1))
    total_height = (max_height * rows) + (spacing * (rows - 1))
    
    # 创建新图片（白色背景）
    new_img = Image.new('RGB', (total_width, total_height), color='white')
    
    # 将图片粘贴到新图片中（考虑间距）
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (max_width + spacing)
        y = row * (max_height + spacing)
        new_img.paste(img, (x, y))
    
    # 保存合并后的图片
    new_img.save(output_path)
    print(f"已保存合并后的图片到: {output_path}")

def case_insensitive_glob(directory, pattern):
    """执行不区分大小写的glob匹配"""
    import re
    
    # 将glob模式转换为正则表达式
    def glob_to_regex(pat):
        """将glob模式转换为正则表达式"""
        i, n = 0, len(pat)
        res = []
        while i < n:
            c = pat[i]
            i += 1
            if c == '*':
                res.append('.*')
            elif c == '?':
                res.append('.')
            elif c == '[':
                j = i
                if j < n and pat[j] == '!':
                    j += 1
                if j < n and pat[j] == ']':
                    j += 1
                while j < n and pat[j] != ']':
                    j += 1
                if j >= n:
                    res.append('\\[')
                else:
                    stuff = pat[i:j].replace('\\', '\\\\')
                    i = j + 1
                    if stuff[0] == '!':
                        stuff = '^' + stuff[1:]
                    elif stuff[0] == '^':
                        stuff = '\\' + stuff
                    res.append('[' + stuff + ']')
            else:
                res.append(re.escape(c))
        return '(?i)^' + ''.join(res) + '$'
    
    # 获取目录下所有文件
    all_files = [f.name for f in Path(directory).iterdir() if f.is_file()]
    
    # 编译正则表达式模式
    regex = re.compile(glob_to_regex(pattern))
    
    # 返回匹配的文件路径
    return [Path(directory) / f for f in all_files if regex.match(f)]

def process_plots(plot_dir, output_dir, config):
    """处理plot目录下的图片（不区分大小写）"""
    plot_dir = Path(plot_dir)
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要处理的图片模式（不区分大小写）
    patterns = [
        '*ROC_DII.png',
        '*DCA_DII.png', 
        '*Threshold.png', 
        '*Learning_Curve.png',
        '*Calibration_Curve.png'
    ]
    
    for pattern in patterns:
        # 查找匹配的图片（不区分大小写）
        image_files = case_insensitive_glob(plot_dir, pattern)
        if not image_files:
            print(f"未找到匹配 {pattern} 的图片（不区分大小写）")
            continue
        
        print(f"\n处理模式 {pattern} 的图片:")
        
        # 打印找到的文件
        print("找到以下文件:")
        for f in image_files:
            print(f"  {f.name}")
        
        # 获取完整路径并按配置排序
        sorted_paths = sort_image_files([str(f) for f in image_files], config)
        
        # 生成输出文件名
        output_name = f"Combine_{pattern.strip('*')}"
        # 确保输出文件扩展名是.png
        if not output_name.endswith('.png'):
            output_name += '.png'
        output_path = output_dir / output_name
        
        # 合并图片
        combine_images(sorted_paths, output_path)

if __name__ == "__main__":
    # 设置输入和输出目录路径
    script_dir = Path(__file__).parent.absolute()
    plot_dir = script_dir.parent / "plot"
    output_dir = script_dir.parent / "plot_combined"
    
    # 确保目录存在
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置文件
    config = load_config()
    
    print(f"输入目录: {plot_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型顺序: {config['models']['order']}")
    
    # 处理图片
    process_plots(plot_dir, output_dir, config)
