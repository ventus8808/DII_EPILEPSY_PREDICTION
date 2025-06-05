#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image, ImageFile
import numpy as np
from pathlib import Path

# 禁用PIL的DecompressionBomb保护
Image.MAX_IMAGE_PIXELS = None
# 允许加载大图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

def sort_image_files(files):
    """按照指定顺序对图片文件进行排序，不区分大小写"""
    # 分离出Logistic、Ensemble和其他模型
    logistic_files = [f for f in files if f.lower().startswith('logistic')]
    ensemble_files = [f for f in files if f.lower().startswith('ensemble')]
    other_files = [f for f in files if not (f.lower().startswith('logistic') or f.lower().startswith('ensemble'))]
    
    # 对其他模型按字母顺序排序（不区分大小写）
    other_files_sorted = sorted(other_files, key=str.lower)
    
    # 合并列表：Logistic + 其他模型 + Ensemble
    sorted_files = logistic_files + other_files_sorted + ensemble_files
    
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

def process_plots(plot_dir, output_dir):
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
        
        # 获取文件名并排序
        file_names = [f.name for f in image_files]
        sorted_files = sort_image_files(file_names)
        
        # 获取完整路径
        sorted_paths = [plot_dir / f for f in sorted_files]
        
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
    
    print(f"输入目录: {plot_dir}")
    print(f"输出目录: {output_dir}")
    
    # 处理图片
    process_plots(plot_dir, output_dir)
