#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np

def read_metrics_files(result_dir):
    """读取所有指标文件并返回包含所有模型指标的字典"""
    metrics_dict = {}
    for file in os.listdir(result_dir):
        if file.endswith('_metrics.json'):
            model_name = file.split('_')[0]
            file_path = os.path.join(result_dir, file)
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                metrics_dict[model_name] = metrics
    return metrics_dict

def create_metrics_table(metrics_dict, decimal_places=3):
    """创建指标表格，保留指定小数位"""
    if not metrics_dict:
        return pd.DataFrame()
    
    # 使用第一个模型的指标顺序作为基准
    first_model = next(iter(metrics_dict.values()))
    column_order = list(first_model.keys())
    
    # 创建数据框
    df = pd.DataFrame(index=list(metrics_dict.keys()), columns=column_order)
    
    # 填充数据
    for model, metrics in metrics_dict.items():
        for metric, value in metrics.items():
            # 格式化为固定三位小数
            df.loc[model, metric] = f"{value:.3f}"
    
    # 排序行：Logistic在第一位，其他按照字母顺序
    model_order = []
    if 'Logistic' in df.index:
        model_order.append('Logistic')
    
    # 添加其他模型（按字母顺序）
    other_models = [model for model in sorted(df.index) if model != 'Logistic']
    model_order.extend(other_models)
    
    df = df.loc[model_order]
    
    # 转置表格，使指标作为行，模型作为列
    df = df.transpose()
    
    return df

def main():
    # 设置路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(project_dir, 'result')
    output_dir = os.path.join(project_dir, 'Table&Figure')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取指标文件
    print("读取指标文件...")
    metrics_dict = read_metrics_files(result_dir)
    
    # 创建表格
    print("创建指标表格...")
    metrics_table = create_metrics_table(metrics_dict)
    
    # 保存CSV文件
    output_file = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_table.to_csv(output_file)
    print(f"表格已保存到: {output_file}")
    
    # 打印表格预览
    print("\n指标表格预览:")
    print(metrics_table)

if __name__ == "__main__":
    main()
