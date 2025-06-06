import json
import yaml
from pathlib import Path

def load_config():
    """加载配置文件"""
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def format_value(v):
    """
    格式化参数值，移除引号和换行符
    """
    if v is None:
        return 'None'
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (list, dict)):
        # 将列表或字典转换为字符串，并移除引号和换行符
        return str(v).replace("'", "").replace('"', '').replace('\n', ' ').replace('\r', '')
    return str(v).replace('"', '').replace("'", "").replace('\n', ' ').replace('\r', '')

def find_param_files(model_dir):
    """递归查找所有参数文件"""
    param_files = {}
    for file_path in model_dir.rglob('*_best_params.json'):
        # 从文件名中提取基础模型名称（移除 _best_params.json 和路径部分）
        base_name = file_path.stem.replace('_best_params', '')
        
        # 处理 Ensemble 模型名称
        if 'Ensemble' in base_name and '_' in base_name:
            # 提取 Ensemble 后面的部分
            parts = base_name.split('_', 2)  # 最多分割两次
            if len(parts) >= 2:
                # 取 Ensemble 后面的部分作为模型名
                model_name = parts[1]
            else:
                model_name = base_name
        else:
            model_name = base_name
            
        # 不再添加父目录名到模型名
        param_files[model_name] = file_path
    return param_files

def generate_params_table():
    # 加载配置
    config = load_config()
    
    # 查找所有参数文件
    script_dir = Path(__file__).parent.absolute()
    model_dir = script_dir.parent / "model"
    model_files = find_param_files(model_dir)
    
    # 收集所有参数
    all_params = {}
    for model_name, file_path in model_files.items():
        if file_path.exists():
            with open(file_path, 'r') as f:
                try:
                    params = json.load(f)
                    # 如果参数在 'best_params' 键中，则提取出来
                    if 'best_params' in params:
                        params = params['best_params']
                    all_params[model_name] = params
                except json.JSONDecodeError:
                    print(f"警告: 无法解析 {file_path}")
        else:
            print(f"警告: 文件不存在 {file_path}")
    
    # 使用配置中的模型顺序
    ordered_models = get_ordered_models(config, all_params)
    
    # 生成表格内容
    table = []
    for model in ordered_models:
        params = all_params[model]
        # 将所有参数格式化为一个字符串
        param_items = []
        for key, value in params.items():
            param_items.append(f"{key}: {format_value(value)}")
        
        # 将所有参数合并为一行，用分号分隔
        param_line = "; ".join(param_items)
        
        # 使用配置中的显示名称（如果有）
        display_name = model
        if 'models' in config and 'display_names' in config['models']:
            # 检查普通模型名称
            if model in config['models']['display_names']:
                display_name = config['models']['display_names'][model]
        
        table.append(f"{display_name}\t{param_line}")
    
    # 添加表头
    output = "Model\tParameters\n" + "\n".join(table)
    
    # 保存为TSV文件
    output_path = script_dir.parent / "Table&Figure" / "best_params.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"已生成 {output_path}")

# 使用配置中的模型顺序
def get_ordered_models(config, all_params):
    """获取按照配置排序的模型列表"""
    ordered_models = []
    if 'models' in config and 'order' in config['models']:
        # 处理配置中指定的模型
        for model in config['models']['order']:
            if model in all_params:
                ordered_models.append(model)
    
    # 添加未在配置中指定但存在的模型
    for model in all_params:
        if model not in ordered_models:
            ordered_models.append(model)
    
    return ordered_models

if __name__ == "__main__":
    generate_params_table()
