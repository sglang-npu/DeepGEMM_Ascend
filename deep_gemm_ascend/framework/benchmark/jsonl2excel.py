import os
import re
import json
import pandas as pd
import glob
import argparse

def params_to_excel(input_jsonl_path, output_xlsx_path):
    data_list = []
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        line_data = json.loads(line)
                        data_list.append(line_data)
                    except json.JSONDecodeError as e:
                        print(f'warnning: parse line {line_num} error:{e}: {line}')

        if not data_list:
            print(f'warnning: Do not parse anything from jsonl file!')
            return
        
        df = pd.DataFrame(data_list)
        df.to_excel(output_xlsx_path, index=False, engine='openpyxl')
    
    except FileNotFoundError:
        print(f'error: input_jsonl_path {input_jsonl_path} not found!')
    except Exception as e:
        print(f'exist unknown error: {e}')

def result_to_excel(input_jsonl_path, output_xlsx_path):
    data_list = []
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        line_data = json.loads(line)
                        parameters = line_data.pop('parameters', {})
                        for key, value in parameters.items():
                            line_data[key] = value
                        data_list.append(line_data)

                    except json.JSONDecodeError as e:
                        print(f'warnning: parse line {line_num} error:{e}: {line}')

        if not data_list:
            print(f'warnning: Do not parse anything from jsonl file!')
            return
        
        df = pd.DataFrame(data_list)
        df.to_excel(output_xlsx_path, index=False, engine='openpyxl')
    
    except FileNotFoundError:
        print(f'error: input_jsonl_path {input_jsonl_path} not found!')
    except Exception as e:
        print(f'exist unknown error: {e}')

def merge_jsonl_by_shape(input_dir, shape_str, output_xlsx_path):
    """
    根据shape合并多个jsonl文件并转换为Excel（排除checkpoint文件）
    :param input_dir: 包含jsonl文件的文件夹路径
    :param shape_str: 要合并的shape字符串（如"1_2_3"）
    :param output_xlsx_path: 输出的Excel文件路径
    """
    # 构建基础匹配模式
    base_pattern = os.path.join(input_dir, f'shape_{shape_str}_rank_*.jsonl')
    all_matching_files = glob.glob(base_pattern)
    
    # 构建精确的正则表达式，排除带有_checkpoint的文件
    # 匹配格式: shape_{shape_str}_rank_数字.jsonl
    pattern = re.compile(
        r"shape_{}_rank_\d+\.jsonl$".format(re.escape(shape_str))
    )
    
    # 过滤出符合条件的文件（排除checkpoint文件）
    jsonl_files = []
    for file_path in all_matching_files:
        file_name = os.path.basename(file_path)
        if pattern.match(file_name):
            jsonl_files.append(file_path)
    
    if not jsonl_files:
        print(f'error: No target jsonl files found for shape {shape_str} in {input_dir}')
        print(f'Checked pattern: {base_pattern}')
        return
    
    print(f'Found {len(jsonl_files)} target files for shape {shape_str}:')
    for file in jsonl_files:
        print(f'- {file}')
    
    # 读取并合并所有文件数据
    all_data = []
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            line_data = json.loads(line)
                            parameters = line_data.pop('parameters', {})
                            for key, value in parameters.items():
                                line_data[key] = value
                            all_data.append(line_data)
                        except json.JSONDecodeError as e:
                            print(f'warning: parse {file_path} line {line_num} error:{e}: {line}')
        except Exception as e:
            print(f'error: failed to read {file_path}: {e}')
    
    if not all_data:
        print(f'warning: No data parsed from any jsonl files')
        return
    
    # 按idx升序排序
    try:
        all_data.sort(key=lambda x: x.get('idx', 0))
        print(f'Successfully sorted {len(all_data)} records by idx')
    except Exception as e:
        print(f'warning: failed to sort data by idx: {e}')
    
    # 转换为Excel
    try:
        df = pd.DataFrame(all_data)
        df.to_excel(output_xlsx_path, index=False, engine='openpyxl')
        print(f'Successfully merged {len(jsonl_files)} files to {output_xlsx_path}')
    except Exception as e:
        print(f'error: failed to convert merged data to Excel: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="待转换的jsonl文件路径")
    parser.add_argument("--output", type=str, help="转换后的xlsx文件路径") 
    parser.add_argument('--shape', type=str, help='要合并的shape')
    parser.add_argument("--type", type=str, help="param, result, merge")
    args = parser.parse_args()
    if args.type == "param":
        params_to_excel(args.input, args.output)   
    elif args.type == "result":
        result_to_excel(args.input, args.output)
    elif args.type == "merge":
        merge_jsonl_by_shape(args.input, args.shape, args.output) 
