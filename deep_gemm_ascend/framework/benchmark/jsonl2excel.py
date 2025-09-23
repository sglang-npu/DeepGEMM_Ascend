import pandas as pd
import json
import argparse

def jsonl_to_excel(input_jsonl_path, output_xlsx_path):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="待转换的jsonl文件路径")
    parser.add_argument("--output", type=str, help="转换后的xlsx文件路径") 
    args = parser.parse_args()
    jsonl_to_excel(args.input, args.output)   