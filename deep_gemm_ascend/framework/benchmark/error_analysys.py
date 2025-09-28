import os
import json
import sys
import glob
from collections import defaultdict

# 尝试导入pandas和openpyxl，若缺失则提示用户安装
try:
    import pandas as pd
    from openpyxl import Workbook
except ImportError:
    print("错误：导出Excel功能需要pandas和openpyxl库，请先安装：")
    print("pip install pandas openpyxl")
    sys.exit(1)

def process_shape_files(directory, output_excel="shape_statistics.xlsx"):
    """
    处理指定目录下的shape jsonl文件，按shape统计各类情况并导出到Excel
    
    参数:
        directory: 包含jsonl文件的目录
        output_excel: 导出的Excel文件名或路径
    """
    # 验证目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在或不是一个有效的目录")
        return
    
    # 存储统计结果的结构
    shape_stats = defaultdict(lambda: {
        'total': 0,
        'normal': 0,
        'operator_error': 0,
        'precision_error': 0,
        'ms_prof_error': 0,
        'seen_idx': set()
    })
    
    # 查找所有符合条件的jsonl文件，排除checkpoint文件
    pattern = os.path.join(directory, 'shape_*_rank_*.jsonl')
    checkpoint_pattern = os.path.join(directory, 'shape_*_rank_*_checkpoint.jsonl')
    
    all_files = set(glob.glob(pattern))
    checkpoint_files = set(glob.glob(checkpoint_pattern))
    target_files = all_files - checkpoint_files
    
    print(f"在目录 '{directory}' 中找到 {len(target_files)} 个目标文件，开始处理...")
    
    for file_path in target_files:
        # 从文件名中提取shape_str
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        if len(parts) >= 3 and parts[0] == 'shape' and 'rank' in parts:
            rank_index = parts.index('rank')
            shape_str = '_'.join(parts[1:rank_index])
        else:
            print(f"警告：文件名 {file_name} 格式不符合预期，跳过处理")
            continue
        
        # 处理文件中的每条数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查是否包含必要的key
                    required_keys = ['idx', 'M', 'N', 'K', 'time', 'diff', 'negative', 'parameters']
                    if not all(key in data for key in required_keys):
                        print(f"警告：文件 {file_name} 第 {line_num} 行缺少必要的key，跳过")
                        continue
                    
                    idx = data['idx']
                    time = data['time']
                    
                    # 只处理第一次出现的idx
                    if idx not in shape_stats[shape_str]['seen_idx']:
                        shape_stats[shape_str]['seen_idx'].add(idx)
                        shape_stats[shape_str]['total'] += 1
                        
                        # 根据time值进行分类统计
                        if time == -1:
                            shape_stats[shape_str]['operator_error'] += 1
                        elif time == float('inf') or str(time).lower() == 'infinity':
                            shape_stats[shape_str]['precision_error'] += 1
                        elif time == 999999999:
                            shape_stats[shape_str]['ms_prof_error'] += 1
                        else:
                            shape_stats[shape_str]['normal'] += 1
                            
                except json.JSONDecodeError:
                    print(f"警告：文件 {file_name} 第 {line_num} 行不是有效的JSON，跳过")
                except Exception as e:
                    print(f"处理文件 {file_name} 第 {line_num} 行时出错：{str(e)}，跳过")
    
    # 准备导出到Excel的数据
    excel_data = []
    for shape_str in sorted(shape_stats.keys()):
        stats = shape_stats[shape_str]
        total = stats['total']
        
        if total == 0:
            continue
            
        # 计算各类情况的比例
        normal_ratio = stats['normal'] / total * 100 if total > 0 else 0
        operator_error_ratio = stats['operator_error'] / total * 100 if total > 0 else 0
        precision_error_ratio = stats['precision_error'] / total * 100 if total > 0 else 0
        ms_prof_error_ratio = stats['ms_prof_error'] / total * 100 if total > 0 else 0
        
        # 添加到Excel数据列表
        excel_data.append({
            'Shape': shape_str,
            '总条数': total,
            '正常情况数量': stats['normal'],
            '正常情况比例(%)': round(normal_ratio, 2),
            '算子执行异常数量(-1)': stats['operator_error'],
            '算子执行异常比例(%)': round(operator_error_ratio, 2),
            '精度异常数量(Infinity)': stats['precision_error'],
            '精度异常比例(%)': round(precision_error_ratio, 2),
            'ms_prof执行异常数量(999999999)': stats['ms_prof_error'],
            'ms_prof执行异常比例(%)': round(ms_prof_error_ratio, 2)
        })
    
    # 导出到Excel
    if excel_data:
        df = pd.DataFrame(excel_data)
        # 调整列的顺序
        columns_order = [
            'Shape', '总条数', 
            '正常情况数量', '正常情况比例(%)',
            '算子执行异常数量(-1)', '算子执行异常比例(%)',
            '精度异常数量(Infinity)', '精度异常比例(%)',
            'ms_prof执行异常数量(999999999)', 'ms_prof执行异常比例(%)'
        ]
        df = df[columns_order]
        
        try:
            df.to_excel(output_excel, index=False, engine='openpyxl')
            print(f"\n统计结果已成功导出到Excel文件: {os.path.abspath(output_excel)}")
        except Exception as e:
            print(f"\n导出Excel文件时出错: {str(e)}")
    else:
        print("\n没有可导出的统计数据")
    
    # 输出控制台结果
    print("\n" + "="*50)
    print("统计结果：")
    print("="*50)
    
    for item in excel_data:
        print(f"\nShape: {item['Shape']}")
        print(f"  总条数: {item['总条数']}")
        print(f"  正常情况: {item['正常情况数量']} ({item['正常情况比例(%)']}%)")
        print(f"  算子执行异常(-1): {item['算子执行异常数量(-1)']} ({item['算子执行异常比例(%)']}%)")
        print(f"  精度异常(Infinity): {item['精度异常数量(Infinity)']} ({item['精度异常比例(%)']}%)")
        print(f"  ms_prof执行异常(999999999): {item['ms_prof执行异常数量(999999999)']} ({item['ms_prof执行异常比例(%)']}%)")
    
    return shape_stats

def print_help():
    """打印帮助信息"""
    print("""
    用法: python shape_statistics.py [选项] <results路径> [输出Excel路径]
    
    功能: 分析指定路径下的shape jsonl文件，按shape统计各类情况的条数及比例，并导出到Excel
    
    选项:
        -h, --help    显示此帮助信息并退出
    
    参数:
        <results路径>      包含jsonl文件的目录路径
        [输出Excel路径]   可选，指定导出的Excel文件路径，默认为当前目录的shape_statistics.xlsx
    
    示例:
        python shape_statistics.py ./results
        python shape_statistics.py /path/to/results ./output.xlsx
    """)

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print("错误：请指定要分析的results路径")
        print_help()
        sys.exit(1)
    
    # 获取路径参数
    results_path = sys.argv[1]
    output_excel = sys.argv[2] if len(sys.argv) > 2 else "shape_statistics.xlsx"
    
    process_shape_files(results_path, output_excel)
