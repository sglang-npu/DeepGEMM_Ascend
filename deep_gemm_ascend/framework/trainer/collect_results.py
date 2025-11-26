import os
import re
import pandas as pd
import glob
from pathlib import Path

def extract_parameters_from_filename(filename):
    """从文件名中提取实验参数"""
    params = {}
    
    # 正则表达式匹配参数模式
    patterns = {
        'loss': r'loss=([^_]+)',
        'optimizer': r'opt=([^_]+)',
        'lr': r'lr=([\d.]+)',
        'hidden_dims': r'hidden=([\d,]+)',
        'batch_size': r'batch=(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            params[key] = match.group(1)
    
    return params

def parse_log_file(log_path):
    """解析log文件，提取训练结果和测试评估指标"""
    results = {}
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
            # 提取总训练轮次
            epoch_match = re.search(r'实际训练轮次: (\d+)', log_content)
            if epoch_match:
                results['total_epochs'] = int(epoch_match.group(1))
            
            # 提取最佳验证损失
            best_loss_match = re.search(r'最佳验证损失: ([\d.]+)', log_content)
            if best_loss_match:
                results['best_val_loss'] = float(best_loss_match.group(1))
            
            # 提取最佳验证损失所在轮次
            best_epoch_match = re.search(r'在第(\d+)轮', log_content)
            if best_epoch_match:
                results['best_epoch'] = int(best_epoch_match.group(1))
            
            # 提取总耗时
            time_match = re.search(r'总耗时: ([\d.]+)秒', log_content)
            if time_match:
                results['total_time_sec'] = float(time_match.group(1))
                
            # 提取使用的NPU
            npu_match = re.search(r'npu(\d+)_exp', os.path.basename(log_path))
            if npu_match:
                results['npu_id'] = int(npu_match.group(1))
            
            # 提取测试集评估指标
            # 处理后数据的测试损失
            test_loss_match = re.search(r'处理后数据的测试损失 \(.+\): ([\d.]+)', log_content)
            if test_loss_match:
                results['test_loss_processed'] = float(test_loss_match.group(1))
            
            # 原始时间尺度的MAE
            mae_match = re.search(r'原始时间尺度的MAE: ([\d.]+) us', log_content)
            if mae_match:
                results['mae_raw_us'] = float(mae_match.group(1))
            
            # 原始时间尺度的RMSE
            rmse_match = re.search(r'原始时间尺度的RMSE: ([\d.]+) us', log_content)
            if rmse_match:
                results['rmse_raw_us'] = float(rmse_match.group(1))
            
            # 平均相对误差
            rel_error_match = re.search(r'平均相对误差: ([\d.]+)%', log_content)
            if rel_error_match:
                results['rel_error_pct'] = float(rel_error_match.group(1))
                
    except Exception as e:
        print(f"解析日志文件 {log_path} 时出错: {str(e)}")
    
    return results

def collect_experiment_results(base_dir, output_file="experiment_results.xlsx"):
    """收集所有实验结果并保存到Excel，包括测试评估指标"""
    # 查找所有log文件
    log_files = glob.glob(os.path.join(base_dir, "**", "*_train.log"), recursive=True)
    print(f"找到 {len(log_files)} 个实验日志文件")
    
    # 准备存储结果的数据列表
    results_list = []
    
    # 遍历每个log文件
    for log_path in log_files:
        # 获取实验目录和文件名
        exp_dir = os.path.dirname(log_path)
        exp_filename = os.path.basename(log_path)
        
        # 提取参数
        params = extract_parameters_from_filename(exp_filename)
        
        # 解析log文件获取结果
        results = parse_log_file(log_path)
        
        # 合并参数和结果
        experiment_data = {
            'experiment_name': os.path.basename(exp_dir),
            'log_path': log_path,
            **params,** results
        }
        
        results_list.append(experiment_data)
    
    # 创建DataFrame
    df = pd.DataFrame(results_list)
    
    # 调整列顺序
    if not df.empty:
        # 确保关键列在前面，包括新增的评估指标
        key_columns = ['experiment_name', 'npu_id', 'loss', 'optimizer', 'lr', 
                      'hidden_dims', 'batch_size', 'best_val_loss', 
                      'test_loss_processed', 'mae_raw_us', 'rmse_raw_us', 
                      'rel_error_pct', 'best_epoch', 'total_epochs', 
                      'total_time_sec', 'log_path']
        
        # 剩下的列
        other_columns = [col for col in df.columns if col not in key_columns]
        
        # 重新排列列
        df = df[key_columns + other_columns]
        
        # 按最佳验证损失排序，也可以改为按mae_raw_us或rel_error_pct排序
        df = df.sort_values(by='best_val_loss')
    
    # 保存到Excel
    df.to_excel(output_file, index=False)
    print(f"实验结果已保存到 {output_file}")
    
    return df

if __name__ == "__main__":
    # 实验结果根目录（与训练脚本中的BASE_OUTPUT_DIR一致）
    BASE_OUTPUT_DIR = "./exp_results"
    
    # 输出Excel文件名
    OUTPUT_FILE = "experiment_results_summary.xlsx"
    
    # 收集并生成结果
    results_df = collect_experiment_results(BASE_OUTPUT_DIR, OUTPUT_FILE)
    
    # 打印一些统计信息
    if not results_df.empty:
        print("\n统计信息:")
        print(f"总实验数: {len(results_df)}")
        print(f"最佳验证损失: {results_df['best_val_loss'].min():.6f}")
        print(f"最小MAE (原始尺度): {results_df['mae_raw_us'].min():.4f} us")
        print(f"最小RMSE (原始尺度): {results_df['rmse_raw_us'].min():.4f} us")
        print(f"最小相对误差: {results_df['rel_error_pct'].min():.2f}%")
        print(f"平均训练时间: {results_df['total_time_sec'].mean():.2f}秒")
    else:
        print("未找到任何实验结果")
    