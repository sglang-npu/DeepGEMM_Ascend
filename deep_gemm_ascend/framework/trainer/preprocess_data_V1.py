import pandas as pd
import os
import numpy as np
import glob

def process_excel_files(folder_path, output_prefix="processed_data_V1"):
    """
    处理文件夹中的所有Excel文件，生成模型训练数据
    修复表头读取问题：正确识别第一行为列名
    """
    # 定义需要保留的特征列和目标列（必须与Excel表头一致）
    feature_columns = [
        'M', 'N', 'K', 
        'm_sections', 'n_sections', 
        'm_sec_o_blocks', 'n_sec_o_blocks', 
        'k_o_iter_blocks', 'db_o_blocks'
    ]
    target_column = 'time'
    required_columns = feature_columns + [target_column] + ['idx', 'diff', 'negative']
    
    # 初始化一个空DataFrame用于存储所有数据
    all_data = pd.DataFrame()
    
    # 获取文件夹中所有xlsx文件
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    if not excel_files:
        print("未找到任何Excel文件，请检查文件夹路径")
        return
    
    # 逐个读取并处理Excel文件
    for file in excel_files:
        try:
            # 关键修改：不跳过表头，第一行作为列名（header=0是默认值，可省略）
            df = pd.read_excel(file, header=0)
            print(f"\n处理文件: {os.path.basename(file)}")
            print(f"文件原始列名: {list(df.columns)}")
            
            # 检查当前文件是否包含所有必需的列
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"警告：文件缺少必要的列 {missing_cols}，已跳过该文件")
                continue
            
            # 将当前文件数据添加到总数据中
            all_data = pd.concat([all_data, df], ignore_index=True)
            print(f"成功加载 {len(df)} 条数据，累计数据量: {len(all_data)}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(file)} 时出错: {str(e)}")
    
    if len(all_data) == 0:
        print("没有加载到有效数据，程序退出")
        return
    
    print(f"\n数据聚合完成，共 {len(all_data)} 条记录")
    
    # 数据清洗：移除time为'inf'、999999999或-1的记录
    print("开始数据清洗...")
    initial_count = len(all_data)
    
    # 处理不同类型的无效值
    # 先将time列转换为字符串处理，避免类型问题
    all_data['time_str'] = all_data['time'].astype(str).str.lower()
    
    # 筛选出有效的记录
    valid_mask = ~(
        (all_data['time_str'] == 'inf') | 
        (all_data['time'] == 999999999) | 
        (all_data['time'] == -1)
    )
    
    cleaned_data = all_data[valid_mask].copy()
    
    # 删除临时列
    cleaned_data = cleaned_data.drop('time_str', axis=1)
    
    # 计算清洗后的数据量
    cleaned_count = len(cleaned_data)
    removed_count = initial_count - cleaned_count
    print(f"数据清洗完成，移除了 {removed_count} 条无效记录，保留 {cleaned_count} 条有效记录")
    
    # 检查是否有足够的数据
    if cleaned_count == 0:
        print("清洗后没有保留任何数据，无法生成训练文件")
        return
    
    # 分离特征和目标变量
    try:
        X = cleaned_data[feature_columns].values  # 输入特征
        y = cleaned_data[target_column].values    # 输出目标
    except KeyError as e:
        print(f"数据中缺少必要的列: {str(e)}")
        print("请检查Excel文件的表头是否与要求一致")
        return
    
    
    # 保存为numpy格式（分开保存特征和目标）
    np.savez(f"{output_prefix}.npz", X=X, y=y)
    print(f"已保存NumPy文件: {output_prefix}.npz")
    

if __name__ == "__main__":
    folder_path = "./raw_data"  # 默认文件夹路径
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建了文件夹: {folder_path}，请将Excel文件放入该文件夹")
    else:
        # 处理Excel文件
        process_excel_files(folder_path)
    