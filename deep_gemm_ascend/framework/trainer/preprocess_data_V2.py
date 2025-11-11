import pandas as pd
import os
import numpy as np
import glob

def align16(x):
    return ((x + 15) // 16) * 16
    
def process_excel_files(folder_path, output_prefix="processed_data_V2"):
    """
    处理文件夹中的所有Excel文件，生成模型训练数据
    修复表头读取问题：正确识别第一行为列名
    """
    # 定义需要保留的特征列和目标列（必须与Excel表头一致）
    original_features = [
        'M', 'N', 'K', 
        'm_sections', 'n_sections', 
        'm_sec_o_blocks', 'n_sec_o_blocks', 
        'k_o_iter_blocks', 'db_o_blocks'
    ]
    target_column = 'time'
    required_columns = original_features + [target_column] + ['idx', 'diff', 'negative']
    
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
            df = pd.read_excel(file, header=0)
            print(f"\n处理文件: {os.path.basename(file)}")
            
            # 检查当前文件是否包含所有必需的列
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"警告：文件缺少必要的列 {missing_cols}，已跳过该文件")
                continue
            
            # 基于get_bench_config函数扩展特征
            # 计算对齐后的块大小
            df['m_blocks'] = df['M'].apply(lambda x: align16(x) // 16)
            df['n_blocks'] = df['N'].apply(lambda x: align16(x) // 16)
            df['k_blocks'] = df['K'].apply(lambda x: align16(x) // 16)
            
            # 计算对齐偏移量
            df['m_o_fix'] = df['M'].apply(lambda x: align16(x) - x)
            df['n_o_fix'] = df['N'].apply(lambda x: align16(x) - x)
            df['k_o_fix'] = df['K'].apply(lambda x: align16(x) - x)
            
            # 计算db_o_num
            df['db_o_num'] = df.apply(lambda row: row['k_o_iter_blocks'] // row['db_o_blocks'], axis=1)
            
            # 计算余数块
            df['r_m_blocks'] = df.apply(lambda row: row['m_blocks'] % row['m_sec_o_blocks'], axis=1)
            df['r_n_blocks'] = df.apply(lambda row: row['n_blocks'] % row['n_sec_o_blocks'], axis=1)
            
            # 处理余数为0的情况
            df['r_m_blocks'] = df.apply(
                lambda row: row['m_sec_o_blocks'] if row['r_m_blocks'] == 0 else row['r_m_blocks'], 
                axis=1
            )
            df['r_n_blocks'] = df.apply(
                lambda row: row['n_sec_o_blocks'] if row['r_n_blocks'] == 0 else row['r_n_blocks'], 
                axis=1
            )
            
            # 计算k迭代次数和尾部块
            df['k_iters'] = df.apply(
                lambda row: (row['k_blocks'] + row['k_o_iter_blocks'] - 1) // row['k_o_iter_blocks'], 
                axis=1
            )
            df['k_tail_blocks'] = df.apply(lambda row: row['k_blocks'] % row['k_o_iter_blocks'], axis=1)
            
            # 计算剩余db_num和k_blocks
            def calculate_remaining(row):
                if row['k_tail_blocks'] == 0:
                    return row['db_o_num'], row['db_o_blocks']
                else:
                    r_db_num = (row['k_tail_blocks'] + row['db_o_blocks'] - 1) // row['db_o_blocks']
                    r_k_blocks = row['k_tail_blocks'] - ((r_db_num - 1) * row['db_o_blocks'])
                    return r_db_num, r_k_blocks
            
            df[['r_db_num', 'r_k_blocks']] = df.apply(
                lambda row: pd.Series(calculate_remaining(row)), 
                axis=1
            )
            
            # 计算m和n的迭代次数
            df['m_iters'] = df.apply(
                lambda row: (row['m_blocks'] + row['m_sec_o_blocks'] - 1) // row['m_sec_o_blocks'], 
                axis=1
            )
            df['n_iters'] = df.apply(
                lambda row: (row['n_blocks'] + row['n_sec_o_blocks'] - 1) // row['n_sec_o_blocks'], 
                axis=1
            )
            
            # 计算部分迭代次数
            df['m_parts'] = df.apply(lambda row: row['m_iters'] // row['m_sections'], axis=1)
            df['n_parts'] = df.apply(lambda row: row['n_iters'] // row['n_sections'], axis=1)
            
            # 计算sc块大小
            df['m_sc_blocks'] = df.apply(lambda row: row['m_parts'] * row['m_sec_o_blocks'], axis=1)
            df['n_sc_blocks'] = df.apply(lambda row: row['n_parts'] * row['n_sec_o_blocks'], axis=1)
            
            # 计算剩余部分
            df['r_m_parts'] = df.apply(
                lambda row: row['m_iters'] - ((row['m_sections'] - 1) * row['m_parts']), 
                axis=1
            )
            df['r_n_parts'] = df.apply(
                lambda row: row['n_iters'] - ((row['n_sections'] - 1) * row['n_parts']), 
                axis=1
            )
            
            # 添加到总数据中
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
    
    extended_features = original_features + [
        'm_blocks', 'n_blocks', 'k_blocks',
        'm_o_fix', 'n_o_fix', 'k_o_fix',
        'db_o_num',
        'r_m_blocks', 'r_n_blocks',
        'k_iters', 'k_tail_blocks',
        'r_db_num', 'r_k_blocks',
        'm_iters', 'n_iters',
        'm_parts', 'n_parts',
        'm_sc_blocks', 'n_sc_blocks',
        'r_m_parts', 'r_n_parts'
    ]
    
    # 分离特征和目标变量
    try:
        X = cleaned_data[extended_features].values  
        y = cleaned_data[target_column].values      
    except KeyError as e:
        print(f"数据中缺少必要的列: {str(e)}")
        print("请检查Excel文件的表头是否与要求一致")
        return
    
    # 保存为numpy格式
    np.savez(f"{output_prefix}.npz", X=X, y=y)
    print(f"已保存NumPy文件: {output_prefix}.npz")
    print(f"特征维度从 {len(original_features)} 扩展到 {len(extended_features)}")
    

if __name__ == "__main__":
    folder_path = "./raw_data"  # 默认文件夹路径
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建了文件夹: {folder_path}，请将Excel文件放入该文件夹")
    else:
        # 处理Excel文件
        process_excel_files(folder_path)
    