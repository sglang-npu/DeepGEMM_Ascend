import os
import pandas as pd
import glob

#定义输入路径列表（可添加任意多个路径）
INPUT_PATHS = [
    r'D:\MyFiles\deepgemm\small_matmul_excel\v2_671\merged_excel',
    r'D:\MyFiles\deepgemm\small_matmul_excel\v3_671\merged_excel',
    r'D:\MyFiles\deepgemm\small_matmul_excel\v4_671\merged_excel',
    r'D:\MyFiles\deepgemm\small_matmul_excel\v5_671\merged_excel',
    r'D:\MyFiles\deepgemm\small_matmul_excel\v6_671\merged_excel',
]
OUTPUT_PATH = r'D:\MyFiles\deepgemm\merged_excel_all'

#创建输出目录（如果不存在）
os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_shape_files_all_paths(input_paths):
    """
    获取所有输入路径下的xlsx文件，以shape为键，文件路径列表为值
    :param input_paths: 输入路径列表
    :return: dict， key=shape名称， value=该shape对应的所有文件路径列表
    """
    shape_files = {}
    for input_path in input_paths:
        #检查路径是否存在
        if not os.path.exists(input_path):
            print(f"警告：路径{input_path}不存在，跳过该路径！")
            continue

        #匹配当前路径下所有xlsx文件
        xlsx_files = glob.glob(os.path.join(input_path, "*.xlsx"))
        for file in xlsx_files:
            #文件名（不含扩展名）即为shape
            shape = os.path.splitext(os.path.basename(file))[0]
            #若shape已存在则追加文件路径，否则新建列表
            if shape in shape_files:
                shape_files[shape].append(file)
            else:
                shape_files[shape]=[file]
    return shape_files

def merge_shape_data(shape, file_paths):
    """
    合并单个shape的多个文件数据
    :param shape: shape名称
    :param file_paths: 该shape对应的所有文件路径列表
    :return: 合并后的DataFrame
    """
    # 必要列检查
    required_cols = ['mTile', 'nTile', 'kTile', 'time']
    all_dfs = []

    #读取所有文件的数据
    for file in file_paths:
        try:
            df = pd.read_excel(file)
            # 检查必要列是否存在
            if not all(col in df.columns for col in required_cols):
                print(f"警告：文件{file}缺少必要的列，跳过该文件！")
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"错误：读取文件{file}失败 - {str(e)}， 跳过该文件！")
            continue
    
    # 无有效数据则返回空DataFrame
    if not all_dfs:
        return pd.DataFrame()

    #合并所有DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)

    #按Tiling分组，计算time的平均值
    merged_df = combined_df.groupby(
        ['mTile', 'nTile', 'kTile'], # 按Tiling组合分组
        as_index = False,
        dropna = False # 保留NA值的分组
    )['time'].mean()   # 计算相同Tiling的time平均值

    # 保留其他列的信息（取每组的第一行数据）
    other_cols = [col for col in combined_df.columns if col not in required_cols]
    if other_cols:
        other_data = combined_df.groupby(
            ['mTile', 'nTile', 'kTile'],
            as_index = False,
            dropna = False 
        )[other_cols].first()
        # 合并平均时间和其他列数据
        merged_df = pd.merge(merged_df, other_data, on=['mTile', 'nTile', 'kTile'], how='left')

    return merged_df

def main():
    #获取所有输入路径下的shape文件（key=shape， value=文件路径列表）
    shape_files = get_shape_files_all_paths(INPUT_PATHS)
    if not shape_files:
        print("错误：未找到任何有效的shape文件！")
        return

    #获取所有unique的shape
    all_shapes = sorted(shape_files.keys())
    print(f"找到{len(all_shapes)}个unique的shape需要处理...")

    #处理每个shape
    processed_count = 0
    for shape in all_shapes:
        file_paths = shape_files[shape]

        # 合并该shape的所有文件数据
        merged_df = merge_shape_data(shape, file_paths)

        if merged_df.empty:
            print(f"跳过shape {shape} （无有效数据）")
            continue
        
        #保存合并后的文件
        output_file = os.path.join(OUTPUT_PATH, f"{shape}.xlsx")
        merged_df.to_excel(output_file, index=False)
        processed_count += 1
        print(f"已处理shape {shape} （共{len(file_paths)}个文件）， 保存到{output_file}")

    # 统计输出文件数量
    output_files = glob.glob(os.path.join(OUTPUT_PATH, "*.xlsx"))
    print(f"\n合并完成! ")
    print(f"输出目录：{OUTPUT_PATH}")
    print(f"处理成功的shape数量：{processed_count}")
    print(f"输出文件总数：{len(output_files)}")

if __name__ == "__main__":
    main()