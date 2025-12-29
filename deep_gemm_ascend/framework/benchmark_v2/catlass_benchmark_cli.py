"""
Catlass GEMM基准测试主入口

提供命令行接口和模块组装功能
"""

import argparse

from .file_io import load_shapes_from_excel, default_shape_group, prepare_shapes_with_qwen3
from .distributed_benchmark_runner import GEMMBenchmarkRunner


def main():
    """主函数：解析参数、组装依赖、启动测试"""
    parser = argparse.ArgumentParser(
        usage='%(prog)s --rank_id [num] --process_num [num] --npu_ids [ids] [--operator_type TYPE]'
    )
    parser.add_argument('--rank_id', required=True, type=int, help='进程的rank ID，从0开始')
    parser.add_argument('--process_num', required=True, type=int, help='总进程数')
    parser.add_argument('--npu_ids', type=str, required=True,
                       help='NPU设备ID列表，用逗号分隔（如: 0,1,2,3），数量必须与process_num一致')
    parser.add_argument('--catlass_bin_path', type=str, default="/home/q30063557/code/catlass/21_dynamic_tiling_matmul_with_layout",
                       help='catlass可执行文件路径')
    parser.add_argument('--result_dir', type=str, default="./catlass_results",
                       help='结果保存目录，默认: ./catlass_results')
    parser.add_argument('--msp_dir', type=str, default="./catlass_msp",
                       help='msprof输出目录，默认: ./catlass_msp')
    parser.add_argument('--operator_type', type=str, required=True,
                       choices=['SmallMatmulKernel', 'CommonMatmulKernel', 'PaddingCommonMatmulKernel', 
                                'PaddingMultiCoreSplitkMatmulKernel', 'PaddingStreamkMatmulKernel'],
                       help='算子类型，必须提供: SmallMatmulKernel, CommonMatmulKernel, '
                            'PaddingCommonMatmulKernel, PaddingMultiCoreSplitkMatmulKernel, PaddingStreamkMatmulKernel')
    parser.add_argument('--core_num', type=int, default=20,
                       help='AI Core数量，默认20')
    parser.add_argument('--shapes_file', type=str, default=None,
                       help='shapes.xlsx文件路径，如果提供则从文件读取shape，否则使用默认shape_group')
    parser.add_argument('--layout_tag_a', type=int, required=True,
                       help='Layout A标签，必须提供: 0=RowMajor, 1=ColumnMajor')
    parser.add_argument('--layout_tag_b', type=int, required=True,
                       help='Layout B标签，必须提供: 0=RowMajor, 1=ColumnMajor')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='对筛选后的shapes进行切片的起始位置（从0开始，包含该位置）')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='对筛选后的shapes进行切片的结束位置（不包含该位置，None表示到末尾）')
    args = parser.parse_args()
    
    # 解析NPU ID列表
    try:
        npu_ids = [int(x.strip()) for x in args.npu_ids.split(',')]
    except ValueError:
        parser.error(f"无效的NPU ID列表格式: {args.npu_ids}。请使用逗号分隔的整数，如: 0,1,2,3")
    
    # 验证NPU ID数量与进程数量一致
    if len(npu_ids) != args.process_num:
        parser.error(f"NPU ID数量 ({len(npu_ids)}) 与进程数量 ({args.process_num}) 不一致")
    
    # 获取当前进程对应的NPU ID
    if args.rank_id < 0 or args.rank_id >= len(npu_ids):
        parser.error(f"rank_id ({args.rank_id}) 超出范围 [0, {len(npu_ids)-1}]")
    npu_id = npu_ids[args.rank_id]
    
    # 根据是否提供shapes_file来决定使用哪个shape_group
    if args.shapes_file:
        print(f"=====Loading shapes from {args.shapes_file}=====")
        shape_group = load_shapes_from_excel(
            args.shapes_file,
            operator_name=args.operator_type,
            layout_tag_a=args.layout_tag_a,
            layout_tag_b=args.layout_tag_b,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
    else:
        print(f"=====Using default shape_group=====")
        # 对default_shape_group也进行打乱、插入qwen3_shapes和切片处理
        shape_group = prepare_shapes_with_qwen3(default_shape_group, args.start_idx, args.end_idx)
    
    print(f"=====STARTING GEMM BENCHMARK (Rank {args.rank_id}/{args.process_num}, NPU ID: {npu_id})=====")
    print(f"Total shapes to test: {len(shape_group)}")
    print(f"NPU IDs: {npu_ids}")
    print(f"Operator type: {args.operator_type}")
    print(f"Core number: {args.core_num}")
    print(f"Layout A: {args.layout_tag_a} ({'RowMajor' if args.layout_tag_a == 0 else 'ColumnMajor'})")
    print(f"Layout B: {args.layout_tag_b} ({'RowMajor' if args.layout_tag_b == 0 else 'ColumnMajor'})")
    
    # 使用命令行参数中的layout值
    layout_a = args.layout_tag_a
    layout_b = args.layout_tag_b
    
    # 运行完整基准测试
    runner = GEMMBenchmarkRunner(
        shape_group,
        rank_id=args.rank_id,
        npu_ids=npu_ids,
        num_processes=args.process_num,
        catlass_bin_path=args.catlass_bin_path,
        result_dir=args.result_dir,
        msp_dir=args.msp_dir,
        operator_type=args.operator_type,
        core_num=args.core_num,
        layout_tag_a=layout_a,
        layout_tag_b=layout_b
    )
    runner.run_benchmarks()    
    print(f"=====GEMM BENCHMARK FINISHED (Rank {args.rank_id}, NPU ID: {npu_id})=====")


if __name__ == "__main__":
    main()
