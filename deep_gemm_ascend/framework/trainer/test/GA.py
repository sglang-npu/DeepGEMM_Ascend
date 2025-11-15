import numpy as np
import random  # 替换numpy随机，增强多样性
import pandas as pd
import argparse
import time  # 添加时间模块用于统计执行时间
from deap import base, creator, tools
from test_om import OMModelInference  # 确保test_om中OMModelInference正常

# --------------------------
# 1. 基础配置
# --------------------------
FIXED_M = 1279
FIXED_N = 5003
FIXED_K = 7681

OPTIMIZE_PARAMS_NAME = [
    "m_sections", "n_sections", 
    "m_sec_o_blocks", "n_sec_o_blocks", 
    "k_o_iter_blocks", "db_o_blocks"
]
ALLOWED_VALUES = {
    "m_sections": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24],
    "n_sections": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24],
    "m_sec_o_blocks": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
    "n_sec_o_blocks": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
    "k_o_iter_blocks": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "db_o_blocks": [1, 2, 4, 8, 16, 32, 64]
}

# --------------------------
# 2. 合法性校验函数
# --------------------------
def is_optimize_params_valid(optimize_params):
    if len(optimize_params) != 6:
        return False
    
    m_sec, n_sec, m_sob, n_sob, k_oib, db_ob = optimize_params
    
    # 可选值校验
    if m_sec not in ALLOWED_VALUES["m_sections"]:
        return False
    if n_sec not in ALLOWED_VALUES["n_sections"]:
        return False
    if m_sob not in ALLOWED_VALUES["m_sec_o_blocks"]:
        return False
    if n_sob not in ALLOWED_VALUES["n_sec_o_blocks"]:
        return False
    if k_oib not in ALLOWED_VALUES["k_o_iter_blocks"]:
        return False
    if db_ob not in ALLOWED_VALUES["db_o_blocks"]:
        return False
    
    # 业务规则校验
    if m_sec * n_sec > 24:
        return False
    if m_sob * n_sob > 128:
        return False
    sum_mn = m_sob + n_sob
    if sum_mn == 0 or k_oib >= 1024 / (sum_mn * 2):
        return False
    if m_sob == 0 or n_sob == 0:
        return False
    max_db = min(128 // m_sob, 128 // n_sob, k_oib)
    if db_ob > max_db:
        return False
    
    return True


# --------------------------
# 3. DEAP配置（含进化监控）
# --------------------------
# 先删除已存在的creator
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

# 定义适应度和个体类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", tuple, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 注册参数生成函数（用random.choice增强多样性）
for param_name in OPTIMIZE_PARAMS_NAME:
    toolbox.register(
        f"attr_{param_name}",
        random.choice,  # 替换numpy随机，避免种子固定导致重复
        ALLOWED_VALUES[param_name]
    )

# 注册个体生成函数（带日志）
def create_individual():
    while True:
        params_tuple = (
            toolbox.attr_m_sections(),
            toolbox.attr_n_sections(),
            toolbox.attr_m_sec_o_blocks(),
            toolbox.attr_n_sec_o_blocks(),
            toolbox.attr_k_o_iter_blocks(),
            toolbox.attr_db_o_blocks()
        )
        if is_optimize_params_valid(params_tuple):
            ind = creator.Individual(params_tuple)
            # 仅初始化时打印前5个个体，避免日志过多
            # if not hasattr(create_individual, "init_count"):
            #     create_individual.init_count = 0
            # if create_individual.init_count < 5:
            #     print(f"初始化个体: {ind}")
            #     create_individual.init_count += 1
            return ind

toolbox.register("individual", create_individual)

# 注册种群生成函数
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 批量评估函数（带详细日志）
def batch_evaluate(individuals, om_inferencer, step_name=""):
    # 组装输入
    batch_input = [
        [FIXED_M, FIXED_N, FIXED_K] + list(ind) 
        for ind in individuals
    ]
    
    # 模型预测（带校验）
    batch_time = om_inferencer.predict(batch_input)
    if batch_time is None:
        raise ValueError(f"{step_name}评估失败：模型返回None")
    if len(batch_time) != len(individuals):
        raise ValueError(
            f"{step_name}评估维度不匹配：输入{len(individuals)}个，返回{len(batch_time)}个"
        )
    
    # 赋值适应度并打印关键日志
    for i, (ind, time) in enumerate(zip(individuals, batch_time)):
        if time is not None and time >= 0:
            ind.fitness.values = (time,)
        else:
            ind.fitness.values = (1e18,)  # 无效个体赋极大值
        # 每代打印前3个个体的适应度
        # if i < 3:
        #     print(f"{step_name}个体{i}：{ind} → time={ind.fitness.values[0]:.6f}")
    
    # 打印适应度分布（监控多样性）
    times = [ind.fitness.values[0] for ind in individuals]
    # print(f"{step_name}适应度分布：min={min(times):.6f}, max={max(times):.6f}, avg={np.mean(times):.6f}")
    return times  # 返回用于后续分析

# 支持动态shape的批量评估函数
def batch_evaluate_with_shape(individuals, om_inferencer, m, n, k, step_name=""):
    # 组装输入
    batch_input = [
        [m, n, k] + list(ind) 
        for ind in individuals
    ]
    
    # 模型预测（带校验）
    batch_time = om_inferencer.predict(batch_input)
    if batch_time is None:
        raise ValueError(f"{step_name}评估失败：模型返回None")
    if len(batch_time) != len(individuals):
        raise ValueError(
            f"{step_name}评估维度不匹配：输入{len(individuals)}个，返回{len(batch_time)}个"
        )
    
    # 赋值适应度并打印关键日志
    for i, (ind, time) in enumerate(zip(individuals, batch_time)):
        if time is not None and time >= 0:
            ind.fitness.values = (time,)
        else:
            ind.fitness.values = (1e18,)  # 无效个体赋极大值
        # 每代打印前3个个体的适应度
        # if i < 3:
        #     print(f"{step_name}个体{i}：{ind} → time={ind.fitness.values[0]:.6f}")
    
    # 打印适应度分布（监控多样性）
    times = [ind.fitness.values[0] for ind in individuals]
    # print(f"{step_name}适应度分布：min={min(times):.6f}, max={max(times):.6f}, avg={np.mean(times):.6f}")
    return times  # 返回用于后续分析

# 选择函数（替换为锦标赛选择，增强筛选能力）
toolbox.register("select", tools.selTournament, tournsize=3)

# 交叉函数（带日志）
def cx_valid(ind1, ind2):
    if not (isinstance(ind1, creator.Individual) and isinstance(ind2, creator.Individual)):
        raise TypeError("交叉对象必须是Individual类型")
    
    # 单点交叉
    offspring1_list, offspring2_list = tools.cxOnePoint(list(ind1), list(ind2))
    offspring1_tuple = tuple(offspring1_list)
    offspring2_tuple = tuple(offspring2_list)
    
    # 合法性校验+替换
    if is_optimize_params_valid(offspring1_tuple):
        offspring1 = creator.Individual(offspring1_tuple)
    else:
        offspring1 = toolbox.individual()
        # print(f"交叉后代1不合法，替换为：{offspring1}")
    
    if is_optimize_params_valid(offspring2_tuple):
        offspring2 = creator.Individual(offspring2_tuple)
    else:
        offspring2 = toolbox.individual()
        # print(f"交叉后代2不合法，替换为：{offspring2}")
    
    # 清空适应度
    del offspring1.fitness.values
    del offspring2.fitness.values
    return offspring1, offspring2

toolbox.register("mate", cx_valid)

# 变异函数（提高变异概率至0.1，增强多样性）
def mutate_valid(individual, indpb=0.1):
    if not isinstance(individual, creator.Individual):
        raise TypeError("变异对象必须是Individual类型")
    
    mutated_list = list(individual)
    for i, param_name in enumerate(OPTIMIZE_PARAMS_NAME):
        if random.random() < indpb:  # 用random增强随机性
            current_val = mutated_list[i]
            other_values = [v for v in ALLOWED_VALUES[param_name] if v != current_val]
            if other_values:
                mutated_list[i] = random.choice(other_values)
    
    mutated_tuple = tuple(mutated_list)
    # 合法性校验+替换
    if is_optimize_params_valid(mutated_tuple):
        mutated_ind = creator.Individual(mutated_tuple)
        # if mutated_ind != individual:
            # print(f"变异成功：{individual} → {mutated_ind}")
    else:
        mutated_ind = toolbox.individual()
        # print(f"变异后代不合法，替换为：{mutated_ind}")
    
    del mutated_ind.fitness.values
    return (mutated_ind,)

toolbox.register("mutate", mutate_valid, indpb=0.1)


# --------------------------
# 4. 主运行函数（带进化监控）
# --------------------------
def run_ga(
    pop_size=50,
    n_generations=40,
    cxpb=0.8,
    mutpb=0.1,
    om_inferencer=None,
    fixed_m=None,
    fixed_n=None,
    fixed_k=None
):
    # 记录算法开始时间
    ga_start_time = time.perf_counter()
    
    # 使用传入的shape参数，如果没有传入则使用全局默认值
    m = fixed_m if fixed_m is not None else FIXED_M
    n = fixed_n if fixed_n is not None else FIXED_N
    k = fixed_k if fixed_k is not None else FIXED_K
    
    # 初始化种群
    pop = toolbox.population(n=pop_size)
    # print(f"\n=== 初始化种群完成（规模{pop_size}）===")

    # 初始评估
    # print("\n=== 初始种群评估 ===")
    batch_evaluate_with_shape(pop, om_inferencer, m, n, k, step_name="初始")

    # 记录全局最优
    best_ind = tools.selBest(pop, 1)[0]
    best_time = best_ind.fitness.values[0]
    # print(f"初始全局最优：{best_ind} → time={best_time:.6f}")

    # 迭代主循环
    for gen in range(n_generations):
        # print(f"\n\n===== 第{gen+1}/{n_generations}代 =====")
        
        # 1. 选择：精英保留+锦标赛选择
        elite_size = max(1, int(pop_size * 0.1))  # 至少保留1个精英
        elite = tools.selBest(pop, elite_size)
        # print(f"选择精英（{elite_size}个）：{[ind.fitness.values[0] for ind in elite]}")
        
        # 选择非精英个体
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, pop_size - elite_size)]
        # print(f"选择非精英个体（{len(offspring)}个）")

        # 2. 交叉
        # print("\n=== 交叉操作 ===")
        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            parent1 = offspring[i]
            parent2 = offspring[i + 1]
            if random.random() < cxpb:
                # print(f"交叉父代：{parent1} × {parent2}")
                child1, child2 = toolbox.mate(parent1, parent2)
                offspring[i] = child1
                offspring[i + 1] = child2

        # 3. 变异
        # print("\n=== 变异操作 ===")
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])

        # 4. 评估新一代（非精英个体）
        # print("\n=== 新一代评估 ===")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print(f"需评估新个体数量：{len(invalid_ind)}")
        batch_evaluate_with_shape(invalid_ind, om_inferencer, m, n, k, step_name=f"第{gen+1}代")

        # 5. 组装新一代种群
        pop = elite + offspring

        # 6. 更新全局最优
        current_best = tools.selBest(pop, 1)[0]
        current_best_time = current_best.fitness.values[0]
        if current_best_time < best_time:
            best_ind = toolbox.clone(current_best)
            best_time = current_best_time
            # print(f"【更新全局最优】{best_ind} → time={best_time:.6f}")
        # else:
        #     print(f"当前最优未更新：{current_best_time:.6f}（全局最优：{best_time:.6f}）")

    # 记录算法结束时间并计算执行时间
    ga_end_time = time.perf_counter()
    ga_execution_time = ga_end_time - ga_start_time
    
    # 输出最终结果
    print("\n" + "="*60)
    print("优化完成！全局最优参数：")
    for name, val in zip(OPTIMIZE_PARAMS_NAME, best_ind):
        print(f"  {name}: {val}")
    print(f"全局最小time：{best_time:.6f}")
    print(f"算法执行时间：{ga_execution_time:.4f}秒")
    print("="*60)

    return dict(zip(OPTIMIZE_PARAMS_NAME, best_ind)), best_time, ga_execution_time


# --------------------------
# 排名比较函数（与annealer_V2保持一致）
# --------------------------
def compare_with_excel(excel_path, best_params):
    """
    与annealer_V2.py中的compare_with_excel函数保持完全一致
    计算GA找到的最优参数在Excel中的排名
    """
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"读取Excel失败：{e}")
        return 0, 0, 0.0, 0

    # 1. 检查Excel是否包含所有必要列（6个参数列+time列）
    required_cols = OPTIMIZE_PARAMS_NAME + ["time"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Excel文件缺少必要列：{missing_cols}")
        return 0, 0, 0.0, 0

    # 2. 转换参数列为int类型（避免Excel数值为float导致匹配失败）
    for col in OPTIMIZE_PARAMS_NAME:
        # 先转换为数值，无效值设为NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 只保留正整数，其他设为NaN
        df[col] = df[col].where((df[col] > 0) & (df[col] == df[col].astype(int)), np.nan)
        # 转换为整数，NaN保持为NaN
        df[col] = df[col].astype('Int64')  # 使用可空整数类型

    # 3. 筛选有效数据（时间有效且所有参数列都是正整数）
    time_valid = df["time"].notna() & df["time"].apply(lambda x: isinstance(x, (int, float)))
    params_valid = df[OPTIMIZE_PARAMS_NAME].notna().all(axis=1)  # 所有参数列都不为NaN
    valid_df = df[time_valid & params_valid]
    total_valid = len(valid_df)
    if total_valid == 0:
        print("Excel中无有效数据（时间或参数无效）")
        return 0, 0, 0.0, 0

    # 4. 用最优参数匹配Excel中的对应行
    # 确保best_params中的值都是整数类型
    # best_params可能是字典或列表，需要统一处理
    if isinstance(best_params, dict):
        # 如果已经是字典，直接使用
        param_dict = {name: int(val) for name, val in best_params.items()}
    else:
        # 如果是列表，转换为字典
        param_dict = {}
        for name, val in zip(OPTIMIZE_PARAMS_NAME, best_params):
            param_dict[name] = int(val)  # 强制转换为整数
    
    matched_df = valid_df.copy()
    # 按每个参数列筛选
    for col, val in param_dict.items():
        matched_df = matched_df[matched_df[col] == val]

    # 5. 处理匹配结果
    if len(matched_df) == 0:
        print("未在Excel中找到与最优Tiling参数匹配的记录")
        return 0, total_valid, 0.0, total_valid + 1  # 无匹配时排名为最后（总有效数+1）
    
    # 取第一个匹配行的实际时间（处理可能的重复记录）
    actual_best_time = matched_df.iloc[0]["time"]
    if actual_best_time < 0:
        print("匹配到的记录时间为无效负值")
        return 0, total_valid, 0.0, total_valid + 1

    # 6. 基于实际时间计算排名和打败比例
    beaten_count = len(valid_df[valid_df["time"] > actual_best_time])
    beaten_ratio = beaten_count / total_valid if total_valid > 0 else 0.0
    actual_rank = total_valid - beaten_count + 1  # 排名规则：总数 - 打败数量 + 1

    return beaten_count, total_valid, beaten_ratio, actual_rank


# --------------------------
# 启动优化
# --------------------------
# 定义所有要测试的shape
SHAPE_GROUP = [
    [4096, 4096, 7168],
    [8, 7168, 18432],
    [8, 18432, 7168],
    [64, 4096, 7168],
    [64, 7168, 18432],
    [64, 18432, 7168],
    [64, 24576, 1536],
    [64, 32768, 512],
    [64, 7168, 16384],
    [128, 4096, 7168],
    [128, 7168, 18432],
    [128, 18432, 7168],
    [1024, 4096, 7168],
    [1024, 18432, 7168],
    [2048, 4096, 7168],
    [1279, 5003, 7681],
    [3511, 6151, 8191],
    [5119, 6997, 9901]
]

def run_single_shape(shape, args, om_inferencer):
    """运行单个shape的GA优化"""
    m, n, k = shape
    shape_str = f"{m}_{n}_{k}"
    excel_file_path = f"../raw_data/shape_{shape_str}.xlsx"
    
    print(f"\n{'='*80}")
    print(f"开始处理Shape: {shape_str} (M={m}, N={n}, K={k})")
    print(f"Excel文件路径: {excel_file_path}")
    print(f"{'='*80}")
    
    # 统计变量
    sum_beaten_ratio = 0
    all_rank = 0
    all_run_time = 0
    all_execution_time = 0  # 新增：统计算法执行时间
    results = []
    
    # 多轮运行
    for epoch in range(args.epochs):
        print(f"\n===== 第 {epoch+1}/{args.epochs} 轮GA优化 =====")
        
        # 运行遗传算法
        best_params, min_time, execution_time = run_ga(
            pop_size=args.pop_size,
            n_generations=args.n_generations,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            om_inferencer=om_inferencer,
            fixed_m=m,
            fixed_n=n,
            fixed_k=k
        )
        
        # 计算排名
        beaten_count, total_valid, beaten_ratio, actual_rank = compare_with_excel(excel_file_path, best_params)
        
        # 累加统计值
        all_rank += actual_rank
        sum_beaten_ratio += beaten_ratio
        all_run_time += min_time
        all_execution_time += execution_time  # 累加执行时间
        
        # 记录单轮结果
        results.append({
            'epoch': epoch + 1,
            'rank': actual_rank,
            'beaten_ratio': beaten_ratio,
            'time': min_time,
            'execution_time': execution_time,  # 新增：记录执行时间
            'params': best_params
        })
        
        print(f"第{epoch+1}轮结果: 排名={actual_rank}, 打败比例={beaten_ratio:.4f}, 时间={min_time:.6f}, 执行时间={execution_time:.4f}秒")
    
    # 计算该shape的统计结果
    avg_rank = all_rank / args.epochs
    avg_beaten_ratio = sum_beaten_ratio / args.epochs
    avg_time = all_run_time / args.epochs
    avg_execution_time = all_execution_time / args.epochs  # 计算平均执行时间
    
    print(f"\n===== Shape {shape_str} 统计结果 =====")
    print(f"平均排名: {avg_rank:.1f}")
    print(f"平均打败benchmark: {avg_beaten_ratio:.4f}% 的Tiling参数组合")
    print(f"平均运行时间: {avg_time:.6f}s")
    print(f"平均算法执行时间: {avg_execution_time:.4f}秒")
    
    return {
        'shape': shape_str,
        'm': m, 'n': n, 'k': k,
        'avg_rank': avg_rank,
        'avg_beaten_ratio': avg_beaten_ratio,
        'avg_time': avg_time,
        'avg_execution_time': avg_execution_time,  # 新增：平均执行时间
        'total_valid': total_valid,
        'results': results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='遗传算法优化程序')
    parser.add_argument('--pop-size', type=int, default=500, help='种群大小 (默认: 500)')
    parser.add_argument('--n-generations', type=int, default=50, help='进化代数 (默认: 50)')
    parser.add_argument('--cxpb', type=float, default=0.8, help='交叉概率 (默认: 0.8)')
    parser.add_argument('--mutpb', type=float, default=0.1, help='变异概率 (默认: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='每个shape运行轮数 (默认: 10)')
    parser.add_argument('--shape', type=str, default="all", help='要测试的shape，格式为M_N_K，或"all"测试所有shape (默认: all)')
    args = parser.parse_args()

    # 确定要测试的shape列表
    if args.shape == "all":
        shapes_to_test = SHAPE_GROUP
        print(f"将测试所有 {len(SHAPE_GROUP)} 个shape")
    else:
        try:
            shape_parts = args.shape.split('_')
            if len(shape_parts) != 3:
                raise ValueError("shape格式错误")
            shapes_to_test = [[int(shape_parts[0]), int(shape_parts[1]), int(shape_parts[2])]]
            print(f"将测试单个shape: {args.shape}")
        except Exception as e:
            print(f"解析shape参数失败: {e}")
            print("使用默认测试所有shape")
            shapes_to_test = SHAPE_GROUP

    # 初始化OM推理器
    om_inferencer = OMModelInference(
        device_id=0,
        model_path="../om_best_model_batch.om",
        scaler_path="../best_result/scaler.npz",
        fixed_batch_size=500
    )

    # 存储所有shape的结果
    all_shape_results = []
    
    # 遍历所有shape
    for i, shape in enumerate(shapes_to_test):
        print(f"\n\n{'#'*100}")
        print(f"进度: {i+1}/{len(shapes_to_test)} - 处理Shape {shape}")
        print(f"{'#'*100}")
        
        try:
            result = run_single_shape(shape, args, om_inferencer)
            all_shape_results.append(result)
        except Exception as e:
            print(f"处理Shape {shape} 时出错: {e}")
            continue

    # 输出汇总统计结果
    if all_shape_results:
        print(f"\n\n{'#'*100}")
        print("汇总统计结果")
        print(f"{'#'*100}")
        
        # 计算总体统计
        total_shapes = len(all_shape_results)
        total_avg_rank = sum(r['avg_rank'] for r in all_shape_results) / total_shapes
        total_avg_beaten_ratio = sum(r['avg_beaten_ratio'] for r in all_shape_results) / total_shapes
        total_avg_time = sum(r['avg_time'] for r in all_shape_results) / total_shapes
        total_avg_execution_time = sum(r['avg_execution_time'] for r in all_shape_results) / total_shapes  # 新增：总体平均执行时间
        
        print(f"测试的Shape数量: {total_shapes}")
        print(f"总体平均排名: {total_avg_rank:.1f}")
        print(f"总体平均打败比例: {total_avg_beaten_ratio:.4f}%")
        print(f"总体平均运行时间: {total_avg_time:.6f}s")
        print(f"总体平均算法执行时间: {total_avg_execution_time:.4f}秒")
        
        # 按排名排序显示详细结果
        print(f"\n详细结果 (按平均排名排序):")
        print(f"{'Shape':<20} {'M':<8} {'N':<8} {'K':<8} {'平均排名':<10} {'打败比例':<12} {'平均时间':<12} {'执行时间':<12}")
        print("-" * 100)
        
        sorted_results = sorted(all_shape_results, key=lambda x: x['avg_rank'])
        for result in sorted_results:
            print(f"{result['shape']:<20} {result['m']:<8} {result['n']:<8} {result['k']:<8} "
                  f"{result['avg_rank']:<10.1f} {result['avg_beaten_ratio']:<12.4f} {result['avg_time']:<12.6f} {result['avg_execution_time']:<12.4f}")
        
        # 找出最佳和最差的shape
        best_shape = min(all_shape_results, key=lambda x: x['avg_rank'])
        worst_shape = max(all_shape_results, key=lambda x: x['avg_rank'])
        
        print(f"\n最佳Shape: {best_shape['shape']} (平均排名: {best_shape['avg_rank']:.1f})")
        print(f"最差Shape: {worst_shape['shape']} (平均排名: {worst_shape['avg_rank']:.1f})")
    else:
        print("没有成功完成任何shape的测试")

    # 释放资源
    del om_inferencer