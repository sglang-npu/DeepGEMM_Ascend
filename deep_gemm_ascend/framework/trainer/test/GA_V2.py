import numpy as np
import random
import pickle
import os
from deap import base, creator, tools
from test_om import OMModelInference  # 确保模型类正常

# --------------------------
# 1. 基础配置（定参数范围+缓存路径）
# --------------------------
# 固定输入参数
FIXED_M = 1279
FIXED_N = 5003
FIXED_K = 7681

# 待优化参数名
OPTIMIZE_PARAMS_NAME = [
    "m_sections", "n_sections", 
    "m_sec_o_blocks", "n_sec_o_blocks", 
    "k_o_iter_blocks", "db_o_blocks"
]

# 待优化参数的遍历范围（需覆盖所有可能合法值，避免遗漏）
PARAM_RANGES = {
    "m_sections": (1, 24),        # 因m*n≤24，m最大24（n=1时）
    "n_sections": (1, 24),        # 同理n最大24（m=1时）
    "m_sec_o_blocks": (1, 128),   # 因m*n≤128，单个最大128（另一个=1时）
    "n_sec_o_blocks": (1, 128),   # 同理
    "k_o_iter_blocks": (1, 512),  # 因k<1024/(sum_mn*2)，sum_mn最小2，k最大511，取512覆盖
    "db_o_blocks": (1, 128)       # 因db≤min(128//m,128//n,k)，最大128（m/n=1、k≥128时）
}

# 本地全量缓存池配置
CACHE_FILE = "full_valid_params_cache.pkl"  # 缓存文件路径
VALID_PARAMS_CACHE = []  # 全局存储全量合法参数


# --------------------------
# 2. 核心工具：合法性校验+全量参数生成
# --------------------------
def is_optimize_params_valid(params):
    """原合法性校验函数（不变，用于筛选全量参数）"""
    if len(params) != 6:
        return False
    m_sec, n_sec, m_sob, n_sob, k_oib, db_ob = params

    # 基础校验：正整数
    if not all(isinstance(p, int) and p > 0 for p in params):
        return False

    # 业务规则校验
    if m_sec * n_sec > 24:
        return False
    if m_sob * n_sob > 128:
        return False
    sum_mn = m_sob + n_sob
    if sum_mn == 0 or k_oib >= 1024 / (sum_mn * 2):
        return False
    max_db = min(128 // m_sob, 128 // n_sob, k_oib)
    if db_ob > max_db:
        return False
    return True


def generate_full_valid_params():
    """生成所有符合约束的参数（全量遍历，无遗漏）"""
    print("开始生成全量合法参数，可能需要1-2分钟...")
    full_valid = []
    # 遍历所有参数的可能组合（按范围遍历，确保不遗漏）
    for m_sec in range(PARAM_RANGES["m_sections"][0], PARAM_RANGES["m_sections"][1] + 1):
        for n_sec in range(PARAM_RANGES["n_sections"][0], PARAM_RANGES["n_sections"][1] + 1):
            # 提前过滤m*n>24的组合，减少后续循环
            if m_sec * n_sec > 24:
                continue
            for m_sob in range(PARAM_RANGES["m_sec_o_blocks"][0], PARAM_RANGES["m_sec_o_blocks"][1] + 1):
                for n_sob in range(PARAM_RANGES["n_sec_o_blocks"][0], PARAM_RANGES["n_sec_o_blocks"][1] + 1):
                    # 提前过滤m_sob*n_sob>128的组合
                    if m_sob * n_sob > 128:
                        continue
                    sum_mn = m_sob + n_sob
                    max_k = int(1024 / (sum_mn * 2)) - 1  # k必须小于1024/(sum_mn*2)
                    if max_k < 1:
                        continue
                    # 遍历k的合法范围
                    for k_oib in range(1, max_k + 1):
                        max_db = min(128 // m_sob, 128 // n_sob, k_oib)
                        if max_db < 1:
                            continue
                        # 遍历db的合法范围
                        for db_ob in range(1, max_db + 1):
                            param = (m_sec, n_sec, m_sob, n_sob, k_oib, db_ob)
                            if is_optimize_params_valid(param):
                                full_valid.append(param)
    # 去重（避免遍历逻辑可能的重复，保险措施）
    full_valid = list(set(full_valid))
    print(f"全量合法参数生成完成，共{len(full_valid)}个")
    return full_valid


# --------------------------
# 3. 缓存池操作：本地读写（优先读缓存）
# --------------------------
def init_full_cache():
    global VALID_PARAMS_CACHE, VALID_PARAMS_SET  # 同时声明两个全局变量
    if os.path.exists(CACHE_FILE):
        # 读取本地list缓存
        with open(CACHE_FILE, "rb") as f:
            VALID_PARAMS_CACHE = pickle.load(f)
    else:
        # 生成全量list缓存（原逻辑不变）
        VALID_PARAMS_CACHE = generate_full_valid_params()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(VALID_PARAMS_CACHE, f)
    # 核心：将list转为set，用于快速查找（仅需执行一次）
    VALID_PARAMS_SET = set(VALID_PARAMS_CACHE)
    print(f"缓存池初始化完成：list={len(VALID_PARAMS_CACHE)}条，set已同步生成")


# --------------------------
# 4. DEAP配置（全缓存取参+约束化变异/交叉）
# --------------------------
# 清理旧creator（避免重复定义报错）
if "FitnessMin" in creator.__dict__:
    del creator.FitnessMin
if "Individual" in creator.__dict__:
    del creator.Individual

# 定义适应度（最小化时间）和个体类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", tuple, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# 4.1 个体生成：直接从全量缓存池取（零校验成本）
def create_individual():
    if len(VALID_PARAMS_CACHE) == 0:
        raise ValueError("全量缓存池未初始化，请先调用init_full_cache()")
    # 随机选择一个合法参数（缓存池已确保100%合法）
    param_tuple = random.choice(VALID_PARAMS_CACHE)
    ind = creator.Individual(param_tuple)
    # 打印前5个初始化个体（日志用）
    if not hasattr(create_individual, "init_count"):
        create_individual.init_count = 0
    if create_individual.init_count < 5:
        print(f"初始化个体：{ind}")
        create_individual.init_count += 1
    return ind

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 4.2 批量评估函数（不变，确保与模型交互正常）
def batch_evaluate(individuals, om_inferencer, step_name=""):
    # 组装模型输入（固定参数+优化参数）
    batch_input = [[FIXED_M, FIXED_N, FIXED_K] + list(ind) for ind in individuals]
    
    # 模型预测（带异常校验）
    batch_time = om_inferencer.predict(batch_input)
    if batch_time is None:
        raise ValueError(f"{step_name}评估失败：模型返回None")
    if len(batch_time) != len(individuals):
        raise ValueError(f"{step_name}评估维度不匹配：输入{len(individuals)}个，返回{len(batch_time)}个")
    
    # 赋值适应度+打印关键日志
    for i, (ind, time) in enumerate(zip(individuals, batch_time)):
        ind.fitness.values = (time,) if (time is not None and time >= 0) else (1e18,)
        if i < 3:
            print(f"{step_name}个体{i}：{ind} → 耗时={ind.fitness.values[0]:.6f}")
    
    # 打印适应度分布（监控进化效果）
    times = [ind.fitness.values[0] for ind in individuals]
    print(f"{step_name}适应度分布：最小={min(times):.6f}，最大={max(times):.6f}，平均={np.mean(times):.6f}")
    return times


# 4.3 选择函数：锦标赛选择（保留优质个体）
toolbox.register("select", tools.selTournament, tournsize=3)


# 4.4 交叉函数：约束化修复（避免无效后代）
def cx_valid(ind1, ind2):
    """
    优化点：
    1. 用set（VALID_PARAMS_SET）的O(1)查找替代list的O(n)查找，判断后代合法性
    2. 保留“无效后代自动修复”逻辑，确保交叉后个体100%合法
    3. 简化日志，仅打印关键交叉信息
    """
    # 基础校验：交叉对象必须是DEAP的Individual类型
    if not (isinstance(ind1, creator.Individual) and isinstance(ind2, creator.Individual)):
        raise TypeError("交叉对象必须是DEAP的Individual类型")
    if len(VALID_PARAMS_CACHE) == 0 or len(VALID_PARAMS_SET) == 0:
        raise ValueError("缓存池未初始化，请先调用init_full_cache()")

    # 步骤1：执行单点交叉（DEAP原生逻辑，保持不变）
    # 将个体转为列表后交叉，再转回元组（候选参数格式）
    offspring1_list, offspring2_list = tools.cxOnePoint(list(ind1), list(ind2))
    offspring1_candidate = tuple(offspring1_list)
    offspring2_candidate = tuple(offspring2_list)

    # 步骤2：判断后代合法性+自动修复（核心优化：用set查找）
    # 原逻辑：offspring1_candidate in VALID_PARAMS_CACHE（O(n)）
    # 新逻辑：offspring1_candidate in VALID_PARAMS_SET（O(1)）
    if offspring1_candidate in VALID_PARAMS_SET:
        offspring1 = creator.Individual(offspring1_candidate)
    else:
        # 后代无效时，从缓存池随机取一个合法个体（修复逻辑不变）
        offspring1 = creator.Individual(random.choice(VALID_PARAMS_CACHE))
    
    if offspring2_candidate in VALID_PARAMS_SET:
        offspring2 = creator.Individual(offspring2_candidate)
    else:
        offspring2 = creator.Individual(random.choice(VALID_PARAMS_CACHE))

    # 步骤3：清空后代适应度（交叉后代需重新评估，DEAP强制要求）
    del offspring1.fitness.values
    del offspring2.fitness.values

    # 日志：打印交叉父代与后代（便于监控进化过程）
    print(f"交叉操作：父代1={ind1} × 父代2={ind2} → 后代1={offspring1}，后代2={offspring2}")

    return offspring1, offspring2  # DEAP要求交叉函数返回两个个体

toolbox.register("mate", cx_valid)


# 4.5 变异函数：约束化变异（仅在合法参数中变异）
def mutate_valid(individual, indpb=0.1, candidate_num=2):
    """
    优化点：
    1. 放弃“全量过滤缓存池”，改用“随机取候选参数”（O(1)复杂度）
    2. 候选数默认2个，兼顾多样性与效率，可根据需求调整
    3. 极端情况兜底（候选全为当前个体时保留原个体）
    """
    # 基础校验：个体类型+缓存池最小规模（至少2个参数才可能变异）
    if not isinstance(individual, creator.Individual):
        raise TypeError("变异对象必须是DEAP的Individual类型")
    if len(VALID_PARAMS_CACHE) < 2 or len(VALID_PARAMS_SET) < 2:
        return (individual,)  # 缓存池太小时不变异，直接返回原个体

    # 按概率决定是否执行变异
    if random.random() < indpb:
        # 步骤1：从缓存池随机取candidate_num个候选参数（无需全量遍历）
        # random.sample支持从大列表快速取样，内部无全量遍历
        candidates = random.sample(VALID_PARAMS_CACHE, candidate_num)
        
        # 步骤2：从候选中筛选“不等于当前个体”的参数（仅扫描10个候选，非1382万条）
        # next函数+生成器：找到第一个符合条件的候选就返回，不扫描全部候选
        current_ind_tuple = tuple(individual)
        mutated_param = next(
            (p for p in candidates if p != current_ind_tuple),  # 找非当前个体的候选
            current_ind_tuple  # 兜底：若所有候选都是当前个体，保留原参数
        )
        
        # 步骤3：生成变异个体+清空适应度（后代需重新评估）
        mutated_ind = creator.Individual(mutated_param)
        del mutated_ind.fitness.values
        
        # 日志：打印变异前后对比（便于调试）
        # print(f"变异操作：{individual} → {mutated_ind}")
        return (mutated_ind,)  # DEAP要求变异函数返回元组
    
    # 不满足变异概率时，返回原个体
    return (individual,)

toolbox.register("mutate", mutate_valid, indpb=0.1)


# --------------------------
# 5. 主运行函数（带进化监控）
# --------------------------
def run_ga(
    pop_size=50,
    n_generations=40,
    cxpb=0.8,
    mutpb=0.1,
    om_inferencer=None
):
    # 初始化种群
    pop = toolbox.population(n=pop_size)
    print(f"\n=== 种群初始化完成（规模：{pop_size}）===")

    # 初始评估（首次计算所有个体适应度）
    print("\n=== 初始种群评估 ===")
    batch_evaluate(pop, om_inferencer, step_name="初始")

    # 记录全局最优（初始为种群最优）
    global_best = tools.selBest(pop, 1)[0]
    global_best_time = global_best.fitness.values[0]
    print(f"初始全局最优：{global_best} → 最小耗时={global_best_time:.6f}")

    # 进化主循环
    for gen in range(n_generations):
        print(f"\n===== 第{gen+1}/{n_generations}代进化 =====")
        
        # 1. 选择：精英保留（10%）+ 锦标赛选择（90%）
        elite_size = max(1, int(pop_size * 0.1))  # 至少保留1个精英
        elite = tools.selBest(pop, elite_size)  # 筛选精英个体
        print(f"步骤1：选择精英（{elite_size}个）→ 耗时分别为：{[ind.fitness.values[0]:.6f for ind in elite]}")
        
        # 选择非精英个体（用于后续交叉变异）
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, pop_size - elite_size)]

        # 2. 交叉：对非精英个体执行交叉
        print("\n步骤2：交叉操作（概率：{cxpb:.1f}）")
        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            if random.random() < cxpb:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])

        # 3. 变异：对非精英个体执行变异
        print("\n步骤3：变异操作（概率：{mutpb:.1f}）")
        for i in range(len(offspring)):
            offspring[i], = toolbox.mutate(offspring[i])

        # 4. 评估新一代：仅评估未计算适应度的个体
        print("\n步骤4：新一代评估")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"需评估的新个体数量：{len(invalid_ind)}")
        if invalid_ind:
            batch_evaluate(invalid_ind, om_inferencer, step_name=f"第{gen+1}代")

        # 5. 组装新一代种群（精英+变异交叉后代）
        pop = elite + offspring

        # 6. 更新全局最优
        current_best = tools.selBest(pop, 1)[0]
        current_best_time = current_best.fitness.values[0]
        if current_best_time < global_best_time:
            global_best = toolbox.clone(current_best)
            global_best_time = current_best_time
            print(f"【更新全局最优】{global_best} → 新最小耗时={global_best_time:.6f}")
        else:
            print(f"当前代最优未超越全局：{current_best_time:.6f}（全局最优：{global_best_time:.6f}）")

    # 输出最终结果
    print("\n" + "="*60)
    print("GA优化完成！最终全局最优结果：")
    for name, val in zip(OPTIMIZE_PARAMS_NAME, global_best):
        print(f"  {name}：{val}")
    print(f"  最小预测耗时：{global_best_time:.6f}")
    print("="*60)

    return dict(zip(OPTIMIZE_PARAMS_NAME, global_best)), global_best_time


# --------------------------
# 6. 启动入口（先初始化缓存，再跑GA）
# --------------------------
if __name__ == "__main__":
    # 第一步：初始化全量缓存池（读本地/生成）
    init_full_cache()
    
    # 第二步：初始化OM模型推理器（按实际路径修改）
    try:
        om_inferencer = OMModelInference(
            device_id=0,
            model_path="../om_best_model_batch.om",
            scaler_path="../best_result/scaler.npz",
            fixed_batch_size=50  # 按模型实际支持的batch_size修改
        )
    except Exception as e:
        raise RuntimeError(f"OM模型推理器初始化失败：{str(e)}") from e
    
    # 第三步：运行遗传算法
    best_params, min_time = run_ga(
        pop_size=50,
        n_generations=40,
        cxpb=0.8,
        mutpb=0.1,
        om_inferencer=om_inferencer
    )
    
    # 第四步：输出结果+释放资源
    print("\n最终全局最优参数字典：", best_params)
    del om_inferencer  # 释放模型资源