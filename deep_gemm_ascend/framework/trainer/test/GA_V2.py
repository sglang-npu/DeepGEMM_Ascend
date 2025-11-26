import numpy as np
import pandas as pd
import random
import math
import pickle
import os
import time
import argparse
from deap import base, creator, tools
from test_om import OMModelInference  # 确保模型类正常

# --------------------------
# 1. 基础配置（定参数范围+缓存路径）
# --------------------------
# 固定输入参数
FIXED_M = 125
FIXED_N = 125
FIXED_K = 29

# 待优化参数名
OPTIMIZE_PARAMS_NAME = [
    "mTile", "nTile", "kTile"
]

# 本地全量缓存池配置
CACHE_FILE = "mnk_tile.xlsx"  # 缓存文件路径
VALID_PARAMS_CACHE = {}  # 全局存储全量合法参数

# --------------------------
# 2. 核心工具：合法性校验+全量参数生成
# --------------------------
class CatlassParameter():
    """
    为 catlass 生成 tiling 网格参数
    参数：mTile, nTile, kTile
    约束条件：
        - 各维度元素数量为 tile × 16，矩阵元素数量需考虑两次 16
        - L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        - L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 < 512KB
    """
    def __init__(self, operator_type=None, core_num=24):
        """
        Args:
            operator_type: 算子类型，可选值: 'smallmatmul', 'commonmatmul', 'paddingcommonmatmul', None(所有)
            core_num: AI Core数量，默认24
        """
        self.operator_type = operator_type
        self.core_num = core_num
        self.grid_parameters = self.grid_generate_parameters()

    def generate_mn_tile_values(self):
        """
        生成mTile和nTile的值（范围1-512）
        策略：多粒度采样，考虑L0C约束（mTile × nTile ≤ 128）
        优化：压缩采样点，确保总组合数在500以内
        
        推理过程：
        1. L0C约束：mTile × nTile ≤ 128，意味着大部分大值组合会被过滤
        2. 步长1范围：1-4（4个值），覆盖最关键的极小值
        3. 其他区域：适当增大步长，减少采样点
        4. 目标：mn_tile_values约15-20个值，k_tile_values约20-25个值
        5. 理论最大组合：20² × 25 = 400 × 25 = 10,000
        6. 实际有效组合（考虑L0C约束）：约300-500
        """
        values = []
        
        # 极小值区域（1-4）：步长1，最密集采样
        # 这个区域对性能影响最大，且L0C约束下最常用
        values.extend(range(1, 5, 1))  # 1,2,3,4 (4个值)
        
        # 小值区域（6-16）：步长2，密集采样
        # 包含关键值8, 12, 16
        values.extend(range(6, 17, 2))  # 6,8,10,12,14,16 (6个值)
        
        # 中小值区域（20-32）：步长4，中等密度
        # 包含关键值24, 28, 32
        values.extend(range(20, 33, 4))  # 20,24,28,32 (4个值)
        
        # 中值区域（40-64）：步长8，稀疏采样
        # 包含关键值48, 56, 64
        values.extend(range(40, 65, 8))  # 40,48,56,64 (4个值)
        
        # 中大值区域（80-128）：步长16，很稀疏采样
        # 包含关键值96, 112, 128
        values.extend(range(80, 129, 16))  # 80,96,112,128 (4个值)
        
        # 大值区域（160-256）：步长32，极稀疏采样
        # 包含关键值192, 224, 256
        values.extend(range(160, 257, 32))  # 160,192,224,256 (4个值)
        
        # 超大值区域（320-512）：步长64，最稀疏采样
        # 包含关键值384, 448, 512
        values.extend(range(320, 513, 64))  # 320,384,448,512 (4个值)
        
        # 去重并排序
        values = sorted(list(set(values)))
        return values
    
    def generate_k_tile_values(self):
        """
        生成kTile的值（范围1-1024）
        策略：多粒度采样，压缩采样点数量
        
        推理过程：
        1. kTile范围更大（1-1024），可以更稀疏采样
        2. 步长1范围：1-4（4个值），覆盖最关键的极小值
        3. 其他区域：适当增大步长
        4. 目标：约20-25个值
        """
        values = []
        
        # 极小值区域（1-4）：步长1，最密集采样
        # kTile对L1缓存影响大，极小值需要完整覆盖
        values.extend(range(1, 5, 1))  # 1,2,3,4 (4个值)
        
        # 小值区域（6-16）：步长2，密集采样
        # 包含关键值8, 12, 16
        values.extend(range(6, 17, 2))  # 6,8,10,12,14,16 (6个值)
        
        # 中小值区域（20-32）：步长4，中等密度
        # 包含关键值24, 28, 32
        values.extend(range(20, 33, 4))  # 20,24,28,32 (4个值)
        
        # 中值区域（40-64）：步长8，稀疏采样
        # 包含关键值48, 56, 64
        values.extend(range(40, 65, 8))  # 40,48,56,64 (4个值)
        
        # 中大值区域（80-128）：步长16，很稀疏采样
        # 包含关键值96, 112, 128
        values.extend(range(80, 129, 16))  # 80,96,112,128 (4个值)
        
        # 大值区域（160-256）：步长32，极稀疏采样
        # 包含关键值192, 224, 256
        values.extend(range(160, 257, 32))  # 160,192,224,256 (4个值)
        
        # 超大值区域（320-512）：步长64，最稀疏采样
        # 包含关键值384, 448, 512
        values.extend(range(320, 513, 64))  # 320,384,448,512 (4个值)
        
        # 极大值区域（640-1024）：步长128，极稀疏采样
        # 包含关键值768, 896, 1024
        values.extend(range(640, 1025, 128))  # 640,768,896,1024 (4个值)
        
        # 去重并排序
        values = sorted(list(set(values)))
        return values
    
    def generate_tile_values(self):
        """
        兼容性方法：返回mTile/nTile的值（保持向后兼容）
        新代码应该使用generate_mn_tile_values()和generate_k_tile_values()
        """
        return self.generate_mn_tile_values()

    def generate_mnk_tiles_linear(self):
        """
        线性生成 mTile, nTile, kTile 的所有组合
        约束条件：
        1. L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        2. L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 x double_buffer < 512KB
        3. mTile/nTile范围：1-512，kTile范围：1-1024
        """
        mn_tile_values = self.generate_mn_tile_values()
        k_tile_values = self.generate_k_tile_values()
        double_buffer = 2
        L1_CACHE_MAX_BYTES = 512 * 1024 // double_buffer  # 256KB
        L0C_MAX_BYTES = 128 * 1024  # 128KB
        
        # 预计算常量，避免重复计算
        L0C_FACTOR = 16 * 16 * 4  # 1024
        L0C_MAX_TILE_PRODUCT = L0C_MAX_BYTES // L0C_FACTOR  # 128
        
        for m_tile in mn_tile_values:
            for n_tile in mn_tile_values:
                # L0C bytes = (mTile*16) * (nTile*16) * 4 = mTile * nTile * 1024
                # 优化：直接比较 mTile * nTile，避免乘法运算
                if m_tile * n_tile > L0C_MAX_TILE_PRODUCT:
                    continue
                
                # 预计算 (mTile*16 + nTile*16)，避免在k循环中重复计算
                m_plus_n_16 = (m_tile + n_tile) * 16
                
                for k_tile in k_tile_values:
                    # L1 bytes = (mTile*16 + nTile*16) * kTile*16 * 2
                    # 优化：使用预计算的值
                    l1_bytes = m_plus_n_16 * k_tile * 32  # 32 = 16 * 2
                    if l1_bytes > L1_CACHE_MAX_BYTES:
                        continue
                    
                    yield m_tile, n_tile, k_tile

    def grid_generate_parameters(self):
        """
        生成所有有效的参数组合
        返回参数列表，每个元素包含 mTile, nTile, kTile
        """
        parameters = []
        mn_tile_values = self.generate_mn_tile_values()
        k_tile_values = self.generate_k_tile_values()
        
        for m_tile, n_tile, k_tile in self.generate_mnk_tiles_linear():
            param_dict = {
                'mTile': m_tile,
                'nTile': n_tile,
                'kTile': k_tile
            }
            parameters.append(param_dict)
        
        # 输出统计信息
        print(f'=' * 60)
        print(f'Tile Values Statistics:')
        print(f'  mTile/nTile values: {len(mn_tile_values)} (range: {min(mn_tile_values)}-{max(mn_tile_values)})')
        print(f'    Values: {mn_tile_values[:10]}...{mn_tile_values[-5:]}')
        print(f'  kTile values: {len(k_tile_values)} (range: {min(k_tile_values)}-{max(k_tile_values)})')
        print(f'    Values: {k_tile_values[:10]}...{k_tile_values[-5:]}')
        print(f'')
        print(f'Combination Analysis:')
        print(f'  Theoretical max (before constraints): {len(mn_tile_values)}² × {len(k_tile_values)} = {len(mn_tile_values)**2 * len(k_tile_values):,}')
        print(f'  Valid combinations (after L0C/L1 constraints): {len(parameters):,}')
        print(f'  Constraint filtering rate: {(1 - len(parameters) / (len(mn_tile_values)**2 * len(k_tile_values))) * 100:.1f}%')
        print(f'=' * 60)
        
        if len(parameters) > 500:
            print(f'⚠️  Warning: Total combinations ({len(parameters)}) exceeds target of 500!')
            print(f'   Consider further reducing tile values or adjusting sampling strategy.')
        elif len(parameters) < 100:
            print(f'⚠️  Warning: Total combinations ({len(parameters)}) is very low (< 100).')
            print(f'   Consider increasing tile values for better coverage.')
        else:
            print(f'✓ Total combinations ({len(parameters)}) is within reasonable range (100-500).')
        
        return parameters

    def check_smallmatmul_constraints(self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0):
        """
        检查shape和tiling参数是否满足SmallMatmul算子的约束条件
        
        SmallMatmul约束条件：
        1. Cache约束: (m1 + n1) × k1 × 2 ≤ 512KB, m1 × n1 × 4 ≤ 128KB
        2. Tile数量: ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        3. K轴: k ≤ k1
        4. Padding: paddingTagA/B/C == PADDING_NONE (简化检查)
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: Tiling参数（实际值，不是tile编号）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否满足SmallMatmul约束
        """
        # 约束1: Cache约束
        # L1约束: (m1 + n1) × k1 × 2 ≤ 512KB = 524288字节
        l1_bytes = (m1 + n1) * k1 * 2
        if l1_bytes > 524288:
            return False
        
        # L0C约束: m1 × n1 × 4 ≤ 128KB = 131072字节
        l0c_bytes = m1 * n1 * 4
        if l0c_bytes > 131072:
            return False
        
        # 约束2: Tile数量约束
        # ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        task_blocks = math.ceil(m / m1) * math.ceil(n / n1)
        if task_blocks > self.core_num:
            return False
        
        # 约束3: K轴约束
        if k > k1:
            return False
        
        # 约束4: Padding约束（简化版本）
        # 这里只做基本的对齐检查，完整的padding检查需要模拟GetPaddingTag的完整逻辑
        # 简化策略：检查可能导致padding的常见情况
        
        # PaddingTagA检查（简化）
        # 如果layoutTagA == RowMajor: innerAxisA = k, outterAxisA = m
        # 如果layoutTagA == ColumnMajor: innerAxisA = m, outterAxisA = k
        if layout_tag_a == 0:  # RowMajor
            inner_axis_a = k
            outter_axis_a = m
        else:  # ColumnMajor
            inner_axis_a = m
            outter_axis_a = k
        
        # 如果innerAxisA < 8 或 (innerAxisA < 32 且 innerAxisA % 16 != 0) 且 outterAxisA > 512
        # 则paddingTagA = PADDING_NZ
        if ((inner_axis_a < 8 or (inner_axis_a < 32 and inner_axis_a % 16 != 0)) and 
            outter_axis_a > 512):
            return False
        
        # PaddingTagB检查（简化）
        if layout_tag_b == 0:  # RowMajor
            inner_axis_b = n
            outter_axis_b = k
        else:  # ColumnMajor
            inner_axis_b = k
            outter_axis_b = n
        
        if ((inner_axis_b < 8 or (inner_axis_b < 32 and inner_axis_b % 16 != 0)) and 
            outter_axis_b > 512):
            return False
        
        # PaddingTagC检查（简化）
        # paddingTagC = PADDING_ND 当: m×n > 2048² 且 n > 256 且 n % 128 != 0
        if (m * n > 2048 * 2048 and n > 256 and n % 128 != 0):
            # 还需要检查totalDataSize < 192MB，这里简化处理
            # 如果满足前面的条件，很可能需要padding
            return False
        
        return True
    
    def filter_parameters(self, shape):
        """
        根据shape和算子类型过滤参数
        
        Args:
            shape: [m, n, k] 矩阵维度
        
        Returns:
            过滤后的参数列表
        """
        m, n, k = shape[0], shape[1], shape[2]
        
        # 如果没有指定算子类型，返回所有参数组合
        if self.operator_type is None:
            return self.grid_parameters
        
        # 根据算子类型进行筛选
        if self.operator_type.lower() == 'smallmatmul':
            filtered_params = []
            for param in self.grid_parameters:
                # 将tile编号转换为实际值（tile × 16）
                m1 = param['mTile'] * 16
                n1 = param['nTile'] * 16
                k1 = param['kTile'] * 16
                
                # 检查是否满足SmallMatmul约束
                # 默认使用RowMajor布局（layout_tag_a=0, layout_tag_b=0）
                # 注意：如果需要支持其他layout组合，可以在这里循环检查所有组合
                if self.check_smallmatmul_constraints(m, n, k, m1, n1, k1, 0, 0):
                    filtered_params.append(param)
            
            if len(filtered_params) == 0:
                print(f"Warning: No valid tiling parameters found for SmallMatmul with shape {shape}")
            
            return filtered_params
        elif self.operator_type.lower() in ['commonmatmul', 'paddingcommonmatmul']:
            # 对于其他算子类型，暂时返回所有参数
            # 可以根据需要添加相应的筛选逻辑
            return self.grid_parameters
        else:
            # 未知的算子类型，返回所有参数
            print(f"Warning: Unknown operator type '{self.operator_type}', returning all parameters")
            return self.grid_parameters

# --------------------------
# 3. 缓存池操作：本地读写（优先读缓存）
# --------------------------
def init_full_cache(shape_list):
    global VALID_PARAMS_CACHE
    global VALID_PARAMS_SET 
    VALID_PARAMS_CACHE = {} 
    VALID_PARAMS_SET  = {}
    parameter = None
    for shape in shape_list:
        filename = '_'.join(map(str, shape)) + '_' + CACHE_FILE
        if os.path.exists(filename):
            # 读取本地mnk_tile缓存
            df = pd.read_excel(filename)
        else:
            # 根据m,n,k生成mnk_tile缓存并保存
            if parameter is None:
                parameter = CatlassParameter(operator_type='smallmatmul', core_num=20)
            data_list = parameter.filter_parameters(shape)
            df = pd.DataFrame(data_list, columns=['mTile', 'nTile', 'kTile'])
            df.to_excel(filename, index=False)
        
        VALID_PARAMS_CACHE[f"{shape}"] = list(zip(df['mTile'], df['nTile'], df['kTile']))   
        # 核心：将list转为set，用于快速查找（仅需执行一次）
        VALID_PARAMS_SET[f"{shape}"] = set(VALID_PARAMS_CACHE)
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
def create_individual(shape):
    if len(VALID_PARAMS_CACHE) == 0:
        raise ValueError("全量缓存池未初始化，请先调用init_full_cache()")
    if shape not in SHAPE_GROUP:
        raise KeyError(f"缓存池中未找到对应的shape: {shape}")
    # 随机选择一个合法参数（缓存池已确保100%合法）
    param_tuple = random.choice(VALID_PARAMS_CACHE[f"{shape}"])
    ind = creator.Individual(param_tuple)
    # 打印前5个初始化个体（日志用）
    if not hasattr(create_individual, "init_count"):
        create_individual.init_count = 0
    if create_individual.init_count < 5:
        print(f"初始化个体：{ind}")
        create_individual.init_count += 1
    return ind

toolbox.register("individual", create_individual)
def create_population(n, shape):
    return [toolbox.individual(shape=shape) for _ in range(n)]
toolbox.register("population", create_population)


# 4.2 批量评估函数（不变，确保与模型交互正常）
def batch_evaluate(individuals, om_inferencer, step_name=""):
    # 组装模型输入（固定参数+优化参数）
    batch_input = [[FIXED_M,FIXED_N,FIXED_K] + list(ind) for ind in individuals]
    
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

# 支持动态shape的批量评估函数
def batch_evaluate_with_shape(individuals, om_inferencer,m, n, k, step_name=""):
    # 组装模型输入（固定参数+优化参数）
    batch_input = [[m,n,k] + list(ind) for ind in individuals]
    
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
def cx_valid(ind1, ind2, shape=None):
    """
    优化点：
    1. 用set（VALID_PARAMS_SET）的O(1)查找替代list的O(n)查找，判断后代合法性
    2. 保留“无效后代自动修复”逻辑，确保交叉后个体100%合法
    3. 简化日志，仅打印关键交叉信息
    """
    if shape == None:
    shape = SHAPE_GROUP[0]
    # 基础校验：交叉对象必须是DEAP的Individual类型
    if not (isinstance(ind1, creator.Individual) and isinstance(ind2, creator.Individual)):
        raise TypeError("交叉对象必须是DEAP的Individual类型")
    if len(VALID_PARAMS_CACHE[f"{shape}"]) == 0 or len(VALID_PARAMS_SET[f"{shape}"]) == 0:
        raise ValueError("缓存池未初始化，请先调用init_full_cache()")

    # 步骤1：执行单点交叉（DEAP原生逻辑，保持不变）
    # 将个体转为列表后交叉，再转回元组（候选参数格式）
    offspring1_list, offspring2_list = tools.cxOnePoint(list(ind1), list(ind2))
    offspring1_candidate = tuple(offspring1_list)
    offspring2_candidate = tuple(offspring2_list)

    # 步骤2：判断后代合法性+自动修复（核心优化：用set查找）
    # 原逻辑：offspring1_candidate in VALID_PARAMS_CACHE（O(n)）
    # 新逻辑：offspring1_candidate in VALID_PARAMS_SET（O(1)）
    if offspring1_candidate in VALID_PARAMS_SET[f"{shape}"]:
        offspring1 = creator.Individual(offspring1_candidate)
    else:
        # 后代无效时，从缓存池随机取一个合法个体（修复逻辑不变）
        offspring1 = creator.Individual(random.choice(VALID_PARAMS_CACHE[f"{shape}"]))
    
    if offspring2_candidate in VALID_PARAMS_SET[f"{shape}"]:
        offspring2 = creator.Individual(offspring2_candidate)
    else:
        offspring2 = creator.Individual(random.choice(VALID_PARAMS_CACHE[f"{shape}"]))

    # 步骤3：清空后代适应度（交叉后代需重新评估，DEAP强制要求）
    del offspring1.fitness.values
    del offspring2.fitness.values

    # 日志：打印交叉父代与后代（便于监控进化过程）
    print(f"交叉操作：父代1={ind1} × 父代2={ind2} → 后代1={offspring1}，后代2={offspring2}")

    return offspring1, offspring2  # DEAP要求交叉函数返回两个个体

toolbox.register("mate", cx_valid)


# 4.5 变异函数：约束化变异（仅在合法参数中变异）
def mutate_valid(individual, shape=None, indpb=0.1, candidate_num=2,):
    """
    优化点：
    1. 放弃“全量过滤缓存池”，改用“随机取候选参数”（O(1)复杂度）
    2. 候选数默认2个，兼顾多样性与效率，可根据需求调整
    3. 极端情况兜底（候选全为当前个体时保留原个体）
    """
    if shape == None:
        shape = SHAPE_GROUP[0]
    # 基础校验：个体类型+缓存池最小规模（至少2个参数才可能变异）
    if not isinstance(individual, creator.Individual):
        raise TypeError("变异对象必须是DEAP的Individual类型")
    if len(VALID_PARAMS_CACHE[f"{shape}"]) < 2 or len(VALID_PARAMS_SET[f"{shape}"]) < 2:
        return (individual,)  # 缓存池太小时不变异，直接返回原个体

    # 按概率决定是否执行变异
    if random.random() < indpb:
        # 步骤1：从缓存池随机取candidate_num个候选参数（无需全量遍历）
        # random.sample支持从大列表快速取样，内部无全量遍历
        candidates = random.sample(VALID_PARAMS_CACHE[f"{shape}"], candidate_num)
        
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
    pop = toolbox.population(n=pop_size, shape=[m,n,k])
    print(f"\n=== 种群初始化完成（规模：{pop_size}）===")

    # 初始评估（首次计算所有个体适应度）
    print("\n=== 初始种群评估 ===")
    batch_evaluate_with_shape(pop, om_inferencer, m, n, k, step_name="初始")

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
        elite_times =  [f"{ind.fitness.values[0]:.6f}" for ind in elite if ind.fitness.values]
        print(f"步骤1：选择精英（{elite_size}个）→ 耗时分别为：{elite_times}")
        
        # 选择非精英个体（用于后续交叉变异）
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, pop_size - elite_size)]

        # 2. 交叉：对非精英个体执行交叉
        print("\n步骤2：交叉操作（概率：{cxpb:.1f}）")
        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            if random.random() < cxpb:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1], shape=[m,n,k])

        # 3. 变异：对非精英个体执行变异
        print("\n步骤3：变异操作（概率：{mutpb:.1f}）")
        for i in range(len(offspring)):
            offspring[i], = toolbox.mutate(offspring[i], shape=[m, n, k])

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

    # 记录算法结束时间
    ga_end_time = time.perf_counter()
    ga_execution_time = ga_end_time - ga_start_time

    # 输出最终结果
    print("\n" + "="*60)
    print("GA优化完成！最终全局最优结果：")
    for name, val in zip(OPTIMIZE_PARAMS_NAME, global_best):
        print(f"  {name}：{val}")
    print(f"  最小预测耗时：{global_best_time:.6f}")
    print(f"\n遗传算法总耗时： {ga_execution_time:.6f} s")
    print("="*60)

    return dict(zip(OPTIMIZE_PARAMS_NAME, global_best)), global_best_time, ga_execution_time

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
    [1, 1, 4],
    [8, 128, 8],
    [9, 9, 9],
    [12, 48, 768],
    [4, 4,16]
]

def run_single_shape(shape, args, om_inferencer):
    """运行单个shape的GA优化"""
    m, n, k = shape
    shape_str = f"{m}_{n}_{k}"
    excel_file_path = f"../merged_excel/{shape_str}.xlsx"
    
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

# --------------------------
# 6. 启动入口（先初始化缓存，再跑GA）
# --------------------------
if __name__ == "__main__":
    # # 第一步：初始化全量缓存池（读本地/生成）
    # init_full_cache()
    
    # # 第二步：初始化OM模型推理器（按实际路径修改）
    # try:
    #     om_inferencer = OMModelInference(
    #         device_id=0,
    #         model_path="../om_best_model.om",
    #         scaler_path="../scaler.npz",
    #         fixed_batch_size=50  # 按模型实际支持的batch_size修改
    #     )
    # except Exception as e:
    #     raise RuntimeError(f"OM模型推理器初始化失败：{str(e)}") from e
    
    # # 第三步：运行遗传算法
    # best_params, min_time, execution_time = run_ga(
    #     pop_size=50,
    #     n_generations=40,
    #     cxpb=0.8,
    #     mutpb=0.1,
    #     om_inferencer=om_inferencer
    # )
    
    # # 第四步：输出结果+释放资源
    # print("\n最终全局最优参数字典：", best_params)
    # del om_inferencer  # 释放模型资源


    parser = argparse.ArgumentParser(description='遗传算法优化程序')
    parser.add_argument('--pop-size', type=int, default=50, help='种群大小 (默认: 500)')
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

    # 第一步：初始化全量缓存池（读本地/生成）
    init_full_cache(shape_list=SHAPE_GROUP)

    # 初始化OM推理器
    om_inferencer = OMModelInference(
        device_id=0,
        model_path="../om_best_model.om",
        scaler_path="../scaler.npz",
        fixed_batch_size=50
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