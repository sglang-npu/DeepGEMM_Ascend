import numpy as np
import pandas as pd
from simanneal import Annealer
from test_om import OMModelInference
import time
import argparse

FIXED_M = 1279
FIXED_N = 5003
FIXED_K = 7681

OPTIMIZE_PARAMS_NAME = [
    "m_sections", "n_sections", 
    "m_sec_o_blocks", "n_sec_o_blocks", 
    "k_o_iter_blocks", "db_o_blocks"
]

TOTAL_PARAMS_ORDER = ["M", "N", "K"] + OPTIMIZE_PARAMS_NAME

OPTIMIZE_PARAMS_IDXS = [3, 4, 5, 6, 7, 8]

# 定义所有六个参数的允许取值列表
ALLOWED_VALUES = {
    "m_sections": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24],
    "n_sections": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24],
    "m_sec_o_blocks": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
    "n_sec_o_blocks": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
    "k_o_iter_blocks": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "db_o_blocks": [1, 2, 4, 8, 16, 32, 64]
}

class SixParamsTimeOptimizer(Annealer):
    def __init__(self, om_inferencer, initial_optimize_params=None):
        if initial_optimize_params is None:
            initial_optimize_params = self._generate_random_optimize_params()
        else:
            if not self._is_optimize_params_valid(initial_optimize_params):
                raise ValueError("初始参数不满足Rule1-Rule4")
        
        self.full_initial_state = [FIXED_M, FIXED_N, FIXED_K] + initial_optimize_params
        
        self.om_inferencer = om_inferencer
        # 所有参数的步长均表示索引移动步数
        self.base_step = {
            "m_sections": 1,
            "n_sections": 1,
            "m_sec_o_blocks": 1,
            "n_sec_o_blocks": 1,
            "k_o_iter_blocks": 1,
            "db_o_blocks": 1
        }
        self.max_retry = 5
        super().__init__(self.full_initial_state)

    def _generate_random_optimize_params(self):
        while True:
            # 所有参数均从允许的列表中随机选择
            m_sections = np.random.choice(ALLOWED_VALUES["m_sections"])
            n_sections = np.random.choice(ALLOWED_VALUES["n_sections"])
            m_sec_o_blocks = np.random.choice(ALLOWED_VALUES["m_sec_o_blocks"])
            n_sec_o_blocks = np.random.choice(ALLOWED_VALUES["n_sec_o_blocks"])
            k_o_iter_blocks = np.random.choice(ALLOWED_VALUES["k_o_iter_blocks"])
            db_o_blocks = np.random.choice(ALLOWED_VALUES["db_o_blocks"])

            random_params = [
                m_sections, n_sections, 
                m_sec_o_blocks, n_sec_o_blocks, 
                k_o_iter_blocks, db_o_blocks
            ]
            
            if self._is_optimize_params_valid(random_params):
                # print(f"生成合法6参数：{random_params}")
                return random_params

    def _is_optimize_params_valid(self, optimize_params): 
        m_sec, n_sec, m_sob, n_sob, k_oib, db_ob = optimize_params

        # 检查所有参数是否在允许的列表中
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

        # 原有规则校验
        if m_sec * n_sec > 24:
            return False
        if m_sob * n_sob > 128:
            return False
        sum_mn = m_sob + n_sob
        if sum_mn == 0 or k_oib >= 1024 / (sum_mn * 2):
            return False
        if m_sob == 0 or n_sob == 0:
            return False
        max_db = min(128//m_sob, 128//n_sob, k_oib)
        if db_ob > max_db:
            return False
        return True

    def _get_neighbor_state(self, current_full_state):
        """生成邻近状态"""
        current_state = current_full_state.copy()
        current_optimize = [current_state[i] for i in OPTIMIZE_PARAMS_IDXS]

        # 选择微调1-2个参数
        param_count = 1 if np.random.random() < 0.7 else 2
        selected_idxs = np.random.choice(OPTIMIZE_PARAMS_IDXS, size=param_count, replace=False)
        # print(f"微调参数索引：{selected_idxs}（共{param_count}个）")

        # 微调参数（所有参数均从允许列表中选择）
        for idx in selected_idxs:
            param_name = TOTAL_PARAMS_ORDER[idx]
            current_val = current_state[idx]
            
            # 所有参数均使用基于索引的微调方式
            values = ALLOWED_VALUES[param_name]
            current_idx = values.index(current_val)
            
            # 计算新索引（基于索引的步长移动）
            base = self.base_step[param_name]
            step = np.random.randint(-base, base + 1)
            new_idx = current_idx + step
            new_idx = max(0, min(len(values) - 1, new_idx))  # 确保索引有效
            
            new_val = values[new_idx]
            current_state[idx] = new_val
            # print(f"  {param_name}：{current_val} → {new_val}（索引步长{step}）")

        # 校验微调后的优化参数
        retry = 0
        while retry < self.max_retry:
            new_optimize = [current_state[i] for i in OPTIMIZE_PARAMS_IDXS]
            
            if self._is_optimize_params_valid(new_optimize):
                return current_state
            else:
                retry +=1
                current_state = current_full_state.copy()  # 回滚

        # print(f"重试{self.max_retry}次失败，生成新6参数")
        new_optimize = self._generate_random_optimize_params()
        for i, idx in enumerate(OPTIMIZE_PARAMS_IDXS):
            current_state[idx] = new_optimize[i]
        return current_state

    def move(self):
        self.state = self._get_neighbor_state(self.state)

    def energy(self):
        input_params = self.state
        # print(f'{input_params=}')
        predicted_time = self.om_inferencer.predict(input_params)
        return predicted_time if (predicted_time is not None and predicted_time >=0) else 1e18


def run_six_params_optimization(inferencer, num_starts, tmax, tmin, steps):
    # 注：原代码此处重复定义inferencer，已删除（避免覆盖传入的inferencer）
    best_overall_params = None
    best_overall_time = float('inf')
    start = time.time()

    # 多次运行模拟退火，每次使用不同初始参数
    for start_idx in range(num_starts):
        # print(f"\n===== 第 {start_idx+1}/{num_starts} 次搜索 =====")
        optimizer = SixParamsTimeOptimizer(om_inferencer=inferencer)
        optimizer.Tmax = tmax
        optimizer.Tmin = tmin
        optimizer.steps = steps
        optimizer.verbose = False
        optimizer.updates = 0
        best_state, best_time = optimizer.anneal()
        # print(f"{best_state=}, {best_time=}")
        # 记录全局最优
        if best_time < best_overall_time:
            best_overall_time = best_time
            best_overall_params = [best_state[i] for i in OPTIMIZE_PARAMS_IDXS]
    end = time.time()
    run_time = end - start
    # 输出全局最优结果
    # print("\n===== 多起点搜索全局最优 =====")
    # for name, val in zip(OPTIMIZE_PARAMS_NAME, best_overall_params):
    #     print(f"  {name}: {val}")
    # print(f"最优时间: {best_overall_time}")
    # print(f"模拟退火运行时间：{run_time}")
    return dict(zip(OPTIMIZE_PARAMS_NAME, best_overall_params)), best_overall_time, run_time

# -------------------------- 核心修改：compare_with_excel函数 --------------------------
def compare_with_excel(excel_path, best_params):
    try:
        df = pd.read_excel(excel_path)
        # print(f"成功读取Excel文件，共包含 {len(df)} 组参数")
    except Exception as e:
        print(f"读取Excel失败：{e}")
        return 0, 0, 0.0, 0  # 增加返回排名，默认0

    # 1. 检查Excel是否包含所有必要列（6个参数列+time列）
    required_cols = OPTIMIZE_PARAMS_NAME + ["time"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Excel文件缺少必要列：{missing_cols}")
        return 0, 0, 0.0, 0

    # 2. 转换参数列为int类型（避免Excel数值为float导致匹配失败）
    for col in OPTIMIZE_PARAMS_NAME:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

    # 3. 筛选有效时间数据（原逻辑保留）
    valid_df = df[df["time"].notna() & df["time"].apply(lambda x: isinstance(x, (int, float)))]
    total_valid = len(valid_df)
    if total_valid == 0:
        print("Excel中无有效时间数据")
        return 0, 0, 0.0, 0

    # 4. 用最优参数匹配Excel中的对应行
    param_dict = dict(zip(OPTIMIZE_PARAMS_NAME, best_params))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模拟退火优化程序')
    parser.add_argument('--num-starts', type=int, default=5, help='多起点搜索的次数 (默认: 5)')
    parser.add_argument('--tmax', type=float, default=90.0, help='模拟退火的起始温度 (默认: 90.0)')
    parser.add_argument('--tmin', type=float, default=1e-5, help='模拟退火的终止温度 (默认: 1e-5)')
    parser.add_argument('--steps', type=int, default=2000, help='模拟退火的步数 (默认: 2000)')
    parser.add_argument('--epochs', type=int, default=10, help='运行轮数 (默认: 10)')
    args = parser.parse_args()

    sum_beaten_ratio = 0
    all_run_time = 0
    all_rank = 0
    model_path = "../om_best_model.om"
    scaler_path = "../best_result/scaler.npz"
    inferencer = OMModelInference(model_path=model_path, scaler_path=scaler_path)
    
    for epoch in range(args.epochs):
        # 1. 运行模拟退火获取最优结果（参数和预测时间）
        best_params, best_time, run_time = run_six_params_optimization(
            inferencer, num_starts=args.num_starts, tmax=args.tmax, tmin=args.tmin, steps=args.steps
        )
        
        # 2. 核心修改：调用时传入最优参数，获取实际排名（不再传预测时间）
        excel_file_path = "../raw_data/shape_1279_5003_7681.xlsx"
        beaten_count, total_valid, beaten_ratio, actual_rank = compare_with_excel(excel_file_path, best_params)
        
        # 3. 累加统计值（排名直接用实际排名，无需计算）
        all_rank += actual_rank
        sum_beaten_ratio += beaten_ratio
        all_run_time += run_time
    
    # 输出统计结果（保持原格式，仅调整排名计算逻辑）
    print(f"模拟退火{args.epochs}轮，平均排名：{all_rank / args.epochs:.1f}，平均打败benchmark {sum_beaten_ratio / args.epochs:.4f}% 的Tiling参数组合，平均运行时间 {all_run_time / args.epochs:.4f}s")