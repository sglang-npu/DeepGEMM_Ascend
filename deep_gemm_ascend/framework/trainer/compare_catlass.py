import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    SKLEARN_AVAILABLE = False

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 确保可以导入同级和test目录下的模块
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "test"))
# 将 framework 目录添加到 sys.path，以便导入 benchmark_v2 模块
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark_v2 import load_shapes_from_excel, CatlassParameter
# 注意：从 benchmark_v2 导入，支持 start_idx 和 end_idx 参数
from utils import MsProfExecutor
from trainer_wm import TimePredictMLP

ORIGINAL_FEATURES = ["M", "N", "K", "m_tile", "n_tile", "k_tile"]
# DERIVED_FEATURES = [
#     "mn_tile",
#     "mk_tile",
#     "nk_tile",
#     "AI_core",
#     "read_bytes",   # 数据读取量 MNK(1/m_tile + 1/n_tile) / 16
#     "write_bytes",  # 数据写出量 MNK / (16 * k_tile)
#     "flops",        # 计算量 MNK
# ]
EXTENDED_FEATURES = ORIGINAL_FEATURES


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def load_scaler_arrays(scaler_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    按照 OM 推理逻辑读取 scalers：npz 文件中包含 mean/std
    """
    scaler_path = Path(scaler_path)
    if scaler_path.suffix.lower() != ".npz":
        raise ValueError(f"当前仅支持npz格式的scaler，收到: {scaler_path.suffix}")

    try:
        scaler_data = np.load(scaler_path)
    except Exception as exc:
        raise ValueError(f"加载scaler失败: {exc}")

    if "mean" not in scaler_data or "std" not in scaler_data:
        raise KeyError(f"scaler文件 {scaler_path} 缺少 mean/std 字段")

    mean_arr = scaler_data["mean"].astype(np.float32)
    print(f"mean_arr: {mean_arr}")
    std_arr = scaler_data["std"].astype(np.float32)
    print(f"std_arr: {std_arr}")
    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
    return mean_arr, std_arr


def load_torch_predictor(model_path: str, scaler_path: str, hidden_dims: List[int]) -> Dict[str, Any]:
    device = detect_device()
    mean_arr, std_arr = load_scaler_arrays(scaler_path)
    feature_dim = mean_arr.shape[0]
    if feature_dim != len(EXTENDED_FEATURES):
        raise ValueError(
            f"scaler维度不匹配，期望{len(EXTENDED_FEATURES)}，实际{feature_dim}"
        )

    model = TimePredictMLP(input_dim=feature_dim, hidden_dims=hidden_dims)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    mean = torch.tensor(mean_arr, dtype=torch.float32, device=device)
    std = torch.tensor(std_arr, dtype=torch.float32, device=device)

    return {"model": model, "mean": mean, "std": std, "device": device, "feature_dim": feature_dim}


def build_feature_matrix(shape: List[int], params: List[Dict[str, int]], feature_dim: int) -> np.ndarray:
    if feature_dim != len(EXTENDED_FEATURES):
        raise ValueError(
            f"特征维度应为{len(EXTENDED_FEATURES)}，但scaler提供的是{feature_dim}"
        )
    m, n, k = shape
    features = np.zeros((len(params), feature_dim), dtype=np.float32)
    for idx, param in enumerate(params):
        data = {
            "M": float(m),
            "N": float(n),
            "K": float(k),
            "m_tile": float(param["mTile"]),
            "n_tile": float(param["nTile"]),
            "k_tile": float(param["kTile"]),
        }
        # data["mn_tile"] = data["m_tile"] * data["n_tile"]
        # data["mk_tile"] = data["m_tile"] * data["k_tile"]
        # data["nk_tile"] = data["n_tile"] * data["k_tile"]
        # data["AI_core"] = np.ceil(data["M"] / data["m_tile"]) * np.ceil(data["N"] / data["n_tile"])
        # data["read_bytes"] = data["M"] * data["N"] * data["K"] * (1 / data["m_tile"] + 1 / data["n_tile"]) / 16
        # data["write_bytes"] = data["M"] * data["N"] * data["K"] / (16 * data["k_tile"])
        # data["flops"] = data["M"] * data["N"] * data["K"]
        features[idx, :] = np.array(
            [data[feat] for feat in EXTENDED_FEATURES], dtype=np.float32
        )
    return features


def predict_batch(
    predict_ctx: Dict[str, Any],
    features: np.ndarray,
    max_batch_size: int = 2048,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """
    批量预测，支持分批处理（每批最多max_batch_size个样本）
    带超时和重试机制，防止推理超时导致程序崩溃
    
    Args:
        predict_ctx: 预测上下文（包含model, mean, std, device）
        features: 特征矩阵，shape=(n_samples, feature_dim)
        max_batch_size: 最大批次大小，默认2048
        max_retries: 最大重试次数，默认3次
        timeout_seconds: 单次推理超时时间（秒），默认30秒
    
    Returns:
        预测结果数组，shape=(n_samples,)
    
    Raises:
        RuntimeError: 如果重试max_retries次后仍然失败
    """
    model = predict_ctx["model"]
    mean = predict_ctx["mean"]
    std = predict_ctx["std"]
    device = predict_ctx["device"]
    
    n_samples = features.shape[0]
    if n_samples == 0:
        raise ValueError("特征矩阵为空，无法进行预测")
    
    all_preds = []
    
    def _predict_single_batch(batch_features: np.ndarray) -> np.ndarray:
        """
        单批次预测，带超时保护
        
        注意：支持任意batch_size（包括1），PyTorch模型在eval模式下可以处理任意batch_size
        """
        batch_size = batch_features.shape[0]
        if batch_size == 0:
            raise ValueError("批次特征为空")
        
        inputs = torch.tensor(batch_features, dtype=torch.float32, device=device)
        inputs = (inputs - mean) / std
        
        with torch.no_grad():
            outputs = model(inputs)
        batch_preds = outputs.cpu().numpy().flatten()
        
        # 验证输出维度
        if len(batch_preds) != batch_size:
            raise RuntimeError(
                f"输出维度不匹配：期望{batch_size}，实际{len(batch_preds)}"
            )
        
        return batch_preds
    
    # 分批处理（支持不足max_batch_size的情况）
    for start_idx in range(0, n_samples, max_batch_size):
        end_idx = min(start_idx + max_batch_size, n_samples)
        batch_features = features[start_idx:end_idx]
        current_batch_size = batch_features.shape[0]
        
        # 重试机制
        batch_preds = None
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # 执行推理
                batch_preds = _predict_single_batch(batch_features)
                
                elapsed_time = time.time() - start_time
                
                # 检查是否超时
                if elapsed_time > timeout_seconds:
                    raise TimeoutError(
                        f"推理超时（{elapsed_time:.2f}秒>{timeout_seconds}秒）"
                    )
                
                # 成功，跳出重试循环
                break
                
            except (TimeoutError, RuntimeError, Exception) as e:
                last_error = e
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                
                if attempt < max_retries:
                    print(
                        f"[predict_batch] 第{attempt+1}次推理失败（耗时{elapsed_time:.2f}秒）: {e}，"
                        f"重试中... (批次 {start_idx}-{end_idx}, batch_size={current_batch_size})"
                    )
                    # 清理GPU缓存（如果有）
                    if device.type in ["cuda", "npu"]:
                        torch.cuda.empty_cache() if device.type == "cuda" else None
                    time.sleep(0.5)  # 短暂等待后重试
                else:
                    # 最后一次重试也失败
                    error_msg = (
                        f"[predict_batch] 推理失败，已重试{max_retries}次 "
                        f"(批次 {start_idx}-{end_idx}, batch_size={current_batch_size}): {last_error}"
                    )
                    print(error_msg)
                    raise RuntimeError(error_msg) from last_error
        
        if batch_preds is not None:
            all_preds.append(batch_preds)
        else:
            raise RuntimeError(
                f"批次 {start_idx}-{end_idx} (batch_size={current_batch_size}) 推理失败，无法获取结果"
            )
    
    # 聚合所有批次的结果
    preds = np.concatenate(all_preds)
    return preds


def measure_tiling_time(
    executor: MsProfExecutor,
    shape: List[int],
    param: Optional[Dict[str, int]],
    timeout_seconds: float = 15.0,
) -> Optional[float]:
    """
    测量tiling执行时间，运行一次
    
    Args:
        executor: MsProfExecutor实例
        shape: 矩阵维度 [M, N, K]
        param: tiling参数，None表示使用默认catlass tiling
        timeout_seconds: 运行超时时间（秒），默认15秒，超过此时间认为是异常
    
    Returns:
        执行时间（微秒），如果运行失败或超时则返回None
    """
    m, n, k = shape
    if param is None:
        param_str = f" {m} {n} {k} 0 0 1 {executor.rank_id}"
    else:
        param_str = (
            f" {m} {n} {k} "
            f"{param['mTile']} {param['nTile']} {param['kTile']} "
            f"0 0 1 {executor.rank_id}"
        )
    
    try:
        start_time = time.time()
        time_us, diff, kernel, _ = executor.ms_prof(param_str, timeout=int(timeout_seconds))
        elapsed_time = time.time() - start_time
        
        # 检查是否超时（超过15秒认为是异常）
        if elapsed_time > timeout_seconds:
            print(
                f"[measure_tiling_time] msprof超时（{elapsed_time:.2f}秒>{timeout_seconds}秒）"
                f" shape={shape}, param={param}"
            )
            return None
        
        # 检查msprof返回结果是否有效
        if time_us is None or time_us >= 999999999:
            print(
                f"[measure_tiling_time] msprof失败 shape={shape}, param={param}"
            )
            return None
        
        print(
            f"[measure_tiling_time] shape={shape}, param={param}, "
            f"time={time_us:.3f}us, diff={diff:.3e}, kernel={kernel}, elapsed={elapsed_time:.2f}s"
        )
        return time_us
    except Exception as e:
        print(
            f"[measure_tiling_time] 执行异常 shape={shape}, param={param}, 错误: {e}"
        )
        return None


class TilingPredictor:
    """
    Tiling参数预测器：封装模型预测逻辑，提供简洁的接口
    
    使用示例:
        predictor = TilingPredictor(
            model_path="model.pth",
            scaler_path="scaler.npz",
            hidden_dims=[64, 128, 64],
            operator_type="SmallMatmulKernel",  # 或 None 表示所有算子
            core_num=20
        )
        best_tiling = predictor.predict(m=128, n=256, k=512)
        # 返回: {"mTile": 16, "nTile": 32, "kTile": 64}
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        hidden_dims: List[int],
        operator_type: Optional[str] = None,
        core_num: int = 20,
        selection_method: str = "greedy",
        selection_top_k: int = 10,
        selector_seed: Optional[int] = None,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 2,
    ):
        """
        初始化Tiling预测器
        
        Args:
            model_path: PyTorch模型参数文件路径(.pth)
            scaler_path: scaler文件路径(.npz)
            hidden_dims: MLP隐藏层配置，如 [64, 128, 64]
            operator_type: Catlass算子类型，可选: "SmallMatmulKernel", "CommonMatmulKernel", 
                          "PaddingMatmulKernel" 或 None(所有)
            core_num: AI Core数量，默认20
        """
        # 加载模型和scaler
        self.predictor_ctx = load_torch_predictor(
            model_path=model_path,
            scaler_path=scaler_path,
            hidden_dims=hidden_dims,
        )
        
        # 初始化Catlass参数生成器
        self.catlass_param_generator = CatlassParameter(
            operator_type=operator_type,
            core_num=core_num,
        )
        self.selection_method = selection_method
        self.selection_top_k = selection_top_k
        self.selector_seed = selector_seed
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        print(f"[TilingPredictor] 初始化完成，模型路径: {model_path}")
    
    def predict(
        self,
        m: int,
        n: int,
        k: int,
    ) -> Optional[Dict[str, int]]:
        """
        根据 M、N、K 预测最优的 tiling 参数
        
        Args:
            m: 矩阵 M 维度
            n: 矩阵 N 维度
            k: 矩阵 K 维度
        
        Returns:
            最优tiling参数字典，格式为 {"mTile": int, "nTile": int, "kTile": int}
            如果没有可用参数，返回None
        """
        shape = [m, n, k]
        
        # 1. 生成所有可用的tiling参数
        params = self.catlass_param_generator.filter_parameters(shape)
        if not params:
            print(f"[TilingPredictor.predict] shape {shape} 无可用参数")
            return None
        
        # 2. 构建特征矩阵
        features = build_feature_matrix(shape, params, self.predictor_ctx["feature_dim"])
        
        # 3. 批量预测（自动处理分批，每批最多2048，带重试机制）
        try:
            preds = predict_batch(self.predictor_ctx, features)
        except RuntimeError as e:
            print(f"[TilingPredictor.predict] 模型推理失败 shape={shape}: {e}")
            return None
        except Exception as e:
            print(f"[TilingPredictor.predict] 模型推理异常 shape={shape}: {e}")
            return None
        
        best_param, _ = predict_best_tiling(
            shape[0],
            shape[1],
            shape[2],
            self.predictor_ctx,
            self.catlass_param_generator,
            selection_method=self.selection_method,
            top_k=self.selection_top_k,
            random_state=self.selector_seed,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
        )
        return best_param


def predict_best_tiling(
    m: int,
    n: int,
    k: int,
    predictor_ctx: Dict[str, Any],
    catlass_param_generator: CatlassParameter,
    selection_method: str = "greedy",
    top_k: int = 10,
    random_state: Optional[int] = None,
    dbscan_eps: float = 0.8,
    dbscan_min_samples: int = 2,
) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
    """
    根据 M、N、K 通过模型预测最优的 tiling 参数（内部函数，保留用于向后兼容）
    
    Args:
        m: 矩阵 M 维度
        n: 矩阵 N 维度
        k: 矩阵 K 维度
        predictor_ctx: 模型预测上下文（包含model, mean, std, device, feature_dim）
        catlass_param_generator: Catlass参数生成器，用于生成可用的tiling参数
    
    Returns:
        (best_param, predicted_time): 
        - best_param: 最优tiling参数字典，格式为 {"mTile": int, "nTile": int, "kTile": int}
          如果没有可用参数，返回None
        - predicted_time: 预测的执行时间（微秒），如果没有可用参数，返回None
    """
    shape = [m, n, k]
    
    # 1. 生成所有可用的tiling参数
    params = catlass_param_generator.filter_parameters(shape)
    if not params:
        print(f"[predict_best_tiling] shape {shape} 无可用参数")
        return None, None
    
    # 2. 构建特征矩阵
    features = build_feature_matrix(shape, params, predictor_ctx["feature_dim"])
    
    # 3. 批量预测（自动处理分批，每批最多512，带重试机制）
    try:
        preds = predict_batch(predictor_ctx, features)
    except RuntimeError as e:
        print(f"[predict_best_tiling] 模型推理失败 shape={shape}: {e}")
        return None, None
    except Exception as e:
        print(f"[predict_best_tiling] 模型推理异常 shape={shape}: {e}")
        return None, None
    
    best_param, predicted_time = select_tiling_strategy(
        params,
        preds,
        method=selection_method,
        top_k=top_k,
        random_state=random_state,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    
    if best_param is not None:
        print(
            f"[predict_best_tiling] shape={shape}, "
            f"最优参数: {best_param}, 预测时间: {predicted_time:.3f}us (method={selection_method})"
        )
    return best_param, predicted_time


def parse_hidden_dims(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def select_tiling_strategy(
    params: List[Dict[str, int]],
    preds: np.ndarray,
    method: str = "greedy",
    top_k: int = 10,
    random_state: Optional[int] = None,
    dbscan_eps: float = 0.8,
    dbscan_min_samples: int = 2,
) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
    """
    根据不同策略从预测结果中选择tiling。
    方法1（greedy）：直接选预测时间最短者。
    方法2（topk_median）：先取TopK，再选中位数。
    方法3（topk_dbscan）：先取TopK，再用DBSCAN挑选性能好且规模大的簇。
    """
    if not params or preds is None or len(preds) == 0:
        return None, None
    
    preds = np.asarray(preds, dtype=np.float32)
    sorted_indices = np.argsort(preds)
    top_k = max(1, min(top_k, len(sorted_indices)))
    top_indices = sorted_indices[:top_k]
    
    if method == "greedy":
        best_idx = sorted_indices[0]
        return params[best_idx], float(preds[best_idx])
    
    if method == "topk_median":
        median_pos = len(top_indices) // 2
        best_idx = top_indices[median_pos]
        return params[best_idx], float(preds[best_idx])
    
    if method == "topk_dbscan":
        if not SKLEARN_AVAILABLE:
            print("[select_tiling_strategy] 警告：scikit-learn未安装，回退到greedy策略。")
            best_idx = sorted_indices[0]
            return params[best_idx], float(preds[best_idx])
        
        features = []
        for idx in top_indices:
            param = params[idx]
            features.append([
                preds[idx],
                float(param["mTile"]),
                float(param["nTile"]),
                float(param["kTile"]),
            ])
        features = np.asarray(features, dtype=np.float32)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0
        scaled_features = (features - mean) / std
        
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = clustering.fit_predict(scaled_features)
        
        clusters: Dict[int, Dict[str, Any]] = {}
        for label, idx in zip(labels, top_indices):
            if label == -1:
                continue
            clusters.setdefault(label, {"indices": []})
            clusters[label]["indices"].append(idx)
        
        if not clusters:
            best_idx = top_indices[0]
            return params[best_idx], float(preds[best_idx])
        
        cluster_scores = []
        for label, data in clusters.items():
            indices = data["indices"]
            times = preds[indices]
            median_time = float(np.median(times))
            cluster_scores.append((label, median_time, len(indices)))
        
        # 同时考虑性能（median时间越小越好）与簇规模（越大越好）
        min_median = min(score[1] for score in cluster_scores)
        max_size = max(score[2] for score in cluster_scores)
        # 避免除零
        max_size = max(max_size, 1)
        weighted_scores = []
        for label, median_time, size in cluster_scores:
            norm_time = median_time / max(min_median, 1e-6)
            norm_size = max_size / size
            combined_score = 0.7 * norm_time + 0.3 * norm_size
            weighted_scores.append((combined_score, label))
        weighted_scores.sort(key=lambda x: x[0])
        best_label = weighted_scores[0][1]
        rng = random.Random(random_state)
        chosen_idx = rng.choice(clusters[best_label]["indices"])
        return params[chosen_idx], float(preds[chosen_idx])
    
    # 默认兜底
    best_idx = sorted_indices[0]
    return params[best_idx], float(preds[best_idx])


def load_processed_shapes(csv_path: Path) -> set:
    """
    从CSV文件中加载已处理的shape集合
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        已处理的shape集合，每个元素为 "M_N_K" 格式的字符串
    """
    processed_shapes = set()
    if not csv_path.exists():
        return processed_shapes
    
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "shape" in row:
                    processed_shapes.add(row["shape"])
                elif "M" in row and "N" in row and "K" in row:
                    # 如果没有shape列，从M、N、K列构建
                    shape_key = f"{row['M']}_{row['N']}_{row['K']}"
                    processed_shapes.add(shape_key)
    except Exception as e:
        print(f"[load_processed_shapes] 读取CSV文件失败: {e}，将重新处理所有shape")
    
    return processed_shapes


def main():
    parser = argparse.ArgumentParser(description="Catlass tiling 预测+验证脚本（无GA）")
    parser.add_argument("--shapes-file", type=str, required=True, help="shapes Excel文件路径")
    parser.add_argument("--output", type=str, default="compare_catlass.csv", help="输出结果文件路径")
    parser.add_argument("--start-idx", type=int, default=None, help="测试shapes的起始索引（从0开始，包含该位置）")
    parser.add_argument("--end-idx", type=int, default=None, help="测试shapes的结束索引（不包含该位置，None表示到末尾）")

    # catlass / msprof
    parser.add_argument("--catlass-bin-path", type=str, default="/home/q30063557/code/cutlass/21_dynamic_tiling_matmul", help="catlass可执行文件路径")
    parser.add_argument("--rank-id", type=int, default=0, help="msprof rank id (默认: 0)")
    parser.add_argument("--msp-dir", type=str, default="./msp", help="msprof输出目录 (默认: ./msp)")
    parser.add_argument("--operator-type", type=str, default=None,
                        choices=["SmallMatmulKernel", "CommonMatmulKernel", "PaddingMatmulKernel"],
                        help="Catlass算子类型，可选: SmallMatmulKernel, CommonMatmulKernel, PaddingMatmulKernel (默认: None 表示所有算子)")
    parser.add_argument("--core-num", type=int, default=20, help="AI Core数量 (默认: 20)")

    # PyTorch 模型
    parser.add_argument("--torch-model-path", type=str, required=True, help="PyTorch模型参数文件(.pth)")
    parser.add_argument("--torch-scaler-path", type=str, required=True, help="scaler文件(.npz)")
    parser.add_argument("--hidden-dims", type=str, default="64, 128, 64",
                        help="MLP隐藏层配置，逗号分隔 (默认: 64, 128, 64)")
    
    # tiling 选择策略
    parser.add_argument(
        "--tiling-selector",
        type=str,
        choices=["greedy", "topk_median", "topk_dbscan"],
        default="greedy",
        help="tiling选择策略：greedy/topk_median/topk_dbscan"
    )
    parser.add_argument(
        "--selector-topk",
        type=int,
        default=10,
        help="topk_* 策略中的TopK数量 (默认: 10)"
    )
    parser.add_argument(
        "--selector-seed",
        type=int,
        default=None,
        help="随机选择时使用的随机种子"
    )
    parser.add_argument(
        "--selector-dbscan-eps",
        type=float,
        default=0.8,
        help="DBSCAN的eps参数 (默认: 0.8)"
    )
    parser.add_argument(
        "--selector-dbscan-min-samples",
        type=int,
        default=2,
        help="DBSCAN的min_samples参数 (默认: 2)"
    )

    args = parser.parse_args()

    # 加载shapes，支持索引切片
    shapes = load_shapes_from_excel(
        args.shapes_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    if not shapes:
        print(f"[main] 未从 {args.shapes_file} 中读取到任何shape")
        return
    
    # 打印索引范围信息
    if args.start_idx is not None or args.end_idx is not None:
        print(f"[main] 使用shapes索引范围: [{args.start_idx or 0}:{args.end_idx or 'end'}]，共 {len(shapes)} 个shape")

    msprof_executor = MsProfExecutor(
        catlass_bin_path=args.catlass_bin_path,
        rank_id=args.rank_id,
        msp_dir=args.msp_dir,
    )

    predictor_ctx = load_torch_predictor(
        model_path=args.torch_model_path,
        scaler_path=args.torch_scaler_path,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
    )

    # operator_type 直接传递给 CatlassParameter，None 表示所有算子
    catlass_param_generator = CatlassParameter(
        operator_type=args.operator_type,
        core_num=args.core_num,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "M",
        "N",
        "K",
        "predicted_time_us",
        "real_time_us",
        "best_param",
        "default_time_us",
    ]
    if not output_path.exists():
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # 加载已处理的shape集合
    processed_shapes = load_processed_shapes(output_path)
    if processed_shapes:
        print(f"[main] 已加载 {len(processed_shapes)} 个已处理的shape，将跳过这些shape")

    # 统计信息
    stats = {
        "total": len(shapes),
        "already_processed": 0,
        "no_params": 0,
        "prediction_failed": 0,
        "success": 0,
    }

    for idx, shape in enumerate(shapes, 1):
        print(f"\n===== [{idx}/{len(shapes)}] 处理Shape: {shape} =====")
        m, n, k = shape
        shape_key = f"{m}_{n}_{k}"
        
        # 检查是否已经处理过
        if shape_key in processed_shapes:
            print(f"[main] shape {shape} 已在CSV文件中，跳过")
            stats["already_processed"] += 1
            continue
        
        # 先检查是否有可用参数，如果没有则跳过模型推理
        params = catlass_param_generator.filter_parameters(shape)
        if not params:
            print(f"[main] shape {shape} 无可用参数，跳过模型推理和msprof测量")
            stats["no_params"] += 1
            continue
        
        # 使用模型预测最优tiling参数
        best_param, predicted_time = predict_best_tiling(
            m,
            n,
            k,
            predictor_ctx,
            catlass_param_generator,
            selection_method=args.tiling_selector,
            top_k=args.selector_topk,
            random_state=args.selector_seed,
            dbscan_eps=args.selector_dbscan_eps,
            dbscan_min_samples=args.selector_dbscan_min_samples,
        )
        
        if best_param is None:
            print(f"[main] shape {shape} 模型推理失败或无可用参数，跳过")
            stats["prediction_failed"] += 1
            continue

        # 测量真实执行时间
        real_time = measure_tiling_time(msprof_executor, shape, best_param)
        default_time_us = measure_tiling_time(msprof_executor, shape, None)

        row = {
            "shape": f"{m}_{n}_{k}",
            "M": m,
            "N": n,
            "K": k,
            "predicted_time_us": predicted_time,
            "best_param": best_param,
            "real_time_us": real_time,
            "default_time_us": default_time_us,
        }
        with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
        
        stats["success"] += 1

    # 打印统计信息
    print("\n" + "=" * 80)
    print("处理统计信息")
    print("=" * 80)
    print(f"总测试数据: {stats['total']}")
    print(f"  ✓ 成功处理: {stats['success']}")
    print(f"  - 已在CSV中（跳过）: {stats['already_processed']}")
    print(f"  - 无可用参数（跳过）: {stats['no_params']}")
    print(f"  - 模型推理失败（跳过）: {stats['prediction_failed']}")
    print(f"  = 已写入CSV: {stats['success'] + stats['already_processed']}")
    print(f"  = 未写入CSV: {stats['no_params'] + stats['prediction_failed']}")
    print("=" * 80)
    
    print(f"\n预测+验证结果写入 {output_path.resolve()}")


# ============================================================================
# 外部调用示例
# ============================================================================
#
# ```python
# from compare_catlass import TilingPredictor
#
# # 初始化预测器
# predictor = TilingPredictor(
#     model_path="path/to/model.pth",
#     scaler_path="path/to/scaler.npz",
#     hidden_dims=[64, 128, 64],
#     operator_type="SmallMatmulKernel",  # 或 None 表示所有算子
#     core_num=20
# )
#
# # 预测任意 M、N、K 的最优 tiling 参数
# best_tiling = predictor.predict(m=128, n=256, k=512)
# # 返回: {"mTile": 16, "nTile": 32, "kTile": 64}
#
# if best_tiling:
#     print(f"最优tiling: mTile={best_tiling['mTile']}, "
#           f"nTile={best_tiling['nTile']}, kTile={best_tiling['kTile']}")
# ```
# ============================================================================


if __name__ == "__main__":
    main()
