import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import os

from catlass_parameter import CatlassParameter
from model import TimePredictMLP
from padding_calculator import PaddingCalculator, PaddingTag
from tiling_calculator import MatmulTilingCalculator

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

# ===================== 版本配置 =====================
VERSION_CONFIG = {
    "A2": {
        "hidden_dims_small": "85, 171, 342, 684, 1369, 2739, 1369, 684, 342, 171, 85",
        "hidden_dims_common": "60, 121, 242, 484, 968, 1937, 968, 484, 242, 121, 60",
        "hidden_dims_padding": "498, 498, 498, 498, 498, 498, 498, 498, 498",
    },
    "A3": {
        "hidden_dims_small": "1116, 1116, 1116, 1116, 1116, 1116, 1116, 1116, 1116",
        "hidden_dims_common": "242, 485, 970, 1940, 3880, 1940, 970, 485, 242",
        "hidden_dims_padding": "5453, 2726, 1363, 681, 340, 170, 85, 42, 21, 17, 17, 17, 17, 17, 17",
    }
}
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(description="根据shape寻找最优tiling")

    parser.add_argument(
        "--model-version",
        type=str,
        choices=["A2", "A3"],
        default="A2",
        help="模型版本：A2 或 A3，指定后自动使用 ./model_{version}/ 下的模型和对应 hidden_dims"
    )

    # -------------------------- 数据相关参数 --------------------------
    parser.add_argument(
        "--selection-method",
        type=str,
        choices=["greedy", "topk_median", "topk_dbscan"],
        default="greedy",
        help="tiling选择策略：greedy/topk_median/topk_dbscan(默认: greedy)"
    )
    parser.add_argument(
        "--selection-topk",
        type=int,
        default=10,
        help="topk_* 策略中的TopK数量 (默认: 10)"
    )
    parser.add_argument(
        "--min-tiling",
        type=int,
        default=60,
        help="最小tiling数量阈值，如果筛选出的tiling数量不足此值，将回退到catlass原生计算方式 (默认: 60)"
    )
    parser.add_argument(
        "--time-diff-threshold",
        type=float,
        default=0.06,
        help="耗时差异阈值（0.0-1.0），如果模型预测的最优tiling耗时 > (1-threshold)*catlass原生tiling耗时，将回退到原生tiling (默认: 0.05，即5%)"
    )

    # -------------------------- 模型相关信息 --------------------------
    parser.add_argument(
        "--model-path-small",
        type=str,
        default="best_mlp_model_small.pth",
        help="small model模型权重路径"
    )
    parser.add_argument(
        "--scaler-path-small",
        type=str,
        default="scaler_small.npz",
        help="small model数据归一化路径"
    )
    parser.add_argument(
        "--hidden-dims-small",
        type=str,
        default="1116, 1116, 1116, 1116, 1116, 1116, 1116, 1116, 1116",
        help="small MLP隐藏层配置，逗号分隔"
    )
    parser.add_argument(
        "--model-path-common",
        type=str,
        default="best_mlp_model_common.pth",
        help="common model模型权重路径"
    )
    parser.add_argument(
        "--scaler-path-common",
        type=str,
        default="scaler_common.npz",
        help="common model数据归一化路径"
    )
    parser.add_argument(
        "--hidden-dims-common",
        type=str,
        default="242, 485, 970, 1940, 3880, 1940, 970, 485, 242",
        help="common MLP隐藏层配置，逗号分隔"
    )
    parser.add_argument(
        "--model-path-padding",
        type=str,
        default="best_mlp_model_padding.pth",
        help="padding model模型权重路径"
    )
    parser.add_argument(
        "--scaler-path-padding",
        type=str,
        default="scaler_padding.npz",
        help="padding model数据归一化路径"
    )
    parser.add_argument(
        "--hidden-dims-padding",
        type=str,
        default="5453, 2726, 1363, 681, 340, 170, 85, 42, 21, 17, 17, 17, 17, 17, 17",
        help="padding MLP隐藏层配置，逗号分隔"
    )
    # 解析参数
    args = parser.parse_args()

    # 若指定了版本，则覆盖路径和隐藏层配置
    if args.model_version is not None:
        ver = args.model_version
        prefix = os.path.dirname(__file__) + f"/model_{ver}/"
        args.model_path_small = prefix + "best_mlp_model_small.pth"
        args.scaler_path_small = prefix + "scaler_small.npz"
        args.model_path_common = prefix + "best_mlp_model_common.pth"
        args.scaler_path_common = prefix + "scaler_common.npz"
        args.model_path_padding = prefix + "best_mlp_model_padding.pth"
        args.scaler_path_padding = prefix + "scaler_padding.npz"

        args.hidden_dims_small = VERSION_CONFIG[ver]["hidden_dims_small"]
        args.hidden_dims_common = VERSION_CONFIG[ver]["hidden_dims_common"]
        args.hidden_dims_padding = VERSION_CONFIG[ver]["hidden_dims_padding"]

    return args

def parse_hidden_dims(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]

# ===================== TilingPredictor 类 =====================
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
            selection_topk: int = 10,
            selector_seed: Optional[int] = None,
            dbscan_eps: float = 0.8,
            dbscan_min_samples: int = 2,
            min_tiling: int = 40,
            time_diff_threshold: float = 0.05,
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
        self.predictor_ctx = self.load_torch_predictor(
            model_path=model_path,
            scaler_path=scaler_path,
            hidden_dims=hidden_dims,
        )
        self.operator_type = operator_type
        self.core_num = core_num
        self.selection_method = selection_method
        self.selection_topk = selection_topk
        self.selector_seed = selector_seed
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_tiling = min_tiling
        self.time_diff_threshold = time_diff_threshold

    def set_catlass_param_generator(self, layout_tag_a, layout_tag_b):
        # 初始化Catlass参数生成器
        self.catlass_param_generator = CatlassParameter(
            operator_type = self.operator_type,
            layout_tag_a = layout_tag_a,
            layout_tag_b = layout_tag_b,
            core_num = self.core_num,
        )

    def detect_device(self) -> torch.device:
        # if torch.cuda.is_available():
        #     return torch.device("cuda:0")
        # if hasattr(torch, "npu") and torch.npu.is_available():
        #     return torch.device("npu:0")
        return torch.device("cpu")

    def load_torch_predictor(self, model_path: str, scaler_path: str, hidden_dims: List[int]):
        device = self.detect_device()
        mean_arr, std_arr = self.load_scaler_arrays(scaler_path)
        feature_dim = mean_arr.shape[0]
        if feature_dim != len(EXTENDED_FEATURES):
            raise ValueError(
                f"scaler维度不匹配，期望{len(EXTENDED_FEATURES)}，实际{feature_dim}"
            )

        model = TimePredictMLP(input_dim=feature_dim, hidden_dims=hidden_dims)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        mean = torch.tensor(mean_arr, dtype=torch.float32, device=device)
        std = torch.tensor(std_arr, dtype=torch.float32, device=device)

        return {"model": model, "mean": mean, "std": std, "device": device, "feature_dim": feature_dim}

    def load_scaler_arrays(self, scaler_path: str) -> Tuple[np.ndarray, np.ndarray]:
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
        std_arr = scaler_data["std"].astype(np.float32)
        std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
        return mean_arr, std_arr

    def build_feature_matrix(self, shape: List[int], params: List[Dict[str, int]], feature_dim: int) -> np.ndarray:
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
            self,
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
                            f"[predict_batch] 第{attempt + 1}次推理失败（耗时{elapsed_time:.2f}秒）: {e}，"
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

    def select_tiling_strategy(
            self,
            params: List[Dict[str, int]],
            preds: np.ndarray,
            method: str = "greedy",
            topk: int = 10,
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
        topk = max(1, min(topk, len(sorted_indices)))
        top_indices = sorted_indices[:topk]

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

    def predict_best_tiling(self,
            m: int,
            n: int,
            k: int,
            predictor_ctx: Dict[str, Any],
            catlass_param_generator: CatlassParameter,
            selection_method: str = "greedy",
            topk: int = 10,
            random_state: Optional[int] = None,
            dbscan_eps: float = 0.8,
            dbscan_min_samples: int = 2,
            tiling: tuple = None,
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
        features = self.build_feature_matrix(shape, params, predictor_ctx["feature_dim"])

        # 3. 批量预测（自动处理分批，每批最多512，带重试机制）
        try:
            preds = self.predict_batch(predictor_ctx, features)
        except RuntimeError as e:
            print(f"[predict_best_tiling] 模型推理失败 shape={shape}: {e}")
            return None, None
        except Exception as e:
            print(f"[predict_best_tiling] 模型推理异常 shape={shape}: {e}")
            return None, None

        best_param, predicted_time = self.select_tiling_strategy(
            params,
            preds,
            method=selection_method,
            topk=topk,
            random_state=random_state,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        # 4. 回退机制
        # 统计筛选出的tiling数量
        num_tilings = len(params)
        print(f"[main] shape {shape} 筛选出 {num_tilings} 个tiling参数")
        #native_tiling = tiling
        native_tiling = [{k: v // 16 for k, v in zip(["mTile", "nTile", "kTile"], tiling)}] if tiling is not None else None
        if num_tilings < self.min_tiling:
            print(
                f"[main] shape {shape} tiling数量不足（{num_tilings} < {self.min_tiling}），fallback到catlass原生计算方式")
            if native_tiling is not None:
                best_param = native_tiling[0]
                features = self.build_feature_matrix(shape, [best_param], predictor_ctx["feature_dim"])
                native_predicted_time = self.predict_batch(predictor_ctx, features).item()
                predicted_time = native_predicted_time
            else:
                print(f"[main] shape {shape} 无法获取catlass原生tiling，跳过")
                return None, None
        else:
            if best_param is None:
                print(f"[main] shape {shape} 模型推理失败或无可用参数，跳过")
                return None, None

            # 预测catlass原生tiling的耗时
            if native_tiling is not None:
                features = self.build_feature_matrix(shape, native_tiling, predictor_ctx["feature_dim"])
                native_predicted_time = self.predict_batch(predictor_ctx, features).item()
                # 比较两个耗时：如果模型预测的最优tiling耗时 > (1-threshold)*catlass原生tiling耗时，则fallback
                if native_predicted_time is not None and predicted_time is not None:
                    threshold = self.time_diff_threshold
                    threshold_time = (1 - threshold) * native_predicted_time
                    if predicted_time > threshold_time:
                        print(
                            f"[main] shape {shape} 模型预测tiling耗时({predicted_time:.3f}us) > (1-{threshold * 100:.1f}%)*catlass原生tiling耗时({threshold_time:.3f}us)，")
                        best_param = native_tiling[0]
                        predicted_time = native_predicted_time

        if best_param is not None:
            if isinstance(best_param, list) and best_param:
                best_param = best_param[0]
            print(
                f"[predict_best_tiling] shape={shape}, "
                f"最优参数: {best_param}, 预测时间: {predicted_time:.3f}us (method={selection_method})"
            )
        return best_param, predicted_time

    def predict(
            self,
            m: int,
            n: int,
            k: int,
            tiling: tuple = None,
            layout_tag_a = 0,
            layout_tag_b = 1
    ) -> Optional[Dict[str, int]]:
        """
        根据 M、N、K 预测最优的 tiling 参数

        Args:
            m: 矩阵 M 维度
            n: 矩阵 N 维度
            k: 矩阵 K 维度
            tiling: catlass 默认tiling用于回退机制
        Returns:
            最优tiling参数字典，格式为 {"mTile": int, "nTile": int, "kTile": int}
            如果没有可用参数，返回None
        """
        shape = [m, n, k]
        # 设置catlass_param_generator的layout
        self.set_catlass_param_generator(layout_tag_a, layout_tag_b)

        best_param, _ = self.predict_best_tiling(
            shape[0],
            shape[1],
            shape[2],
            self.predictor_ctx,
            self.catlass_param_generator,
            selection_method=self.selection_method,
            topk=self.selection_topk,
            random_state=self.selector_seed,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            tiling=tiling,
        )
        return best_param

# ===================== GetBestConfig 类 =====================
class GetBestConfig:
    def __init__(self):
        args = parse_args()
        self.predictor_small = TilingPredictor(
            model_path=args.model_path_small,
            scaler_path=args.scaler_path_small,
            hidden_dims=parse_hidden_dims(args.hidden_dims_small),
            operator_type="SmallMatmul",  # 或 None 表示所有算子
            core_num=20,
            min_tiling=args.min_tiling,
            time_diff_threshold=args.time_diff_threshold,
            selection_method=args.selection_method,
            selection_topk=args.selection_topk,
        )
        self.predictor_common = TilingPredictor(
            model_path=args.model_path_common,
            scaler_path=args.scaler_path_common,
            hidden_dims=parse_hidden_dims(args.hidden_dims_common),
            operator_type="CommonMatmul",  # 或 None 表示所有算子
            core_num=20,
            min_tiling=args.min_tiling,
            time_diff_threshold=args.time_diff_threshold,
            selection_method=args.selection_method,
            selection_topk=args.selection_topk,
        )
        self.predictor_padding = TilingPredictor(
            model_path=args.model_path_padding,
            scaler_path=args.scaler_path_padding,
            hidden_dims=parse_hidden_dims(args.hidden_dims_padding),
            operator_type="PaddingCommonMatmul",  # 或 None 表示所有算子
            core_num=20,
            min_tiling=args.min_tiling,
            time_diff_threshold=args.time_diff_threshold,
            selection_method=args.selection_method,
            selection_topk=args.selection_topk,
        )

        self.matmul_tiling_calculator = MatmulTilingCalculator()

    def predict(self, m, n, k, layout_tag_a, layout_tag_b):
        shape = [m, n, k]
        args = shape + [layout_tag_a, layout_tag_b]  # 合并参数
        # 1.计算catlass的tiling和kerneltype
        result = self.matmul_tiling_calculator.calculate(*args)
        # 2.使用训练好的模型寻找最优tiling（包含回退机制）
        if result["operator_type"] == "SmallMatmul":
            result["predict_tiling"] = self.predictor_small.predict(*shape, result["tiling"], layout_tag_a, layout_tag_b)
        elif result["operator_type"] == "CommonMatmul":
            result["predict_tiling"] = self.predictor_common.predict(*shape, result["tiling"], layout_tag_a, layout_tag_b)
        elif result["operator_type"] == "PaddingCommonMatmul":
            result["predict_tiling"] = self.predictor_padding.predict(*shape, result["tiling"], layout_tag_a, layout_tag_b)

        if result.get("predict_tiling") is None:
            result["predict_tiling"] = dict(zip(["m1", "n1", "k1"], result["tiling"]))
        else:
            mapping = {"mTile": "m1", "nTile": "n1", "kTile": "k1"}
            result["predict_tiling"] = {
                mapping[k]: v * 16
                for k, v in result["predict_tiling"].items()
                if k in mapping
            }
            result["block_dim"] = self.matmul_tiling_calculator.calculate_block_dim(
                m, n, k,
                result["predict_tiling"]["m1"],
                result["predict_tiling"]["n1"],
                result["predict_tiling"]["k1"],
                result["layout"][0],
                result["layout"][1],
                result["paddingTagA"],
                result["paddingTagB"],
                result["splitkFactor"],
            )
        return result

# ===================== main =====================
def main():
    get_best_config = GetBestConfig()

    # 预测任意 M、N、K 的最优 tiling 参数
    best_tiling = get_best_config.predict(m=72, n=7392, k=8192, layout_tag_a=0, layout_tag_b=1)
    best_tiling = get_best_config.predict(m=1, n=4, k=4, layout_tag_a=0, layout_tag_b=0)
    # 返回: {"m1": 16, "n1": 32, "k1": 64}

    if best_tiling:
        for key, value in best_tiling.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()