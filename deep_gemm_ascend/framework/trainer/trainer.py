import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time
import argparse

torch.random.manual_seed(2025)
np.random.seed(2025)

# -------------------------- 1. 命令行参数解析函数（仅保留核心参数） --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="时间预测模型")
    
    # -------------------------- 数据相关参数 --------------------------
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="merged_excel", 
        help="预处理数据文件路径（默认：processed_data.npz）"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=512, 
        help="训练/验证/测试的批次大小（默认：512，需为正整数）"
    )
    
    # -------------------------- 模型结构参数 --------------------------
    parser.add_argument(
        "--hidden-dims", 
        type=str, 
        default="64, 128, 64", 
        help="隐藏层维度列表（格式：逗号分隔，如'64,32,16'，默认：64,32,12）"
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        default=13,
        help="输入维度"
    )
    # -------------------------- 训练策略参数 --------------------------
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000, 
        help="最大训练轮次（默认：1000，需为正整数）"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=25, 
        help="早停耐心值（连续N轮验证损失无改善则停止，默认：25）"
    )
    parser.add_argument(
        "--loss", 
        type=str, 
        default="mse", 
        choices=["mse", "mae", "smoothl1"], 
        help="选择损失函数（mse:均方误差, mae:平均绝对误差, smoothl1:平滑L1损失，默认: mse）"
    )
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="adam", 
        choices=["adam", "sgd", "rmsprop", "adamw"], 
        help="选择优化器（默认: adam）"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        help="初始学习率（默认：0.01，建议范围：1e-4 ~ 1e-2）"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="动量（仅SGD优化器生效，默认: 0.9）")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减（默认: 1e-5）")
    parser.add_argument("--alpha", type=float, default=0.99, help="平滑常数（仅RMSprop生效，默认: 0.99）")
    # -------------------------- 设备参数 --------------------------
    parser.add_argument(
        "--npu-id", 
        type=int, 
        default=0, 
        help="指定使用的NPU设备编号（如0、1，默认：0；仅当NPU可用时生效）"
    )
    # -------------------------- 保存参数 --------------------------
    parser.add_argument(
        "--model-save-path", 
        type=str, 
        default="best_mlp_model.pth", 
        help="最佳模型保存路径（默认：best_mlp_model.pth）"
    )
    parser.add_argument(
        "--scaler-save-path", 
        type=str, 
        default="scaler.npz", 
        help="特征标准化参数保存路径（默认：scaler.npz）"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # -------------------------- 参数合法性校验 --------------------------
    if args.batch_size <= 0:
        raise ValueError("batch-size必须为正整数！")
    try:
        args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
        if any(dim <= 0 for dim in args.hidden_dims):
            raise ValueError("隐藏层维度必须为正整数！")
    except ValueError:
        raise ValueError("--hidden-dims格式错误！需为逗号分隔的正整数（如64,32）")
    if args.lr <= 0:
        raise ValueError("学习率必须为正数！")
    if args.npu_id < 0:
        raise ValueError("NPU编号--npu-id必须为非负整数！")
    if not (0 < args.momentum < 1):
        raise ValueError("动量值必须在(0, 1)范围内！")
    
    return args

# -------------------------- 2. 设备配置（支持指定NPU编号） --------------------------
def get_device(args):
    """根据命令行指定的--npu-id选择设备，若NPU不可用则使用CPU"""
    # 1. 检查NPU是否可用
    if torch.npu.is_available():
        # 2. 检查指定的NPU编号是否在可用范围内
        available_npu_count = torch.npu.device_count()
        if args.npu_id >= available_npu_count:
            raise RuntimeError(
                f"指定的NPU设备编号{args.npu_id}超出可用范围！"
                f"当前环境仅可用{available_npu_count}个NPU（编号0~{available_npu_count-1}）"
            )
        # 3. 使用指定的NPU设备
        device = torch.device(f"npu:{args.npu_id}")
        print(f"使用指定的NPU设备: {device}（当前环境共{available_npu_count}个NPU可用）")
    else:
        # 4. NPU不可用，切换到CPU
        device = torch.device("cpu")
        print(f"未检测到NPU设备，使用CPU设备: {device}")
    return device

# -------------------------- 3. 数据类（无修改） --------------------------
def process_data(args):
    # 定义文件夹路径
    folder_path = args.data_path

    # 初始化空列表存储所有数据
    input_data = []
    labels = []
    test_input_data = []
    test_labels = []

    # 统计文件夹中Excel文件个数
    file_count = 0
    try:
        contents = os.listdir(folder_path)
        for item in contents:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                file_count += 1
        print(f"文件夹 '{folder_path}' 中有 {file_count} 个文件。")
    except FileNotFoundError:
        print(f"路径'{folder_path}' 不存在。")
    except PermissionError:
        print(f"没有权限访问路径 '{folder_path}' 。")

    count = 0
    # 遍历文件夹中的所有Excel文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            count += 1
        if count < 0.95 * file_count:
            file_path = os.path.join(folder_path, filename)
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 提取输入数据和标签数据
            inputs = df[['M', 'N', 'K', 'mTile', 'nTile', 'kTile']].values
            # 计算交叉项
            cross_term = [
                inputs[:, 0] * inputs[:, 1], # M * N
                inputs[:, 0] * inputs[:, 0], # M * K
                inputs[:, 1] * inputs[:, 2], # N * K
                inputs[:, 3] * inputs[:, 4], # mTile * nTile
                inputs[:, 3] * inputs[:, 5], # mTile * kTile
                inputs[:, 4] * inputs[:, 5], # nTile * kTile
                np.ceil(inputs[:, 1] / inputs[:, 3]) * np.ceil(inputs[:, 2] / inputs[:, 4]), # AI core
            ]

            # 将交叉项转换为二维数组
            cross_term_array = np.column_stack(cross_term)

            # 拼接原始输入和交叉项
            combined_inputs = np.concatenate([inputs, cross_term_array], axis=1)

            # 转换为Pytorch Tensor
            inputs = torch.tensor(combined_inputs, dtype=torch.float32)
            target = df['time'].values

            # 将数据添加到列表中
            input_data.append(inputs)
            labels.append(target)
        else:
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)

            # 提取输入数据和标签数据
            inputs = df[['M', 'N', 'K', 'mTile', 'nTile', 'kTile']].values
            # 计算交叉项
            cross_term = [
                inputs[:, 0] * inputs[:, 1], # M * N
                inputs[:, 0] * inputs[:, 0], # M * K
                inputs[:, 1] * inputs[:, 2], # N * K
                inputs[:, 3] * inputs[:, 4], # mTile * nTile
                inputs[:, 3] * inputs[:, 5], # mTile * kTile
                inputs[:, 4] * inputs[:, 5], # nTile * kTile
                np.ceil(inputs[:, 0] / inputs[:, 3]) * np.ceil(inputs[:, 1] / inputs[:, 4]), # AI core
            ]

            # 将交叉项转换为二维数组
            cross_term_array = np.column_stack(cross_term)

            # 拼接原始输入和交叉项
            combined_inputs = np.concatenate([inputs, cross_term_array], axis=1)

            # 转换为Pytorch Tensor
            inputs = torch.tensor(combined_inputs, dtype=torch.float32)
            target = df['time'].values

            # 将数据添加到列表中
            test_input_data.append(inputs)
            test_labels.append(target)

    # 合并所有数据
    input_data = np.concatenate(input_data, axis=0)
    labels = np.concatenate(labels, axis=0)
    test_input_data = np.concatenate(test_input_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    def filter_invalid_data(input_data, labels):
        # 剔除异常数据
        prior_len = len(labels)
        valid_mask = ~(
            (labels == 'inf') |
            (labels == 999999999) |
            (labels == -1) 
        )
        input_data = input_data[valid_mask]
        labels = labels[valid_mask]
        print(f"根据自定义异常，筛掉了{prior_len - len(labels)} 条异常数据")

        # 根据3σ原则筛选异常值
        mean = np.mean(labels)
        std = np.std(labels)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # 找出满足条件的索引
        prior_len = len(labels)
        valid_indices = np.where((labels >= lower_bound) & (labels <= upper_bound))[0]
        invalid_indices1 = np.where((labels < lower_bound))[0]
        invalid_indices2 = np.where((labels > upper_bound))[0]

        # 筛选数据
        input_data = input_data[valid_indices]
        labels = labels[valid_indices]

        # 打印筛掉的数据数量
        removed_count = prior_len - len(valid_indices)
        print(f'根据3σ原则, 筛掉了 {removed_count}条异常数据，共{len(invalid_indices1)}条低于下界的数据， {len(invalid_indices2)}条高于上界的数据')
        print(f'保留了{len(labels)}条数据')

        return input_data, labels

    input_data, labels = filter_invalid_data(input_data, labels)
    test_input_data, test_labels = filter_invalid_data(test_input_data, test_labels)

    # 划分数据集
    train_size = int(0.7 * len(input_data))
    val_size = int(0.2 * len(input_data))
    test_size = len(input_data) - train_size - val_size

    print(f"数据范围: min={labels.min():.4f} us, max={labels.max():.4f} us, mean={labels.mean():.4f} us")
    print(f"时间跨度比例 (max/min): {labels.max()/labels.min():.2f} 倍")

    # 随机打乱数据
    indices = np.random.permutation(len(input_data))
    input_data = input_data[indices]
    labels = labels[indices]

    # 划分训练集、验证集和测试集
    X_train = input_data[:train_size]
    y_train = labels[:train_size]
    X_val = input_data[train_size:train_size + val_size]
    y_val = labels[train_size:train_size + val_size]
    # 见过shape但没见过tiling的测试集
    X_test = input_data[train_size + val_size:]
    y_test = labels[train_size + val_size:]
    # 添加未见过shape数据到测试集上（混合模式）
    X_test_mix = np.concatenate((X_test, test_input_data), axis=0)
    y_test_mix = np.concatenate((y_test, test_labels), axis=0)

    # 转换为Pytorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1,1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).view(-1,1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1,1)
    X_test_mix = torch.FloatTensor(X_test_mix)
    y_test_mix = torch.FloatTensor(y_test_mix).view(-1,1)

    # 创建数据集
    train_dataset = TimeDataset(
        X_train, y_train,
        log_transform=False
    )
    # 保存特征标准化系数
    np.savez(args.scaler_save_path, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    print(f"特征值标准化参数已保存为：{args.scaler_save_path}")

    # 验证集/测试集
    val_dataset = TimeDataset(
        X_val, y_val,
        mean=train_dataset.mean,
        std=train_dataset.std,
        log_transform=False
    )
    test_dataset = TimeDataset(
        X_test, y_test,
        mean=train_dataset.mean,
        std=train_dataset.std,
        log_transform=False
    )
    test_dataset_mix = TimeDataset(
        X_test_mix, y_test_mix,
        mean=train_dataset.mean,
        std=train_dataset.std,
        log_transform=False
    )

    print(f"特征标准化均值范围：[{train_dataset.mean.min():.4f}, {train_dataset.mean.max():.4f}]")
    print(f"train dataset: {len(train_dataset)}, batch size: {args.batch_size}")
    print(f"val dataset: {len(val_dataset)}, batch size: {args.batch_size}")
    print(f"test dataset: {len(test_dataset)}, batch size: {args.batch_size}")
    print(f"mix test dataset: {len(test_dataset_mix)}, batch size: {args.batch_size}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader_mix = DataLoader(test_dataset_mix, batch_size=batch_size, num_workers=2, shuffle=True)
    return train_dataset, val_dataset, test_dataset, test_dataset_mix, train_loader, val_loader, test_loader, test_loader_mix

class TimeDataset(Dataset):
    def __init__(self, X, y, mean=None, std=None, log_transform=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.log_transform = log_transform
        # 1. 特征标准化
        if mean is None or std is None:
            self.mean = torch.mean(self.X, dim=0)
            self.std = torch.std(self.X, dim=0)
            # 避免除以零
            self.std = torch.where(self.std < 1e-8, torch.tensor(1.0, dtype=torch.float32), self.std)
        else:
            self.mean = mean
            self.std = std
        
        self.X_normalized = (self.X - self.mean) / self.std
        
        # 2. 目标值对数变换
        if self.log_transform:
            # 添加微小值避免log(0)
            self.y_processed = torch.log(self.y + 1e-6)
        else:
            self.y_processed = self.y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X_normalized[idx], self.y_processed[idx]
    
    # 逆变换：将模型输出还原为原始时间尺度
    def inverse_transform_y(self, y_processed):
        if self.log_transform:
            return torch.clamp(torch.exp(y_processed) - 1e-6, min=1e-9)
        else:
            return y_processed

# -------------------------- 4. MLP回归模型（无修改） --------------------------
class TimePredictMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
        super(TimePredictMLP, self).__init__()
        # 构建网络层
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)
        
    def forward(self, x):
        return self.layers(x)

# -------------------------- 5. 训练函数（无修改，仅接收device参数） --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, args, best_model_save_path):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    counter = 0  # 早停计数器

    print(f"\n开始训练，共{args.epochs}轮，设备：{device}，早停耐心值：{args.patience}")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)
        
        # 计算平均损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度器
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        # 早停逻辑（固定min_delta=1e-8）
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_save_path)
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            print(f"早停计数器: {counter}/{args.patience}")
            if counter >= args.patience:
                print(f"早停触发：连续{args.patience}轮验证损失无改善")
                break
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f}秒，实际训练轮次: {epoch+1}")
    print(f"最佳验证损失: {best_val_loss:.6f} (在第{best_epoch}轮)")
    print(f"最佳模型已保存至: {best_model_save_path}")
    
    return model, train_losses, val_losses, best_val_loss


# -------------------------- 6. 评估模型（无修改） --------------------------
def evaluate_model(model, test_loader, criterion, device, test_dataset, best_model_save_path, mode="normal"):
    model.load_state_dict(torch.load(best_model_save_path))
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    all_preds_raw = []
    all_targets_raw = []
    count = 0
    mean_inference_time = 0
    with torch.no_grad():
        for features, targets_processed in test_loader:
            count += 1
            features, targets_processed = features.to(device), targets_processed.to(device)
            inference_start_time = time.time()
            outputs_processed = model(features)
            mean_inference_time = (mean_inference_time * (count - 1) + (time.time() - inference_start_time)) / count
            # 计算处理后的损失
            loss = criterion(outputs_processed, targets_processed)
            test_loss += loss.item() * features.size(0)
            
            # 逆变换到原始尺度
            preds_raw = test_dataset.inverse_transform_y(outputs_processed.cpu())
            targets_raw = test_dataset.inverse_transform_y(targets_processed.cpu())
            
            all_preds_raw.extend(preds_raw.numpy())
            all_targets_raw.extend(targets_raw.numpy())
    
    # 计算评估指标
    test_loss /= len(test_loader.dataset)
    preds_np = np.array(all_preds_raw)
    targets_np = np.array(all_targets_raw)
    mae_raw = np.mean(np.abs(preds_np - targets_np))
    rmse_raw = np.sqrt(np.mean((preds_np - targets_np) **2))
    rel_error = np.mean(np.abs(preds_np - targets_np) / (targets_np + 1e-6)) * 100
    
    if mode == "normal":
        print(f"\n测试集评估结果:")
    elif mode == "mix":
        print(f"\n混合测试集评估结果:")
    print(f"处理后数据的测试损失 ({args.loss.upper()}): {test_loss:.6f}")
    print(f"原始时间尺度的MAE: {mae_raw:.4f} us")
    print(f"原始时间尺度的RMSE: {rmse_raw:.4f} us")
    print(f"平均相对误差: {rel_error:.2f}%")
    print(f"平均推理时间: {mean_inference_time:.6f} s")
    return test_loss, mae_raw, rmse_raw, rel_error, all_preds_raw, all_targets_raw


# -------------------------- 7. 可视化函数（无修改） --------------------------
def plot_loss_curves(train_losses, val_losses, loss_type="mse", save_path="loss_curves.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    # 根据损失函数类型设置y轴标签
    loss_label = {"mse": "MSE Loss", "mae": "MAE Loss", "smoothl1": "SmoothL1 Loss"}[loss_type]
    plt.ylabel(loss_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"损失曲线已保存为: {save_path}")
    plt.close()


def plot_predictions(preds_raw, targets_raw, save_path="predictions_vs_actual.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets_raw, preds_raw, alpha=0.5)
    
    min_val = min(min(targets_raw), min(preds_raw)) * 0.9
    max_val = max(max(targets_raw), max(preds_raw)) * 1.1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Real Time (us)')
    plt.ylabel('Predict Time (us)')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"预测对比图已保存为: {save_path}")
    plt.close()


def plot_log_predictions(preds_raw, targets_raw, save_path="log_predictions_vs_actual.png"):
    preds_log = np.log(np.array(preds_raw) + 1)
    targets_log = np.log(np.array(targets_raw) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets_log, preds_log, alpha=0.5)
    
    min_val = min(min(targets_log), min(preds_log)) * 0.9
    max_val = max(max(targets_log), max(preds_log)) * 1.1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Real Time（log）')
    plt.ylabel('Predict Time（log）')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"对数尺度预测对比图已保存为: {save_path}")
    plt.close()

# -------------------------- 8. 损失函数和优化器创建函数（新增） --------------------------
def create_loss_function(args):
    if args.loss == "mse":
        return nn.MSELoss()
    elif args.loss == "mae":
        return nn.L1Loss()
    elif args.loss == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"不支持的损失函数: {args.loss}")

def create_optimizer(model_parameters, args):
    """根据命令行参数创建对应的优化器"""
    if args.optimizer == "adam":
        return optim.Adam(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        return optim.SGD(
            model_parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(
            model_parameters,
            lr=args.lr,
            alpha=args.alpha,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")

# -------------------------- 9. 主函数（设备获取传入args） --------------------------
def main(args):
    # 1. 加载数据
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"错误：未找到数据文件 {args.data_path}，请先运行数据预处理脚本")
    
    # 2. 创建数据加载器
    train_dataset, val_dataset, test_dataset, test_dataset_mix, train_loader, val_loader, test_loader, test_loader_mix = process_data(args)

    # 3. 初始化模型、损失函数和优化器（通过函数调用使用命令行参数）
    device = get_device(args)
    model = TimePredictMLP(input_dim=args.input_dim, hidden_dims=args.hidden_dims)
    print(model)

    # 调用自定义函数，根据命令行参数创建损失函数和优化器
    criterion = create_loss_function(args)  # 使用命令行--loss参数
    optimizer = create_optimizer(model.parameters(), args)  # 使用命令行--optimizer及相关参数
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # 4. 训练模型
    model, train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, args, best_model_save_path=args.model_save_path
    )
    
    # 5. 评估模型
    test_loss, mae_raw, rmse_raw, rel_error, preds_raw, targets_raw = evaluate_model(
        model, test_loader, criterion, device, test_dataset, best_model_save_path=args.model_save_path, mode="normal"
    )
    
    test_loss_mix, mae_raw_mix, rmse_raw_mix, rel_error_mix, preds_raw_mix, targets_raw_mix = evaluate_model(
        model, test_loader_mix, criterion, device, test_dataset_mix, best_model_save_path=args.model_save_path, mode="mix"
    )

    # 6. 可视化结果
    plot_loss_curves(train_losses, val_losses, loss_type=args.loss)
    plot_predictions(preds_raw, targets_raw)
    plot_log_predictions(preds_raw, targets_raw)

# -------------------------- 10. 程序入口 --------------------------
if __name__ == "__main__":
    args = parse_args()
    # 打印当前实验配置（包含损失函数和优化器信息）
    print("="*50)
    print("当前实验配置：")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*50)
    main(args)