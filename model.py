import torch
import torch.nn as nn

class TimePredictMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
        super(TimePredictMLP, self).__init__()
        # 构建网络层
        layers = []
        # 添加输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        # 添加中间隐藏层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
        # 添加中间隐藏层
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)