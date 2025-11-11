import torch
from trainer import TimePredictMLP

model = TimePredictMLP(input_dim=30, hidden_dims=[128, 64, 32])
weights_path = './best_result/best_model.pth'

model.load_state_dict(
    torch.load(
        weights_path,
        map_location=torch.device("cpu")  # 无论训练用GPU/CPU，此处用CPU加载更通用
    )
)
model.eval()

dummy_input = torch.randn(512, 30)
onnx_save_path = "./best_model.onnx"

torch.onnx.export(
        model=model,                # pytorch网络模型
        args=dummy_input,          # 随机的模拟输入
        f=onnx_save_path,        # 导出的onnx文件位置
        export_params=True,   # 导出训练好的模型参数
        verbose=True,         # verbose=True，支持打印onnx节点和对应的PyTorch代码行
        training=torch.onnx.TrainingMode.EVAL,  # 导出模型调整到推理状态，将dropout，BatchNorm等涉及的超参数固定
        input_names=["input"],    # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
        output_names=["output"],  # 为输出节点设置别名
        dynamic_axes={
            "input":{0: "-1"},
            "output":{0: "-1"}
        },
        keep_initializers_as_inputs=None,  #是否将模型参数作为输入数据的一部分进行导出
        opset_version=11                  # ONNX 运算符的版本号
)
