import numpy as np
from ais_bench.infer.interface import InferSession

class OMModelInference:
    def __init__(self, 
                 device_id=0, 
                 model_path="../om_best_model.om", 
                 scaler_path="../best_result/scaler.npz",
                 fixed_batch_size=50):  # 新增：模型支持的固定batch_size
        """
        初始化OM模型推理器（含固定batch_size适配）
        :param fixed_batch_size: 模型要求的固定batch_size（必须与模型导出时一致）
        """
        # 1. 初始化模型会话（不变）
        self.session = InferSession(device_id=device_id, model_path=model_path)
        
        # 2. 特征定义（不变）
        self.original_features = [
            'M', 'N', 'K', 
            'm_sections', 'n_sections', 
            'm_sec_o_blocks', 'n_sec_o_blocks', 
            'k_o_iter_blocks', 'db_o_blocks'
        ]
        self.extended_features = self.original_features + [
            'm_blocks', 'n_blocks', 'k_blocks',
            'm_o_fix', 'n_o_fix', 'k_o_fix',
            'db_o_num', 'r_m_blocks', 'r_n_blocks',
            'k_iters', 'k_tail_blocks', 'r_db_num', 'r_k_blocks',
            'm_iters', 'n_iters', 'm_parts', 'n_parts',
            'm_sc_blocks', 'n_sc_blocks', 'r_m_parts', 'r_n_parts'
        ]
        self.feature_dim = len(self.extended_features)  # 30维
        
        # 3. 加载scaler（不变）
        try:
            scaler_data = np.load(scaler_path)
            self.mean = scaler_data['mean'].astype(np.float32)
            self.std = scaler_data['std'].astype(np.float32)
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
        except Exception as e:
            raise ValueError(f"加载scaler失败：{str(e)}")
        
        if len(self.mean) != self.feature_dim or len(self.std) != self.feature_dim:
            raise ValueError(f"scaler维度不匹配（期望{self.feature_dim}维）")
        
        # 新增：固定batch_size配置
        self.fixed_batch_size = fixed_batch_size
        print(f"模型固定batch_size: {self.fixed_batch_size}，将自动填充/截断输入")

    def align16(self, x):
        """复现训练时的16倍对齐逻辑（不变）"""
        return (x + 15) // 16 * 16

    def preprocess(self, input_batch):
        """
        预处理（新增：填充至固定batch_size）
        :return: (preprocessed_data, original_batch_size) 
                 预处理后的数据（固定batch_size）+ 原始batch_size（用于后处理截断）
        """
        # 1. 输入格式检查（不变）
        if not isinstance(input_batch, (list, np.ndarray)):
            raise ValueError(f"输入必须是list或numpy数组，实际为{type(input_batch)}")
        
        if isinstance(input_batch, list):
            input_np = np.array(input_batch, dtype=np.float32)
        else:
            input_np = input_batch.astype(np.float32)
        
        if input_np.ndim != 2:
            raise ValueError(f"输入必须是二维（batch_size, 9），实际为{input_np.ndim}维")
        
        original_batch_size, input_dim = input_np.shape
        if input_dim != 9:
            raise ValueError(f"输入特征维度必须为9，实际为{input_dim}")

        # 2. 填充至固定batch_size（核心修改）
        if original_batch_size > self.fixed_batch_size:
            raise ValueError(f"输入batch_size({original_batch_size})超过模型最大支持({self.fixed_batch_size})")
        
        # 用最后一个样本填充（避免引入异常值）
        pad_size = self.fixed_batch_size - original_batch_size
        if pad_size > 0:
            # 取最后一个有效样本作为填充值
            last_sample = input_np[-1:]  # shape=(1, 9)
            # 复制pad_size次，拼接至输入
            input_np = np.concatenate([input_np, np.repeat(last_sample, pad_size, axis=0)], axis=0)
            # 确认填充后尺寸正确
            assert input_np.shape[0] == self.fixed_batch_size, f"填充后batch_size错误：{input_np.shape[0]}"

        # 3. 批量计算扩展特征（不变）
        extended_features_list = []
        for i in range(self.fixed_batch_size):  # 循环固定batch_size
            sample = input_np[i]
            input_data = dict(zip(self.original_features, sample))
            
            # 扩展特征计算（与原逻辑完全一致）
            input_data['m_blocks'] = self.align16(input_data['M']) // 16
            input_data['n_blocks'] = self.align16(input_data['N']) // 16
            input_data['k_blocks'] = self.align16(input_data['K']) // 16
            input_data['m_o_fix'] = self.align16(input_data['M']) - input_data['M']
            input_data['n_o_fix'] = self.align16(input_data['N']) - input_data['N']
            input_data['k_o_fix'] = self.align16(input_data['K']) - input_data['K']
            input_data['db_o_num'] = input_data['k_o_iter_blocks'] // input_data['db_o_blocks']
            
            input_data['r_m_blocks'] = input_data['m_blocks'] % input_data['m_sec_o_blocks']
            input_data['r_n_blocks'] = input_data['n_blocks'] % input_data['n_sec_o_blocks']
            input_data['r_m_blocks'] = input_data['m_sec_o_blocks'] if input_data['r_m_blocks'] == 0 else input_data['r_m_blocks']
            input_data['r_n_blocks'] = input_data['n_sec_o_blocks'] if input_data['r_n_blocks'] == 0 else input_data['r_n_blocks']
            
            input_data['k_iters'] = (input_data['k_blocks'] + input_data['k_o_iter_blocks'] - 1) // input_data['k_o_iter_blocks']
            input_data['k_tail_blocks'] = input_data['k_blocks'] % input_data['k_o_iter_blocks']
            if input_data['k_tail_blocks'] == 0:
                input_data['r_db_num'] = input_data['db_o_num']
                input_data['r_k_blocks'] = input_data['db_o_blocks']
            else:
                input_data['r_db_num'] = (input_data['k_tail_blocks'] + input_data['db_o_blocks'] - 1) // input_data['db_o_blocks']
                input_data['r_k_blocks'] = input_data['k_tail_blocks'] - ((input_data['r_db_num'] - 1) * input_data['db_o_blocks'])
            
            input_data['m_iters'] = (input_data['m_blocks'] + input_data['m_sec_o_blocks'] - 1) // input_data['m_sec_o_blocks']
            input_data['n_iters'] = (input_data['n_blocks'] + input_data['n_sec_o_blocks'] - 1) // input_data['n_sec_o_blocks']
            input_data['m_parts'] = input_data['m_iters'] // input_data['m_sections']
            input_data['n_parts'] = input_data['n_iters'] // input_data['n_sections']
            input_data['m_sc_blocks'] = input_data['m_parts'] * input_data['m_sec_o_blocks']
            input_data['n_sc_blocks'] = input_data['n_parts'] * input_data['n_sec_o_blocks']
            input_data['r_m_parts'] = input_data['m_iters'] - ((input_data['m_sections'] - 1) * input_data['m_parts'])
            input_data['r_n_parts'] = input_data['n_iters'] - ((input_data['n_sections'] - 1) * input_data['n_parts'])
            
            extended_sample = np.array([input_data[feat] for feat in self.extended_features], dtype=np.float32)
            extended_features_list.append(extended_sample)
        
        # 4. 堆叠并标准化（不变）
        input_array = np.stack(extended_features_list, axis=0)
        input_normalized = (input_array - self.mean) / self.std
        
        # 返回预处理数据+原始batch_size（用于后处理）
        return input_normalized, original_batch_size

    def infer(self, preprocessed_data):
        """模型推理（不变，使用固定batch_size输入）"""
        return self.session.infer(feeds=[preprocessed_data], mode="static")

    def postprocess(self, model_output, original_batch_size):
        """
        后处理（新增：按原始batch_size截断）
        :param original_batch_size: 原始输入的batch_size（非填充后）
        """
        try:
            # 1. 解析模型输出（不变）
            if isinstance(model_output, list):
                output_array = model_output[0]
            else:
                output_array = model_output
            output_array = np.array(output_array, dtype=np.float32).flatten()
            
            # 2. 截断至原始batch_size（核心修改）
            output_array = output_array[:original_batch_size]
            
        except Exception as e:
            raise ValueError(f"解析模型输出失败：{str(e)}")
        
        # 3. 逆变换（不变）
        y_original = np.exp(output_array) - 1e-6
        y_original = np.clip(y_original, a_min=1e-9, a_max=None)
        
        return y_original

    def predict(self, input_batch):
        """完整预测流程（适配填充和截断）"""
        try:
            # 1. 预处理（返回填充后的数据和原始batch_size）
            preprocessed, original_batch_size = self.preprocess(input_batch)
            # 2. 推理
            output = self.infer(preprocessed)
            # 3. 后处理（按原始batch_size截断）
            time_values = self.postprocess(output, original_batch_size)
            return time_values
        except Exception as e:
            print(f"预测流程出错: {str(e)}")
            return None

    def __del__(self):
        """释放资源（不变）"""
        if hasattr(self, 'session'):
            self.session.free_resource()
            print("模型资源已释放")


# 使用示例（验证固定batch_size适配）
if __name__ == "__main__":
    # 1. 初始化推理器（指定模型支持的固定batch_size，如32）
    inferencer = OMModelInference(
        device_id=0,
        model_path="../om_best_model_batch.om",
        scaler_path="../best_result/scaler.npz",
        fixed_batch_size=50
    )
    
    # 2. 准备输入（batch_size=2，小于固定值32）
    input_batch = [
        [1024, 1024, 1024, 2, 2, 16, 16, 32, 8],
        [2048, 2048, 2048, 4, 4, 32, 32, 64, 16]
    ]
    
    # 3. 批量预测（自动填充至32，后处理截断回2）
    predicted_times = inferencer.predict(input_batch)
    
    # 4. 输出结果（应仅含2个有效样本）
    if predicted_times is not None:
        print(f"预测结果（原始batch_size={len(predicted_times)}）: {predicted_times}")
        for i, time in enumerate(predicted_times, 1):
            print(f"样本{i}预测时间: {time:.6f}")