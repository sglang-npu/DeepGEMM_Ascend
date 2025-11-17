"""
工具函数模块：提供msprof执行相关的工具类
从benchmark_catlass.py中提取，供GA.py等模块复用
"""
import subprocess
import re
import os
import time
import glob
import shutil
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class MsProfExecutor:
    """
    msprof命令执行器，用于获取真实执行时间
    从benchmark_catlass.py中提取，提供独立可复用的msprof执行功能
    """
    
    def __init__(self, catlass_bin_path: str, rank_id: int = 0, msp_dir: str = "./msp"):
        """
        初始化msprof执行器
        
        Args:
            catlass_bin_path: catlass可执行文件路径
            rank_id: rank ID，默认为0
            msp_dir: msprof输出目录，默认为"./msp"
        """
        self.catlass_bin_path = catlass_bin_path
        self.rank_id = rank_id
        self.msp_dir = msp_dir
        
        # 为每个进程创建独立的msp目录，避免多进程冲突
        self.rank_msp_dir = os.path.join(msp_dir, f"rank_{rank_id}")
        os.makedirs(self.rank_msp_dir, exist_ok=True)
        
        # 预编译正则表达式，避免重复编译
        self.error_pattern = re.compile(r'Max Relative Error:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)')
        self.time_pattern = re.compile(r'Task Duration\(us\): (\d+\.\d+)')
        self.kernel_func_name_pattern = re.compile(r'Kernel Func Name:\s*(.+)', re.MULTILINE)
        
        # 预构建msprof命令模板，使用进程特定的输出目录
        self.msprof_cmd_template = f"msprof op --output={self.rank_msp_dir} --aic-metrics='PipeUtilization' {self.catlass_bin_path}"
    
    def find_opprof_folder(self, max_wait_time: float = 2.0) -> Optional[str]:
        """
        在当前进程的msp目录中查找最新的OPPROF文件夹
        增加重试机制，等待文件生成完成
        
        Args:
            max_wait_time: 最大等待时间（秒），默认2秒
        
        Returns:
            OPPROF文件夹路径，如果找不到则返回None
        """
        if not os.path.exists(self.rank_msp_dir):
            return None
        
        # 重试查找，等待文件生成（优化：减少等待时间）
        start_time = time.time()
        sleep_interval = 0.05  # 减少到50ms
        while time.time() - start_time < max_wait_time:
            # 查找所有OPPROF_开头的文件夹
            opprof_pattern = os.path.join(self.rank_msp_dir, "OPPROF_*")
            opprof_folders = glob.glob(opprof_pattern)
            
            if not opprof_folders:
                time.sleep(sleep_interval)
                continue
            
            # 按修改时间排序，获取最新的文件夹
            try:
                opprof_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_folder = opprof_folders[0]
                
                # 检查文件夹中是否有PipeUtilization.xlsx或PipeUtilization.csv文件
                xlsx_path = os.path.join(latest_folder, "PipeUtilization.xlsx")
                csv_path = os.path.join(latest_folder, "PipeUtilization.csv")
                
                # 优先检查xlsx，然后检查csv
                file_path = None
                if os.path.exists(xlsx_path):
                    file_path = xlsx_path
                elif os.path.exists(csv_path):
                    file_path = csv_path
                
                if file_path:
                    # 额外检查文件是否可读（确保文件已完全写入）
                    try:
                        # 尝试打开文件，确保文件已完全生成
                        with open(file_path, 'rb') as f:
                            f.read(1)  # 读取第一个字节，检查文件是否可读
                        return latest_folder
                    except (IOError, PermissionError):
                        # 文件可能还在写入中，继续等待
                        time.sleep(sleep_interval)
                        continue
            except Exception as e:
                # 只在verbose模式下输出错误
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    print(f"[find_opprof_folder] Rank {self.rank_id} 错误: {e}")
                time.sleep(sleep_interval)
                continue
        
        return None
    
    def parse_pipe_utilization(self, opprof_folder: str) -> Dict[str, Any]:
        """
        解析OPPROF文件夹中的PipeUtilization文件（支持xlsx和csv格式）
        第一行是表头，数据从第二行开始
        
        Args:
            opprof_folder: OPPROF文件夹路径
        
        Returns:
            解析后的数据字典，如果解析失败返回空字典
        """
        # 优先查找xlsx文件，然后查找csv文件
        xlsx_path = os.path.join(opprof_folder, "PipeUtilization.xlsx")
        csv_path = os.path.join(opprof_folder, "PipeUtilization.csv")
        
        file_path = None
        file_type = None
        
        if os.path.exists(xlsx_path):
            file_path = xlsx_path
            file_type = "xlsx"
        elif os.path.exists(csv_path):
            file_path = csv_path
            file_type = "csv"
        else:
            if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件不存在（xlsx和csv都不存在）")
            return {}
        
        try:
            # 根据文件类型选择相应的读取方法
            if file_type == "xlsx":
                # 读取xlsx文件，第一行作为表头（header=0是默认值）
                # 使用上下文管理器确保文件及时关闭
                with pd.ExcelFile(file_path, engine='openpyxl') as excel_file:
                    df = pd.read_excel(excel_file, header=0)
            else:  # csv
                # 读取csv文件，第一行作为表头
                df = pd.read_csv(file_path, header=0, encoding='utf-8')
            
            if df.empty:
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件为空（只有表头，没有数据行）")
                # 即使没有数据行，也返回列名信息
                return {col: None for col in df.columns}
            
            # 将DataFrame转换为字典
            # 取第一行数据（索引0），因为第一行是表头，数据从第二行开始
            if len(df) > 0:
                result = df.iloc[0].to_dict()
            else:
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件没有数据行")
                return {col: None for col in df.columns}
            
            # 处理NaN值，转换为None
            def clean_value(v):
                try:
                    if pd.isna(v):
                        return None
                    # 尝试转换为Python原生类型
                    if isinstance(v, (int, float)):
                        return int(v) if isinstance(v, (int, float)) and v == int(v) else float(v)
                    return v
                except:
                    return None
            
            result = {k: clean_value(v) for k, v in result.items()}
            return result
        except Exception as e:
            # 只在verbose模式下输出详细错误
            if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                print(f"[parse_pipe_utilization] Rank {self.rank_id} 解析失败: {e}")
                import traceback
                traceback.print_exc()
            return {}
    
    def cleanup_opprof_folder(self, opprof_folder: str) -> None:
        """
        删除OPPROF文件夹以节省空间
        增加重试机制，确保删除成功
        
        Args:
            opprof_folder: 要删除的OPPROF文件夹路径
        """
        if not opprof_folder or not os.path.exists(opprof_folder):
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 先尝试关闭可能打开的文件句柄
                shutil.rmtree(opprof_folder)
                # 验证删除是否成功
                if not os.path.exists(opprof_folder):
                    return
                else:
                    # 删除失败，等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
            except PermissionError as e:
                # 文件可能被占用，等待后重试
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                    continue
                else:
                    if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                        print(f"[cleanup_opprof_folder] Rank {self.rank_id} 删除失败（权限错误）: {e}")
            except Exception as e:
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    print(f"[cleanup_opprof_folder] Rank {self.rank_id} 删除失败: {e}")
                    import traceback
                    traceback.print_exc()
                break
    
    def ms_prof(self, param_str, max_retries: int = 3, timeout: int = 15) -> Tuple[Optional[float], float, str, Dict[str, Any]]:
        """
        执行msprof命令并同时解析时间、精度、kernel函数名和PipeUtilization数据
        如果解析到的时间是0.0，会重新执行命令（最多重试max_retries次）
        如果命令执行超过timeout秒，会跳过该组合，返回None标记
        
        Args:
            param_str: 命令参数字符串，格式: " m n k mTile nTile kTile 0 0 1 rank_id"
            max_retries: 最大重试次数，默认3次
            timeout: 命令执行超时时间（秒），默认15秒
        
        Returns:
            (time_us, diff, kernel_func_name, pipe_utilization_data)
            - time_us: 执行时间（微秒），如果超时或失败则为None或999999999
            - diff: 精度误差
            - kernel_func_name: kernel函数名
            - pipe_utilization_data: PipeUtilization数据字典
        """
        # 优化：使用预构建的命令模板，避免重复字符串拼接
        full_cmd = self.msprof_cmd_template + param_str
        
        # 打印将要执行的命令（可选，可以通过环境变量控制）
        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
            print(f"[ms_prof] 执行命令: {full_cmd}")
        
        def parse_output(output: str):
            """解析输出中的时间、精度和kernel函数名（优化版本）"""
            time_us = None
            diff = float('inf')
            kernel_func_name = ""
            
            # 优化：一次性搜索所有模式，减少字符串遍历次数
            time_match = self.time_pattern.search(output)
            if time_match:
                time_us = float(time_match.group(1))
            
            error_match = self.error_pattern.search(output)
            if error_match:
                try:
                    diff = float(error_match.group(1))
                except ValueError:
                    pass  # 保持默认值 float('inf')
            
            kernel_match = self.kernel_func_name_pattern.search(output)
            if kernel_match:
                # 优化：减少字符串操作，直接使用strip和split的组合
                kernel_func_name = ' '.join(kernel_match.group(1).split())
            
            return time_us, diff, kernel_func_name
        
        # 重试逻辑
        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    full_cmd, 
                    shell=True, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=timeout  # 添加15秒超时
                )
                
                # 优化：合并 stdout 和 stderr，减少字符串操作
                # 因为 msprof 可能将信息输出到 stderr，先检查stderr是否为空
                if result.stderr:
                    stdout_text = result.stdout.decode('utf-8', errors='ignore')
                    stderr_text = result.stderr.decode('utf-8', errors='ignore')
                    combined_output = stdout_text + stderr_text
                else:
                    combined_output = result.stdout.decode('utf-8', errors='ignore')
                
                time_us, diff, kernel_func_name = parse_output(combined_output)
                
                if time_us is None:
                    # 解析失败，静默重试
                    if attempt < max_retries:
                        continue
                    else:
                        # 只在verbose模式下输出错误
                        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                            print(f"[ms_prof] Rank {self.rank_id} 解析失败，输出内容（前200字符）: {combined_output[:200]}")
                        return (999999999, diff, kernel_func_name, {})
                
                # 检查时间值是否为0.0或异常值
                if time_us == 0.0:
                    if attempt < max_retries:
                        continue
                    else:
                        # 只在verbose模式下输出警告
                        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                            print(f"[ms_prof] Rank {self.rank_id} 警告：重试{max_retries}次后仍为0.0，返回999999999")
                        return (999999999, diff, kernel_func_name, {})
                
                # 时间值有效，查找并解析PipeUtilization.xlsx
                pipe_utilization_data = {}
                # 等待一小段时间，确保OPPROF文件夹已完全生成（优化：减少等待时间）
                time.sleep(0.3)  # 从500ms减少到300ms
                
                # 查找OPPROF文件夹（带重试机制，最多等待1.5秒，优化：减少等待时间）
                opprof_folder = self.find_opprof_folder(max_wait_time=1.5)
                if opprof_folder:
                    # 先解析文件（支持xlsx和csv格式）
                    pipe_utilization_data = self.parse_pipe_utilization(opprof_folder)
                    # 等待一小段时间，确保文件句柄已关闭（优化：减少等待时间）
                    time.sleep(0.05)  # 从100ms减少到50ms
                    # 读取后立即删除文件夹以节省空间
                    self.cleanup_opprof_folder(opprof_folder)
                else:
                    # 只在verbose模式下输出警告
                    if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                        print(f"[ms_prof] Rank {self.rank_id} 警告：未找到OPPROF文件夹或PipeUtilization文件（xlsx/csv）")
                
                return (time_us, diff, kernel_func_name, pipe_utilization_data)
                    
            except subprocess.TimeoutExpired:
                # 超时情况：不重试，直接返回None标记，让调用者跳过该组合
                print(f"[ms_prof] Rank {self.rank_id} 命令执行超时（>{timeout}秒），跳过该tile组合")
                return (None, float('inf'), "", {})
                
            except Exception as e:
                if attempt < max_retries:
                    # 静默重试，只在verbose模式下输出
                    if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                        print(f"[ms_prof] Rank {self.rank_id} 第{attempt+1}次尝试失败: {e}，重试中...")
                    continue
                else:
                    # 只在最后一次失败时输出错误
                    print(f"[ms_prof] Rank {self.rank_id} 执行失败，错误: {e}")
                    return (999999999, float('inf'), "", {})
        
        # 理论上不会到达这里，但为了安全起见
        return (999999999, float('inf'), "", {})

