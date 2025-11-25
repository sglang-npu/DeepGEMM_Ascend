import os
import signal
import subprocess
from logger import logger


def kill_process_group(proc):
    """
    功能: 终止进程
    输入: subprocess.Popen打开的进程对象
    """
    try:
        if proc.poll() is None:
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    logger.error("SIGKILL failed.")
                    try:
                        proc.kill()
                    except (ProcessLookupError, OSError):
                        logger.error("proc kill failed.")
            else:
                proc.kill()

            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.error("proc kill timeout.")
    except (ProcessLookupError, OSError):
        logger.error("proc kill failed.")
    except Exception as e:
        logger.error(f"proc kill failed. err msg: {e}")


class MsProfExecutor:
    """
    功能: msprof op的执行器
    使用: 
    1. 创建对象 executor = MsProfExecutor('./output', 'PipeUtilization', kernel_name, 1, 15)
    2. 然后调用 executor.process('./executable param1 param2')
    3. 也可以调用 executor.print_cmd() 查看 msprof op 的命令是否正确(需要INFO等级日志)
    """
    def __init__(self,
                 output: str,
                 aic_metrics: str,
                 kernel_name="_Z",
                 launch_count=0,
                 timeout=15):
        self.ms_prof_cmd = "msprof op "
        self.ms_prof_cmd += f"--output={output} "
        self.ms_prof_cmd += f"--aic-metrics={aic_metrics} "
        if kernel_name != '':
            self.ms_prof_cmd += f"--kernel-name={kernel_name} "
        if launch_count != 0:
            self.ms_prof_cmd += f"--launch-count={launch_count} "

        self.timeout = timeout

    def print_cmd(self):
        logger.info(f"msprof exec cmd head : '{self.ms_prof_cmd}'")

    def process(self, program: str) -> str:
        proc = None
        full_cmd = self.ms_prof_cmd + program
        try:
            popen_kwargs = {
                'shell': True,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
            }
            if hasattr(os, 'setsid'):
                popen_kwargs['start_new_session'] = True
            proc = subprocess.Popen(full_cmd, **popen_kwargs)

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                logger.error(f"[ms_prof] exec timeout({self.timeout}s).")
                kill_process_group(proc)
                return ""
            
            if returncode != 0:
                logger.error(f"[ms_prof] exec failed, returncode is {returncode}.")

            combined_output = stdout.decode('utf-8')
            if stderr:
                stderr_text = stderr.decode('utf-8')
                combined_output += stderr_text
            return combined_output
        except Exception as e:
            if proc is not None:
                kill_process_group(proc)
            else:
                logger.error(f"proc not existed. err msg: {e}")
            return ""

