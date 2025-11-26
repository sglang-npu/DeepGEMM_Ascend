import os
import signal
import subprocess
from typing import Optional

from .logger import logger


def kill_process_group(proc):
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
                pass
    except (ProcessLookupError, OSError):
        logger.error("proc kill failed.")
        pass
    except Exception as e:
        logger.error(f"proc kill failed. err msg: {e}")
        pass


class MsProfExecutor:
    def __init__(self,
                 output: str,
                 aic_metrics: str,
                 kernel_name="_Z",
                 launch_count=0,
                 timeout=15):
        self.base_cmd = "msprof op "
        self.base_cmd += f"--output={output} "
        self.base_cmd += f"--aic-metrics={aic_metrics} "
        if kernel_name != '':
            self.base_cmd += f"--kernel-name={kernel_name} "

        self.default_launch_count = launch_count
        self.default_timeout = timeout

    def _build_cmd(self, program: str, launch_count: Optional[int]) -> str:
        cmd = self.base_cmd
        effective_launch = launch_count if launch_count is not None else self.default_launch_count
        if effective_launch and effective_launch > 0:
            cmd += f"--launch-count={effective_launch} "
        return cmd + program

    def print_cmd(self):
        logger.debug(f"msprof exec cmd head : '{self.base_cmd}'")

    def process(self,
                program: str,
                launch_count: Optional[int] = None,
                timeout: Optional[int] = None) -> str:
        proc = None
        full_cmd = self._build_cmd(program, launch_count)
        logger.debug(f"msprof exec cmd : '{full_cmd}'")
        effective_timeout = timeout if timeout is not None else self.default_timeout
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
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                logger.error(f"[ms_prof] exec timeout({effective_timeout}s).")
                kill_process_group(proc)
                return ""
            
            combined_output = stdout.decode('utf-8') if stdout else ""
            stderr_text = ""
            if stderr:
                stderr_text = stderr.decode('utf-8')
                combined_output += stderr_text
            
            if returncode != 0:
                logger.error(f"[ms_prof] exec failed, returncode is {returncode}. stderr: {stderr_text[:500] if stderr_text else 'None'}")
            
            return combined_output
        except Exception as e:
            if proc is not None:
                kill_process_group(proc)
                logger.error(f"[ms_prof] 执行异常，已终止进程组. err msg: {e}, program: {program}")
            else:
                logger.error(f"[ms_prof] 进程不存在，执行异常. err msg: {e}, program: {program}")
            return ""

