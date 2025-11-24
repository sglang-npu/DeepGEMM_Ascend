import logging
import os


def get_log_level_from_env():
    level = os.getenv("BENCHMARK_LOG_LEVEL", "INFO").upper()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(level, logging.INFO)


def setup_logger():
    level = get_log_level_from_env()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datafmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler("process_log.log", encoding="utf-8")
        ]
    )
    return logging.getlogger()


logger = setup_logger()
