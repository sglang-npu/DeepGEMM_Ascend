"""
conftest.py for get_best_config tests.

Handles path configuration and shared fixtures.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_get_best_config_path = os.path.join(_project_root, "get_best_config")

if _get_best_config_path not in sys.path:
    sys.path.insert(0, _get_best_config_path)