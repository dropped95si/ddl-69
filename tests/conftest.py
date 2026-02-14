import os
import sys


def _ensure_src_on_path() -> None:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(tests_dir)
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_src_on_path()
