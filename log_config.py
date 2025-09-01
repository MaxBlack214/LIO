import os
import sys

# 自动推断 workload 名字
def detect_workload():
    for arg in sys.argv:
        if "*" not in arg and "/" in arg:
            return os.path.basename(os.path.dirname(arg))
    return "default"

workload = detect_workload()
log_dir = f"log/{workload}"
os.makedirs(log_dir, exist_ok=True)  #  自动创建目录