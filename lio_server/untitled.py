import shutil
import os
from datetime import datetime

def backup_bao_db(backup_dir="db_backups", tag=None):
    """
    备份当前 bao.db 文件到指定目录。
    :param backup_dir: 备份目录（默认 db_backups）
    :param tag: 可选标签，用于标记备份文件名
    :return: 备份文件路径
    """
    src_path = os.path.join(os.path.dirname(__file__), "bao.db")
    os.makedirs(backup_dir, exist_ok=True)

    # 用时间戳 + tag 构造文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        filename = f"bao_{tag}_{timestamp}.db"
    else:
        filename = f"bao_backup_{timestamp}.db"

    dest_path = os.path.join(backup_dir, filename)
    shutil.copy(src_path, dest_path)
    print(f"✅ 备份成功：{dest_path}")
    return dest_path