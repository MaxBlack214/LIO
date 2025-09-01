import sqlite3
import json

def print_experience_samples(db_path="lio_server/lio.db", limit=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, plan, reward, pg_pid FROM experience ORDER BY id DESC LIMIT {limit}")
    rows = cursor.fetchall()
    for row in rows:
        id_, plan_json, reward, pid = row
        plan = json.loads(plan_json)
        print(f"id: {id_}, reward: {reward}, pg_pid: {pid}")
        print(f"plan sample (truncated): {str(plan)[:200]}")  # 打印前200字符做示例
        print("-" * 40)
    conn.close()

def get_experience_count(db_path="lio_server/lio.db"):
    """返回经验池中数据的数量"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM experience")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def clear_experience_pool(db_path="lio_server/lio.db"):
    """清空经验池并 VACUUM 释放磁盘空间"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experience")
    conn.commit()
    cursor.execute("VACUUM")  # 释放磁盘空间
    conn.close()