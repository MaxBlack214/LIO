import os
import psycopg2
import time
import shutil

# ========== 配置项 ==========
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "dsb",
    "user": "postgres",
    "password": "123456"
}
THRESHOLD = 300  # 秒
SQL_FOLDERS = ["dsb_arithmetic", "dsb_aggregation"]
OUTPUT_BASE_DIR = "output"  # 全局输出目录
# ==========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def read_sql(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def execute_and_time(sql, conn, cursor):
    try:
        cursor.execute("SET statement_timeout = 300000;")  # 设置 300s 超时
        start = time.time()
        cursor.execute("EXPLAIN (ANALYZE, TIMING OFF) " + sql)
        result = cursor.fetchall()
        end = time.time()

        for row in result:
            line = row[0]
            if "Execution Time" in line:
                exec_time_ms = float(line.split("Execution Time:")[1].split("ms")[0].strip())
                return exec_time_ms / 1000.0  # 转换为秒

        return end - start  # fallback
    except Exception as e:
        conn.rollback()  # 关键：回滚事务，避免事务挂起导致后续失败
        error_msg = str(e)
        if "canceling statement due to statement timeout" in error_msg:
            return THRESHOLD + 1  # 超时算作慢
        print(f"[ERROR] 执行失败: {e}")
        return None  # 其它错误

def process_folder(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    fast_dir = ensure_dir(os.path.join(OUTPUT_BASE_DIR, f"{folder_name}_fast"))
    slow_dir = ensure_dir(os.path.join(OUTPUT_BASE_DIR, f"{folder_name}_slow"))
    fail_dir = ensure_dir(os.path.join(OUTPUT_BASE_DIR, f"{folder_name}_fail"))

    sql_files = [f for f in os.listdir(folder_path) if f.endswith(".sql")]
    print(f"[INFO] 正在处理目录: {folder_name}, 共 {len(sql_files)} 条 SQL")

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            for sql_file in sql_files:
                file_path = os.path.join(folder_path, sql_file)
                sql = read_sql(file_path)
                print(f"→ 执行 {sql_file}...", end=' ')
                exec_time = execute_and_time(sql, conn, cursor)

                if exec_time is None:
                    shutil.copy(file_path, os.path.join(fail_dir, sql_file))
                    print("❌ 执行失败，归为 FAIL")
                elif exec_time <= THRESHOLD:
                    shutil.copy(file_path, os.path.join(fast_dir, sql_file))
                    print(f"✅ {exec_time:.2f}s，归为 FAST")
                else:
                    shutil.copy(file_path, os.path.join(slow_dir, sql_file))
                    print(f"🐢 {exec_time:.2f}s，归为 SLOW")

def main():
    for folder in SQL_FOLDERS:
        if not os.path.exists(folder):
            print(f"[WARN] 跳过不存在的目录: {folder}")
            continue
        process_folder(folder)

if __name__ == "__main__":
    main()