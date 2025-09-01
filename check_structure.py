import os
import re

tables = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
    "customer_address", "customer_demographics", "date_dim", "household_demographics",
    "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
    "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]

sql_path = "/root/autodl-tmp/TPC-DS/tpcds-kit/tools/tpcds.sql"
data_path = "/root/autodl-tmp/TPC-DS/data"

def get_column_count_from_sql(table_name):
    with open(sql_path, 'r') as f:
        sql = f.read()

    # 使用正则提取建表语句
    pattern = rf"create table {table_name}\s*\((.*?)\);"
    match = re.search(pattern, sql, re.DOTALL | re.IGNORECASE)
    if not match:
        return -1

    content = match.group(1)
    # 每一行是一个字段定义或主键等，过滤掉空行和主键
    lines = content.splitlines()
    field_lines = [line.strip() for line in lines if line.strip() and not line.strip().lower().startswith("primary key")]
    return len(field_lines)

def get_column_count_from_data(table_name):
    file_path = os.path.join(data_path, f"{table_name}.dat")
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.strip()
            if line:  # 找第一行非空行
                return line.count("|") + 1
    return -1

# 主逻辑
for table in tables:
    sql_cols = get_column_count_from_sql(table)
    dat_cols = get_column_count_from_data(table)
    status = "✅ MATCH" if sql_cols == dat_cols else "❌ MISMATCH"
    print(f"{table:25} | SQL: {sql_cols:2} | DAT: {dat_cols:2} | {status}")