import psycopg2
import json

# 假设这是你用来控制启用操作的函数（参考前文）
def set_enable_params(cursor, hint_set):
    ops = ['hashjoin', 'mergejoin', 'nestloop',
           'seqscan', 'indexscan', 'indexonlyscan', 'bitmapscan']
    for op in ops:
        on_off = 'on' if op in hint_set else 'off'
        cursor.execute(f"SET enable_{op} = {on_off};")

def print_plan_json(cursor, sql):
    cursor.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0]  # plan是长度1的list，包含dict
    print(json.dumps(plan, indent=2))  # 美化输出

def main():
    hint_set = [
        {"mergejoin", "bitmapscan"},
        {"seqscan", "mergejoin"},
        {"seqscan", "bitmapscan", "mergejoin", "indexscan", "hashjoin", "nestloop", "indexonlyscan"}
    ]

    base_sql = """
    SELECT movie_info.info, title.title
    FROM movie_info
    JOIN title ON movie_info.movie_id = title.id
    WHERE movie_info.info_type_id = 101
    LIMIT 10;
    """

    conn = psycopg2.connect(
        dbname="imdb",
        user="postgres",
        password="123456",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    cur.execute("SET client_min_messages = log;")

    for i, hs in enumerate(hint_set):
        print(f"\n>>> Applying hint set {i + 1}: {hs}")
        set_enable_params(cur, hs)
        print_plan_json(cur, base_sql)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()