import psycopg2

def apply_hint_set(cursor, hint_set):
    """
    根据传入的 hint 集合，开启对应的参数，其他全部关闭。

    参数：
    - cursor: psycopg2 游标
    - hint_set: set[str]，包含任意 join 和 scan 类型的字符串
      支持：
        Join: "nestloop", "hashjoin", "mergejoin"
        Scan: "seqscan", "indexscan", "indexonlyscan", "bitmapscan"
    """

    hint_map = {
        "nestloop": "enable_nestloop",
        "hashjoin": "enable_hashjoin",
        "mergejoin": "enable_mergejoin",
        "seqscan": "enable_seqscan",
        "indexscan": "enable_indexscan",
        "indexonlyscan": "enable_indexonlyscan",
        "bitmapscan": "enable_bitmapscan"
    }

    # 先全部关闭
    for param in hint_map.values():
        cursor.execute(f"SET {param} = off;")

    # 开启 hint_set 中的
    for hint in hint_set:
        hint_lower = hint.lower()
        if hint_lower not in hint_map:
            raise ValueError(f"Unknown hint option: {hint}")
        cursor.execute(f"SET {hint_map[hint_lower]} = on;")


def print_explain(cursor, sql, description):
    print(f"\n--- {description} ---")
    cursor.execute(f"EXPLAIN {sql}")
    for row in cursor.fetchall():
        print(row[0])

def main():
    conn = psycopg2.connect(
        dbname="imdb",
        user="postgres",
        password="your_password",  # 替换成你的密码
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    sql = """
    SELECT movie_info.info, title.title
    FROM movie_info
    JOIN title ON movie_info.movie_id = title.id
    WHERE movie_info.info_type_id = 101
    LIMIT 10;
    """

    # 示例 hint_set
    hint_set = {"nestloop", "seqscan"}

    print_explain(cur, sql, "WITHOUT any hint set")

    apply_hint_set(cur, hint_set)
    print_explain(cur, sql, f"WITH hints: {hint_set}")

    # 改变 hint_set，示范补集关闭
    all_hints = {"nestloop", "hashjoin", "mergejoin", "seqscan", "indexscan", "indexonlyscan", "bitmapscan"}
    hint_set = all_hints - {"nestloop", "seqscan"}  # 关闭 nestloop 和 seqscan
    apply_hint_set(cur, hint_set)
    print_explain(cur, sql, f"WITH hints: {hint_set}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
    