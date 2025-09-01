from hint_excute import apply_hint_set
from sklearn.inspection import permutation_importance
import numpy as np
import dill
import time
import re
from collections import defaultdict
from sympy import latex, parse_expr
from evolutionary_forest.utils import gene_to_string, individual_to_tuple,get_feature_importance

def extract_args(expr):
    """
    提取表达式中出现的 ARG 的编号
    例如 Sub(ARG56, ARG43) => [56, 43]
    """
    return list(map(int, re.findall(r'ARG(\d+)', expr)))

def get_vector_from_importance_dict(importance_dict):
    """
    把表达式 importance 字典转为 112维 importance 向量，importance 均分给各个 ARG
    """
    vector = np.zeros(112)
    for expr, importance in importance_dict.items():
        args = extract_args(expr)
        if not args:
            continue
        share = importance / len(args)
        for arg in args:
            if arg < 112:
                vector[arg] += share
    return vector

def clean_and_collapse_importance(raw_dict):
    """
    去除 lambda 头并将 importance 列表求和。
    """
    collapsed = {}
    for full_key, importance_list in raw_dict.items():
        # Debug: 打印每个 key 看是否是我们想要处理的格式

        if ":" in full_key:
            expr = full_key.split(":", 1)[1].strip()
        else:
            expr = full_key.strip()

        if isinstance(importance_list, list):
            total = sum(importance_list)
        else:
            total = importance_list

        collapsed[expr] = total
    return collapsed

def get_top_hint_set_and_complement(vector_112d):
    """
    根据 112 维向量，返回最重要操作的 hint 集合和补集。
    返回格式： (top_hint_set, complement_set)
    各自是“set（hint）”组成的一个子集
    """
    if len(vector_112d) != 112:
        raise ValueError("输入向量长度必须是112")

    # 全集
    hint_mapping = [
        ((28, 31), {"nestloop", "seqscan"}),
        ((32, 35), {"nestloop", "indexscan"}),
        ((36, 39), {"nestloop", "indexonlyscan"}),

        ((44, 47), {"hashjoin", "seqscan"}),
        ((48, 51), {"hashjoin", "indexscan"}),
        ((52, 55), {"hashjoin", "indexonlyscan"}),

        ((60, 63), {"mergejoin", "seqscan"}),
        ((64, 67), {"mergejoin", "indexscan"}),
        ((68, 71), {"mergejoin", "indexonlyscan"}),
    ]
    #JOB-D
    # hint_mapping = [
    # 
    #     ((44, 47), {"hashjoin", "seqscan"}),
    #     ((52, 55), {"hashjoin", "indexonlyscan"}),
    # 
    # ]
    #CEB-EF
    # hint_mapping = [
    # 
    #     ((52, 55), {"hashjoin", "indexonlyscan"}),
    #     ((64, 67), {"mergejoin", "indexscan"}),
    #     ((68, 71), {"mergejoin", "indexonlyscan"}),
    # ]


    scores = []
    all_hint_sets = []
    for (start, end), hint_combo in hint_mapping:
        score = sum(vector_112d[start:end + 1])
        scores.append((score, hint_combo))
        all_hint_sets.append(hint_combo)

    top_score, top_hint = max(scores, key=lambda x: x[0])

    # 固定全集
    universe = {"hashjoin", "indexscan", "mergejoin", "nestloop", "seqscan", "indexonlyscan"}

    complement = universe - top_hint

    return [top_hint], [complement]

def print_explain(cursor, sql, description):
    print(f"\n--- {description} ---")
    cursor.execute(f"EXPLAIN {sql}")
    for row in cursor.fetchall():
        print(row[0])

if __name__ == "__main__":
    import psycopg2

    # 初始 hint_set，可以指定或空
    hint_set = [
    ]

    model_dir = "bao_server/EF_Default_Model"
    with open(f"{model_dir}/EF.pkl", "rb") as f:
        model = dill.load(f)
    # 开始计时
    start_time = time.time()
    #result = permutation_importance(model, x, y, n_repeats=10, random_state=0, scoring='r2')
    #vector = result.importances_mean
    feature_importance_dict = get_feature_importance(
        model,  # 你的 EF 模型
        latex_version=False,
        fitness_weighted=False,
        mean_fitness=True,  
        ensemble_weighted=False
    )
    cleaned = clean_and_collapse_importance(feature_importance_dict)
    vector = get_vector_from_importance_dict(cleaned)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n计算 importance 所花费的时间：{elapsed_time:.4f} 秒")


    top_hint, complement_hint = get_top_hint_set_and_complement(vector)
    hint_set.extend(top_hint)
    hint_set.extend(complement_hint)
    print(hint_set)

    # # 连接数据库
    # conn = psycopg2.connect(
    #     dbname="imdb",
    #     user="postgres",
    #     password="123456",
    #     host="localhost",
    #     port=5432
    # )
    # cur = conn.cursor()
    # cur.execute("SET client_min_messages = log;")
    # 
    # base_sql = """
    # SELECT movie_info.info, title.title
    # FROM movie_info
    # JOIN title ON movie_info.movie_id = title.id
    # WHERE movie_info.info_type_id = 101
    # LIMIT 10;
    # """
    # 
    # print_explain(cur, base_sql, "WITHOUT HINT")
    # 
    # for i, hs in enumerate(hint_set):
    #     print(f"\n>>> Applying hint set {i + 1}: {hs}")
    #     apply_hint_set(cur, hs)
    #     print_explain(cur, base_sql, f"WITH HINT SET {i + 1}")
    # 
    # cur.close()
    # conn.close()
