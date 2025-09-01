import psycopg2
import json
import os
import sys
import time
import csv
import numpy as np
# 加载 EF 模型所需路径
sys.path.append(os.path.join(os.path.dirname(__file__), "bao_server"))
from model import Regression
# 设置主目录为 sys.path，导入 log_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import log_config

# 设置 enable 操作
def set_enable_params(cursor, hint_set):
    ops = ['hashjoin', 'mergejoin', 'nestloop',
           'seqscan', 'indexscan', 'indexonlyscan']
    for op in ops:
        on_off = 'on' if op in hint_set else 'off'
        cursor.execute(f"SET enable_{op} = {on_off};")

# 获取 EXPLAIN JSON（不执行语句本身）
def get_plan_json(cursor, sql):
    cursor.execute(f"EXPLAIN (VERBOSE, FORMAT JSON) {sql}")
    plan = cursor.fetchone()[0]
    return plan[0]  # plan 是 list，取第一个 dict
def select_hint(hint_sets, base_sql, model, cur):
    cur.execute("SET client_min_messages = log;")

    predictions = []
    plan_cache = {}
    all_mean_preds = []  # 保存每个 hint 的均值
    all_std_preds = []   # 保存每个 hint 的方差
    final_preds_per_hint = []  # 保存每个 hint 的最终预测值，用于计算 hint_set 方差

    start_time = time.time()  # 推理开始时间
    for i, hs in enumerate(hint_sets):
        set_enable_params(cur, hs)
        plan_json = get_plan_json(cur, base_sql)
        plan_str = json.dumps(plan_json)

        final_pred, all_preds_i, mean_pred_i, std_pred_i, hof_size,std_pred = model.predict([plan_str], return_all=True)
        #pred = final_pred[0][0]
        pred = std_pred[0][0]
        # print(f"pred:{pred}")
        predictions.append((pred, i))
        plan_cache[i] = plan_json

        # 保存每个 hint 的均值和方差
        all_mean_preds.append(mean_pred_i[0])
        all_std_preds.append(std_pred_i[0])

        # 保存每个 hint 的最终预测值
        final_preds_per_hint.append(pred)

    best_idx = min(predictions, key=lambda x: x[0])[1]

    # 取最优 hint 对应的均值和方差
    mean_pred = all_mean_preds[best_idx]
    std_pred = all_std_preds[best_idx]
    

    # ====== 增改部分：计算 5 个 hint 的预测值方差 ======
    hintset_pred_variance = np.var(final_preds_per_hint)

    inference_time = time.time() - start_time  # 推理耗时

    all_hints_str = ["+".join(sorted(hs)) for hs in hint_sets]
    all_hints_joined = "|".join(all_hints_str)
    csv_path = os.path.join(log_config.log_dir, "idx&predicted_reward.csv")
    sql_clean = base_sql.replace("\n", " ").replace("\r", " ").replace(",", "，")
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            predictions[best_idx][0],  # 最佳 hint 的预测 reward
            best_idx,                  # 最佳 hint 的下标
            round(inference_time, 6),  # 推理时间（秒）
            all_hints_joined,          # 所有 hint_set，字符串形式
            sql_clean
        ])
    if best_idx == 0:
        pred_diff_from_zero = -1
    else:
        pred_diff_from_zero = predictions[best_idx][0] - predictions[0][0]
    #若判定为复杂sql且预测可靠性不高 则用原版优化器，不给予hint
    # if std_pred >0.185 and final_preds_per_hint[0]>0.34:
    #     print("预测结果可靠性不高,不添加hint")
    #     best_idx = 0
    #     hof_size = -1#标记为不可靠预测
    return best_idx, plan_cache[best_idx], all_preds_i, mean_pred, std_pred, hof_size, hintset_pred_variance,pred_diff_from_zero,final_preds_per_hint
# def select_hint(hint_sets,base_sql,model,cur):
# 
#     
#     cur.execute("SET client_min_messages = log;")
# 
#     predictions = []
#     plan_cache = {}
#     start_time = time.time()  #  推理开始时间
#     for i, hs in enumerate(hint_sets):
#         set_enable_params(cur, hs)
#         plan_json = get_plan_json(cur, base_sql)
#         plan_str = json.dumps(plan_json)
#         #pred = model.predict([plan_str])[0][0]
#         #print("===========调试森林的确定性==============")
#         final_pred, all_preds, mean_pred, std_pred, hof_size = model.predict([plan_str],return_all=True)
# 
#         pred = final_pred[0][0]
#         #print("===========调试森林的确定性==============")
#         predictions.append((pred, i))
#         plan_cache[i] = plan_json
#     best_idx = min(predictions, key=lambda x: x[0])[1]
#     inference_time = time.time() - start_time  #  推理耗时
#     # 写入日志：选中hint的预测值、索引、推理时间、全部hint、sql内容
#     # 为避免写入嵌套 list（会让 CSV 读取不方便），将 hint sets 转成字符串形式
#     all_hints_str = ["+".join(sorted(hs)) for hs in hint_sets]
#     all_hints_joined = "|".join(all_hints_str)  # 多个 hint_set 用 | 分隔
#     csv_path = os.path.join(log_config.log_dir, "idx&predicted_reward.csv")
#     sql_clean = base_sql.replace("\n", " ").replace("\r", " ").replace(",", "，")
#     with open(csv_path, 'a', newline='') as file:
#         writer = csv.writer(file)
#         # 可以灵活扩展字段，比如加入 base_sql[:30] 截断内容标识是哪条 SQL
#         writer.writerow([
#             predictions[best_idx][0],        # 最佳 hint 的预测 reward
#             best_idx,                        # 最佳 hint 的下标
#             round(inference_time, 6),        # 推理时间（秒）
#             all_hints_joined,                # 所有 hint_set，字符串形式
#             sql_clean 
#         ])
#     return best_idx,plan_cache[best_idx],all_preds, mean_pred, std_pred, hof_size

if __name__ == "__main__":
    # 连接数据库
    conn = psycopg2.connect(
        dbname="imdb",
        user="postgres",
        password="123456",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    # 加载训练好的模型
    model = BaoRegression()
    model.load("bao_server/EF_Default_Model")
    hint_sets = [
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
    #print(select_hint(hint_sets,base_sql,cur))
    cur.close()
    conn.close()