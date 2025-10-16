import csv

import psycopg2
import os
import sys
import random
from time import time, sleep

import numpy as np
from sklearn.inspection import permutation_importance
import dill
import shutil
import glob
import re
from collections import defaultdict
from sympy import latex, parse_expr

from evolutionary_forest.utils import gene_to_string, individual_to_tuple,get_feature_importance
import test_getset

sys.path.append(os.path.join(os.path.dirname(__file__), "lio_server"))
from model import Regression
from test_predict import select_hint,get_plan_json
from hint_excute import apply_hint_set
from test_getset import get_top_hint_set_and_complement,clean_and_collapse_importance,get_vector_from_importance_dict
from lio_server.storage import record_reward 
from lio_server.constants import DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH
from lio_server.train import train_and_swap  
from test_checkdb import get_experience_count,clear_experience_pool
import log_config

Is_empty = False

#PG_CONNECTION_STR = "dbname=imdb user=postgres host=localhost password='123456' port='5432'"

model_dir = "lio_server/EF_Default_Model"



def extract_relative_sql_path(fp, workload):
    parts = fp.split(os.sep)
    for i, part in enumerate(parts):
        if part == workload:
            return os.path.join(part, *parts[i+1:])
    return os.path.basename(fp)


# https://stackoverflow.com/questions/312443/
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_query(sql, hint_set, Is_empty):
    model = Regression()
    model.load(DEFAULT_MODEL_PATH)
    start1 = time()
    state = 0
    select_index = -1  
    used_hint = set()  
    all_preds, mean_pred, std_pred, hof_size = None, -1, -1, -1
    hintset_pred_variance = -1
    pred_diff_from_zero = -2 
    final_preds_per_hint = [-1,-1,-1,-1,-1]
    while time() - start1 < 300:
        conn = None
        try:
            conn = psycopg2.connect(PG_CONNECTION_STR)
            cur = conn.cursor()

            cur.execute("SET statement_timeout = '300s';")

            if not Is_empty:
                select_index, plan_json,all_preds_i, mean_pred, std_pred, hof_size, hintset_pred_variance,pred_diff_from_zero,final_preds_per_hint = select_hint(hint_set, sql, model, cur)
                print(f"select_index: {select_index}")
                used_hint = hint_set[select_index]
                apply_hint_set(cur, used_hint)

            start2 = time()
            cur.execute(sql)
            stop2 = time()

            reward = stop2 - start2
            pid = conn.get_backend_pid()

            if Is_empty:
                plan_json = get_plan_json(cur, sql)

            record_reward(plan_json, reward, pid)
            return stop2 - start1 ,select_index,used_hint,all_preds, mean_pred, std_pred, hof_size, hintset_pred_variance,pred_diff_from_zero,final_preds_per_hint # 正常返回总耗时

        except Exception as e:
            if state == 0:
                print("❌ Error in run_query:", e)
                state = 1
                print("⚠️ Will retry if time allows...")
            sleep(1)

        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e_close:
                    print("⚠️ Error closing connection:", e_close)


    total_time = time() - start1
    print(f"⚠️ Query failed or timeout after {total_time:.2f} seconds.")
    return total_time,select_index,used_hint,all_preds, mean_pred, std_pred, hof_size, hintset_pred_variance,pred_diff_from_zero,final_preds_per_hint
if len(sys.argv) < 5:
    print("Usage: python run_queries.py '<sql_glob>' <param_a> <param_b> <db_name>")
    sys.exit(1)


sql_glob = sys.argv[1]
param_a = int(sys.argv[2])
param_b = int(sys.argv[3])
db_name = sys.argv[4] 

PG_CONNECTION_STR = f"dbname={db_name} user=postgres host=localhost password='123456' port='5432'"
print(PG_CONNECTION_STR)
print("a,b:")
print(param_a,param_b)

query_paths = glob.glob(sql_glob)
#query_paths = sys.argv[1:]
workload_name = os.path.basename(os.path.dirname(query_paths[0]))  
log_config.log_dir = os.path.join("log", workload_name)
os.makedirs(log_config.log_dir, exist_ok=True)


queries = []

for fp in query_paths:
    with open(fp) as f:
        query = f.read()
    queries.append((fp, query))
print("Read", len(queries), "queries.")

random.seed(42)
query_sequence = random.choices(queries, k=param_b)
# query_sequence = queries
bao_chunks = list(chunks(query_sequence, param_a))


clear_experience_pool()

#hint_set = []
# add your hint_set
hint_set = [
    {'hashjoin', 'indexscan', 'mergejoin', 'nestloop', 'seqscan', 'indexonlyscan'},  # case 0
    {'hashjoin', 'indexonlyscan', 'indexscan', 'mergejoin', 'seqscan'},  # case 1
    {'hashjoin', 'indexonlyscan', 'seqscan'},  # case 3
    {'hashjoin', 'indexonlyscan', 'indexscan', 'nestloop', 'seqscan'},  # case 4
]



for c_idx, chunk in enumerate(bao_chunks):
    if c_idx == -1:
        break
    print(get_experience_count())
    if get_experience_count() == 0:
        Is_empty = True
    else:
        Is_empty = False
    print(Is_empty)
    if c_idx == -1:
        print(f"c_idx == {c_idx+1}   skip")
        
    else:
        print(f"======== Progress: {c_idx+1}/{len(bao_chunks)} ========")
        for q_idx, (fp, q) in enumerate(chunk):
            # if q_idx < 50:
            #     continue
            print(f"Chunk {c_idx+1}/{len(bao_chunks)} - Query {q_idx+1}/{len(chunk)}")
            print("fp:")
            print(fp)
            q_time,select_index,used_hint,all_preds, mean_pred, std_pred, hof_size, hintset_pred_variance,pred_diff_from_zero, final_preds_per_hint= run_query(q,hint_set,Is_empty)
            print(c_idx, q_idx, time(), fp, q_time, flush=True)
            exe_time_path = os.path.join(log_config.log_dir, 'exe_time.csv')
            with open(exe_time_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([c_idx, q_idx, q_time, time(), extract_relative_sql_path(fp, workload_name),select_index, mean_pred, std_pred, hintset_pred_variance, pred_diff_from_zero,hof_size,final_preds_per_hint[0],final_preds_per_hint[1],final_preds_per_hint[2],used_hint])
            print("get_experience_count:")
            print(get_experience_count())

        if Is_empty:
            src_db_path = os.path.join("lio_server", "lio.db")
            dest_db_path = os.path.join(log_config.log_dir, "lio.db")
            shutil.copy(src_db_path, dest_db_path)   
    print("----------------start train----------------------")
    start = time()
    train_and_swap(DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH, verbose=True)  # 替代 os.system
    # print("-----------------get  importance------------------")
    # with open(f"{DEFAULT_MODEL_PATH}/EF.pkl", "rb") as f:
    #     model = dill.load(f)

    # feature_importance_dict = get_feature_importance(
    #     model,  
    #     latex_version=False,
    #     fitness_weighted=False,
    #     mean_fitness=True,
    #     ensemble_weighted=False
    # )
    # cleaned = clean_and_collapse_importance(feature_importance_dict)
    # importance = get_vector_from_importance_dict(cleaned)
    # top_hint, complement_hint = get_top_hint_set_and_complement(importance)


    hint_set = [
        {'hashjoin', 'indexscan', 'mergejoin', 'nestloop', 'seqscan', 'indexonlyscan'},  # case 0
        {'hashjoin', 'indexonlyscan', 'indexscan', 'mergejoin', 'seqscan'},  # case 1
        {'hashjoin', 'indexonlyscan', 'seqscan'},  # case 3
        {'hashjoin', 'indexonlyscan', 'indexscan', 'nestloop', 'seqscan'},  # case 4
    ]


    # hint_set.extend(top_hint)
    # hint_set.extend(complement_hint)
    print("new hint_set:")
    print(hint_set)
    end = time()
    train_time = end - start
    train_time_path = os.path.join(log_config.log_dir, 'train_time.csv')
    with open(train_time_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([c_idx, train_time])
