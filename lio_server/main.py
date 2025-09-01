import csv
import pickle
import socketserver
import json
import struct
import sys
import time
import os

import dill
from numba.core.typing.builtins import Print

import storage
import model
import train
import baoctl
import math
import reg_blocker
import pandas
import numpy as np
from constants import (PG_OPTIMIZER_INDEX, DEFAULT_MODEL_PATH,
                       OLD_MODEL_PATH, TMP_MODEL_PATH)

row = 0


def add_buffer_info_to_plans(buffer_info, plans):
    for p in plans:
        p["Buffers"] = buffer_info
    return plans


class BaoModel:
    def __init__(self):
        self.__current_model = None

    def select_plan(self, messages):
        start = time.time()
        # the last message is the buffer state
        *arms, buffers = messages
        # print('messages', messages)

        # if we don't have a model, default to the PG optimizer
        if self.__current_model is None:
            return PG_OPTIMIZER_INDEX

        # if we do have a model, make predictions for each plan.
        arms = add_buffer_info_to_plans(buffers, arms)
        # with open('planwithtreestruct.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([arms])
        #修改部分
        #final
        res = self.__current_model.predict(arms)
        idx = res.argmin()
        # plan_set = self.__current_model.transform_to_1dvector()
        # np_plan_Set = np.array(plan_set)
        # print(np_plan_Set)
        # np.savetxt('data_example.csv', np_plan_Set[idx], fmt='%f')
        # df = pandas.DataFrame()
        # for i in range(112):
        #     df.at[row, i] = plan_set[idx][i]
        #     row += 1
        # dp = pandas.DataFrame(plan_set[idx])
        # print(type(dp))
        # print(dp)
        # with open('ef_test_beginner500.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([plan_set[idx]])
        stop = time.time()
        inferencetime = stop - start
        print("Selected index", idx,
              "after", f"{round((stop - start) * 1000)}ms",
              "Predicted reward / PG:", res[idx][0],
              "/", res[0][0])
        # print(idx, 'idx')
        with open('CEB_ef_paramtest8_idx&predicted_reward_3k2_200_frombegin_random42.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([res[idx][0], idx, inferencetime])
        return idx

    def predict(self, messages):
        # the last message is the buffer state
        plan, buffers = messages

        # if we don't have a model, make a prediction of NaN
        if self.__current_model is None:
            return math.nan

        # if we do have a model, make predictions for each plan.
        plans = add_buffer_info_to_plans(buffers, [plan])
        # print('plans', plans)
        res = self.__current_model.predict(plans)
        print('RES', res[0][0])
        return res[0][0]

    def load_model(self, fp):
        try:
            self.__current_model = model.BaoRegression(have_cache_data=True)
            # with open(DEFAULT_MODEL_PATH + '/EF.pkl', "rb") as f:
            self.__current_model.load(DEFAULT_MODEL_PATH)

            # if reg_blocker.should_replace_model(
            #         self.__current_model,
            #         new_model):
            #     self.__current_model = new_model
            #     print("Accepted new model.")
            # else:
            #     print("Rejecting load of new model due to regresison profile.")

        except Exception as e:
            print("Failed to load Bao model from", fp,
                  "Exception:", sys.exc_info()[0])
            raise e


class JSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            str_buf += self.request.recv(1024).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return

            if (null_loc := str_buf.find("\n")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON:", json_msg)
                        break


class BaoJSONHandler(JSONTCPHandler):
    def setup(self):
        self.__messages = []

    def handle_json(self, data):
        # print(data, 'baojsonhandler')
        if "final" in data:
            message_type = self.__messages[0]["type"]
            self.__messages = self.__messages[1:]

            if message_type == "query":
                #--1. 选择一个arm对应的plan
                result = self.server.bao_model.select_plan(self.__messages)
                self.request.sendall(struct.pack("I", result))
                self.request.close()
                #--final
                #--2. 每个arm都执行一遍
                # for i in range(9):
                #     self.request.sendall(struct.pack("I", i))
                # self.request.close()
                #--final
            elif message_type == "predict":
                result = self.server.bao_model.predict(self.__messages)
                self.request.sendall(struct.pack("d", result))
                self.request.close()
            elif message_type == "reward":
                plan, buffers, obs_reward = self.__messages
                plan = add_buffer_info_to_plans(buffers, [plan])[0]
                storage.record_reward(plan, obs_reward["reward"], obs_reward["pid"])
                print("store experience")
            elif message_type == "load model":
                path = self.__messages[0]["path"]
                self.server.bao_model.load_model(path)
            else:
                print("Unknown message type:", message_type)

            return True

        self.__messages.append(data)
        return False


def start_server(listen_on, port):
    model = BaoModel()

    if os.path.exists(DEFAULT_MODEL_PATH):
        print("Loading existing model")
        model.load_model(DEFAULT_MODEL_PATH)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), BaoJSONHandler) as server:
        server.bao_model = model
        server.serve_forever()


if __name__ == "__main__":
    from multiprocessing import Process
    from config import read_config

    config = read_config()
    port = int(config["Port"])
    listen_on = config["ListenOn"]

    print(f"Listening on {listen_on} port {port}")

    server = Process(target=start_server, args=[listen_on, port])

    print("Spawning server process...")
    server.start()
