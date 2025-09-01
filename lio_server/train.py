import pickle

import dill
from numba.core.ir import Print

import numpy as np
import storage
import model
import os
import shutil
import reg_blocker
from lio_server.constants import DEFAULT_MODEL_PATH


from datetime import datetime


def get_current_time_string():
    # 获取当前时间

    now = datetime.now()

    # 格式化时间为仅包含小时、分钟和秒的字符串

    time_string = now.strftime("%H%M%S")

    return time_string


class BaoTrainingException(Exception):
    pass


def train_and_swap(fn, old, tmp, verbose=False):
    if os.path.exists(fn):
        old_model = model.Regression(have_cache_data=True)
        old_model.load(fn)
    else:
        old_model = None

    new_model = train_and_save_model(tmp, verbose=verbose)
    # max_retries = 5
    # current_retry = 1
    # while not reg_blocker.should_replace_model(old_model, new_model):
    #     if current_retry >= max_retries == 0:
    #         print("Could not train model with better regression profile.")
    #         return
    #
    #     print("New model rejected when compared with old model. "
    #           + "Trying to retrain with emphasis on regressions.")
    #     print("Retry #", current_retry)
    #     new_model = train_and_save_model(tmp, verbose=verbose,
    #                                      emphasize_experiments=current_retry)
    #     current_retry += 1
    #
    if os.path.exists(fn):
        # shutil.rmtree(old, ignore_errors=True)
        ranstr = get_current_time_string()
        os.rename(fn, old + ranstr)
    os.rename(tmp, fn)

# 为了保存每个旧模型修改的代码
# def train_and_swap(fn, old_prefix, tmp, verbose=False):
#     # index = 0
#     # def deal():
#     #     nonlocal index
#         # 假设 fn 是当前模型的完整路径，old_prefix 是旧文件名前缀，tmp 是新模型的临时路径
#         print(fn,  old_prefix, tmp)
#         if os.path.exists(fn):
#             # 加载旧模型（这里假设 model.BaoRegression 和其 load 方法已经定义）
#             old_model = model.BaoRegression(have_cache_data=True)
#             old_model.load(fn)
#         else:
#             old_model = None
#
#         # 训练并保存新模型（这里假设 train_and_save_model 函数已经定义）
#         new_model_path = train_and_save_model(tmp, verbose=verbose)
#         # 注意：train_and_save_model 应该返回新模型的完整路径，或者您可以在这里构建它
#
#         # 如果旧模型文件存在，则重命名它
#         if os.path.exists(fn):
#             # 构建旧文件的完整路径（假设 .model 是文件扩展名）
#             oldfile = f"{old_prefix}_{0}"
#             os.rename(fn, oldfile)
#             # index += 1  # 递增全局索引
#         os.rename(tmp, fn)  # 这里假设 new_model_path 和 fn 在同一个文件系统上
        # print(tmp, fn , oldfile)
    # return deal
    #final
def train_and_save_model(fn, verbose=True, emphasize_experiments=0):
    all_experience = storage.experience()
    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    print('experience size', storage.experience_size())
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")

    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.Regression(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    return reg


if __name__ == "__main__":
    import sys

    # print(sys.argv, '000')
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1])
    # print(sys.argv[1], 't')

    print("Model saved, attempting load...")
    reg = model.BaoRegression(have_cache_data=True)
    reg.load(sys.argv[1])
