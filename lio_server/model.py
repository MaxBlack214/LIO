import csv
import json
import sys
import shutil
import dill
import joblib
import numpy as np
import torch
import torch.optim
import os
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from featurize import TreeFeaturizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import log_config  # 注意是导入整个模块


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evolutionary_forest.forest import EvolutionaryForestRegressor

from sklearn.ensemble import RandomForestRegressor

CUDA = torch.cuda.is_available()


def _x_transform_path(base):
    return os.path.join(base, "x_transform")


def _y_transform_path(base):
    return os.path.join(base, "y_transform")


def ef_path(base):
    return os.path.join(base, "EF.pkl")


def _inv_log1p(x):
    return np.exp(x) - 1




def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    # targets = torch.tensor(targets)
    # 上面这行代码说效率太低了，转换成下面的是否会好一点
    targets_np = np.array(targets)  # 将列表转换为单一的 numpy 数组
    targets = torch.from_numpy(targets_np)  # 将 numpy 数组转换为 PyTorch 张量
    return trees, targets


class Regression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.EF = EFregressor()
        self.__verbose = verbose

        #  我们可以用np.log1p(x)，即取对数，这样可以使得数据在一定程度上符合正态分布的特征。还原过程就是log1p的逆运算expm1.
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])

        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0

    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n

    def load(self, path):  # 在机器学习中我们训练模型后，需要把模型保存到本地，这里我们采用joblib来保存 joblib.load
        # with open(_n_path(path), "rb") as f:  # 每次都写close()比较繁琐，Python引入with语句，这样能够确保最后文件一定被关闭，且不用手动再调用close方法.
        #     self.__n = joblib.load(f)
        # with open(_channels_path(path), "rb") as f:
        #     self.__in_channels = joblib.load(f)

        # self.__net = net.BaoNet(self.__in_channels)
        with open(ef_path(path), "rb") as f:
            self.EF.regressor = dill.load(f)
        # self.__net.eval()
        #
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        if not os.path.exists(path):
            # try to create a directory here
            os.makedirs(path, exist_ok=True)
        else:
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        # torch.save(self.__net.state_dict(), _nn_path(path))
        # joblib.dump(self.__net, ef_path(path))
        with open(ef_path(path), "wb") as f:
            dill.dump(self.EF.regressor, f)
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        # with open(_channels_path(path), "wb") as f:
        #     joblib.dump(self.__in_channels, f)
        # with open(_n_path(path), "wb") as f:
        #     joblib.dump(self.__n, f)

    def fit(self, X, y):
        # 会认为子类是一种父类类型，考虑继承关系。
        # 如果要判断两个类型是否相同推荐使用
        # isinstance()。
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
    
        # print('TEST', X)
        self.__n = len(X)

        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)



        pairs = list(zip(X, y))
        dataset = DataLoader(pairs,
                             batch_size=2000,
                             shuffle=True,
                             collate_fn=collate)
        print(len(dataset), 'length')
        # determine the initial number of channels
        # for inp, _tar in dataset:
        #     in_channels = inp[0][0].shape[0]
        #     break

        # self.__log("Initial input channels:", in_channels)
        #
        # if self.__have_cache_data:
        #     assert in_channels == self.__tree_transform.num_operators() + 3
        # else:
        #     assert in_channels == self.__tree_transform.num_operators() + 2
        #
        if isinstance(X, list):
            X = np.array(X)
        for x, y in dataset:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            # -------- y_train和y_test的修改，仅对rf有效，非rf请删除
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            # ----------5

            self.EF.regressor.fit(x_train, y_train)
            
            
            log_path = os.path.join(log_config.log_dir, 'x_train_y_train.csv')
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([x_train, y_train])
            r2 = r2_score(y_test, self.EF.regressor.predict(x_test))
            self.__log("rscore:", r2)

            base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 model.py 所在目录绝对路径
            testdata_dir = os.path.join(base_dir, "testdata")
            
            
            
            os.makedirs(testdata_dir, exist_ok=True)  # 确保目录存在
            np.save(os.path.join(testdata_dir, 'x_test.npy'), x_test)
            np.save(os.path.join(testdata_dir, 'y_test.npy'), y_test)
            rscore_path = os.path.join(log_config.log_dir, 'rscore.csv')
            with open(rscore_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([r2])
            break
        # if epoch % 15 == 0:
        #     self.__log("Epoch", epoch, "training loss:", loss_accum)

        # stopping condition
        # if len(losses) > 10 and losses[-1] < 0.1:
        #     last_two = np.min(losses[-2:])
        #     if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
        #         self.__log("Stopped training from convergence condition at epoch", epoch)
        #         break
        # else:
        #     self.__log("Stopped training after max epochs")

    # new function
    def transform_to_1dvector(self):
        return self.__tree_transform.get_1dvector()

    # final

    def predict(self, X, return_all=False):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        # print('TEST2', X)
        X = self.__tree_transform.transform(X)
        # print('TRANSFORM', X)
        # self.__net.eval()
        # pred = self.__net(X).cpu().detach().numpy()
        X = np.array(X)
        if not return_all:
            pred = self.EF.regressor.predict(X,return_all=return_all)
            # print(X, 'X')
            #仅rf----
            pred = pred.reshape(-1, 1)
            #-------
            print(pred, 'pred')
            return self.__pipeline.inverse_transform(pred)
        else:
            pred, predictions, mean_prediction, std_prediction, hof_size = self.EF.regressor.predict(X,return_all=return_all)
            pred = pred.reshape(-1, 1)
            #-------
            print(pred, 'pred')
            return self.__pipeline.inverse_transform(pred), predictions, mean_prediction, std_prediction, hof_size,pred


class EFregressor:
    def __init__(self):
        self.regressor = EvolutionaryForestRegressor(max_height=6,gene_num=3, boost_size=3, n_gen=5, n_pop=50, cross_pb=1,elitism=3                                 ,base_learner='Random-DT', normalize=False, select='AutomaticLexicase', 
                                                     verbose=True, n_process=1)

    # def __init__(self):
    #     self.regressor = RandomForestRegressor(max_depth=5, n_estimators=75, verbose=1)


if __name__ == '__main__':
    print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'path')
