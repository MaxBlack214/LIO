from sklearn.inspection import permutation_importance
import numpy as np
import dill

model_dir = "bao_server/EF_Default_Model"

with open(f"{model_dir}/EF.pkl", "rb") as f:
    model = dill.load(f)

x = np.load('bao_server/testdata/x_test.npy')
y = np.load('bao_server/testdata/y_test.npy')

result = permutation_importance(model, x, y, n_repeats=10, random_state=0, scoring='r2')
print(result.importances_mean)