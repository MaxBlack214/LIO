import dill
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from IPython.display import Image
from sklearn import tree
# import pydotplus
import graphviz
# from evolutionary_forest.forest import EvolutionaryForestRegressor
from sklearn.tree import _tree
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance, feature_append


def tree_to_rules(dtree, feature_name, index):
    tree_ = dtree.tree_
    feature_name = [
        feature_name[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []
    values = tree_.value.tolist()

    def recurse(node, depth, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_branch = tree_.children_left[node]
            right_branch = tree_.children_right[node]

            rule_left = list(rule)
            rule_left.append((name, '<=', threshold))
            recurse(left_branch, depth + 1, rule_left)

            rule_right = list(rule)
            rule_right.append((name, '>', threshold))
            recurse(right_branch, depth + 1, rule_right)
        else:
            class_ = values[node][0]
            # if class_[0] > class_[1]:
            #     rules.append((rule, 0))
            # else:
            #     rules.append((rule, 1))
            # print(class_)
            rules.append((rule, class_[0]))

    recurse(0, 1, [])
    return rules


# estimator用于sklearn中的随机森林
# for index, model in enumerate(Estimators):
#     rules = tree_to_rules(model, diabetes.feature_names, 0)
#     print(rules)
#     break
# 输出某一个例子的判断规则
# def simple_example():
#     X, y = make_regression(n_samples=100, n_features=5, n_informative=5)
#     gp = EvolutionaryForestRegressor(max_height=8, normalize=True, select='AutomaticLexicase', boost_size=10, n_gen=2,
#                                      gene_num=5, base_learner='Random-DT')
#     gp.fit(X, y)
#     assert_almost_equal(mean_squared_error(y, gp.predict(X)), 0)
#     # assert len(gp.hof) == 10
#     # assert len(gp.hof[0].gene) == 5
#     print(str(gp.hof[0].gene[0]))


def graph_rules(ef, graph_name):
    for index, individual in enumerate(ef.hof):
        feature_name = []
        for j in range(len(individual.gene)):
            feature_name.append(str(individual.gene[j]))
        # filename = 'test_' + str(index) + '.pdf'
        # 导出决策树为 Graphviz 格式
        model = individual.pipe['Ridge']
        dot_data = tree.export_graphviz(model, out_file=None,
                                        feature_names=feature_name,
                                        filled=True, rounded=True,
                                        special_characters=True)

        # 使用 Graphviz 显示决策树
        graph = graphviz.Source(dot_data)
        graph.render('result/pic/tpcds_200_422' + str(index)) # 保存为文件
        graph.view()  # 在新窗口中显示图形
        rules = tree_to_rules(model, feature_name, 0)
        print(rules)
        break


def feature_name(model):
    feature_importance_dict = get_feature_importance(model)
    print(feature_importance_dict, 'feature_importance_dict')
    plot_feature_importance(feature_importance_dict, True)


if __name__ == '__main__':
    # with open('/home/lab505/EF_for_SQL/EF4PostgreSQL/bao_server/EF_Previous_Model/EF.pkl', 'rb') as file:
    with open('/home/lab505/msj/tpcds_1_model/EF_Previous_Model/EF.pkl', 'rb') as file:
        model = dill.load(file)
        graph_rules(model, 'tpcds_rf')
    feature_name(model)