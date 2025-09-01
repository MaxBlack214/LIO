import numpy as np

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES

# 加入的编码信息
additional_pos = {"aka_title": [0, 29], "char_name": [30, 47], "role_type": [48, 50], "comp_cast_type": [51, 53],
                  "movie_link": [54, 56], "cast_info": [57, 62], "title": [63, 89], "aka_name": [90, 110],
                  "kind_type": [111, 113], "name": [114, 137], "company_type": [138, 140], "movie_info": [141, 146],
                  "person_info": [147, 152], "info_type": [153, 155], "company_name": [156, 170], "keyword": [171, 176],
                  "movie_info_idx": [177, 182]}

additional_content = {
    "aka_title": [0.99, 0, 0, 0.99, 0, 0, 0.99, 0.01, 0, 0.81, 0.07, 0.06, 0, 0, 0, 0.11, 0.09, 0.08, 0.01, 0, 0, 0.02,
                  0.02, 0.02, 0.99, 0, 0, 0, 0, 0],
    "char_name": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.01, 0, 0.15, 0.01, 0], "role_type": [0.08, 0.08, 0.08],
    "comp_cast_type": [0.25, 0.25, 0.25], "movie_link": [0, 0, 0], "cast_info": [0.61, 0.04, 0.03, 0.7, 0.04, 0.03],
    "title": [0.56, 0.03, 0.03, 0.39, 0, 0, 1, 0, 0, 0.99, 0, 0, 0, 0, 0, 0.28, 0, 0, 0.06, 0.06, 0.06, 0.56, 0.23,
              0.06, 0.96, 0, 0], "aka_name": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.12, 0, 0, 0.31, 0.01, 0],
    "kind_type": [0.14, 0.14, 0.14],
    "name": [0.42, 0.35, 0.23, 1, 0, 0, 0.79, 0.06, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.01, 0.01, 0.22, 0.01,
             0.01], "company_type": [0.25, 0.25, 0.25], "movie_info": [0.07, 0.05, 0.04, 0.9, 0.01, 0],
    "person_info": [0, 0, 0, 0.97, 0, 0], "info_type": [0.01, 0.01, 0.01],
    "company_name": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0], "keyword": [0, 0, 0, 0.06, 0, 0],
    "movie_info_idx": [0.02, 0.02, 0.02, 1, 0, 0]}


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg


def is_join(node):  # 判断是否有上述列表中的join
    return node["Node Type"] in JOIN_TYPES


def is_scan(node):  # 判断是否有上述列表中的scan
    return node["Node Type"] in LEAF_TYPES


class TreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.sum_vector = []
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)
        # print('stats_extractor', stats_extractor)
        # print('all_rels', relations)

        # 生成的适应RF的特征序列组合
        self.scan_All = np.zeros(16)

        join1 = [0, 0, 0, 0]
        join2 = [0, 0, 0, 0]
        join3 = [0, 0, 0, 0]
        self.join_All = np.zeros(12)
        # 1：Nested Loop + Nested Loop；
        # 2：Nested Loop + Hash Join；
        # 3：Nested Loop + Merge Join；
        # 4：Hash Join + Nested Loop；
        # 5：Hash Join + Hash Join；
        # 6：Hash Join + Merge Join；
        # 7：Merge Join + Nested Loop；
        # 8：Merge Join + Hash Join；
        # 9：Merge Join + Merge Join；
        self.join_join_All = np.zeros(36)

        # 1：Nested Loop + Seq Scan；
        # 2：Nested Loop + Index Scan；
        # 3：Nested Loop + Index Only Scan；
        # 4：Nested Loop + Bitmap Index Scan；
        # 5：Hash Join + Seq Scan；
        # 6：Hash Join + Index Scan；
        # 7：Hash Join + Index Only Scan；
        # 8：Hash Join + Bitmap Index Scan；
        # 9：Merge Join + Seq Scan；
        # 10：Merge Join + Index Scan；
        # 11：Merge Join + Index Only Scan；
        # 12：Merge Join + Bitmap Index Scan；
        self.join_scan_All = np.zeros(48)

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":  # 位图索引扫描
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")

    def __featurize_join(self, node):  # 生成向量
        assert is_join(node)  # 条件为 true 正常执行，条件为 false 触发异常
        # arr = np.zeros(len(ALL_TYPES)) # 可删
        # arr[ALL_TYPES.index(node["Node Type"])] = 1  # 可删

        # 下面是新的RF编码
        self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4] += 1
        if len(self.__stats(node)) > 2:
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 1] += self.__stats(node)[0]
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 2] += self.__stats(node)[1]
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 3] += self.__stats(node)[2]
        else:
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 1] += 0
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 2] += self.__stats(node)[0]
            self.join_All[JOIN_TYPES.index(node["Node Type"]) * 4 + 3] += self.__stats(node)[1]
        # final
        # return np.concatenate((arr, self.__stats(node)))
        return

    def __find_all_relation(self, node, relations):
        if not node or not node.get("Plans"):
            return
        if "Relation Name" in node:
            if node["Relation Name"] in additional_pos:
                relations.append(node["Relation Name"])
        children = node["Plans"]
        if len(children) > 0:
            self.__find_all_relation(children[0], relations)
        if len(children) > 1:
            self.__find_all_relation(children[1], relations)

    # 添加新的编码的join特征化
    # def __featurize_join(self, node):  # 生成向量
    #     assert is_join(node)  # 条件为 true 正常执行，条件为 false 触发异常
    #     # 先全部遍历一遍找到所有的表
    #     relations_for_join = []
    #     self.__find_all_relation(node, relations_for_join)
    #     # print('relations_for_join', relations_for_join)
    #     # 下面是添加编码的部分,查找node当中是否有RelationName，有的话查找这个表是否是在字典里面，有的话加入该表的所有编码
    #     arr_add = np.zeros(183)
    #     # 下面是把得到的所有relation列名都编入arr_add
    #     for ele in relations_for_join:
    #         pos = additional_pos[ele]
    #         index = 0
    #         for num in range(pos[0], pos[1] + 1):
    #             arr_add[num] = additional_content[ele][index]
    #             index += 1
    #     # print('对所有表的编码', arr_add)
    #     # left = children[0]
    #     # right = children[1]
    #     # if "Relation Name" in node:
    #     #     if node["Relation Name"] in additional_pos:
    #     #         pos = additional_pos[node["Relation Name"]]
    #     #         index = 0
    #     #         for num in range(pos[0], pos[1] + 1):
    #     #             arr_add[num] = additional_content[node["Relation Name"]][index]
    #     #             index += 1
    #     # if "Relation Name" not in left:
    #     #     temp = children[0]
    #     #     if left["Relation Name"] in additional_pos:
    #     #         pos_l = additional_pos[left["Relation Name"]]
    #     #         index = 0
    #     #         for num in range(pos_l[0], pos_l[1] + 1):
    #     #             arr_add[num] = additional_content[left["Relation Name"]][index]
    #     #             index += 1
    #     # if "Relation Name" in right:
    #     #     if right["Relation Name"] in additional_pos:
    #     #         pos_r = additional_pos[right["Relation Name"]]
    #     #         index = 0
    #     #         for num in range(pos_r[0], pos_r[1] + 1):
    #     #             arr_add[num] = additional_content[right["Relation Name"]][index]
    #     #             index += 1
    #     # final
    #     arr = np.zeros(len(ALL_TYPES))
    #     arr[ALL_TYPES.index(node["Node Type"])] = 1
    #     # 合并起来
    #     # np.concatenate((arr, arr_add), axis=0)
    #     # final
    #     return np.concatenate((arr, arr_add, self.__stats(node)))

    def __featurize_scan(self, node):  # 生成向量
        assert is_scan(node)
        # 下面是添加编码的部分,查找node当中是否有RelationName，有的话查找这个表是否是在字典里面，有的话加入该表的所有编码
        # arr_add = np.zeros(183)
        # if "Relation Name" in node:
        #     if node["Relation Name"] in additional_pos:
        #         pos = additional_pos[node["Relation Name"]]
        #         index = 0
        #         for num in range(pos[0], pos[1] + 1):
        #             arr_add[num] = additional_content[node["Relation Name"]][index]
        #             index += 1
        # final

        # arr = np.zeros(len(ALL_TYPES))  #可删
        # arr[ALL_TYPES.index(node["Node Type"])] = 1  #可删

        # 合并起来
        # np.concatenate((arr, arr_add), axis=0)
        # final
        # return (np.concatenate((arr, arr_add, self.__stats(node))),
        #         self.__relation_name(node))

        # 下面是RF的编码
        self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4] += 1
        if len(self.__stats(node)) > 2:
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 1] += self.__stats(node)[0]
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 2] += self.__stats(node)[1]
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 3] += self.__stats(node)[2]
        else:
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 1] += 0
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 2] += self.__stats(node)[0]
            self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 4 + 3] += self.__stats(node)[1]

        # 下面是关于数据分布的，稍后用其他方法处理
        # if "Relation Name" in node:
        #     if node["Relation Name"] in additional_pos:
        #         pos = additional_pos[node["Relation Name"]]
        #         sum1 = 0
        #         sum2 = 0
        #         sum3 = 0
        #         max1 = 0
        #         max2 = 0
        #         max3 = 0
        #         count = 0
        #         for num in range(pos[0], pos[1] + 1, 3):
        #             sum1 += additional_content[node["Relation Name"]][num]
        #             sum2 += additional_content[node["Relation Name"]][num + 1]
        #             sum3 += additional_content[node["Relation Name"]][num + 2]
        #             max1 = max(max1, additional_content[node["Relation Name"]][num])
        #             max2 = max(max2, additional_content[node["Relation Name"]][num + 1])
        #             max3 = max(max3, additional_content[node["Relation Name"]][num + 2])
        #             count += 1
        #         average1 = sum1 / count
        #         average2 = sum2 / count
        #         average3 = sum3 / count
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 4] = average1
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 6] = average2
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 8] = average3
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 5] = max1
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 7] = max2
        #         self.scan_All[LEAF_TYPES.index(node["Node Type"]) * 10 + 9] = max3
        # final
        return
        # return (np.concatenate((arr, self.__stats(node))),
        #         self.__relation_name(node))

    def father_son_structure_encode(self, plan, child, typeName):
        if typeName == 'join_join':
            arr1 = JOIN_TYPES
            arr2 = JOIN_TYPES
            self.join_join_All[(arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4] += 1
            self.join_join_All[
                (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 1] += (
                    self.__stats(plan)[0] + self.__stats(child)[0])
            self.join_join_All[
                (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 2] += (
                    self.__stats(plan)[1] + self.__stats(child)[1])
            if len(self.__stats(plan)) > 2 and len(self.__stats(child)) > 2:
                self.join_join_All[
                    (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 3] += (
                        self.__stats(plan)[2] + self.__stats(child)[2])
        else:
            arr1 = JOIN_TYPES
            arr2 = LEAF_TYPES
            self.join_scan_All[(arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4] += 1
            self.join_scan_All[
                (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 1] += (
                    self.__stats(plan)[0] + self.__stats(child)[0])
            self.join_scan_All[
                (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 2] += (
                    self.__stats(plan)[1] + self.__stats(child)[1])
            if len(self.__stats(plan)) > 2 and len(self.__stats(child)) > 2:
                self.join_scan_All[
                    (arr1.index(plan["Node Type"]) * 3 + arr2.index(child["Node Type"])) * 4 + 3] += (
                        self.__stats(plan)[2] + self.__stats(child)[2])

    def plan_to_feature_tree(self, plan):  # 生成特征向量树
        children = plan["Plans"] if "Plans" in plan else []

        # 下面是适应RF的编码，join-join结构和join-scan结构类型
        if len(children) == 1:
            if is_join(plan) and is_join(children[0]):
                self.father_son_structure_encode(plan, children[0], "join_join")
            if is_join(plan) and is_scan(children[0]):
                self.father_son_structure_encode(plan, children[0], "join_scan")

        if len(children) == 2:
            if is_join(plan) and is_join(children[0]):
                self.father_son_structure_encode(plan, children[0], "join_join")
            if is_join(plan) and is_join(children[1]):
                self.father_son_structure_encode(plan, children[1], "join_join")
            if is_join(plan) and is_scan(children[0]):
                self.father_son_structure_encode(plan, children[0], "join_scan")
            if is_join(plan) and is_scan(children[1]):
                self.father_son_structure_encode(plan, children[1], "join_scan")
        # final

        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])

        if is_join(plan):
            assert len(children) == 2
            self.__featurize_join(plan)
            self.plan_to_feature_tree(children[0])
            self.plan_to_feature_tree(children[1])
            # return (my_vec, left, right)
            self.__featurize_join(plan)
            return

        if is_scan(plan):
            assert not children
            # return self.__featurize_scan(plan)
            self.__featurize_scan(plan)
            return

        for child in children:
            self.plan_to_feature_tree(child)
        return


def norm(x, lo, hi):  # 正态分布
    return (np.log(x + 1) - lo) / (hi - lo)


def get_buffer_count_for_leaf(leaf, buffers):
    total = 0
    if "Relation Name" in leaf:
        total += buffers.get(leaf["Relation Name"], 0)

    if "Index Name" in leaf:
        total += buffers.get(leaf["Index Name"], 0)

    return total


class StatExtractor:
    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        res = []
        for f, lo, hi in zip(self.__fields, self.__mins, self.__maxs):
            if f not in inp:
                res.append(0)
            else:
                res.append(norm(inp[f], lo, hi))
        return res


def get_plan_stats(data):  # 生成计划的代价、基数、buffer
    costs = []
    rows = []
    bufs = []

    def recurse(n, buffers=None):
        costs.append(n["Total Cost"])
        rows.append(n["Plan Rows"])
        if "Buffers" in n:
            bufs.append(n["Buffers"])

        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)

    for plan in data:
        recurse(plan["Plan"], buffers=plan.get("Buffers", None))

    costs = np.array(costs)
    rows = np.array(rows)
    bufs = np.array(bufs)

    costs = np.log(costs + 1)
    rows = np.log(rows + 1)
    bufs = np.log(bufs + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    bufs_min = np.min(bufs) if len(bufs) != 0 else 0
    bufs_max = np.max(bufs) if len(bufs) != 0 else 0

    if len(bufs) != 0:
        return StatExtractor(
            ["Buffers", "Total Cost", "Plan Rows"],
            [bufs_min, costs_min, rows_min],
            [bufs_max, costs_max, rows_max]
        )
    else:
        return StatExtractor(
            ["Total Cost", "Plan Rows"],
            [costs_min, rows_min],
            [costs_max, rows_max]
        )


def get_all_relations(data):
    all_rels = []

    def recurse(plan):
        if "Relation Name" in plan:
            yield plan["Relation Name"]

        if "Plans" in plan:
            for child in plan["Plans"]:
                yield from recurse(child)

    for plan in data:
        all_rels.extend(list(recurse(plan["Plan"])))

    return set(all_rels)


def get_featurized_trees(data):
    all_rels = get_all_relations(data)
    stats_extractor = get_plan_stats(data)

    t = TreeBuilder(stats_extractor, all_rels)
    trees = []

    for plan in data:
        tree = t.plan_to_feature_tree(plan)
        trees.append(tree)

    # print('join_all', t.join_All)
    # print('scan_all', t.scan_All)
    # print('join_scan_all', t.join_scan_All)
    # print('join_join_all', t.join_join_All)

    return trees


def _attach_buf_data(tree):
    if "Buffers" not in tree:
        return

    buffers = tree["Buffers"]

    def recurse(n):
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)
            return

        # it is a leaf
        n["Buffers"] = get_buffer_count_for_leaf(n, buffers)

    recurse(tree["Plan"])


class TreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        for t in trees:
            _attach_buf_data(t)
        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        # self.print_new()
        # print('stats_extractor', stats_extractor)
        # print('all_rels', stats_extractor)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def clear_old_data_in_vectors(self):
        self.__tree_builder.scan_All = np.zeros(16)
        self.__tree_builder.join_All = np.zeros(12)
        self.__tree_builder.join_scan_All = np.zeros(48)
        self.__tree_builder.join_join_All = np.zeros(36)

    def get_1dvector(self):
        # print('join_all', self.__tree_builder.join_All)
        # print('scan_all', self.__tree_builder.scan_All)
        # print('join_scan_all', self.__tree_builder.join_scan_All)
        # print('join_join_all', self.__tree_builder.join_join_All)
        return self.__tree_builder.sum_vector

    def transform(self, trees):
        # if self.__tree_builder is None:
        #     self.fit(trees)
        for t in trees:
            _attach_buf_data(t)
        # result = []
        self.__tree_builder.sum_vector = []
        for x in trees:
            temp = np.array([])
            self.clear_old_data_in_vectors()
            # result.append(self.__tree_builder.plan_to_feature_tree(x["Plan"]))
            self.__tree_builder.plan_to_feature_tree(x["Plan"])
            temp = np.concatenate((self.__tree_builder.join_All, self.__tree_builder.scan_All,
                                   self.__tree_builder.join_scan_All, self.__tree_builder.join_join_All))
            self.__tree_builder.sum_vector.append(temp)
        return self.__tree_builder.sum_vector
        # return [self.__tree_builder.plan_to_feature_tree(x["Plan"]) for x in trees]

    def num_operators(self):
        return len(ALL_TYPES)
