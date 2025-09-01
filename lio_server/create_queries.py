data = []
for line in open("/home/lab505/dhy/sql/01_train/fast_after_agg_train.sql", "r"):
    data.append(line)
    continue


def sql_create(name, msg):
    first_path = "/home/lab505/queries_for_beginner/"
    full_path = first_path + name + '.sql'
    with open(full_path, 'a') as file:
        file.write(msg)
        file.close()


count = 1
for i in data:
    sql_create('q' + str(count), i)
    count += 1
    continue