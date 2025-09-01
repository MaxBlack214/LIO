import sqlite3

# 连接到SQLite数据库
conn = sqlite3.connect('bao.db')  # 将'example.db'替换为你的文件名

# 创建一个Cursor对象
cursor = conn.cursor()

# 执行SQL查询
cursor.execute('SELECT * FROM experience')  # 将'table_name'替换为你的表名

# 获取查询结果
rows = cursor.fetchall()

# 打印查询结果
for row in rows:
    print(row)

# 关闭连接
conn.close()