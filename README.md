# LIO

**LIO** is designed to run on **PostgreSQL version 16.9**.

This is the prototype implementation of **LIO**, a Learned Query Optimization model.

## Usage

Before using this method, you need to prepare a folder to store your SQL files. Then, run the program with the following command:

```bash
python run_queries.py "/sql_path/*sql" 200 num_sql dbname
