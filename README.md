# LIO

**LIO** is designed to run on **PostgreSQL version 16.9**.

This is the prototype implementation of **LIO**, a Learned Query Optimization model.

## Usage

Before using this method, please install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
Then, prepare a folder to store your SQL files.

- `num_sql` should be set to the total number of SQL files in your folder.

Run the program with the following command:

```bash
python run_queries.py "/sql_path/*sql" 200 num_sql dbname
