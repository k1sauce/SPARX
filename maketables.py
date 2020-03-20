# !/usr/bin/python
import os
from db import connect

if __name__ == '__main__':

    conn = connect()
    cur = conn.cursor()

    datasets = os.listdir('SPaRX/SPARX Data Sets')

    for ds in datasets:

        tname = os.path.splitext(ds)[0]

        # SQL string concat is very bad for security but OK

        sql = " SELECT EXISTS ("
        sql += " SELECT FROM information_schema.tables"
        sql += " WHERE table_schema = 'public'"
        sql += f" AND table_name = '{tname}'"
        sql += ")"

        cur.execute(sql)

        if cur.fetchone()[0]:
            continue
        else:
            sql = f"CREATE TABLE {tname}()"
            cur.execute(sql)
            conn.commit()

    conn.close()









