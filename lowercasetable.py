# !/usr/bin/python
from db import connect

if __name__ == '__main__':

    conn = connect()
    cur = conn.cursor()

    # get the table names
    cur.execute("SELECT table_name  FROM information_schema.tables WHERE table_schema = 'public';")
    tablenames = cur.fetchall()
    tablenames = [x[0] for x in tablenames]

    # make all the table names lower case if they exist
    for table in tablenames:
        if not table.islower():
            sql = f'ALTER TABLE public."{table}" RENAME TO '
            sql += table.lower()
            print(sql)
            cur.execute(sql)
            conn.commit()
        else:
            continue

    # get the table names
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tablenames = cur.fetchall()
    tablenames = [x[0] for x in tablenames]


    # make all the column names lower
    for table in tablenames:

        #get the column names
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name= '{table}'")
        cnames = cur.fetchall()
        cnames = [x[0] for x in cnames]

        for col in cnames:
            if not col.islower():
                sql = f'ALTER TABLE {table} RENAME COLUMN "{col}" TO '
                sql += col.lower()
                print(sql)
                cur.execute(sql)
                conn.commit()
            else:
                continue

    conn.close()
