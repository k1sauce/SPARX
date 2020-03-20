# !/usr/bin/python
from db import connect

if __name__ == '__main__':

    conn = connect()
    cur = conn.cursor()

    sql = "SELECT table_name "
    sql += "FROM INFORMATION_SCHEMA.TABLES "
    sql += "WHERE table_schema = 'public' "

    cur.execute(sql)
    table_names = cur.fetchall()

    for tname in table_names:

        tname = tname[0]
        print('Fixing ascii in table: ' + tname)

        sql = "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS"
        sql += f" WHERE data_type = 'text' AND table_name = '{tname}'"


        cur.execute(sql)
        columnames = cur.fetchall()
        for this_c_name in columnames:

            cnlower = False

            this_c_name = this_c_name[0]

            if this_c_name.islower():

                cnlower = True

                sql = f'SELECT index, {this_c_name} FROM public."{tname}"'

            else:

                sql = f'SELECT index, "{this_c_name}" FROM public."{tname}"'


            cur.execute(sql)

            this_c = cur.fetchall()

            # try to convert if the first row in this_c starts with \\x
            try:
                if this_c[0][1].startswith('\\x'):

                    for row in this_c:

                        index = row[0]
                        ascii = bytearray.fromhex(row[1].split('x')[1]).decode()

                        ascii = ascii.replace("'","-")
                        ascii = ascii.replace('"', "--")


                        sql = f'UPDATE public."{tname}" '

                        if cnlower:
                            sql += f"SET {this_c_name} "
                        else:
                            sql += f'SET "{this_c_name}" '

                        sql += f"= '{ascii}' WHERE index = {index} "
                        print(sql)
                        cur.execute(sql)
                        conn.commit()

            except:

                # if the first row in this_c is None check the rows and convert
                if this_c[0][1] is None:

                    for row in this_c:

                        # continue if the value is None
                        if row[1] is None:
                            continue

                        # else convert if starts with \\x
                        else:
                            if row[1].startswith('\\x'):

                                index = row[0]
                                ascii = bytearray.fromhex(row[1].split('x')[1]).decode()

                                ascii = ascii.replace("'", "-")
                                ascii = ascii.replace('"', "--")

                                sql = f'UPDATE public."{tname}" '

                                if cnlower:
                                    sql += f"SET {this_c_name} "
                                else:
                                    sql += f'SET "{this_c_name}" '

                                sql += f"= '{ascii}' WHERE index = {index} "

                                print(sql)
                                cur.execute(sql)
                                conn.commit()

                            else:
                                continue

