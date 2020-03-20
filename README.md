Kyle Hazen - 20 March 2020

SPARX Data Set were provided in the sas7bdat file format.

To ease exploratory analysis these files were imported into a 
PostgreSQL database named "pdsas". 

The code in the jupyter notebook requires a connection to the pdsas database,
hosted at localhost. To set up the pdsas database on OSX you can do the following:

To install postgres on mac use homebrew
```
brew install postgresql
```

To start the postgres background service use
```
brew services start postgresql
```


Make a database called pdsas
```
psql postgres
>postgres=# CREATE DATABASE pdsas; 
```

import 'pdsas.pgsql' to postgres with the following command. 
You should have the postgres background service running. 

"user" is the Owner of the pdsas table.

```
psql -U <user> pdsas < pdsas.pgsql
```

Edit the database.ini file with your username and password

[postgresql]
host=localhost
database=pdsas
user=<user>
password=password

Then open the jupyter notebook sparxv1.ipynb
```
jupyter notebook sparxv1.ipynb
```


Below steps documents how the data was imported into postgresql

After initializing a postgresql database (pdsas) tables 
for each data set were created by running.
```bash
python maketables.py
```

The tables were then populated with the bash script.
```bash
#!/usr/bin/env bash
# Run in SPARK Data Sets directory to populate tables.
for f in *.sas7bdat; do sas2db --db postgresql+psycopg2://kyle:password@localhost:5432/pdsas $f; done;
```

And the ASCII encoding was fixed in the database by running.
```bash
python converteascii.py
```

The case of the table names and table columns names was converted to lowercase
by running
```bash
python lowercasetables.py
```

