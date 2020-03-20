#!/usr/bin/env bash
# Run in SPARK Data Sets directory to populate tables.
for f in *.sas7bdat; do sas2db --db postgresql+psycopg2://kyle:password@localhost:5432/pdsas $f; done;