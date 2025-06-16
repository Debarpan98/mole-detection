#!/bin/bash
export DS_DB="playground-datascience-289517:northamerica-northeast1:datascience"
#export DERMINATOR_DB='staging-tools-dermago:northamerica-northeast1:derminator-staging-t5r86:datascience'
cd /
#/cloud_sql_proxy -credential_file=/tmp/auth_keys/auth.json -instances=$DS_DB=tcp:5432 &
#/cloud_sql_proxy -credential_file=/tmp/auth_keys/auth.json -instances=$DERMINATOR_DB=tcp:5433 &

/cloud_sql_proxy  -instances=$DS_DB=tcp:5432 &
# /cloud_sql_proxy  -instances=$DERMINATOR_DB=tcp:5433 &

#/cloud_sql_proxy -credential_file=playground-datascience-289517-4a213c02630a.json -instances=$DS_DB=tcp:5432 &
#/cloud_sql_proxy -credential_file=staging-tools-dermago-d3bd8d788ba5.json -instances=$DERMINATOR_DB=tcp:5433 &
python /app/src/main.py