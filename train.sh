#!/bin/bash
export DS_DB="GCP SQL instance name"
#export DERMINATOR_DB='GCP SQL instance name'
cd /
#/cloud_sql_proxy -credential_file=/path -instances=$DS_DB=tcp: &
#/cloud_sql_proxy -credential_file=/path -instances=$DERMINATOR_DB=tcp: &

/cloud_sql_proxy  -instances=$DS_DB=tcp:portNumber &
# /cloud_sql_proxy  -instances=$DERMINATOR_DB=tcp:portnumber &

#/cloud_sql_proxy -credential_file=/path -instances=$DS_DB=tcp:portnumber &
#/cloud_sql_proxy -credential_file=/path -instances=$DERMINATOR_DB=tcp:portnumber &
python /app/src/main.py