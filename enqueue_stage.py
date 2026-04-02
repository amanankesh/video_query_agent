from glob import glob
from utils.aud_db_utils import get_pg_conn
from psycopg2.extras import execute_values

import time
import pandas as pd
from config import SLEEP_DURATION, DEBUG_MODE, PIPELINE_TABLE

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--old', type=str)
args = parser.parse_args()

sleep_time = SLEEP_DURATION
debug = DEBUG_MODE

conn = get_pg_conn()

old_job = args.old
table = PIPELINE_TABLE


while True:
    cursor = conn.cursor()
    files = glob('jobs/*.xlsx')
    files.sort()

    if files:
        new_job = files[-1]

        if new_job != old_job:
            print("Enqueuing:", new_job)
            df = pd.read_excel(new_job, sheet_name='enqueue')
            old_job = files[-1]
            data = list(df.itertuples(index=False, name=None))

            if data:
                execute_values(cursor, f"INSERT INTO {table} \
                                         (stage, priority, s3_key, filename, config) VALUES %s \
                                         ON CONFLICT (s3_key) DO UPDATE SET \
                                            priority = EXCLUDED.priority", data)
                conn.commit()
            old_job = new_job

        cursor.close()

        if debug:
            print("Exiting due to debug mode")
            break

    print(f"enqueue_stage : sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)