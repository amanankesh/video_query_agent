import os
import json
import time
from utils.job_queue import fetch_next_job, update_job_stage, mark_job_failed
from utils.inference import get_meta_data
from utils.aud_db_utils import get_pg_conn
from utils.json_to_excel import merge_prompt3_prompt4, merge_prompt1_prompt2

from config import (
    SLEEP_DURATION, CHUNK_DURATION, 
    DEBUG_MODE, PROMPT_TEMPLATES_DIR, 
    PROJECT, LOCATION, 
    MODEL, MAX_WORKERS, TEMPERATURE
)

conn = get_pg_conn()

sleep_time = SLEEP_DURATION
debug = DEBUG_MODE

args = {
    'model': MODEL,
    'project': PROJECT,
    'location': LOCATION,
    'temperature': TEMPERATURE,
    'chunk_size': CHUNK_DURATION,
    'max_workers': MAX_WORKERS,
    'prompt_dir': PROMPT_TEMPLATES_DIR,
}

status = "pending"

while True:

    job = fetch_next_job(conn, 'inference', status=status)
    if job:
        try:
            start = time.time()
            local_path = job['local_path']
            dir_path = os.path.dirname(local_path)
            video_name = os.path.splitext(os.path.basename(local_path))[0]
            combined = os.path.join(dir_path, video_name)

            args['output_dir'] = combined
            usage_stats = get_meta_data(args)
            merge_prompt1_prompt2(combined)
            merge_prompt3_prompt4(combined)

            inference_time = time.time() - start
            usage_stats["inference_time"] = f"{inference_time:0.2f}"
            usage_stats["model"] = MODEL
            usage_stats["temperature"] = TEMPERATURE
            update_job_stage(
                conn,
                job['id'],
                'shot_description',
                new_status='pending',
                addons=[
                    f"inference_stats = '{json.dumps(usage_stats)}'::jsonb"
                ]
            )
            print("Inference time:", inference_time)
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            mark_job_failed(conn, job['id'])

        if debug:
            print("Exiting due to debug mode")
            break

    else:
        print(f"inference_stage : sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
