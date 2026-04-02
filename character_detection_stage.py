import os
import time
import json
from utils.detect_and_cluster import process_video_faces
from utils.job_queue import update_job_stage, fetch_next_job, mark_job_failed
from config import SLEEP_DURATION, DEBUG_MODE
from utils.aud_db_utils import get_pg_conn

conn = get_pg_conn()
debug = DEBUG_MODE
sleep_time = SLEEP_DURATION
status = "pending"

while True:
    job = fetch_next_job(conn, 'character_detection', status=status) 
    
    if job:
        # try:
        start = time.time()
        
        local_path = job.get('local_path', None)
        if local_path:
            
            dir_path = os.path.dirname(local_path)
            video_name = os.path.splitext(os.path.basename(local_path))[0]
            combined = os.path.join(dir_path, video_name)

            print("\nlocal_path : ", local_path, combined, "\n")

            usage_stats = process_video_faces(combined)
            character_detection_time = time.time() - start
            usage_stats["character_detection_time"] = f"{character_detection_time:0.2f}"
            update_job_stage(conn, job['id'], 'inference', new_status='pending', addons=[f"local_path = '{local_path}'", f"character_detection_stats = '{json.dumps(usage_stats)}'::jsonb"])
        else:
            mark_job_failed(conn, job['id'])
            raise ValueError("No local_path found in job")
       
        # except Exception as e:
        #     print("Download failed for:", e)

        if debug:
            print("Exiting due to debug mode")
            break
    else:
        print(f"character_detection_stage : sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
