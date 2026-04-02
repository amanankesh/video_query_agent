import os
import time
from utils.download import download_local_file

from utils.job_queue import update_job_stage, fetch_next_job, mark_job_failed
from config import LOCAL_VIDEO_DIR, SLEEP_DURATION, DEBUG_MODE, SCENE_THRESHOLD, CHUNK_DURATION
from utils.aud_db_utils import get_pg_conn
from utils.video_utils import split_video
from utils.detect_shots import detect_and_split_shots
conn = get_pg_conn()
debug = DEBUG_MODE
sleep_time = SLEEP_DURATION

status = "pending"
local_base_dir = LOCAL_VIDEO_DIR

i = 0
while True:
    job = fetch_next_job(conn, 'download', status=status) # instead of fetching one job at a time, we can fetch multiple jobs and process in parallel
    
    if job:
        try:
            start = time.time()
            s3_key = job['s3_key']
            new_filename = job['filename'].replace('/', '\/')
            config = job['config']
            print("config : ", new_filename, "\n")

            local_dir = os.path.join(config['download_dir'], config['network'], config['media_type'], config['language'], config['channel'] if config['channel'] is not None else '')
 
            print(s3_key, local_dir, new_filename, config['download_dir'])
            local_path = download_local_file(local_base_dir, s3_key, local_dir, new_filename, config['download_dir'])
            download_time = time.time() - start

            if local_path:
                ### these two functions can be run parallelly
                start = time.time()
                split_video(local_path, local_dir, split_duration=CHUNK_DURATION)
                start = time.time()
                detect_and_split_shots(video_path=local_path, threshold=SCENE_THRESHOLD)
                shot_detection_time = time.time() - start
            else:
                print("\nlocal_path : ", local_path,"\n")
                raise ValueError("Download returned no local_path")

            if local_path:
                print("Downloaded to:", local_path)
                update_job_stage(conn, job['id'], 'character_detection', new_status='pending', addons=[f"local_path = '{local_path}'", f"download_time = {download_time:0.2f}", f"shot_detection_time = {shot_detection_time:0.2f}"])

        except Exception as e:
            print("Download failed for:", s3_key, e)
            mark_job_failed(conn, job['id'])

        if debug:
            print("Exiting due to debug mode")
            break
    else:
        print(f"download_stage : sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
