
PROJECT="js-titan-dslabs"
LOCATION="us-central1"
MODEL="gemini-2.5-flash"
TEMPERATURE=0.5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Milvus audio collections in this project are built with 768-dim mpnet embeddings (see utils/aud_db_utils.py).
QUERY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LOCAL_VIDEO_DIR = "/Users/amana1/working_dir/videos"
LOCAL_PROCESSING_DIR = "/Users/amana1/working_dir/Meta_Extraction/out"
MAX_WORKERS = 10
LOG_DIR = "/Users/amana1/working_dir/Meta_Extraction/Logs"
PROMPT_TEMPLATES_DIR = "/Users/amana1/working_dir/Meta_Extraction/prompts"
DEBUG_MODE = False

#character detection params
RESIZE_WIDTH = 640
SCENE_THRESHOLD = 30.0
FACE_SIM_THRESHOLD = 0.45          
FACE_PAD = 0.2 

CHUNK_DURATION = 5  # in seconds
SLEEP_DURATION = 60  # in seconds


###Database Configs###

DB_NAME="postgres"
DB_USER="postgres"
PASSWORD="postgres"
HOST="localhost"
PORT="5432"

PIPELINE_TABLE = "pipeline_jobs_table"

VIDEO_TABLE = "video_meta"
AUDIO_TABLE = "audio_meta"

VIDEO_COLUMNS = [
    "chunk_id", "description", "frame_idx", "timestamp", "gender",
    "objects", "noticeable", "unnoticeable"
]

AUDIO_COLUMNS = [
    "chunk_id", "content_summary", "overall_tone", "overall_sentiment",
    "background_type", "background_description"
]

#####milvus configs#####

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

VIDEO_COLLECTION = "video_embeddings"
AUDIO_COLLECTION = "audio_embeddings"

#######################################################



"""
Directory Structure:

    output_dir/media_name/{time_stamp}/
        frames/
            frame_000.png
            ....

        audio_chunks/
            audio_000.wav
            ....

        video_chunks/
            video_000.mp4
            ....

        annotated_frames/
            frame_000.png
            ....

        shots/
            shot_{start}_{end}.mp4
            ....

        prompt1/
            chunk_000.json
            ....

        prompt2/
            chunk_000.json
            ....

        prompt3/
            chunk_000.json
            ....

        prompt4/
            chunk_000.json
            ....

        shots.json
        shots_description.json
        prompt1_prompt2.xlsx
        prompt3_prompt4.xlsx
        scenes_description.json

"""