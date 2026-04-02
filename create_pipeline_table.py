from utils.aud_db_utils import get_pg_conn
from config import PIPELINE_TABLE

conn = get_pg_conn()
table = PIPELINE_TABLE

query = f"""
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    stage TEXT CHECK (stage IN ('download', 'inference', 'db_insertion', 'character_detection', 'shot_description', 'scene_detection', 'scene_description')) NOT NULL,
    priority INTEGER NOT NULL,
    s3_key TEXT UNIQUE,
    filename TEXT NOT NULL,
    config JSONB NOT NULL,
    metadata JSONB DEFAULT NULL,
    local_path TEXT DEFAULT NULL,
    processed_output TEXT DEFAULT NULL,

    download_time REAL,
    db_insertion_time REAL,
    shot_detection_time REAL,

    character_detection_stats JSONB,
    inference_stats JSONB,
    shot_description_stats JSONB,
    scene_detection_stats JSONB,
    scene_description_stats JSONB,

    infer_logs JSONB,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'done', 'failed')) DEFAULT 'pending',
    updated_at TIMESTAMP DEFAULT NOW()
);
"""
cursor = conn.cursor()
cursor.execute(query)
conn.commit()
conn.close()
print(f"Table '{table}' created successfully.")
