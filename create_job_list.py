import os
import json
import pandas as pd
from utils import read_yaml
from datetime import datetime
from argparse import ArgumentParser
from utils.vid_db_utils import create_frame_table
from utils.aud_db_utils import get_pg_conn, create_audio_table
from utils.download import list_local_files, generate_new_filename, check_filename


def build_table_name(cfg, suffix):
    channel = cfg.get("channel")
    base = "_".join([cfg["network"], cfg["media_type"], cfg["language"]])
    if channel:
        base = f"{base}_{channel}"
    return f"{base}_{suffix}"

def main():
    parser = ArgumentParser(description="Create job sheet and required DB tables.")
    parser.add_argument("--config", type=str, default="configs/movies_hindi.yaml")
    parser.add_argument("--local_dir", type=str, default="/Users/amana1/working_dir/videos")
    parser.add_argument("--priority", type=int, default=2)
    args = parser.parse_args()

    conn = get_pg_conn()
    config = read_yaml(args.config)

    os.makedirs("jobs", exist_ok=True)

    files = list_local_files(
        args.local_dir,
        max_size_gb=config["max_size_gb"],
        num_movies=config["num_files"],
    )

    frame_table = build_table_name(config, "frame")
    audio_table = build_table_name(config, "audio")
    print("Using frame table:", frame_table)
    print("Using audio table:", audio_table)
    create_frame_table(conn, frame_table)
    create_audio_table(conn, audio_table)

    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT movie FROM {frame_table};")
    existing = pd.DataFrame(cursor.fetchall())
    cursor.close()

    df = pd.DataFrame(files, columns=["s3_key"])
    df["filename"] = df["s3_key"].apply(lambda x: generate_new_filename(x, config["media_type"]))
    df["movie"] = df["filename"].str.split(".").str[0]
    df["check_filename"] = df["filename"].apply(check_filename)
    df["config"] = json.dumps(config)
    df["stage"] = "download"
    df["priority"] = args.priority

    mask = True if existing.empty else (~df["movie"].isin(existing["movie"]))
    enqueue_df = df[mask & (df["check_filename"])]
    check_df = df[mask & (~df["check_filename"])]

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join("jobs", f"job_{date_time}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        enqueue_df[["stage", "priority", "s3_key", "filename", "config"]].to_excel(
            writer,
            sheet_name="enqueue",
            index=False,
        )
        check_df[["s3_key", "filename", "config"]].to_excel(
            writer,
            sheet_name="check",
            index=False,
        )
    conn.close()
    print("Created:", output_file)


if __name__ == "__main__":
    main()