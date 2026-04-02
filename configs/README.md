# Job config YAML

Files here are passed to **`create_job_list.py`** (`--config`) and serialized into each pipeline job’s `config` JSON column.

Common keys:

| Key | Role |
|-----|------|
| `network`, `media_type`, `language` | Used in output paths and Postgres table names (`{network}_{media_type}_{language}[_channel]_frame\|audio`). |
| `channel` | Optional; omit or `null` if not applicable. |
| `download_dir` | Base directory for processed assets (e.g. `media_files/`). |
| `num_files`, `max_size_gb` | Limits when listing local or S3 objects. |
| `bucket_name`, `s3_prefix` | S3 source when using cloud listing (see `utils/download.py`). |
| `download_log`, `infer_meta` | Optional metadata paths for your deployment. |

Example: [`movies_hindi.yaml`](movies_hindi.yaml).
