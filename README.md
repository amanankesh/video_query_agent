# Video metadata extraction pipeline

Automated processing for long-form video: job-queue–driven workers download or copy media, split and chunk video/audio, detect shots, run face clustering, call **Google Gemini** for per-chunk visual and audio metadata, merge results to Excel, and (optionally) load structured rows plus embeddings into **PostgreSQL** and **Milvus** for semantic search.

---

## Table of contents

1. [Overview](#overview)  
2. [Repository layout](#repository-layout)  
3. [Prerequisites](#prerequisites)  
4. [Python dependencies](#python-dependencies)  
5. [Configuration (`config.py`)](#configuration-configpy)  
6. [Environment variables](#environment-variables)  
7. [Databases](#databases)  
8. [End-to-end workflow](#end-to-end-workflow)  
9. [Pipeline stages](#pipeline-stages)  
10. [Per-title output layout](#per-title-output-layout)  
11. [Metadata attributes (reference)](#metadata-attributes-reference)  
12. [Semantic query (`query.py`)](#semantic-query-querypy)  
13. [Troubleshooting](#troubleshooting)  

---

## Overview

| Component | Purpose |
|-----------|---------|
| **`pipeline_jobs_table`** (Postgres) | Queue of work items: stage name, priority, S3/local key, filename, JSON `config`, status, paths, timing and stats. |
| **Stage workers** (`*_stage.py`) | Long-running processes that `SELECT … FOR UPDATE SKIP LOCKED` the next job for their stage, process it, then advance the row to the next stage or mark failure. |
| **`configs/*.yaml`** | Network, language, channel, local/S3 roots, file limits—serialized into each job’s `config` column and used to build output paths and DB table names. |
| **`jobs/*.xlsx`** | Generated job batches; `enqueue_stage.py` inserts the `enqueue` sheet into the pipeline table. |
| **`prompts/`** | Text templates for Gemini (chunk visual/audio prompts, shot description, etc.—used by `utils/inference.py` and related code). |
| **Milvus + Sentence-Transformers** | Vector index over chunk text (e.g. `content_summary`); query side must use the **same embedding model** as insert (see below). |

---

## Repository layout

| Path | Description |
|------|-------------|
| `config.py` | Global paths, Gemini project/location/model, DB/Milvus defaults, chunk duration, face/shot thresholds, table name constants. |
| `configs/*.yaml` | Per-campaign or per-network settings for `create_job_list.py` (see [`configs/README.md`](configs/README.md)). |
| `jobs/` | Timestamped `.xlsx` files produced by `create_job_list.py`; consumed by `enqueue_stage.py`. |
| `prompts/` | Gemini prompt templates. |
| `utils/` | DB helpers, download, inference, shot/face detection, Excel merge, **`utils/search.py`** (semantic search helpers). |
| `requirements.txt` | Pip dependencies (see [Python dependencies](#python-dependencies)). |
| `query.py` | CLI for natural-language search against Milvus + Postgres. |
| `create_pipeline_table.py` | Creates `PIPELINE_TABLE` if it does not exist. |
| `create_job_list.py` | Builds job Excel + creates per-network **frame** and **audio** Postgres tables. |
| `enqueue_stage.py` | Watches `jobs/*.xlsx` and upserts rows into the pipeline table. |
| `download_stage.py`, `character_detection_stage.py`, `inference_stage.py`, … | Stage workers (see [Pipeline stages](#pipeline-stages)). |

### Repository tree (top level)

```text
video_understading/
├── README.md
├── requirements.txt
├── query.py
├── config.py
├── create_pipeline_table.py
├── create_job_list.py
├── enqueue_stage.py
├── download_stage.py
├── character_detection_stage.py
├── inference_stage.py
├── db_insertion_stage.py
├── configs/
├── jobs/
├── prompts/
└── utils/
```

Additional stage scripts may live at the repo root or under `utils/` depending on your branch.

YAML keys for job configs are summarized in [`configs/README.md`](configs/README.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10+ recommended. |
| **PostgreSQL** | For `pipeline_jobs_table` and dynamic `{network}_{media_type}_{language}[_channel]_{frame\|audio}` tables. |
| **Milvus** | 2.x, reachable at `MILVUS_HOST`:`MILVUS_PORT` (default `localhost:19530`). Needed after embeddings are inserted; required for `query.py`. |
| **Google Cloud / Vertex AI** | Project and credentials for `google-genai` / Gemini (`PROJECT`, `LOCATION`, `MODEL` in `config.py`). |
| **Media sources** | Local directory (e.g. `LOCAL_VIDEO_DIR`) and/or S3 (`boto3`) depending on `utils/download.py` usage. |
| **Hardware** | GPU optional; face models use ONNX Runtime; Apple Silicon users often use `onnxruntime-silicon` instead of `onnxruntime` for speed. |

---

## Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

**Included groups (see `requirements.txt` for versions):**

- **Core:** `numpy<2`, `pandas`, `opencv-python-headless`, `scenedetect[opencv]`, `google-genai`, `openpyxl`, `boto3`  
- **Face / vision:** `insightface`, `tqdm`, `scikit-learn`, `onnxruntime`  
- **Embeddings & vectors:** `sentence-transformers[torch]`, `pymilvus`, `psycopg2-binary`  
- **Config:** `python-dotenv` (optional; used by `utils/aud_db_utils.py` to load `.env`)

---

## Configuration (`config.py`)

These are the main symbols used across stages (adjust paths for your machine):

| Symbol | Role |
|--------|------|
| `PROJECT` | GCP project id for Gemini. |
| `LOCATION` | Region (e.g. `us-central1`). |
| `MODEL` | Gemini model id (e.g. `gemini-2.5-flash`). |
| `TEMPERATURE` | Sampling temperature for generation. |
| `EMBEDDING_MODEL` | General embedding label in config (`all-MiniLM-L6-v2` is 384-d; not used for the default Milvus audio schema). |
| `QUERY_EMBEDDING_MODEL` | **Use for Milvus search**; must match vectors stored in collections (**`all-mpnet-base-v2`**, 768-d, see `utils/aud_db_utils.py`). |
| `LOCAL_VIDEO_DIR` | Root for local media when downloading/copying. |
| `LOCAL_PROCESSING_DIR` | Example output base (some docs reference `media_files/` under YAML `download_dir`). |
| `MAX_WORKERS` | Parallelism hint for inference (`inference_stage.py`). |
| `LOG_DIR` | Log directory. |
| `PROMPT_TEMPLATES_DIR` | Directory containing prompt `.txt` files for Gemini. |
| `DEBUG_MODE` | If `True`, many stages exit after one job (useful for testing). |
| `RESIZE_WIDTH` | Frame width for character detection path. |
| `SCENE_THRESHOLD` | PySceneDetect threshold (see `download_stage` / shot detection). |
| `FACE_SIM_THRESHOLD`, `FACE_PAD` | Face clustering parameters. |
| `CHUNK_DURATION` | Seconds per audio/video chunk for inference. |
| `SLEEP_DURATION` | Seconds to sleep when a stage has no jobs. |
| `DB_NAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT` | PostgreSQL connection defaults. |
| `PIPELINE_TABLE` | Name of the jobs table (default `pipeline_jobs_table`). |
| `VIDEO_TABLE`, `AUDIO_TABLE` | Legacy/example names in config (`video_meta`, `audio_meta`); actual tables are often dynamic (see [Databases](#databases)). |
| `VIDEO_COLUMNS`, `AUDIO_COLUMNS` | Example column lists for metadata schemas. |
| `MILVUS_HOST`, `MILVUS_PORT` | Milvus endpoint. |
| `VIDEO_COLLECTION`, `AUDIO_COLLECTION` | Example collection names (`video_embeddings`, `audio_embeddings`); production names are often derived like Postgres tables. |

The multiline string at the bottom of `config.py` documents an example **per-title directory tree** under `output_dir/media_name/{time_stamp}/` (frames, chunks, prompts, JSON, Excel).

---

## Environment variables

`utils/aud_db_utils.py` loads `.env` and expects **database** variables with these exact names:

| Variable | Meaning |
|----------|---------|
| `DB_NAME` | PostgreSQL database name |
| `DB_USER` | User |
| `PASSWORD` | Password (**not** `DB_PASSWORD`) |
| `HOST` | Host |
| `PORT` | Port |
| `MILVUS_HOST` | Milvus host |
| `MILVUS_PORT` | Milvus port |

If these are **not** set, `utils/search.py` falls back to the same fields from `config.py` for connections.

---

## Databases

### Pipeline jobs table

Created by `create_pipeline_table.py` as `PIPELINE_TABLE` (default `pipeline_jobs_table`). Notable columns:

- **`stage`** — One of: `download`, `inference`, `db_insertion`, `character_detection`, `shot_description`, `scene_detection`, `scene_description`.  
- **`priority`** — Higher runs first (`ORDER BY priority DESC`).  
- **`s3_key`** — Unique key (local or S3 style path identifier).  
- **`filename`**, **`config`** (JSONB) — File name and full YAML-derived config.  
- **`local_path`**, **`processed_output`** — Paths on disk after processing.  
- **`status`** — `pending`, `in_progress`, `done`, `failed`.  
- **Timing / stats** — `download_time`, `db_insertion_time`, `shot_detection_time`, JSON blobs for inference and downstream stages.

Workers use **`FOR UPDATE SKIP LOCKED`** so multiple workers can run the same stage safely.

### Dynamic frame and audio tables

`create_job_list.py` creates Postgres tables named:

```text
{network}_{media_type}_{language}[_{channel}]_frame
{network}_{media_type}_{language}[_{channel}]_audio
```

Example: `viacom18_movies_hindi_frame`, `viacom18_movies_hindi_audio` when `channel` is null (the `_channel` segment is omitted).

Schema details are defined in `utils/vid_db_utils.py` (frame) and `utils/aud_db_utils.py` (audio)—including `id TEXT PRIMARY KEY`, `movie`, `chunk_id`, transcript fields, background/song/brand fields, etc.

### Milvus

Audio collections are created in `utils/aud_db_utils.py` with **768-dimensional** float vectors, **COSINE** metric, **`all-mpnet-base-v2`**-compatible embeddings, and a **VARCHAR** `id` aligned with Postgres rows. Collection names in production often mirror the audio table name (see `db_insertion_stage.py` intent).

### Database sketches (illustrative)

> **Source of truth:** `create_pipeline_table.py` defines the jobs table; `create_job_list.py` creates dynamic tables via `utils/vid_db_utils.py` and `utils/aud_db_utils.py`. This section is reference only.

**Pipeline jobs table (older sketch):**

```text
id SERIAL PRIMARY KEY,
stage TEXT CHECK (stage IN ('download', 'inference', 'db_insertion')),
priority INTEGER NOT NULL,
s3_key TEXT NOT NULL,
filename TEXT NOT NULL,
config JSONB NOT NULL,
metadata JSONB DEFAULT NULL,
local_path TEXT DEFAULT NULL,
processed_output TEXT DEFAULT NULL,
download_time REAL,
inference_time REAL,
db_insertion_time REAL,
infer_logs JSONB,
status TEXT CHECK (status IN ('pending', 'in_progress', 'done', 'failed')) DEFAULT 'pending',
updated_at TIMESTAMP DEFAULT NOW(),
CONSTRAINT unique_s3_key UNIQUE (s3_key)
```

Current code allows additional `stage` values such as `character_detection`, `shot_description`, `scene_detection`, `scene_description`.

**Frame / video metadata table (sketch):**

```text
id TEXT PRIMARY KEY,
movie TEXT,
second INT,
model TEXT,
temperature DECIMAL(3, 2),
objects TEXT[],
object_count JSONB,
gender TEXT[],
ocr_text TEXT[],
noticeable_objects_top TEXT[],
noticeable_objects_bottom TEXT[],
unnoticeable_objects_top TEXT[],
unnoticeable_objects_bottom TEXT[],
scene_emotion TEXT[],
age_group TEXT[],
scene_tags TEXT[],
scene_label TEXT,
weather TEXT,
day_night INT,
person_emotion TEXT[],
clarity_of_image TEXT,
actions TEXT[],
celebrity TEXT[],
timestamp TEXT,
brand_based_on_logos TEXT[],
location TEXT[],
setting TEXT[],
description TEXT,
sentiment TEXT
```

Align actual columns with `utils/vid_db_utils.py` and your inference JSON schema.

---

## End-to-end workflow

1. **Create the jobs table** (once per database):

   ```bash
   python create_pipeline_table.py
   ```

2. **Prepare YAML** under `configs/` (see `configs/movies_hindi.yaml` for fields: `network`, `media_type`, `language`, `channel`, `download_dir`, `num_files`, `max_size_gb`, optional S3 fields).

3. **Generate jobs and DB tables**:

   ```bash
   python create_job_list.py --config configs/movies_hindi.yaml --local_dir /path/to/videos
   ```

   This creates `jobs/job_<timestamp>.xlsx` (sheets `enqueue` and `check`) and ensures frame/audio tables exist.

4. **Enqueue** (run continuously or once):

   ```bash
   python enqueue_stage.py
   ```

   It picks the latest `jobs/*.xlsx` and `INSERT … ON CONFLICT (s3_key) DO UPDATE` into `PIPELINE_TABLE` from the `enqueue` sheet.

5. **Run stage workers** in separate terminals or processes (order of stages follows job `stage` field transitions in code—typically download → character detection → inference → …):

   ```bash
   python download_stage.py
   python character_detection_stage.py
   python inference_stage.py
   # … additional stages as implemented (shot/scene/db insertion)
   ```

6. **Query** embedded content (after Milvus + Postgres are populated):

   ```bash
   python query.py "your question" -c COLLECTION -t TABLE --top-k 10
   ```

---

## Pipeline stages

Documented order (see also [Per-title output layout](#per-title-output-layout)):

| Order | Stage | Script / module | Role |
|------|--------|------------------|------|
| — | Job table | `create_pipeline_table.py` | DDL for `PIPELINE_TABLE`. |
| — | Job list | `create_job_list.py` | Excel jobs + `frame`/`audio` Postgres tables. |
| — | Enqueue | `enqueue_stage.py` | Sync Excel → Postgres queue. |
| 1 | Download | `download_stage.py` | Next job with `stage=download`: copy/download file, split video/audio (`CHUNK_DURATION`), run shot detection (`SCENE_THRESHOLD`). Advances job toward `character_detection`. |
| 2 | Character detection | `character_detection_stage.py` | Face detection / clustering on frames (`utils/detect_and_cluster.py`). |
| 3 | Inference | `inference_stage.py` | Gemini over annotated frames + audio chunks (`utils/inference.py`), merges JSON to Excel (`utils/json_to_excel.py`), advances toward `shot_description`. |
| 4+ | Shot / scene pipelines | `shot_description_stage.py`, `shot_detection_stage.py`, … (if present) | See [Per-title output layout](#per-title-output-layout) for artifacts. |
| … | DB insertion | `db_insertion_stage.py` | Intended to read processed Excel, insert into Postgres + Milvus embeddings (**requires `utils.db` and related helpers in your deployment**). |

**Note:** `inference_stage.py` uses `DEBUG_MODE` from `config.py` to exit after one job when testing.

---

## Per-title output layout

Paths are relative to your configured output base (often `download_dir` from YAML + `network` / `media_type` / `language` / optional `channel` + sanitized title). Some deployments use a timestamp segment under `media_name/`.

### Processing steps (logical)

| Step | Input → output |
|------|----------------|
| **0 — Video processing** | `media_dir` → split video → `…/frames`, `…/audio_chunks`, `…/video_chunks` |
| **1 — Shot detection** | Video → `…/shots`, `shots.json` |
| **2 — Character recognition** | `frames/` → `annotated_frames/` |
| **3 — Inference** | `annotated_frames/` → `prompt1/`, `prompt2/` · `audio_chunks/` → `prompt3/`, `prompt4/` |
| **4 — Post-processing** | `prompt1/` + `prompt2/` → merged Excel · `prompt3/` + `prompt4/` → merged Excel |
| **5 — Shot description** | `shots.json` + merged visual Excel → `shots_description.json` (and optionally `.xlsx`) |
| **6 — Scene description** | `shots_description.json` → `scenes_description.json` (and optionally `.xlsx`) |

### Directory tree (typical)

```text
output_dir/media_name/{time_stamp}/
├── video.mp4
├── frames/
├── audio_chunks/
├── video_chunks/
├── annotated_frames/
├── shots/
├── prompt1/
├── prompt2/
├── prompt3/
├── prompt4/
├── shots.json
├── shots_description.json
├── shots_description.xlsx
├── prompt1_prompt2.xlsx
├── prompt3_prompt4.xlsx
├── scenes_description.json
└── scenes_description.xlsx
```

### Artifacts (summary)

| Path / artifact | Description |
|-----------------|-------------|
| `frames/` | Extracted frames. |
| `audio_chunks/` | `audio_*.wav` (length ≈ `CHUNK_DURATION`). |
| `video_chunks/` | Short MP4 chunks. |
| `annotated_frames/` | Frames with face annotations. |
| `shots/` | Per-shot clips; `shots.json` lists shot boundaries. |
| `prompt1/` … `prompt4/` | Per-chunk JSON from Gemini (visual/audio prompts). |
| `prompt1_prompt2.xlsx`, `prompt3_prompt4.xlsx` | Merged spreadsheets. |
| `shots_description.json` (+ `.xlsx`) | Shot-level descriptions when that stage runs. |
| `scenes_description.json` (+ `.xlsx`) | Scene-level rollups when that stage runs. |

See also the multiline docstring at the end of `config.py`.

---

## Metadata attributes (reference)

Used for scene detection, prompting, and semantic search field selection.

### Frame-oriented

Core field groups:

`object`, `scene_tags`, `scene_label`, `scene_location`, `scene_setting`, `scene_description`, `character_info`

Suggested facets:

1. Location  
2. Setting  
3. Scene tags  
4. Scene label  
5. Characters  
6. Description  
7. Objects (scene-defining)  
8. Day / night  
9. Weather  

### Audio-oriented

Core field groups:

`background_emotion`, `background_type`, `background_description`, `content_description`, `translation_approximate`, `character_info`

Suggested facets:

1. Background description  
2. Background type  
3. Background emotion  
4. Overall tone  
5. Overall audio emotion  
6. Audio events  
7. Speakers  
8. Character info  
9. **Content summary**  
10. **Translation (approximate)**  
11. **Transcript full text** (strong candidate for semantic similarity / embeddings)

---

## Semantic query (`query.py`)

CLI entrypoint:

```bash
python query.py "<natural language>" \
  --collection MILVUS_COLLECTION_NAME \
  --table POSTGRES_TABLE_NAME \
  [--top-k N] [--json]
```

| Option | Short | Meaning |
|--------|-------|---------|
| `text` | (positional) | Query string; if omitted, stdin is read. |
| `--collection` | `-c` | Milvus collection (must match how embeddings were inserted). |
| `--table` | `-t` | Postgres table whose `id` values match Milvus primary keys. |
| `--top-k` | `-k` | Number of hits (default `10`). |
| `--json` | | Print structured JSON only (for automation). |

**Embedding consistency:** collections built with `utils/aud_db_utils.py` use **768-d** **`all-mpnet-base-v2`**-style vectors. Set `QUERY_EMBEDDING_MODEL` in `config.py` accordingly (default is already `sentence-transformers/all-mpnet-base-v2`).

**Infrastructure:** Milvus must be running and reachable; otherwise connection fails at query time.

**Alternative:** `python -m utils.search` can run an interactive loop using **`multimodal_search`** with default collection/table names (`milvus_audio_table` / `pg_audio_table`, etc.)—handy only if you use those exact names.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| **Cannot connect to Milvus** | Start Milvus 2.x; verify `MILVUS_HOST` / `MILVUS_PORT` (firewall, Docker port mapping). |
| **Empty or wrong search results** | Same embedding model at query and insert time; collection and table names match production; Postgres rows exist for returned ids. |
| **Postgres auth errors** | `DB_NAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT` in `.env` or `config.py`. |
| **No jobs picked up** | Job `stage` and `status` must match what each worker queries (`fetch_next_job` uses `status` e.g. `pending` vs `in_progress` depending on script). |
| **`db_insertion_stage` import errors** | That script imports `utils.db`; supply that module in your environment or align the stage with `utils/aud_db_utils.py` + Milvus helpers you actually use. |
| **Hardcoded paths in `create_job_list.py`** | Default `--config` / `--local_dir` in the file may point to another machine; pass `--config` and `--local_dir` explicitly. |

---

## Summary

This README describes how to install dependencies, configure `config.py` and environment variables, create and enqueue jobs, run stage workers, understand output layout and metadata fields, and run **`query.py`** against Milvus and PostgreSQL.
