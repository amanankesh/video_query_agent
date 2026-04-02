import json
import os
import subprocess
import time
import logging
import concurrent.futures
from typing import List, Dict, Any

from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

# =========================================================
# 🔧 CONFIGURATION
# =========================================================

VIDEO_PATH = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0.mp4"
SHOT_JSON_PATH = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0/shots_description.json"

OUTPUT_DIR = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0/output/scene_clips8"

# Save Stage-1 + Final
OUTPUT_STAGE1_JSON = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0/output/scenes_stage1_raw8.json"
OUTPUT_FINAL_JSON  = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0/output/scenes_final8.json"

# Prompt files
PROMPT_STAGE1_PATH = "/Users/amana1/working_dir/Meta_Extraction/prompts/scene_detection_prompt.txt"
PROMPT_STAGE2_PATH = "/Users/amana1/working_dir/Meta_Extraction/prompts/scene_merge_prompt.txt"

USE_VIDEO_EXPORT = True
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
MAX_WORKERS = 4
MODEL_NAME = "gemini-2.5-pro"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD PROMPTS
# =========================================================

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SCENE_DETECTION_PROMPT = load_prompt(PROMPT_STAGE1_PATH)
SEMANTIC_MERGE_PROMPT  = load_prompt(PROMPT_STAGE2_PATH)

# =========================================================
# GEMINI SETUP
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("🔌 Connecting to Vertex AI...")

client = genai.Client(
    vertexai=True,
    project="js-titan-dslabs",
    location="us-central1",
    http_options=HttpOptions(api_version="v1")
)

generation_config = GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.1
)

# =========================================================
# STAGE 1 — DETECT SCENES
# =========================================================

def chunk_shots(shots, window=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, i = [], 0
    while i < len(shots):
        chunks.append(shots[i:i + window])
        i += window - overlap
    return chunks

def process_single_chunk(chunk_data):
    idx = chunk_data["index"]
    shots = chunk_data["shots"]

    shots_json = json.dumps(shots, ensure_ascii=False, indent=2)
    prompt = SCENE_DETECTION_PROMPT.replace("{{shots_json}}", shots_json)

    for _ in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=generation_config
            )
            scenes = json.loads(response.text)
            return scenes if isinstance(scenes, list) else scenes.get("scenes", [])
        except:
            time.sleep(3)

    return []

def detect_scenes_parallel(shots):
    chunks = chunk_shots(shots)
    payloads = [{"index": i, "shots": c} for i, c in enumerate(chunks)]

    all_scenes = []
    with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as ex:
        for r in ex.map(process_single_chunk, payloads):
            all_scenes.extend(r)

    # =========================================================
    # ADD FIRST/LAST SHOT DESCRIPTIONS HERE
    # =========================================================
    for scene in all_scenes:
        ss = scene.get("start_shot")
        es = scene.get("end_shot")

        if ss is not None and ss < len(shots):
            scene["first_shot_description"] = shots[ss].get("description")
        else:
            scene["first_shot_description"] = None

        if es is not None and es < len(shots):
            scene["last_shot_description"] = shots[es].get("description")
        else:
            scene["last_shot_description"] = None

    # SAVE
    with open(OUTPUT_STAGE1_JSON, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ STAGE-1 SAVED → {OUTPUT_STAGE1_JSON}")
    return all_scenes

# =========================================================
# STAGE 2 — NUMERIC REPAIR
# =========================================================

def repair_scene_continuity(scenes, shots):
    indices = [s.get("shot_index", i) for i,s in enumerate(shots)]
    min_idx, max_idx = min(indices), max(indices)

    scenes = [s for s in scenes if "start_shot" in s and "end_shot" in s]
    scenes.sort(key=lambda x: x["start_shot"])

    if scenes[0]["start_shot"] > min_idx:
        scenes[0]["start_shot"] = min_idx

    merged, last = [], None

    for s in scenes:
        if not last:
            merged.append(s)
            last = s
            continue

        if s["start_shot"] <= last["end_shot"]:
            last["end_shot"] = max(last["end_shot"], s["end_shot"])
            continue

        merged.append(s)
        last = s

    if merged[-1]["end_shot"] < max_idx:
        merged[-1]["end_shot"] = max_idx

    for i, s in enumerate(merged, 1):
        s["scene_id"] = i

    return merged

# =========================================================
# STAGE 3 — SEMANTIC MERGE
# =========================================================

def perform_semantic_merge(scenes):
    if len(scenes) <= 1:
        for i, s in enumerate(scenes, 1):
            s["scene_id"] = i
        return scenes

    # Send compact representation WITH shot descriptions
    mini = [{
        "scene_id": s["scene_id"],
        "scene_location": s.get("scene_location"),
        "main_characters": s.get("main_characters", []),
        "scene_summary": s.get("scene_summary", ""),
        "narrative_context": s.get("narrative_context", ""),
        "time_continuity": s.get("time_continuity", "continuous"),
        "first_shot_description": s.get("first_shot_description"),
        "last_shot_description": s.get("last_shot_description")
    } for s in scenes]

    prompt = SEMANTIC_MERGE_PROMPT.replace(
        "{{scenes_json}}",
        json.dumps(mini, indent=2, ensure_ascii=False)
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=generation_config
        )
        merge_groups = json.loads(response.text)
    except:
        return scenes

    scene_map = {s["scene_id"]: s for s in scenes}
    used = set()
    final = []
    new_id = 1

    for group in merge_groups:
        ids = [gid for gid in group if gid in scene_map]
        if not ids:
            continue

        ids.sort(key=lambda x: scene_map[x]["start_shot"])
        used.update(ids)

        # LOCATION + CHARACTER SAFETY CHECKS
        locs = {scene_map[i].get("scene_location") for i in ids}
        base_chars = set(scene_map[ids[0]].get("main_characters", []))

        valid = True
        for i in ids[1:]:
            if scene_map[i].get("scene_location") not in locs:
                valid = False
            if not base_chars.intersection(scene_map[i].get("main_characters", [])):
                valid = False

        if not valid:
            for i in ids:
                cp = scene_map[i].copy()
                cp["scene_id"] = new_id
                new_id += 1
                final.append(cp)
            continue

        base = scene_map[ids[0]].copy()
        base["scene_id"] = new_id
        new_id += 1
        base["start_shot"] = min(scene_map[i]["start_shot"] for i in ids)
        base["end_shot"]   = max(scene_map[i]["end_shot"] for i in ids)

        final.append(base)

    for s in scenes:
        if s["scene_id"] not in used:
            cp = s.copy()
            cp["scene_id"] = new_id
            new_id += 1
            final.append(cp)

    final.sort(key=lambda x: x["start_shot"])
    return final

# =========================================================
# STAGE 4 — TIMESTAMPS + EXPORT
# =========================================================

def attach_time_ranges_and_export(scenes, shots):
    shot_map = {s.get("shot_index", i): s for i,s in enumerate(shots)}
    keys = sorted(shot_map.keys())

    for s in scenes:
        ss, es = s["start_shot"], s["end_shot"]
        ss = ss if ss in shot_map else keys[0]
        es = es if es in shot_map else keys[-1]

        st = float(shot_map[ss].get("start", 0))
        et = float(shot_map[es].get("end", 0))
        s["start_time"], s["end_time"] = st, et

        if USE_VIDEO_EXPORT:
            out_file = os.path.join(
                OUTPUT_DIR,
                f"scene_{s['scene_id']:03d}_{int(st)}s_{int(et)}s.mp4"
            )

            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{st:.3f}",
                "-i", VIDEO_PATH,
                "-t", f"{(et - st):.3f}",
                "-c", "copy", out_file
            ])

            s["video_path"] = out_file

    return scenes

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    with open(SHOT_JSON_PATH, "r", encoding="utf-8") as f:
        shots = json.load(f)

    raw_scenes      = detect_scenes_parallel(shots)
    repaired_scenes = repair_scene_continuity(raw_scenes, shots)
    final_scenes    = perform_semantic_merge(repaired_scenes)
    final_scenes    = attach_time_ranges_and_export(final_scenes, shots)

    with open(OUTPUT_FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_scenes, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ FINAL JSON SAVED → {OUTPUT_FINAL_JSON}")
