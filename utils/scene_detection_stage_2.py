#!/usr/bin/env python3
"""
stage2_llm_merge.py
Stage-2 LLM Merge (FULL DATA PAIRWISE VERSION)

Features:
✓ Uses FULL scene JSON for decision
✓ Reads REAL timestamps from shots_description.json (Ground Truth)
✓ Validates LLM response
✓ Generates Final JSON, Screenplay Text, and Video Clips
"""

import json, os, time, logging, subprocess, glob
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# Base Directory
BASE_DIR = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0"

# Inputs
STAGE1_JSON = os.path.join(BASE_DIR, "output", "scenes_stage1.json") # Ensure this matches Stage 1 output
SHOTS_JSON = os.path.join(BASE_DIR, "shots_description.json")

# Outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FINAL_JSON = os.path.join(OUTPUT_DIR, "scenes_final.json")
SCRIPT_TXT = os.path.join(OUTPUT_DIR, "final_screenplay.txt")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "scene_clips")

PROMPT_PATH = "/Users/amana1/working_dir/Meta_Extraction/prompts/scene_pair_prompt.txt"

# Video Source Locator (Auto-detect mp4/mkv)
possible_videos = glob.glob(os.path.join(BASE_DIR, "*.mp4")) + glob.glob(os.path.join(BASE_DIR, "*.mkv"))
VIDEO_SOURCE = possible_videos[0] if possible_videos else None

MODEL_NAME = "gemini-2.5-pro"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


# ---------------------------------------------------------
# INIT LLM
# ---------------------------------------------------------

client = genai.Client(
    vertexai=True,
    project="js-titan-dslabs",
    location="us-central1",
    http_options=HttpOptions(api_version="v1")
)

gen_cfg = GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.0
)


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def load_text(path):
    try:
        return open(path, "r", encoding="utf-8").read()
    except:
        logger.error(f"Could not load prompt: {path}")
        return ""

PAIR_PROMPT = load_text(PROMPT_PATH)


def safe_float(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except:
        return default


# ---------------------------------------------------------
# LOAD SHOTS & ATTACH REAL TIMESTAMPS
# ---------------------------------------------------------

def load_shots_and_attach(scenes):
    """
    Enrich Stage 1 scenes with the ACTUAL start/end times from the shots file.
    This fixes any hallucinated timestamps from the LLM.
    """
    print("--> Loading ground-truth shots...")
    if not os.path.exists(SHOTS_JSON):
        print("!! Error: Shots JSON not found.")
        return scenes

    shots = json.load(open(SHOTS_JSON, "r", encoding="utf-8"))
    shot_map = {s.get("shot_index"): s for s in shots}

    for sc in scenes:
        start_idx = sc.get("start_shot")
        end_idx = sc.get("end_shot")

        fs = shot_map.get(start_idx, {})
        ls = shot_map.get(end_idx, {})

        # Overwrite with real data
        sc["start_sec"] = safe_float(fs.get("start") or fs.get("start_sec"))
        sc["end_sec"] = safe_float(ls.get("end") or ls.get("end_sec"))

        sc["first_shot_description"] = fs.get("shot_description") or fs.get("description", "")
        sc["last_shot_description"] = ls.get("shot_description") or ls.get("description", "")

    return scenes


# ---------------------------------------------------------
# LLM MERGE DECISION (VALIDATED)
# ---------------------------------------------------------

def ask_merge_full(sceneA, sceneB):
    """
    Sends two scenes to LLM to decide if they are one continuous scene.
    """
    if not PAIR_PROMPT:
        return False

    # Create a cleaner payload for the LLM to save tokens, but keep logic
    payload_A = {
        "scene_id": sceneA.get("scene_id"),
        "location": sceneA.get("scene_location"),
        "characters": sceneA.get("main_characters"),
        "last_shot": sceneA.get("last_shot_description"),
        "script_end": sceneA.get("play_script", [])[-3:] if sceneA.get("play_script") else []
    }
    
    payload_B = {
        "scene_id": sceneB.get("scene_id"),
        "location": sceneB.get("scene_location"),
        "characters": sceneB.get("main_characters"),
        "first_shot": sceneB.get("first_shot_description"),
        "script_start": sceneB.get("play_script", [])[:3] if sceneB.get("play_script") else []
    }

    pair_payload = {"scene_A": payload_A, "scene_B": payload_B}

    prompt = PAIR_PROMPT.replace(
        "{{pair_json}}",
        json.dumps(pair_payload, indent=2, ensure_ascii=False)
    )

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=gen_cfg
            )

            data = json.loads(resp.text)

            if isinstance(data, dict) and "merge" in data:
                reason = data.get("reasoning", "No reason given")
                print(f"    LLM: {reason}")
                return bool(data["merge"])
            
            # If list returned (rare), check first item
            if isinstance(data, list) and len(data) > 0:
                return bool(data[0].get("merge", False))

        except Exception as e:
            time.sleep(1)
            
    return False


# ---------------------------------------------------------
# MERGE TWO SCENES
# ---------------------------------------------------------

def merge_two(a, b):
    """
    Combines Scene A and Scene B into a single object.
    B is merged INTO A (A is extended).
    """
    merged = a.copy()

    # 1. Update Endpoints
    merged["end_shot"] = b["end_shot"]
    merged["end_sec"] = b["end_sec"]
    merged["last_shot_description"] = b.get("last_shot_description")

    # 2. Merge Characters
    merged["main_characters"] = sorted(
        list(set(a.get("main_characters", [])) |
             set(b.get("main_characters", [])))
    )

    # 3. Merge Scripts
    merged["play_script"] = a.get("play_script", []) + b.get("play_script", [])

    # 4. Merge Arcs
    arcA = a.get("character_arc", {})
    arcB = b.get("character_arc", {})
    
    # Update Arc logic: 
    # If char exists in both, keep A's start state and take B's end state.
    for char, data in arcB.items():
        if char in arcA:
            arcA[char]["ending_state"] = data.get("ending_state")
            arcA[char]["intent_shift"] = f"{arcA[char].get('starting_state')} -> {data.get('ending_state')}"
        else:
            arcA[char] = data
            
    merged["character_arc"] = arcA

    # 5. Merge Summary
    merged["scene_summary"] = (a.get("scene_summary", "") + " " +
                               b.get("scene_summary", "")).strip()

    return merged


# ---------------------------------------------------------
# SAVE SCREENPLAY
# ---------------------------------------------------------

def write_screenplay(scenes):
    print(f"--> Writing screenplay to {SCRIPT_TXT}...")
    with open(SCRIPT_TXT, "w", encoding="utf-8") as f:
        f.write("FINAL SCREENPLAY\n=================\n\n")

        for sc in scenes:
            sid = sc.get("scene_id")
            start = time.strftime("%H:%M:%S", time.gmtime(sc["start_sec"]))
            end = time.strftime("%H:%M:%S", time.gmtime(sc["end_sec"]))
            loc = sc.get('scene_location', 'UNKNOWN').upper()

            f.write(f"SCENE {sid} - {loc} [{start} - {end}]\n")
            f.write("Characters: " + ", ".join(sc.get("main_characters", [])) + "\n")
            f.write("-" * 50 + "\n")

            for line in sc.get("play_script", []):
                f.write(line + "\n")
            f.write("\n\n")


# ---------------------------------------------------------
# EXTRACT CLIPS (FFMPEG)
# ---------------------------------------------------------

def extract_clips(scenes):
    if not VIDEO_SOURCE or not os.path.exists(VIDEO_SOURCE):
        print(f"!! Warning: Video file not found at {VIDEO_SOURCE or 'Unknown'}. Skipping clips.")
        return

    print(f"\n--> Extracting clips from {os.path.basename(VIDEO_SOURCE)}...\n")

    for sc in scenes:
        sid = sc["scene_id"]
        start = safe_float(sc["start_sec"])
        end = safe_float(sc["end_sec"])

        # Validation
        if end <= start:
            print(f" ❌ Scene {sid}: Invalid timestamps ({start} -> {end}), skipping.")
            continue
        
        # Minimum duration check (0.5s)
        if end - start < 0.5:
             print(f" ⚠️ Scene {sid}: Too short (<0.5s), skipping.")
             continue

        # Format filename
        filename = f"scene_{sid:03d}.mp4"
        clip_path = os.path.join(CLIPS_DIR, filename)

        # FFmpeg Command
        # Using -ss before -i for fast seek, -to for duration
        # Using 'libx264' with 'ultrafast' to ensure frame accuracy (copy can fail on non-keyframes)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", VIDEO_SOURCE,
            "-c:v", "libx264",     # Re-encode for accuracy
            "-preset", "ultrafast",
            "-c:a", "copy",
            "-loglevel", "error",
            clip_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f" ✔ Saved Scene {sid}: {filename} ({end-start:.1f}s)")
        except Exception as e:
            print(f" ✖ Failed to save Scene {sid}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("==========================================")
    print("      SCENE MERGING - STAGE 2 (PAIRWISE)")
    print("==========================================")

    if not PAIR_PROMPT:
        print(f"CRITICAL ERROR: Prompt file not found at {PROMPT_PATH}")
        return

    # 1. LOAD DATA
    if not os.path.exists(STAGE1_JSON):
        print(f"Error: Stage 1 output not found at {STAGE1_JSON}")
        return
        
    scenes = json.load(open(STAGE1_JSON, "r", encoding="utf-8"))
    
    # Assign temp IDs for logging
    for i, sc in enumerate(scenes):
        sc["scene_id"] = i + 1

    # 2. ENRICH WITH REAL TIMESTAMPS
    scenes = load_shots_and_attach(scenes)
    
    # 3. ROLLING MERGE LOOP
    merged_list = []
    i = 0
    total = len(scenes)

    print(f"--> Processing {total} raw fragments...")
    
    while i < total:
        # If last element, just append it
        if i == total - 1:
            merged_list.append(scenes[i])
            break

        A = scenes[i]
        B = scenes[i + 1]

        # Optimization: Don't ask LLM if timestamps are wildly far apart (> 5s gap)
        time_gap = B["start_sec"] - A["end_sec"]
        if time_gap > 5.0:
            print(f"   [Pair {A['scene_id']}+{B['scene_id']}] ↛ SPLIT (Time gap {time_gap:.1f}s)")
            merged_list.append(A)
            i += 1
            continue

        print(f"   [Pair {A['scene_id']}+{B['scene_id']}] Comparing...", end=" ")
        
        # Ask LLM
        should_merge = ask_merge_full(A, B)

        if should_merge:
            print("→ MERGE ✔")
            # Merge B into A, then set B as the new A (rolling forward)
            scenes[i + 1] = merge_two(A, B)
            # We do NOT append A to merged_list yet; the combined result continues to check against the next one
        else:
            print("→ SPLIT")
            # A is finished, add to final list
            merged_list.append(A)

        i += 1

    # 4. FINAL ID REASSIGNMENT
    for i, sc in enumerate(merged_list):
        sc["scene_id"] = i + 1

    # 5. SAVE OUTPUTS
    print(f"\n--> Saving Final JSON to {FINAL_JSON}")
    json.dump(merged_list, open(FINAL_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    write_screenplay(merged_list)

    extract_clips(merged_list)

    print("\n==========================================")
    print("      STAGE-2 COMPLETE ✔")
    print("==========================================")


if __name__ == "__main__":
    main()