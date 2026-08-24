"""
Multi-angle goal augmentation with inline CLIP embedding (parallelized).

Renders goal objects from multiple VIEWING ANGLES (camera positions around
the object, always facing the object) using parallel AI2-THOR controllers,
then CLIP-encodes all frames in one batch on GPU.

Architecture:
    - N worker processes, each with its own AI2-THOR controller
    - Shared work queue: workers pull files one at a time (files complete ASAP)
    - Main process: collects frames, CLIP-encodes on GPU, writes outputs

Key fix vs prior version:
    The offset was being added to the camera HEADING (looking away from the
    object at 90/180/270 offsets -> visibility always fails -> ~1:1 ratio).
    Now the offset selects candidate POSITIONS at different angles around
    the object, and the camera always faces the object -> 4:1 ratio.

Usage:
    python -m see2seek.utils.augment_goal_angles \
        --input_dir /path/to/imagenav_dataset/train \
        --output_dir /path/to/augmented_train \
        --split train \
        --num_angles 4 \
        --num_workers 24
"""

import argparse
import gzip
import io
import json
import math
import multiprocessing as mp
import os
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

NUM_ANGLES = 4
HORIZONS = [15, 30]
EMBED_BATCH_SIZE = 128
ANGLE_TOLERANCE_DEG = 60.0


# ===========================================================================
# Worker process
# ===========================================================================

def _worker_main(worker_id, work_queue, result_queue, num_angles):
    """
    Worker entry point. Pulls files from shared work_queue until it's empty.
    Sends results and progress to result_queue.
    """
    from ai2thor.controller import Controller

    ctrl = Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        scene="FloorPlan_Train1_1",
        gridSize=0.25,
        rotateStepDegrees=30,
        snapToGrid=False,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=224,
        height=224,
        fieldOfView=79,
    )

    angle_offsets = [i * (360.0 / num_angles) for i in range(num_angles)]
    current_scene = None

    while True:
        try:
            file_path, id_start = work_queue.get_nowait()
        except Exception:
            break

        fname = os.path.basename(file_path)

        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    episodes = json.load(f)
            else:
                with open(file_path, "r") as f:
                    episodes = json.load(f)

            if isinstance(episodes, dict):
                episodes = episodes.get("episodes", [])

            results = []
            image_id_counter = id_start
            episodes_processed = 0

            for episode in episodes:
                scene_name = episode.get("scene")
                object_type = episode.get("object_type")
                episodes_processed += 1

                if not episode.get("shortest_path"):
                    continue

                # Reset scene if needed
                if scene_name != current_scene:
                    try:
                        ctrl.reset(scene=scene_name)
                        current_scene = scene_name
                    except Exception:
                        try:
                            ctrl.stop()
                        except Exception:
                            pass
                        ctrl = Controller(
                            agentMode="locobot",
                            visibilityDistance=1.5,
                            scene=scene_name,
                            gridSize=0.25,
                            rotateStepDegrees=30,
                            snapToGrid=False,
                            renderDepthImage=False,
                            renderInstanceSegmentation=False,
                            width=224,
                            height=224,
                            fieldOfView=79,
                        )
                        current_scene = scene_name

                event = ctrl.step(action="Pass")
                path_end = episode["shortest_path"][-1]

                # Find target object closest to goal position
                target_obj = None
                min_dist = float("inf")
                for obj in event.metadata["objects"]:
                    if obj["objectType"] == object_type:
                        pos = obj["position"]
                        dist = math.sqrt(
                            (pos["x"] - path_end["x"]) ** 2
                            + (pos["z"] - path_end["z"]) ** 2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            target_obj = obj

                if target_obj is None:
                    continue

                obj_pos = target_obj["position"]
                target_id = target_obj["objectId"]

                # Get reachable positions
                event = ctrl.step(action="GetReachablePositions")
                if not event.metadata["lastActionSuccess"]:
                    continue

                reachable = event.metadata["actionReturn"]

                # For each candidate, compute its bearing FROM the object
                # angle = atan2(cand_x - obj_x, cand_z - obj_z)
                candidates_with_angle = []
                for pos in reachable:
                    dx = pos["x"] - obj_pos["x"]
                    dz = pos["z"] - obj_pos["z"]
                    dist = math.sqrt(dx * dx + dz * dz)
                    if 0.5 <= dist <= 2.5:
                        angle_deg = math.degrees(math.atan2(dx, dz)) % 360
                        candidates_with_angle.append((pos, dist, angle_deg))

                if not candidates_with_angle:
                    continue

                # For each desired viewing angle, find the best candidate position
                for angle_idx, offset in enumerate(angle_offsets):
                    target_angle = offset

                    # Sort candidates by angular proximity to target_angle
                    def angle_dist(c):
                        diff = abs(c[2] - target_angle)
                        if diff > 180:
                            diff = 360 - diff
                        return diff

                    nearby = [c for c in candidates_with_angle if angle_dist(c) < ANGLE_TOLERANCE_DEG]
                    if not nearby:
                        nearby = sorted(candidates_with_angle, key=angle_dist)[:5]

                    # Prefer closer positions for cleaner views
                    nearby.sort(key=lambda c: c[1])

                    found = False
                    for pos, _, _ in nearby[:8]:
                        if found:
                            break
                        # Always FACE the object from this position
                        dx = obj_pos["x"] - pos["x"]
                        dz = obj_pos["z"] - pos["z"]
                        face_yaw = math.degrees(math.atan2(dx, dz))

                        for horizon in HORIZONS:
                            event = ctrl.step(
                                action="TeleportFull",
                                x=pos["x"], y=pos["y"], z=pos["z"],
                                rotation=dict(x=0, y=face_yaw, z=0),
                                horizon=horizon,
                            )
                            if not event.metadata.get("lastActionSuccess", False):
                                continue

                            for obj in event.metadata["objects"]:
                                if obj["objectId"] == target_id and obj["visible"]:
                                    pil_img = Image.fromarray(event.frame)
                                    buf = io.BytesIO()
                                    pil_img.save(buf, format="PNG")
                                    frame_bytes = buf.getvalue()

                                    aug_id = f"{episode['id']}_angle{angle_idx}"
                                    filename = f"id_{image_id_counter:06d}_{scene_name}_{aug_id}_goal.png"

                                    aug_episode = episode.copy()
                                    aug_episode["id"] = aug_id
                                    aug_episode["goal_image_path"] = f"../images/{filename}"
                                    aug_episode["optimal_goal_pose"] = {
                                        "x": pos["x"], "y": pos["y"], "z": pos["z"],
                                        "rotation": face_yaw, "horizon": horizon,
                                    }
                                    aug_episode["angle_offset"] = offset

                                    results.append((aug_episode, frame_bytes, filename))
                                    image_id_counter += 1
                                    found = True
                                    break
                            if found:
                                break

                # Send progress every 10 episodes
                if episodes_processed % 10 == 0:
                    result_queue.put(("progress", worker_id, fname, episodes_processed))

            # File complete
            result_queue.put(("file_done", worker_id, fname, results))

        except Exception as e:
            result_queue.put(("error", worker_id, fname, str(e)))

    try:
        ctrl.stop()
    except Exception:
        pass
    result_queue.put(("worker_done", worker_id, None, None))


# ===========================================================================
# Main process
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-angle goal augmentation (parallel AI2-THOR + CLIP)",
    )
    parser.add_argument("--input_dir", required=True,
                        help="Path to imagenav_dataset/<split> (contains episodes/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (creates episodes/ and embeddings.pt)")
    parser.add_argument("--split", default="train",
                        help="Split name for embedding keys (default: train)")
    parser.add_argument("--num_angles", type=int, default=NUM_ANGLES)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel AI2-THOR instances")
    parser.add_argument("--save_images", action="store_true",
                        help="Also save rendered goal images to output_dir/images/")
    parser.add_argument("--device", default=None,
                        help="Device for CLIP encoder (default: cuda if available)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_episodes_dir = os.path.join(args.input_dir, "episodes")
    output_episodes_dir = os.path.join(args.output_dir, "episodes")

    if not os.path.isdir(input_episodes_dir):
        raise FileNotFoundError(f"Episodes dir not found: {input_episodes_dir}")

    os.makedirs(output_episodes_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    episode_files = sorted([
        f for f in os.listdir(input_episodes_dir)
        if (f.endswith(".json.gz") or f.endswith(".json")) and not f.endswith(".bak")
    ])

    if not episode_files:
        raise FileNotFoundError(f"No episode files in {input_episodes_dir}")

    # Pre-count episodes for progress tracking
    print(f"Input: {input_episodes_dir} ({len(episode_files)} files)")
    print("Counting episodes...")
    total_source_episodes = 0
    for fname in episode_files:
        path = os.path.join(input_episodes_dir, fname)
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                eps = json.load(f)
        else:
            with open(path) as f:
                eps = json.load(f)
        if isinstance(eps, dict):
            eps = eps.get("episodes", [])
        total_source_episodes += len(eps)

    print(f"Total source episodes: {total_source_episodes}")
    print(f"Output: {args.output_dir}")
    print(f"Angles: {args.num_angles} | Workers: {args.num_workers}")

    if args.dry_run:
        print(f"\n[DRY RUN]")
        print(f"Max augmented (x{args.num_angles}): {total_source_episodes * args.num_angles}")
        print(f"Estimated time with {args.num_workers} workers: "
              f"~{total_source_episodes * 0.5 / args.num_workers / 60:.0f} minutes")
        return

    # Build shared work queue -- files are pulled in order, completing sequentially
    ctx = mp.get_context("spawn")
    work_queue = ctx.Queue()
    max_per_file = 2000 * args.num_angles
    for i, fname in enumerate(episode_files):
        path = os.path.join(input_episodes_dir, fname)
        id_start = i * max_per_file
        work_queue.put((path, id_start))

    # Launch workers
    print(f"\nLaunching {args.num_workers} AI2-THOR workers...")
    result_queue = ctx.Queue()
    processes = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=_worker_main,
            args=(wid, work_queue, result_queue, args.num_angles),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print(f"Started {len(processes)} workers")

    # CLIP encoder on GPU in main process
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CLIP encoder on: {device}")
    from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
    clip_encoder = CLIPGoalEncoder(device=device)

    # Load existing embeddings if resuming
    embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
    if os.path.exists(embeddings_path):
        embedding_dict = torch.load(embeddings_path, map_location="cpu")
        print(f"Resuming: {len(embedding_dict)} existing embeddings")
    else:
        embedding_dict = {}

    # Collect results
    workers_done = 0
    total_augmented = 0
    files_completed = 0
    all_pending_images: List[Image.Image] = []
    all_pending_keys: List[str] = []

    ep_pbar = tqdm(total=total_source_episodes, desc="Episodes", unit="ep")
    file_pbar = tqdm(total=len(episode_files), desc="Files done", unit="file", position=1)

    while workers_done < len(processes):
        msg_type, wid, fname, data = result_queue.get()

        if msg_type == "worker_done":
            workers_done += 1
            continue

        elif msg_type == "progress":
            ep_pbar.n = min(ep_pbar.n + 10, ep_pbar.total)
            ep_pbar.set_postfix(aug=total_augmented, w=wid)
            ep_pbar.refresh()
            continue

        elif msg_type == "error":
            tqdm.write(f"  [ERROR] Worker {wid} on {fname}: {data}")
            files_completed += 1
            file_pbar.update(1)
            continue

        elif msg_type == "file_done":
            results = data
            files_completed += 1
            file_pbar.update(1)
            total_augmented += len(results)
            ep_pbar.set_postfix(aug=total_augmented, ratio=f"{total_augmented/max(ep_pbar.n,1):.1f}x")

            # Save episode file immediately
            if results:
                aug_episodes = []
                for aug_episode, frame_bytes, filename in results:
                    aug_episodes.append(aug_episode)

                    key = f"{args.split}/images/{filename}"
                    if key not in embedding_dict:
                        pil_img = Image.open(io.BytesIO(frame_bytes))
                        all_pending_images.append(pil_img)
                        all_pending_keys.append(key)

                        if args.save_images:
                            save_path = os.path.join(args.output_dir, "images", filename)
                            pil_img.save(save_path)

                # Write episode file
                out_path = os.path.join(output_episodes_dir, fname)
                if not out_path.endswith(".gz"):
                    out_path += ".gz"
                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    json.dump(aug_episodes, f)

            # Batch-encode accumulated images
            while len(all_pending_images) >= EMBED_BATCH_SIZE:
                with torch.no_grad():
                    embs = clip_encoder.encode_image(all_pending_images[:EMBED_BATCH_SIZE]).cpu()
                for k, e in zip(all_pending_keys[:EMBED_BATCH_SIZE], embs):
                    embedding_dict[k] = e
                all_pending_images = all_pending_images[EMBED_BATCH_SIZE:]
                all_pending_keys = all_pending_keys[EMBED_BATCH_SIZE:]

            # Periodic checkpoint save (every 50 files)
            if files_completed % 50 == 0:
                torch.save(embedding_dict, embeddings_path)
                tqdm.write(f"  Checkpoint: {len(embedding_dict)} embeddings saved")

    ep_pbar.close()
    file_pbar.close()

    # Flush remaining images
    if all_pending_images:
        print(f"\nEncoding remaining {len(all_pending_images)} images...")
        for i in tqdm(range(0, len(all_pending_images), EMBED_BATCH_SIZE), desc="CLIP"):
            batch_imgs = all_pending_images[i:i + EMBED_BATCH_SIZE]
            batch_keys = all_pending_keys[i:i + EMBED_BATCH_SIZE]
            with torch.no_grad():
                embs = clip_encoder.encode_image(batch_imgs).cpu()
            for k, e in zip(batch_keys, embs):
                embedding_dict[k] = e

    # Wait for workers
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Final save
    torch.save(embedding_dict, embeddings_path)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Source episodes:    {total_source_episodes}")
    print(f"  Augmented episodes: {total_augmented}")
    print(f"  Ratio:              {total_augmented/max(total_source_episodes,1):.2f}x")
    print(f"  Embeddings:         {len(embedding_dict)} -> {embeddings_path}")
    print(f"  Episode files:      {output_episodes_dir}")


if __name__ == "__main__":
    main()
