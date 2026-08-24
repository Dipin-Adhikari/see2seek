"""
Multi-angle goal augmentation with inline CLIP embedding (parallelized).

Renders goal objects from multiple heading angles using a pool of parallel
AI2-THOR controllers, then CLIP-encodes all frames in one batch on GPU.

Architecture (optimized for i9-14900K + RTX 4090):
    - N worker processes, each running an independent AI2-THOR/Unity instance
    - Workers render frames and return (episode_json, frame_bytes) tuples
    - Main process: CLIP-encodes all frames on GPU, writes embeddings.pt + episodes

CHANGES vs original:
    - Progress is now reported per-episode (not per-file), via periodic
      "progress" messages on the queue, so you can see real throughput.
    - Candidate search is capped (MAX_CANDIDATES_TRIED) instead of trying
      every reachable position in range, which was the main source of
      "spinning for a long time with nothing to show for it".

Output:
    - Augmented episode JSON files in output_dir/episodes/
    - embeddings.pt in output_dir/ (same key format the training env expects)
    - Optionally: goal images in output_dir/images/ (--save_images)

Usage:
    python -m see2seek.utils.augment_goal_angles \
        --input_dir /path/to/imagenav_dataset/train \
        --output_dir /path/to/augmented_train \
        --split train \
        --num_angles 4 \
        --num_workers 8

    # Dry run:
    python -m see2seek.utils.augment_goal_angles --input_dir ... --output_dir ... --dry_run
"""

import argparse
import gzip
import io
import json
import math
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

NUM_ANGLES = 4
HORIZONS = [15, 30]
EMBED_BATCH_SIZE = 128

# How many of the closest reachable-position candidates to try per angle
# before giving up on that angle. Trying *all* candidates (could be 50-200+)
# is the main reason episodes were taking forever with no visible progress.
MAX_CANDIDATES_TRIED = 8

# Push a "progress" message to the main process every this-many episodes,
# so the main process can update a live per-episode progress bar instead
# of waiting for an entire file to finish.
PROGRESS_EVERY = 5


# ===========================================================================
# Worker process
# ===========================================================================

def _worker_init(worker_id: int, num_angles: int):
    """Initialize a worker with its own AI2-THOR controller."""
    global _ctrl, _num_angles, _worker_id
    _worker_id = worker_id
    _num_angles = num_angles

    from ai2thor.controller import Controller
    _ctrl = Controller(
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


def _worker_process_file(args_tuple, queue=None, worker_id=None):
    """
    Process a single scene file in this worker's AI2-THOR instance.

    Emits periodic ("progress", worker_id, fname, {...}) messages on `queue`
    (if provided) every PROGRESS_EVERY episodes, plus one final flush at the
    end of the file so the counts add up exactly.

    Returns:
        List of (aug_episode_dict, frame_png_bytes, filename) tuples
    """
    input_path, image_id_start = args_tuple
    global _ctrl, _num_angles

    fname = os.path.basename(input_path)

    if input_path.endswith(".gz"):
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            episodes = json.load(f)
    else:
        with open(input_path, "r") as f:
            episodes = json.load(f)

    if isinstance(episodes, dict):
        episodes = episodes.get("episodes", [])

    total_eps_in_file = len(episodes)
    angle_offsets = [i * (360.0 / _num_angles) for i in range(_num_angles)]
    results = []
    image_id_counter = image_id_start
    current_scene = None

    episodes_since_checkpoint = 0
    augmented_since_checkpoint = 0

    def _flush_progress(ep_idx):
        if queue is None:
            return
        queue.put((
            "progress", worker_id, fname,
            {
                "episodes_delta": episodes_since_checkpoint,
                "augmented_delta": augmented_since_checkpoint,
                "episode_idx": ep_idx + 1,
                "total_episodes": total_eps_in_file,
            },
        ))

    for ep_idx, episode in enumerate(episodes):
        scene_name = episode.get("scene")
        object_type = episode.get("object_type")

        episodes_since_checkpoint += 1

        if not episode.get("shortest_path"):
            if episodes_since_checkpoint >= PROGRESS_EVERY:
                _flush_progress(ep_idx)
                episodes_since_checkpoint = 0
                augmented_since_checkpoint = 0
            continue

        # Reset scene if needed
        if scene_name != current_scene:
            try:
                _ctrl.reset(scene=scene_name)
                current_scene = scene_name
            except Exception:
                try:
                    _ctrl.stop()
                except Exception:
                    pass
                from ai2thor.controller import Controller
                _ctrl = Controller(
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

        event = _ctrl.step(action="Pass")
        path_end = episode["shortest_path"][-1]

        # Find target object
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
            if episodes_since_checkpoint >= PROGRESS_EVERY:
                _flush_progress(ep_idx)
                episodes_since_checkpoint = 0
                augmented_since_checkpoint = 0
            continue

        obj_pos = target_obj["position"]
        target_id = target_obj["objectId"]

        # Get reachable positions
        event = _ctrl.step(action="GetReachablePositions")
        if not event.metadata["lastActionSuccess"]:
            if episodes_since_checkpoint >= PROGRESS_EVERY:
                _flush_progress(ep_idx)
                episodes_since_checkpoint = 0
                augmented_since_checkpoint = 0
            continue

        reachable = event.metadata["actionReturn"]
        candidates = []
        for pos in reachable:
            dist = math.sqrt(
                (pos["x"] - obj_pos["x"]) ** 2
                + (pos["z"] - obj_pos["z"]) ** 2
            )
            if 0.5 <= dist <= 2.5:
                candidates.append((pos, dist))

        candidates.sort(key=lambda p: p[1])
        # Cap how many candidates we're willing to try. The closest ones
        # almost always work; trying all of them (could be 50-200+) is what
        # made episodes take forever with nothing to show for it.
        candidates = candidates[:MAX_CANDIDATES_TRIED]

        if not candidates:
            if episodes_since_checkpoint >= PROGRESS_EVERY:
                _flush_progress(ep_idx)
                episodes_since_checkpoint = 0
                augmented_since_checkpoint = 0
            continue

        # Render from each angle
        for angle_idx, offset in enumerate(angle_offsets):
            found = False
            for pos, _ in candidates:
                if found:
                    break
                dx = obj_pos["x"] - pos["x"]
                dz = obj_pos["z"] - pos["z"]
                base_yaw = math.degrees(math.atan2(dx, dz))
                yaw = (base_yaw + offset) % 360

                for horizon in HORIZONS:
                    event = _ctrl.step(
                        action="TeleportFull",
                        x=pos["x"], y=pos["y"], z=pos["z"],
                        rotation=dict(x=0, y=yaw, z=0),
                        horizon=horizon,
                    )
                    if not event.metadata.get("lastActionSuccess", False):
                        continue

                    for obj in event.metadata["objects"]:
                        if obj["objectId"] == target_id and obj["visible"]:
                            # Encode frame as PNG bytes (compact for IPC)
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
                                "rotation": yaw, "horizon": horizon,
                            }
                            aug_episode["angle_offset"] = offset

                            results.append((aug_episode, frame_bytes, filename))
                            image_id_counter += 1
                            augmented_since_checkpoint += 1
                            found = True
                            break
                    if found:
                        break

        if episodes_since_checkpoint >= PROGRESS_EVERY:
            _flush_progress(ep_idx)
            episodes_since_checkpoint = 0
            augmented_since_checkpoint = 0

    # Final flush for any leftover episodes since the last checkpoint
    if episodes_since_checkpoint > 0:
        _flush_progress(total_eps_in_file - 1)

    return results


def _cleanup_worker(_):
    """Shut down this worker's controller."""
    global _ctrl
    try:
        _ctrl.stop()
    except Exception:
        pass


def _worker_main(worker_id, assigned_files, num_angles, queue):
    """Worker process entry point: init controller, process files, send results via queue."""
    from ai2thor.controller import Controller

    global _ctrl, _num_angles, _worker_id
    _worker_id = worker_id
    _num_angles = num_angles

    _ctrl = Controller(
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

    for file_args in assigned_files:
        try:
            results = _worker_process_file(file_args, queue=queue, worker_id=worker_id)
            fname = os.path.basename(file_args[0])
            queue.put(("results", worker_id, fname, results))
        except Exception as e:
            fname = os.path.basename(file_args[0])
            queue.put(("error", worker_id, fname, str(e)))

    try:
        _ctrl.stop()
    except Exception:
        pass
    queue.put(("done", worker_id, None, None))


# ===========================================================================
# Main process
# ===========================================================================

def _count_total_episodes(input_episodes_dir, episode_files):
    """Pre-scan all files to get a total episode count for the progress bar."""
    total = 0
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
        total += len(eps)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Multi-angle goal augmentation (parallel AI2-THOR + CLIP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Path to imagenav_dataset/<split> (contains episodes/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (creates episodes/ and embeddings.pt)")
    parser.add_argument("--split", default="train",
                        help="Split name for embedding keys (default: train)")
    parser.add_argument("--num_angles", type=int, default=NUM_ANGLES)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel AI2-THOR instances (default: 8)")
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

    print(f"Input: {input_episodes_dir} ({len(episode_files)} files)")
    print(f"Output: {args.output_dir}")
    print(f"Angles: {args.num_angles} | Workers: {args.num_workers}")
    print(f"Hardware: using {args.num_workers} parallel AI2-THOR controllers")

    if args.dry_run:
        total_eps = _count_total_episodes(input_episodes_dir, episode_files)
        print(f"\n[DRY RUN]")
        print(f"Total source episodes: {total_eps}")
        print(f"Max augmented (x{args.num_angles}): {total_eps * args.num_angles}")
        print(f"Estimated time with {args.num_workers} workers: "
              f"~{total_eps * args.num_angles * 0.3 / args.num_workers / 60:.0f} minutes")
        return

    # Assign image_id ranges to each file so IDs don't collide across workers
    max_per_file = 2000 * args.num_angles  # generous upper bound
    work_items = []
    for i, fname in enumerate(episode_files):
        path = os.path.join(input_episodes_dir, fname)
        id_start = i * max_per_file
        work_items.append((path, id_start))

    # Pre-scan for total episode count so the progress bar has a real total
    print("\nCounting source episodes for progress bar...")
    total_source_episodes = _count_total_episodes(input_episodes_dir, episode_files)
    print(f"Total source episodes: {total_source_episodes}")

    # Launch workers
    print(f"\nLaunching {args.num_workers} AI2-THOR workers...")
    ctx = mp.get_context("spawn")

    # Distribute work across workers using Process + Queue
    result_queue = ctx.Queue()
    file_chunks = [[] for _ in range(args.num_workers)]
    for i, item in enumerate(work_items):
        file_chunks[i % args.num_workers].append(item)

    # Start workers
    processes = []
    for wid in range(args.num_workers):
        if not file_chunks[wid]:
            continue
        p = ctx.Process(
            target=_worker_main,
            args=(wid, file_chunks[wid], args.num_angles, result_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print(f"Started {len(processes)} workers, processing {len(work_items)} scene files...")

    # Collect results from workers
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CLIP encoder will run on: {device}")
    from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
    clip_encoder = CLIPGoalEncoder(device=device)

    # Load existing embeddings if resuming
    embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
    if os.path.exists(embeddings_path):
        embedding_dict = torch.load(embeddings_path, map_location="cpu")
        print(f"Resuming: {len(embedding_dict)} existing embeddings")
    else:
        embedding_dict = {}

    # Collect all results, grouped by source file
    workers_done = 0
    total_augmented = 0
    all_pending_images: List[Image.Image] = []
    all_pending_keys: List[str] = []
    file_episodes: Dict[str, list] = {}

    # Primary progress bar: per-episode, updates live as workers process
    ep_pbar = tqdm(total=total_source_episodes, desc="Episodes processed", unit="ep")
    files_done = 0
    file_pbar = tqdm(total=len(work_items), desc="Files completed", unit="file")

    while workers_done < len(processes):
        msg_type, wid, fname, data = result_queue.get()

        if msg_type == "done":
            workers_done += 1
            continue

        elif msg_type == "error":
            tqdm.write(f"  Worker {wid} error on {fname}: {data}")
            files_done += 1
            file_pbar.update(1)
            continue

        elif msg_type == "progress":
            ep_pbar.update(data["episodes_delta"])
            ep_pbar.set_postfix({
                "augmented": total_augmented + data["augmented_delta"],
                "worker": wid,
                "file": fname[:20],
            })
            continue

        # msg_type == "results"
        results = data
        files_done += 1
        file_pbar.update(1)
        total_augmented += len(results)

        # Collect episodes for this file
        if fname not in file_episodes:
            file_episodes[fname] = []

        for aug_episode, frame_bytes, filename in results:
            file_episodes[fname].append(aug_episode)

            key = f"{args.split}/images/{filename}"
            if key not in embedding_dict:
                pil_img = Image.open(io.BytesIO(frame_bytes))
                all_pending_images.append(pil_img)
                all_pending_keys.append(key)

                if args.save_images:
                    save_path = os.path.join(args.output_dir, "images", filename)
                    pil_img.save(save_path)

        # Batch-encode when we have enough
        if len(all_pending_images) >= EMBED_BATCH_SIZE:
            with torch.no_grad():
                embs = clip_encoder.encode_image(all_pending_images[:EMBED_BATCH_SIZE]).cpu()
            for k, e in zip(all_pending_keys[:EMBED_BATCH_SIZE], embs):
                embedding_dict[k] = e
            all_pending_images = all_pending_images[EMBED_BATCH_SIZE:]
            all_pending_keys = all_pending_keys[EMBED_BATCH_SIZE:]

    ep_pbar.close()
    file_pbar.close()

    # Flush remaining images
    print(f"\nEncoding remaining {len(all_pending_images)} images...")
    for i in tqdm(range(0, len(all_pending_images), EMBED_BATCH_SIZE), desc="CLIP encoding"):
        batch_imgs = all_pending_images[i:i + EMBED_BATCH_SIZE]
        batch_keys = all_pending_keys[i:i + EMBED_BATCH_SIZE]
        with torch.no_grad():
            embs = clip_encoder.encode_image(batch_imgs).cpu()
        for k, e in zip(batch_keys, embs):
            embedding_dict[k] = e

    # Wait for all worker processes
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Save episode files
    print(f"\nSaving {len(file_episodes)} episode files...")
    for fname, episodes in file_episodes.items():
        if not episodes:
            continue
        out_path = os.path.join(output_episodes_dir, fname)
        if not out_path.endswith(".gz"):
            out_path += ".gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(episodes, f)

    # Save embeddings
    torch.save(embedding_dict, embeddings_path)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Augmented episodes: {total_augmented}")
    print(f"  Embeddings: {len(embedding_dict)} -> {embeddings_path}")
    print(f"  Episode files: {output_episodes_dir}")
    if args.save_images:
        print(f"  Images: {args.output_dir}/images/")


if __name__ == "__main__":
    main()