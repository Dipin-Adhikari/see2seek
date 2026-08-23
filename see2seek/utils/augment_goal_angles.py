"""
Multi-angle goal augmentation with inline CLIP embedding.

For each episode, renders the goal object from multiple heading angles
(default 4 evenly-spaced). Each angle that produces a visible goal image
is CLIP-encoded on the fly — no intermediate image files are saved to disk.

Output:
    - Augmented episode JSON files (one per input scene file) in output_dir/episodes/
    - A single embeddings.pt in output_dir/ containing the CLIP embeddings
      keyed as "<split>/images/<filename>" (same format the training env expects)
    - Optionally saves goal images to output_dir/images/ (--save_images flag)

This replaces the separate augment + image2vec pipeline with a single pass.

Usage:
    python -m see2seek.utils.augment_goal_angles \
        --input_dir /path/to/imagenav_dataset/train \
        --output_dir /path/to/output_train \
        --split train \
        --num_angles 4

    # Also save the rendered images (for debugging / visualization):
    python -m see2seek.utils.augment_goal_angles \
        --input_dir ... --output_dir ... --save_images

    # Dry run:
    python -m see2seek.utils.augment_goal_angles \
        --input_dir ... --output_dir ... --dry_run
"""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

NUM_ANGLES = 4
HORIZONS = [15, 30]
CONTROLLER_RESTART_INTERVAL = 200
EMBED_BATCH_SIZE = 64
CHECKPOINT_EVERY = 500


def launch_controller(scene="FloorPlan_Train1_1"):
    from ai2thor.controller import Controller
    return Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        scene=scene,
        gridSize=0.25,
        rotateStepDegrees=30,
        snapToGrid=False,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=224,
        height=224,
        fieldOfView=79,
    )


def find_target_object(event, object_type, path_end):
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
    return target_obj


def try_render_angles(controller, target_id, obj_pos, candidates, num_angles, horizons):
    """
    Try to render the goal object from multiple heading angles.
    Returns list of (PIL_image, pos, yaw, horizon, angle_idx) for successful renders.
    """
    angle_offsets = [i * (360.0 / num_angles) for i in range(num_angles)]
    results = []

    for angle_idx, offset in enumerate(angle_offsets):
        found = False
        for pos, _ in candidates:
            if found:
                break
            dx = obj_pos["x"] - pos["x"]
            dz = obj_pos["z"] - pos["z"]
            base_yaw = math.degrees(math.atan2(dx, dz))
            yaw = (base_yaw + offset) % 360

            for horizon in horizons:
                event = controller.step(
                    action="TeleportFull",
                    x=pos["x"], y=pos["y"], z=pos["z"],
                    rotation=dict(x=0, y=yaw, z=0),
                    horizon=horizon,
                )
                if not event.metadata.get("lastActionSuccess", False):
                    continue

                for obj in event.metadata["objects"]:
                    if obj["objectId"] == target_id and obj["visible"]:
                        pil_img = Image.fromarray(event.frame)
                        results.append((pil_img, pos, yaw, horizon, angle_idx))
                        found = True
                        break
                if found:
                    break

    return results


def process_scene_file(input_path, controller, num_angles, image_id_counter, dry_run):
    """Process one scene file. Returns (augmented_episodes, rendered_images, updated_counter)."""
    if input_path.endswith(".gz"):
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            episodes = json.load(f)
    else:
        with open(input_path, "r") as f:
            episodes = json.load(f)

    if isinstance(episodes, dict):
        episodes = episodes.get("episodes", [])

    augmented_episodes = []
    rendered_images = []  # list of (key, PIL_image)

    if dry_run:
        return augmented_episodes, rendered_images, image_id_counter, len(episodes)

    current_scene = None

    for ep_idx, episode in enumerate(episodes):
        scene_name = episode.get("scene")
        object_type = episode.get("object_type")

        if not episode.get("shortest_path"):
            continue

        if scene_name != current_scene:
            try:
                controller.reset(scene=scene_name)
                current_scene = scene_name
            except Exception:
                controller.stop()
                controller = launch_controller(scene_name)
                current_scene = scene_name

        event = controller.step(action="Pass")
        path_end = episode["shortest_path"][-1]
        target_obj = find_target_object(event, object_type, path_end)

        if target_obj is None:
            continue

        obj_pos = target_obj["position"]
        target_id = target_obj["objectId"]

        event = controller.step(action="GetReachablePositions")
        if not event.metadata["lastActionSuccess"]:
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
        if not candidates:
            continue

        results = try_render_angles(controller, target_id, obj_pos, candidates, num_angles, HORIZONS)

        for pil_img, pos, yaw, horizon, angle_idx in results:
            aug_id = f"{episode['id']}_angle{angle_idx}"
            filename = f"id_{image_id_counter:06d}_{scene_name}_{aug_id}_goal.png"

            aug_episode = episode.copy()
            aug_episode["id"] = aug_id
            aug_episode["goal_image_path"] = f"../images/{filename}"
            aug_episode["optimal_goal_pose"] = {
                "x": pos["x"], "y": pos["y"], "z": pos["z"],
                "rotation": yaw, "horizon": horizon,
            }
            aug_episode["angle_offset"] = angle_idx * (360.0 / num_angles)
            augmented_episodes.append(aug_episode)
            rendered_images.append((filename, pil_img))
            image_id_counter += 1

    return augmented_episodes, rendered_images, image_id_counter, len(episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-angle goal augmentation with inline CLIP embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Path to imagenav_dataset/<split> (contains episodes/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (creates episodes/ and embeddings.pt)")
    parser.add_argument("--split", default="train",
                        help="Split name for embedding keys (default: train)")
    parser.add_argument("--num_angles", type=int, default=NUM_ANGLES)
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
    print(f"Angles per episode: {args.num_angles}")
    print(f"Split key: {args.split}")

    if args.dry_run:
        print("[DRY RUN] Estimating counts...")
        total_eps = 0
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
            total_eps += len(eps)
        print(f"Total episodes: {total_eps}")
        print(f"Max augmented (x{args.num_angles}): {total_eps * args.num_angles}")
        return

    # Initialize CLIP encoder
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing CLIP encoder on {device}...")
    from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
    clip_encoder = CLIPGoalEncoder(device=device)

    # Load existing embeddings if resuming
    embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
    if os.path.exists(embeddings_path):
        embedding_dict = torch.load(embeddings_path, map_location="cpu")
        print(f"Resuming: {len(embedding_dict)} existing embeddings loaded")
    else:
        embedding_dict = {}

    controller = launch_controller()
    image_id_counter = 0
    total_original = 0
    total_augmented = 0
    global_ep_count = 0

    # Batch buffer for CLIP encoding
    pending_images: List[Image.Image] = []
    pending_keys: List[str] = []

    def flush_embeddings():
        """Encode pending images in batch and add to embedding_dict."""
        nonlocal pending_images, pending_keys
        if not pending_images:
            return
        with torch.no_grad():
            embs = clip_encoder.encode_image(pending_images).cpu()
        for key, emb in zip(pending_keys, embs):
            embedding_dict[key] = emb
        pending_images = []
        pending_keys = []

    for file_idx, fname in enumerate(tqdm(episode_files, desc="Scene files")):
        input_path = os.path.join(input_episodes_dir, fname)

        augmented, rendered, image_id_counter, n_original = process_scene_file(
            input_path, controller, args.num_angles, image_id_counter, dry_run=False,
        )

        total_original += n_original
        total_augmented += len(augmented)

        # Queue rendered images for CLIP encoding
        for filename, pil_img in rendered:
            key = f"{args.split}/images/{filename}"
            if key not in embedding_dict:
                pending_images.append(pil_img)
                pending_keys.append(key)

            if args.save_images:
                pil_img.save(os.path.join(args.output_dir, "images", filename))

            # Flush batch when full
            if len(pending_images) >= EMBED_BATCH_SIZE:
                flush_embeddings()

        # Save augmented episodes
        if augmented:
            out_path = os.path.join(output_episodes_dir, fname)
            if not out_path.endswith(".gz"):
                out_path += ".gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(augmented, f)

        # Periodic checkpoint
        global_ep_count += n_original
        if global_ep_count >= CHECKPOINT_EVERY:
            flush_embeddings()
            torch.save(embedding_dict, embeddings_path)
            tqdm.write(f"  Checkpoint: {len(embedding_dict)} embeddings saved")
            global_ep_count = 0

        # Restart controller periodically
        if (file_idx + 1) % 10 == 0:
            controller.stop()
            controller = launch_controller()

    # Final flush
    flush_embeddings()
    controller.stop()

    # Save final embeddings
    torch.save(embedding_dict, embeddings_path)

    print(f"\n{'='*60}")
    print(f"DONE: {total_original} original -> {total_augmented} augmented episodes")
    print(f"Multiplier: {total_augmented / max(total_original, 1):.1f}x")
    print(f"Embeddings: {len(embedding_dict)} saved to {embeddings_path}")
    print(f"Episodes: {output_episodes_dir}")


if __name__ == "__main__":
    main()
