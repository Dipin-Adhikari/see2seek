import os
import json
import math
import gzip
from ai2thor.controller import Controller
from PIL import Image

# ==========================================
# Configuration Paths
# ==========================================
INPUT_DATASET_DIR = "/home/dipin/imagenav/dataset"     # Your dataset folder
OUTPUT_DATASET_DIR = "/home/dipin/imagenav/imagenav_dataset"   # Converted structure output

# Memory Management
CHECKPOINT_INTERVAL = 50
CONTROLLER_RESTART_INTERVAL = 250

def launch_controller(initial_scene="FloorPlan_Train1_1"):
    """Helper to spin up a clean AI2-THOR instance."""
    print("🤖 Initializing/Restarting AI2-THOR Controller Instance...")
    return Controller(
        agentMode="bot",
        scene=initial_scene,
        gridSize=0.25,
        width=300,
        height=300
    )

def main():
    # Only tracking your requested splits (test removed)
    splits = ["debug", "train", "val"]
    all_files = []
    
    print(f"Scanning {INPUT_DATASET_DIR} for compressed (.json.gz) episode files...")
    
    # 1. Look explicitly for .json.gz files and bind them to their split context
    for root, dirs, files in os.walk(INPUT_DATASET_DIR):
        for file in files:
            if file.endswith(".json.gz"):
                full_input_path = os.path.join(root, file)
                
                # Get path relative to the input root (e.g., 'train/episode/ep1.json.gz')
                rel_path = os.path.relpath(full_input_path, INPUT_DATASET_DIR)
                path_parts = rel_path.split(os.sep)
                
                # Ensure the file belongs to one of our target splits
                if path_parts[0] in splits:
                    full_output_path = os.path.join(OUTPUT_DATASET_DIR, rel_path)
                    # Store input path, output path, and the specific split name
                    all_files.append((full_input_path, full_output_path, path_parts[0]))
                
    if not all_files:
        print(f"❌ No .json.gz files found. Verify the file path name structures.")
        return

    print(f"✔ Success! Found {len(all_files)} compressed JSON files across splits: {splits}")
    
    # Initialize the controller and global auto-incrementing ID tracker
    controller = launch_controller()
    global_episodes_processed = 0
    image_id_counter = 0  # CRITICAL: Unique auto-increment ID tracker

    # 2. Process file by file
    for file_in, file_out, split_name in all_files:
        print(f"\n==========================================")
        print(f"📂 Processing File: {file_in}")
        print(f"📂 Target Split Folder: {split_name}")
        print(f"==========================================")
        
        # Setup specific local image directory inside the current split folder
        split_image_dir = os.path.join(OUTPUT_DATASET_DIR, split_name, "images")
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        
        # Open using gzip in Read-Text ("rt") mode
        with gzip.open(file_in, "rt", encoding="utf-8") as f:
            try:
                dataset = json.load(f)
            except Exception as e:
                print(f"   ⚠️ Skipped: Could not decode Gzip/JSON. Error: {e}")
                continue
                
        # Handle formats: Raw List OR a Dict like {"episodes": [...]}
        episodes = dataset.get("episodes", dataset) if isinstance(dataset, dict) else dataset
        if not isinstance(episodes, list):
            print(f"   ⚠️ Skipped: Unrecognized JSON structure.")
            continue
            
        # Optimize rendering by sorting this file's episodes by scene
        episodes.sort(key=lambda x: x.get("scene", ""))
        
        imagenav_episodes = []
        successful_count = 0
        
        # 3. Process every episode in the current compressed file
        for idx, episode in enumerate(episodes):
            episode_id = episode.get("id", f"ep_{idx}")
            scene_name = episode.get("scene")
            target_object_type = episode.get("object_type")
            
            global_episodes_processed += 1
            
            # Periodic Controller Restart to clear out Unity's VRAM cache
            if global_episodes_processed % CONTROLLER_RESTART_INTERVAL == 0:
                controller.stop()
                controller = launch_controller(scene_name)
            
            if "shortest_path" not in episode or not episode["shortest_path"]:
                print(f"   ⚠️ [{idx+1}/{len(episodes)}] Skipped {episode_id}: Missing 'shortest_path'.")
                continue
                
            final_path_pos = episode["shortest_path"][-1]
            
            try:
                event = controller.reset(scene=scene_name)
            except Exception as sim_err:
                print(f"   ❌ Simulator Error resetting {scene_name}: {sim_err}. Re-launching...")
                controller.stop()
                controller = launch_controller(scene_name)
                event = controller.step(action="Pass")

            # Locate the correct target instance using path proximity
            target_obj = None
            min_dist = float("inf")
            for obj in event.metadata["objects"]:
                if obj["objectType"] == target_object_type:
                    obj_coords = obj["position"]
                    dist_to_path_end = math.sqrt(
                        (obj_coords["x"] - final_path_pos["x"])**2 + 
                        (obj_coords["z"] - final_path_pos["z"])**2
                    )
                    if dist_to_path_end < min_dist:
                        min_dist = dist_to_path_end
                        target_obj = obj
                        
            if target_obj is None:
                print(f"   ⚠️ [{idx+1}/{len(episodes)}] Skipped {episode_id}: Object missing.")
                continue
                
            actual_target_id = target_obj["objectId"]
            obj_pos = target_obj["position"]
            
            # Fetch valid viewpoints
            event = controller.step(action="GetReachablePositions")
            if not event.metadata["lastActionSuccess"]:
                continue
                
            reachable_positions = event.metadata["actionReturn"]
            candidate_positions = []
            for pos in reachable_positions:
                dist = math.sqrt((pos['x'] - obj_pos['x'])**2 + (pos['z'] - obj_pos['z'])**2)
                if 0.5 <= dist <= 2.5:
                    candidate_positions.append((pos, dist))
                    
            candidate_positions.sort(key=lambda p: math.sqrt(
                (p[0]['x'] - final_path_pos['x'])**2 + (p[0]['z'] - final_path_pos['z'])**2
            ))
            
            success = False
            
            # CRITICAL FIX: Prepended sequential ID to prevent any filename clashes
            goal_image_filename = f"id_{image_id_counter:06d}_{scene_name}_{episode_id}_goal.png"
            goal_image_full_path = os.path.join(split_image_dir, goal_image_filename)
            
            # Find the unblocked viewpoint
            for pos, _ in candidate_positions:
                dx = obj_pos['x'] - pos['x']
                dz = obj_pos['z'] - pos['z']
                yaw = math.degrees(math.atan2(dx, dz))
                if yaw < 0:
                    yaw += 360
                    
                for horizon in [15, 30]:
                    event = controller.step(
                        action="TeleportFull",
                        x=pos['x'], y=pos['y'], z=pos['z'],
                        rotation=dict(x=0, y=yaw, z=0),
                        horizon=horizon
                    )
                    
                    for obj in event.metadata["objects"]:
                        if obj["objectId"] == actual_target_id and obj["visible"]:
                            img = Image.fromarray(event.frame)
                            img.save(goal_image_full_path)
                            
                            # Log relative path to target goal image frame from output JSON position
                            episode["goal_image_path"] = os.path.relpath(goal_image_full_path, os.path.dirname(file_out))
                            episode["optimal_goal_pose"] = {
                                "x": pos['x'], "y": pos['y'], "z": pos['z'],
                                "rotation": yaw, "horizon": horizon
                            }
                            success = True
                            break
                    if success: break
                
                if success:
                    imagenav_episodes.append(episode)
                    successful_count += 1
                    image_id_counter += 1  # Safely increment unique image tracker ID
                    print(f"   ✔ [{idx+1}/{len(episodes)}] Success: {episode_id} -> Saved to {split_name}/images/")
                    break
                    
            if not success:
                print(f"   ❌ [{idx+1}/{len(episodes)}] Failed: No camera view found.")

            # Checkpoint step: Fixed double file open syntax bug to stream cleanly through gzip
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                out_data = dataset.copy() if isinstance(dataset, dict) else []
                if isinstance(dataset, dict):
                    out_data["episodes"] = imagenav_episodes
                else:
                    out_data = imagenav_episodes
                    
                with gzip.open(file_out, "wt", encoding="utf-8") as f_out:
                    json.dump(out_data, f_out, indent=4)
                    
        # 4. Final compression save for the current completed file
        out_data = dataset.copy() if isinstance(dataset, dict) else []
        if isinstance(dataset, dict):
            out_data["episodes"] = imagenav_episodes
        else:
            out_data = imagenav_episodes
            
        with gzip.open(file_out, "wt", encoding="utf-8") as f_out:
            json.dump(out_data, f_out, indent=4)
            
        print(f"💾 File Complete! Compressed and saved {successful_count}/{len(episodes)} episodes to: {file_out}")

    print("\n🎉 ALL COMPRESSED SPLITS CONVERTED SUCCESSFULLY WITH LOCALIZED IMAGES!")
    controller.stop()

if __name__ == "__main__":
    main()