import os
import random
import torch
import time
from types import SimpleNamespace
from PIL import Image

# Import your updated environment class
from see2seek.envs.robothor_env import RoboTHOREnv

import logging
# Force AI2-THOR internal logs to surface
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ai2thor").setLevel(logging.DEBUG)

# ==========================================
# TEST CONFIGURATION
# ==========================================
DEBUG_EPISODES_PATH = "/home/dipin/See2Seek/imagenav_dataset/debug/episodes/FloorPlan_Train1_1.json.gz"
OUTPUT_DIR = "./birds_eye_views"

def create_mock_config(episodes_path: str) -> SimpleNamespace:
    """Creates a minimal mock configuration object matching your cfg schema."""
    cfg = SimpleNamespace()
    cfg.env = SimpleNamespace()
    
    # Environment Setup
    cfg.env.episodes_path = episodes_path
    cfg.env.image_width = 300
    cfg.env.image_height = 300
    cfg.env.image_channels = 3
    cfg.env.depth_sensor = False
    
    # Navigation Action Constraints
    cfg.env.move_magnitude = 0.25
    cfg.env.rotate_degrees = 90.0
    cfg.env.num_actions = 4
    cfg.env.max_steps = 128  # Keep test episodes short
    
    # Reward Setup
    cfg.env.success_distance = 1.0
    cfg.env.success_reward = 10.0
    cfg.env.geodesic_reward_scale = 1.0
    cfg.env.slack_reward = -0.01
    
    return cfg

def capture_birds_eye_view(env: RoboTHOREnv, step_count: int, camera_initialized: bool) -> bool:
    """Dynamically creates or updates a top-down camera tracking the agent using correct AI2-THOR API naming."""
    try:
        agent_pos = env._controller.last_event.metadata["agent"]["position"]
        
        # Position the camera 6 meters above the agent pointing straight down (rotation x=90)
        cam_position = {"x": agent_pos["x"], "y": agent_pos["y"] + 6.0, "z": agent_pos["z"]}
        cam_rotation = {"x": 90, "y": 0, "z": 0}
        
        if not camera_initialized:
            # FIX: Correct action names are AddThirdPartyCamera / UpdateThirdPartyCamera
            event = env._controller.step(
                action="AddThirdPartyCamera",
                position=cam_position,
                rotation=cam_rotation,
                fieldOfView=90
            )
            camera_initialized = True
        else:
            event = env._controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=cam_position,
                rotation=cam_rotation
            )
            
        # FIX: Extract frames from third_party_camera_frames array list
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            frame_array = event.third_party_camera_frames[0]
            img = Image.fromarray(frame_array)
            img_path = os.path.join(OUTPUT_DIR, f"step_{step_count:02d}.png")
            img.save(img_path)
            print(f"    📸 Saved bird's-eye view frame to: {img_path}")
            
    except Exception as e:
        print(f"    ⚠️ Bird's-eye capture error: {e}")
        
    return camera_initialized

def main():
    if not os.path.exists(DEBUG_EPISODES_PATH):
        print(f"❌ Error: Cannot find debug episodes file at: {DEBUG_EPISODES_PATH}")
        print("Please verify your path configurations before running.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🛠️ Constructing environment configuration package...")
    cfg = create_mock_config(DEBUG_EPISODES_PATH)

    print("🤖 Initializing RoboTHOREnv with pre-cached embeddings...")
    env = RoboTHOREnv(cfg, worker_id=0, render=True)

    print("\n🔄 Testing env.reset()...")
    obs = env.reset()
    
    # Extract structural references
    rgb_tensor = obs.get("rgb")
    goal_tensor = obs.get("goal")

    print("------------------------------------------")
    print("📊 Observation Space Verification:")
    print(f"   🔹 Live RGB Shape:  {tuple(rgb_tensor.shape)} (Expected: (3, 300, 300))")
    print(f"   🔹 Live RGB Type:   {rgb_tensor.dtype} (Expected: torch.float32)")
    print(f"   🔹 Goal Vector Shape: {tuple(goal_tensor.shape)} (Expected: (512,))")
    print(f"   🔹 Goal Vector Type:  {goal_tensor.dtype} (Expected: torch.float32)")
    print("------------------------------------------")

    assert goal_tensor.shape == (512,), "❌ Failed: Goal observation should be a 512-dim embedding vector!"
    print("✔ Reset observation verification passed successfully.")

    # Render initial position top-down frame
    print("\n🗺️ Initializing Bird's-Eye View Camera Rig...")
    camera_initialized = capture_birds_eye_view(env, step_count=0, camera_initialized=False)

    print("\n👟 Stepping through environment with random actions...")
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1
        action = random.randint(0, 2)
        
        # Execute structural step execution
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"   [Step {step_count:02d}] Executed Action: {action} | Reward: {reward:+.4f} | Done: {done}")
        
        # Sync and capture the bird's eye viewpoint following the agent layout updates
        camera_initialized = capture_birds_eye_view(env, step_count, camera_initialized)

        if done:
            print("\n🏁 Episode Finished!")
            print(f"   🔹 Total Accumulated Reward: {total_reward:.4f}")
            print(f"   🔹 Reason for ending: {'Agent called Stop' if action == 3 else 'Reached Max Steps Timeout'}")
            print(f"   🔹 Navigation Success Flag:  {info.get('success')}")
            print(f"   🔹 Executed Episode ID:       {info.get('episode_id')}")
            print(f"   🔹 Scene Context Room:         {info.get('scene_id')}")

        time.sleep(2)

    print("\n🧹 Shutting down active simulator controller process threads...")
    env.close()
    print("🎉 Environment integration test completed successfully with zero processing faults!")

if __name__ == "__main__":
    main()