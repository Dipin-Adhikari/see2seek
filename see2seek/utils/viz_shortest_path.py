"""
Visualize shortest paths for RoboTHOR/ImageNav debug episodes.

For every episode in a `*.json.gz` episodes file, this script:
  1. Grabs a calibrated, orthographic top-down camera frame of the scene.
  2. Projects each waypoint of `shortest_path` (world x, z) into image pixels.
  3. Draws the path as numbered circles connected by lines (green = start,
     blue = intermediate, red = goal), plus outlined markers for
     `initial_position` and `optimal_goal_pose` so you can sanity-check them
     against the path endpoints.
  4. Saves one PNG per episode.

Usage:
    python viz_shortest_path.py \
        --episodes /home/dipin/See2Seek/imagenav_dataset/debug/episodes/FloorPlan_Train1_1.json.gz \
        --output ./birds_eye_views_with_paths \
        --image-size 1024

On a headless GCP VM, uncomment the `x_display="0"` line in `build_controller()`
(same as your working AllenAct startx.py setup).
"""

import argparse
import gzip
import json
import os

from PIL import Image, ImageDraw
from ai2thor.controller import Controller


# ==========================================
# World -> pixel projection for an orthographic top-down camera
# ==========================================
class TopDownProjector:
    """Converts AI2-THOR world (x, z) coordinates into pixel coordinates on
    an orthographic top-down camera frame.

    For an orthographic camera looking straight down, the frame shows a
    square window of world space centered on the camera's (x, z), spanning
    `2 * orthographic_size` world units in each direction. That makes the
    mapping from world space to pixel space a simple linear rescale.
    """

    def __init__(self, frame_shape, cam_position, orthographic_size):
        self.h, self.w = frame_shape[0], frame_shape[1]
        self.cam_x = cam_position["x"]
        self.cam_z = cam_position["z"]
        self.orth_size = orthographic_size

        self.min_x = self.cam_x - self.orth_size
        self.min_z = self.cam_z - self.orth_size
        self.span = 2.0 * self.orth_size

    def to_pixel(self, x, z):
        norm_x = (x - self.min_x) / self.span
        norm_z = (z - self.min_z) / self.span
        px = int(round(norm_x * self.w))
        # Row 0 of the image is the top of the frame (max z), so flip.
        py = int(round((1.0 - norm_z) * self.h))
        return px, py


# ==========================================
# Controller / top-down camera helpers
# ==========================================
def build_controller(scene, image_size):
    return Controller(
        scene=scene,
        width=image_size,
        height=image_size,
        gridSize=0.25,
        # x_display="0",  # uncomment on headless GCP VM (Xorg display)
    )


def get_top_down_frame(controller, image_size, camera_initialized):
    """Adds (first call) or updates (subsequent calls) an orthographic
    top-down third-party camera and returns (frame, projector, initialized)."""
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    cam_props = dict(event.metadata["actionReturn"])

    # Field names have shifted slightly across ai2thor versions. Print
    # cam_props once if this KeyErrors on your install and adjust.
    orth_size = cam_props.get("orthographicSize")
    if orth_size is None:
        # Some versions return a nested "orthographicSize" under a
        # different key, or you need event.metadata["cameraOrthSize"].
        raise KeyError(
            f"Couldn't find orthographicSize in cam_props: {cam_props}. "
            "Print cam_props and adjust the key name for your ai2thor version."
        )

    cam_props["orthographic"] = True
    cam_props["farClippingPlane"] = 50
    cam_props["skyboxColor"] = "white"

    if not camera_initialized:
        event = controller.step(action="AddThirdPartyCamera", **cam_props)
        camera_initialized = True
    else:
        event = controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=cam_props["position"],
            rotation=cam_props["rotation"],
            fieldOfView=cam_props.get("fieldOfView", 90),
        )

    frame = event.third_party_camera_frames[0]
    projector = TopDownProjector(frame.shape, cam_props["position"], orth_size)
    return frame, projector, camera_initialized


# ==========================================
# Drawing
# ==========================================
def draw_episode_path(frame, episode, projector, output_path):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    path = episode["shortest_path"]
    pixel_path = [projector.to_pixel(p["x"], p["z"]) for p in path]

    # Connecting line
    for i in range(len(pixel_path) - 1):
        draw.line([pixel_path[i], pixel_path[i + 1]], fill=(255, 140, 0), width=4)

    # Waypoints, numbered
    r = 8
    for i, (px, py) in enumerate(pixel_path):
        if i == 0:
            color = (0, 200, 0)        # start
        elif i == len(pixel_path) - 1:
            color = (220, 20, 60)      # goal (last shortest_path point)
        else:
            color = (30, 100, 220)     # intermediate
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0), width=2)
        draw.text((px + r + 2, py - r), str(i), fill=(0, 0, 0))

    # Cross-check markers: initial_position and optimal_goal_pose (may not
    # exactly coincide with shortest_path[0]/[-1] due to agent height offset)
    init_p = episode["initial_position"]
    goal_p = episode["optimal_goal_pose"]
    ipx, ipy = projector.to_pixel(init_p["x"], init_p["z"])
    gpx, gpy = projector.to_pixel(goal_p["x"], goal_p["z"])
    draw.ellipse([ipx - 4, ipy - 4, ipx + 4, ipy + 4], outline=(0, 0, 0), width=2)
    draw.ellipse([gpx - 4, gpy - 4, gpx + 4, gpy + 4], outline=(0, 0, 0), width=2)

    draw.text((10, 10), episode["id"], fill=(0, 0, 0))
    draw.text((10, 28), f"path_len={episode.get('shortest_path_length', 0):.2f}m", fill=(0, 0, 0))

    img.save(output_path)


# ==========================================
# Main
# ==========================================
def load_episodes(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Path to episodes .json.gz file")
    parser.add_argument("--output", default="./birds_eye_views_with_paths")
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()

    if not os.path.exists(args.episodes):
        raise FileNotFoundError(f"Cannot find episodes file at: {args.episodes}")

    os.makedirs(args.output, exist_ok=True)
    episodes = load_episodes(args.episodes)
    print(f"Loaded {len(episodes)} episodes")

    scene = episodes[0]["scene"]
    controller = build_controller(scene, args.image_size)
    camera_initialized = False
    frame, projector, camera_initialized = get_top_down_frame(
        controller, args.image_size, camera_initialized
    )

    for episode in episodes:
        if episode["scene"] != scene:
            scene = episode["scene"]
            controller.reset(scene=scene)
            camera_initialized = False
            frame, projector, camera_initialized = get_top_down_frame(
                controller, args.image_size, camera_initialized
            )

        out_path = os.path.join(args.output, f"{episode['id']}.png")
        draw_episode_path(frame, episode, projector, out_path)
        print(f"Saved {out_path}")

    controller.stop()
    print("Done.")


if __name__ == "__main__":
    main()