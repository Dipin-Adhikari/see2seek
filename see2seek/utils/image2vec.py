import os
import torch
from PIL import Image
from tqdm import tqdm

from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder

DATASET_DIR = "/home/dipin/See2Seek/imagenav_dataset"
SPLITS = ["train"]
BATCH_SIZE = 32
CHECKPOINT_EVERY = 50  # save every 50 batches 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing CLIP Goal Encoder on device: {device}")
    clip_encoder = CLIPGoalEncoder(device=device)

    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        images_dir = os.path.join(split_dir, "images")

        if not os.path.exists(images_dir):
            print(f"Skipping split [{split.upper()}]: No 'images' folder found at {images_dir}")
            continue

        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_filenames = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]

        if not image_filenames:
            print(f"No images found inside {images_dir}")
            continue

        output_pt_path = os.path.join(split_dir, "embeddings.pt")

        # RESUME: load existing embeddings if present
        embedding_dict = {}
        if os.path.exists(output_pt_path):
            embedding_dict = torch.load(output_pt_path)
            print(f"Resuming: found {len(embedding_dict)} existing embeddings, skipping those.")

        # Filter out already-embedded images
        remaining = [
            f for f in image_filenames
            if f"{split}/images/{f}" not in embedding_dict
        ]
        print(f"\n⚡ {len(remaining)} / {len(image_filenames)} images remaining for split: {split.upper()}")

        if not remaining:
            print("Nothing to do, all images already embedded.")
            continue

        batch_counter = 0
        for i in tqdm(range(0, len(remaining), BATCH_SIZE)):
            batch_files = remaining[i : i + BATCH_SIZE]
            batch_images = []
            batch_keys = []

            for fname in batch_files:
                img_path = os.path.join(images_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_keys.append(f"{split}/images/{fname}")
                except Exception as e:
                    print(f"\nError opening image {img_path}: {e}")

            if not batch_images:
                continue

            with torch.no_grad():
                embeddings = clip_encoder.encode_image(batch_images).cpu()

            for key, embedding in zip(batch_keys, embeddings):
                embedding_dict[key] = embedding

            batch_counter += 1
            if batch_counter % CHECKPOINT_EVERY == 0:
                torch.save(embedding_dict, output_pt_path)
                tqdm.write(f"Checkpoint saved: {len(embedding_dict)} embeddings so far.")

        # Final save
        torch.save(embedding_dict, output_pt_path)
        print(f"Success! Saved {len(embedding_dict)} embeddings to: {output_pt_path}")

    print("\nALL IMAGES CONVERTED TO CLIP EMBEDDINGS SUCCESSFULLY!")

if __name__ == "__main__":
    main()