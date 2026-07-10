import os
import torch
from PIL import Image
from tqdm import tqdm

from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder

# ==========================================
# Configuration Paths
# ==========================================
# Path to your generated dataset containing debug, train, and val
DATASET_DIR = "/home/dipin/See2Seek/imagenav_dataset"
SPLITS = ["debug"]
BATCH_SIZE = 64  # Process images in parallel batches for massive speedup

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

        # Collect all images inside the folder
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_filenames = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
        
        if not image_filenames:
            print(f"No images found inside {images_dir}")
            continue

        print(f"\n⚡ Processing {len(image_filenames)} images for split: {split.upper()}")
        embedding_dict = {}

        # Process images in chunks/batches
        for i in tqdm(range(0, len(image_filenames), BATCH_SIZE)):
            batch_files = image_filenames[i : i + BATCH_SIZE]
            batch_images = []
            batch_keys = []

            for fname in batch_files:
                img_path = os.path.join(images_dir, fname)
                try:
                    # Open and ensure it is in standard RGB format
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    
                    # The key matches the exact relative path string used in your JSON files
                    # e.g., "train/images/id_000001_FloorPlan..._goal.png"
                    dict_key = f"{split}/images/{fname}"
                    batch_keys.append(dict_key)
                except Exception as e:
                    print(f"\nError opening image {img_path}: {e}")

            if not batch_images:
                continue

            # Generate embeddings for the entire batch in one GPU forward pass
            with torch.no_grad():
                # Shape: (N, 512)
                embeddings = clip_encoder.encode_image(batch_images)
                # Move tensors to CPU memory to keep the file lightweight and safe to load anywhere
                embeddings = embeddings.cpu()

            # Map the unique path keys to their corresponding embedding tensors
            for key, embedding in zip(batch_keys, embeddings):
                embedding_dict[key] = embedding

        # Save the finalized dictionary map for this split
        output_pt_path = os.path.join(split_dir, "embeddings.pt")
        torch.save(embedding_dict, output_pt_path)
        print(f"Success! Saved {len(embedding_dict)} embeddings to: {output_pt_path}")

    print("\nALL IMAGES CONVERTED TO CLIP EMBEDDINGS SUCCESSFULLY!")

if __name__ == "__main__":
    main()



