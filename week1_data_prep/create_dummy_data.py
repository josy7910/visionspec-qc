# week1_data_prep/create_dummy_data.py
# Run this if you don't have real PCB images yet
# Creates colored placeholder images for testing

import os
import numpy as np
from PIL import Image
import random

def create_dummy_dataset(base_dir="data", num_images=50):
    """
    Creates fake PCB-like images for testing the pipeline.
    Green-ish images = PASS, Red-ish images = DEFECT
    """
    splits  = ["train", "val"]
    classes = ["pass", "defect"]

    for split in splits:
        for cls in classes:
            folder = os.path.join(base_dir, split, cls)
            os.makedirs(folder, exist_ok=True)

            count = num_images if split == "train" else num_images // 5

            for i in range(count):
                # Create a 224x224 image
                img_array = np.random.randint(100, 200,
                                              (224, 224, 3),
                                              dtype=np.uint8)
                if cls == "pass":
                    # Greenish tint for good boards
                    img_array[:,:,1] = np.clip(
                        img_array[:,:,1] + 50, 0, 255)
                else:
                    # Reddish tint + a "defect" mark for bad boards
                    img_array[:,:,0] = np.clip(
                        img_array[:,:,0] + 50, 0, 255)
                    # Draw a dark spot simulating defect
                    x, y = random.randint(50,174), random.randint(50,174)
                    img_array[y:y+20, x:x+20] = [20, 20, 20]

                img = Image.fromarray(img_array)
                img.save(os.path.join(folder, f"{cls}_{i:04d}.jpg"))

            print(f"  Created {count} {split}/{cls} images")

    print("\nDummy dataset created in data/ folder ✅")


if __name__ == "__main__":
    create_dummy_dataset()