import os
import csv

CAPTION_FILE = os.environ.get("CAPTION_FILE", "/workspace/results_20130124.token")
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", "/workspace/flickr30k-images")
OUTPUT_CSV   = os.environ.get("OUTPUT_CSV",   "/workspace/captions.csv")

MIN_WORDS = 5
MAX_WORDS = 50
MAX_IMAGES = 10000  

rows         = []
missing      = 0
filtered     = 0
seen_images  = set()

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        image_part, caption = parts
        image_name = image_part.split("#")[0]

        if len(seen_images) >= MAX_IMAGES and image_name not in seen_images:
            continue

        caption = caption.lower().strip()
        words   = caption.split()

        if len(words) < MIN_WORDS:
            filtered += 1
            continue

        if len(words) > MAX_WORDS:
            caption = " ".join(words[:MAX_WORDS]).rstrip(",:;-")

        image_path = os.path.join(IMAGE_FOLDER, image_name)
        if not os.path.exists(image_path):
            missing += 1
            continue

        seen_images.add(image_name)
        rows.append([image_name, caption])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(rows)

print("=" * 40)
print(f"Total pairs:        {len(rows)}")
print(f"Unique images:      {len(seen_images)}")
print(f"Filtered captions:  {filtered}")
print(f"Missing images:     {missing}")
print(f"Saved to:           {OUTPUT_CSV}")
print("=" * 40)
