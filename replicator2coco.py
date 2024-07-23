#%%
import numpy as np
import json
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process and organize image data for COCO-style annotation.")
parser.add_argument("base_directory", type=str, help="The root folder path of the data subfolders.")
args = parser.parse_args()

base_directory = Path(args.base_directory)

def is_similar(box1, box2):
    """ Check if two boxes are similar based on the 0.5% criterion. """
    for b1, b2 in zip(box1, box2):
        if abs(b1 - b2) > 0.005 * max(b1, b2):
            return False
    return True

def filter_boxes(boxes):
    """ Filter out bounding boxes that are too similar to others already accepted. """
    accepted = []
    for new_box in boxes:
        if not any(is_similar(new_box, acc_box) for acc_box in accepted):
            accepted.append(new_box)
    return accepted# Set up logging

# Processed data directory
processed_data_dir = base_directory / "processed_data"
processed_data_dir.mkdir(exist_ok=True)
logging.info("Processed data directory created at %s", processed_data_dir)

# Directory for processed images
images_dir = processed_data_dir / "images"
images_dir.mkdir(exist_ok=True)
logging.info("Images directory created at %s", images_dir)

# COCO JSON structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "chair"}]  # Assuming only chairs are of interest
}

# Annotation ID counter
annotation_id = 1

# Process each folder
for folder_number in range(1, 25):
    folder_path = base_directory / str(folder_number)
    if folder_path.exists():
        json_files = list(folder_path.glob("bounding_box_2d_tight_labels_*.json"))
        logging.info("Processing folder: %s", folder_path)

        for json_file in json_files:
            with open(json_file) as f:
                labels = json.load(f)
                chair_id = next((int(key) for key, value in labels.items() if value['class'] == 'chair'), None)
            
            if chair_id is not None:
                image_id = json_file.stem.split("_")[-1]
                unique_image_id = int(f"{folder_number}{image_id}")  # Unique ID across folders
                npy_file = folder_path / f"bounding_box_2d_tight_{image_id}.npy"
                image_file = folder_path / f"rgb_{image_id}.png"
                target_image_path = images_dir / f"rgb_{unique_image_id}.png"

                if npy_file.exists() and image_file.exists():
                    data = np.load(npy_file, allow_pickle=True)
                    valid_boxes = []
                    for bbox in data:
                        if bbox['semanticId'] == chair_id and bbox['occlusionRatio'] < 0.4:
                            valid_boxes.append((bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']))
                    filtered_boxes = filter_boxes(valid_boxes)
                    
                    # Copy image to processed directory
                    shutil.copy(image_file, target_image_path)
                    logging.info("Image %s copied to %s", image_file, target_image_path)

                    # Add image info to COCO data
                    img = Image.open(image_file)
                    width, height = img.size
                    coco_data['images'].append({
                        "id": unique_image_id,
                        "width": width,
                        "height": height,
                        "file_name": f"rgb_{unique_image_id}.png"
                    })

                    # Add annotations for each box
                    for box in filtered_boxes:
                        coco_data['annotations'].append({
                            "id": int(annotation_id),  # Ensure Python native int type
                            "image_id": int(unique_image_id),
                            "category_id": 1,
                            "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                            "area": int((box[2] - box[0]) * (box[3] - box[1])),
                            "iscrowd": 0
                        })
                        annotation_id += 1
                        logging.info("Annotation %d added for image %d", annotation_id, unique_image_id)

# Write COCO data to JSON
with open(processed_data_dir / "coco.json", "w") as f:
    json.dump(coco_data, f, indent=4)
    logging.info("COCO annotation file written at %s", processed_data_dir / "coco.json")

# # Define path to the folder
# folder_path = Path("/home/segreto/omni.replicator_out/_output/")

# # Get all json and numpy files related to bounding boxes and labels
# json_files = list(folder_path.glob("bounding_box_2d_tight_labels_*.json"))
# npy_files = list(folder_path.glob("bounding_box_2d_tight_*.npy"))
# image_files = list(folder_path.glob("rgb_*.png"))



# # Process each file
# for json_file in json_files:
#     with open(json_file) as f:
#         labels = json.load(f)
#         # Find the ID for 'chair'
#         chair_id = None
#         for key, value in labels.items():
#             if value['class'] == 'chair':
#                 chair_id = int(key)
#                 break

#     if chair_id is not None:
#         # Corresponding npy file
#         image_id = json_file.stem.split("_")[-1]
#         npy_file = folder_path / f"bounding_box_2d_tight_{image_id}.npy"
#         if npy_file.exists():
#             data = np.load(npy_file, allow_pickle=True)
#             valid_boxes = []

#             for bbox in data:
#                 if bbox['semanticId'] == chair_id and bbox['occlusionRatio'] < 0.4:
#                     valid_boxes.append((bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']))
            
#             filtered_boxes = filter_boxes(valid_boxes)

#             image_file = folder_path / f"rgb_{image_id}.png"
#             if image_file.exists():
#                 img = Image.open(image_file)
#                 plt.figure(figsize=(12, 8))
#                 plt.imshow(img)
#                 ax = plt.gca()

#                 for box in filtered_boxes:
#                     rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
#                     ax.add_patch(rect)
                
#                 plt.show()



# %%
