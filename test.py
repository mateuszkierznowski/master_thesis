import pandas as pd
import os
import cPickle as pickle
stat_dict = {"olx": 0,
             "autoscout": 0,
             "otomoto": 0}

pd.DataFrame(stat_dict, index=[0]).to_csv("hej.csv")
df = pd.read_csv("hej.csv")

import datetime
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import os
import pandas as pd

# specify the path to the main folder
folder_path = r'C:\Users\User\Desktop\Master_videos_all\train'

# create an empty list to store the dataframes for each subdirectory


df = create_dataframe(folder_path)

##
import tensorflow as tf
import pickle

# Load the .pkl model
import torch
model = torch.load(r"C:\Users\User\Downloads\r2plus1d_152_ft_kinetics_from_sports1m_f128957437.pkl", encoding='latin1',map_location=torch.device('cpu'))


import fiftyone as fo
import fiftyone.zoo as foz

#
# Only the required images will be downloaded (if necessary).
# By default, only detections are loaded


dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["validation","train"],
    classes=["person", "car"],
    max_samples=200,
    dataset_dir="./COCO"
)

import json
import os
import shutil

# Define the categories you want to include in your dataset
categories = [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
]

# Create a directory to store the images
os.makedirs("dataset/images", exist_ok=True)

# Create a list to store the image annotations
annotations = []

# Loop through each category and copy 50 images to the dataset directory
for category in categories:
    category_id = category["id"]
    category_name = category["name"]
    source_dir = f"coco/train2017/{category_name}"
    target_dir = f"dataset/images/{category_name}"
    os.makedirs(target_dir, exist_ok=True)
    file_names = os.listdir(source_dir)[:50]
    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        shutil.copyfile(source_path, target_path)
        # Create an annotation for the image
        annotation = {
            "id": len(annotations) + 1,
            "image_id": len(annotations) + 1,
            "category_id": category_id,
            "bbox": [0, 0, 0, 0],
            "area": 0,
            "iscrowd": 0
        }
        annotations.append(annotation)

# Write the annotations to a JSON file
with open("dataset/annotations.json", "w") as f:
    json.dump({
        "images": [
            {"id": i+1, "file_name": f"{category}/{file_name}"}
            for i, category in enumerate(categories)
            for file_name in os.listdir(f"dataset/images/{category['name']}")
        ],
        "annotations": annotations,
        "categories": categories
    }, f, indent=4)



import tensorflow as tf
from utils import *
import pathlib


# Example usage
image = tf.random.uniform(shape=[12, 112, 112, 3], minval=0, maxval=255, dtype=tf.float32)
augmented_image = image_augmentation(image)

# Print the shape of the augmented image tensor
print(augmented_image.shape)



subset_paths = {'train_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\train_2"),
                'val_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\val_2")}

n_frames = 36
batch_size = 24

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train_master'], n_frames, training=True),
                                          output_signature=output_signature)

frames, label = next(iter(train_ds))
print(frames.shape)
import matplotlib.pyplot as plt
plt.imshow(frames[35])
plt.show()

import cv2
from scikit_image.io import Video


cap = Video(videofile)
fc = cap.frame_count()
for i in np.arange(fc):
   z = cap.get_index_frame(i)

import os
[name.replace("_avi", ".avi") for name in os.listdir(r"C:\Users\User\Desktop\Master_videos_all\train_2\Pullup")]

path_to_rename = r"C:\Users\User\Desktop\Master_videos_all\train_2\Pullup"
for name in os.listdir():
    os.rename(f"{path_to_rename}/{name}", f"{path_to_rename}/{name.replace('_avi', '.avi')}")

path_to_rename = r"C:\Users\User\Desktop\Master_videos_all\val_2\Pullup"
for name in os.listdir(path_to_rename):
    if not name.startswith(".g"):
        os.rename(f"{path_to_rename}/{name}", f"{path_to_rename}/{name.replace('_avi', '.avi')}")

