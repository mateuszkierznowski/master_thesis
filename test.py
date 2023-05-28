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
from utils import FrameGenerator
import matplotlib.pyplot as plt

subset_paths = {'train_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\train_3"),
                'val_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\val_3")}

n_frames = 12
batch_size = 24

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds_no_aug = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train_master'], n_frames, augmentation=False, training=False, frame_step=52),
                                          output_signature=output_signature)

train_ds_aug = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train_master'], n_frames, augmentation=True, training=False, frame_step=52),
                                          output_signature=output_signature)


train_gen = FrameGenerator(subset_paths['train_master'], n_frames, augmentation=True, training=False, frame_step=1)
train_gen16 = FrameGenerator(subset_paths['train_master'], n_frames, augmentation=True, training=False, frame_step=16)

def plot_frames(frames):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

    # Iterate over the frames and plot them in the subplots
    for i, frame in enumerate(frames):
        # Determine the subplot index
        row = i // 4
        col = i % 4

        # Plot the frame in the corresponding subplot
        axes[row, col].imshow(frame)  # Assuming 'frame' is a numpy array

        # Optionally, you can turn off the axis labels and ticks
        axes[row, col].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


from video_preprocessing.vidaug.vidaug import augmentors as va

sometimes_05 = lambda aug: va.Sometimes(.5, aug)
sometimes_02 = lambda aug: va.Sometimes(.2, aug)
sometimes_01 = lambda aug: va.Sometimes(.1, aug)

seq = va.Sequential([
    va.RandomRotate(degrees=15), # randomly rotates the video with a degree randomly choosen from [-10, 10]
    sometimes_05(va.HorizontalFlip()),
    sometimes_01(va.PiecewiseAffineTransform()),
    sometimes_01(va.Superpixel(10)),
    sometimes_02(va.SomeOf([va.Add(10), va.Multiply(1.2)], N=1)),
    sometimes_02(va.SomeOf([va.Downsample(.8), va.Upsample(2)], N=1))
])

for i in range(1000):
    frames, label = next(iter(train_ds_no_aug))
    frames_2, label_2 = next(iter(train_ds_aug))

plot_frames(frames)
plot_frames(frames_2)

for frame in frames_2:
    print(frame.shape)

# frame = tf.image.convert_image_dtype(frames[0:10], tf.float32)
# frame = tf.image.resize_with_pad(frame, *(224, 224))

frames_ = seq(np.array(frames))
plot_frames(frames_)

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


