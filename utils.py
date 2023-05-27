import tqdm
import random
import pathlib
import collections
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import numpy as np

import tensorflow as tf

from video_preprocessing.vidaugm.vidaug import augmentors as va



def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class


def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder


def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (112,112), frame_step = 2):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  if ret:
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
      for _ in range(frame_step):
        ret, frame = src.read()
      if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
      else:
        result.append(result[-1])

  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result




class FrameGenerator:
  def __init__(self, path, n_frames, training=False, augmentation=True, frame_shape=112, frame_step=2):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
    self.augmentation = augmentation
    self.frame_shape = frame_shape
    self.frame_step = frame_step

  def image_augmentation(self, image_tensor):
    sometimes_05 = lambda aug: va.Sometimes(.5, aug)
    sometimes_02 = lambda aug: va.Sometimes(.2, aug)
    sometimes_01 = lambda aug: va.Sometimes(.1, aug)

    seq = va.Sequential([
      va.RandomRotate(degrees=15),  # randomly rotates the video with a degree randomly choosen from [-10, 10]
      sometimes_05(va.HorizontalFlip()),
      # sometimes_01(va.PiecewiseAffineTransform()),
      # sometimes_01(va.Superpixel(10)),
      #sometimes_02(va.SomeOf([va.Add(10), va.Multiply(1.2)], N=1)),
      #sometimes_02(va.SomeOf([va.Downsample(.8), va.Upsample(2)], N=1))
    ])

    frames: np.array = seq(np.array(image_tensor))

    image_tensor: np.array = np.array(frames)

    return image_tensor

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    video_paths += list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      try:
        if self.augmentation:
          video_frames = frames_from_video_file(path, self.n_frames, output_size=(self.frame_shape, self.frame_shape), frame_step = self.frame_step)
          video_frames = self.image_augmentation(video_frames)
        else:
          video_frames = frames_from_video_file(path, self.n_frames,output_size=(self.frame_shape, self.frame_shape), frame_step = self.frame_step)
      except IndexError:
        continue

      label = self.class_ids_for_name[name] # Encode labels

      if self.n_frames == 1:
        if self.augmentation:
          video_frames = video_frames.reshape(self.frame_shape, self.frame_shape, 3)
          video_frames = self.image_augmentation(video_frames)
        else:
          video_frames = video_frames.reshape(self.frame_shape, self.frame_shape, 3)
      yield video_frames, label

def create_dataframe(folder_path):
  dfs = []

  # iterate through the subdirectories
  for subfolder_name in os.listdir(folder_path):
    # check if the item in the folder is a subdirectory
    if os.path.isdir(os.path.join(folder_path, subfolder_name)):
      # create a list of the picture files in the subdirectory
      picture_files = [f for f in os.listdir(os.path.join(folder_path, subfolder_name))]
      # create a dataframe for the picture files in the subdirectory
      df = pd.DataFrame({'video_name': picture_files})
      # add a column to the dataframe with the subdirectory name
      df['tag'] = subfolder_name
      # append the dataframe to the list of dataframes
      dfs.append(df)

  # concatenate the dataframes for each subdirectory into a single dataframe
  df = pd.concat(dfs, ignore_index=True)

  return df

def get_recall_precision(cm):
  recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
  precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
  f1 = 2 * (recall * precision) / (recall + precision)

  return recall, precision, f1

def get_actual_predicted_labels_per_video(dataset, model):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def get_actual_predicted_labels_per_batch(dataset, model):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type, save_plot=False):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize': (12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  if save_plot:
    plt.savefig(save_plot)

  plt.show()

  return cm