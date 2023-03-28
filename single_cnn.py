from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Input, MaxPooling2D, Dropout
from keras import layers
from utils import *
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications

# define input image size
img_width, img_height = 224, 224

# define number of classes
num_classes = 10

# load InceptionV3 model without top layer (i.e., without the fully connected layer)
base_model = applications.InceptionV3(weights='imagenet',
                                include_top=False,
                                input_shape=(img_width, img_height, 3))
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(num_classes, activation='softmax'))

model = add_model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

subset_paths = {'train_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\train"),
                'val_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\val")}

n_frames = 1
batch_size = 24

output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train_master'], n_frames, training=True),
                                          output_signature=output_signature)

# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val_master'], n_frames),
                                        output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

frames, label = next(iter(train_ds))
#model.build(frames)

history = model.fit(x=train_ds,
                    epochs=50,
                    validation_data=val_ds)


#take labels
fg = FrameGenerator(subset_paths['train_master'], n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())





