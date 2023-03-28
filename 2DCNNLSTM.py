from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Input, MaxPooling2D, Dropout
from keras import layers
from utils import *

HEIGHT = 224
WIDTH = 224

input_shape = (None, 10, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input
x = ConvLSTM2D(filters=32, kernel_size=(5, 5), return_sequences=True,
    activation="relu")(x)
x = layers.BatchNormalization()(x)
x = ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=True,
    activation="relu")(x)
x = layers.BatchNormalization()(x)
x = ConvLSTM2D(filters=32, kernel_size=(1, 1),
    activation="relu")(x)
x = layers.BatchNormalization()(x)
#x = layers.ReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(1024)(x)
x = layers.Dropout(.3)(x)
x = layers.Dense(10)(x)
model = keras.Model(input, x)

############
# model = Sequential()
# # input = layers.Input(shape=(input_shape[1:]))
#
# model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), input_shape=(10, HEIGHT, WIDTH, 3)))
# ### model.add(...more layers)
# model.add(Dense(units=10))
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


subset_paths = {'train_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\train"),
                'val_master': pathlib.Path(r"C:\Users\User\Desktop\Master_videos_all\val")}

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train_master'], n_frames, training=True),
                                          output_signature=output_signature)

# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val_master'], n_frames),
                                        output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

frames, label = next(iter(train_ds))
model.build(frames)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x=train_ds,
                    epochs=50,
                    validation_data=val_ds)


#take labels
fg = FrameGenerator(subset_paths['train_master'], n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())

