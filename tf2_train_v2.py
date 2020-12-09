import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import matplotlib.pyplot as plt
import numpy as np
import PIL
from natsort import natsorted
from matplotlib import image

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

data_dir2 = pathlib.Path('E:\Codes\TFWP_training')
# image_count = len(list(data_dir2.glob('*/*.png')))
# print(image_count)

batch_size = 32
img_height = 96
img_width = 96

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir2,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir2,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Pre-trained model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height,img_width,3), 
    include_top= False, 
    weights='imagenet')


# Freeze the pre-trained model weights
base_model.trainable = True

# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(img_height, activation='relu'),
    layers.Dense(num_classes,activation='sigmoid')
])



learning_rate = 0.0001
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
              loss='binary_crossentropy',
              metrics=['accuracy']
)

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_best_only=True,
    save_weights_only=True)#,
    #save_freq=5)

es_Callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',patience=3)

num_epochs = 100
steps_per_epoch = round(3000/batch_size)
val_steps = 20
model.fit(train_ds.repeat(),
          epochs=num_epochs,
          steps_per_epoch = steps_per_epoch,
          validation_data=val_ds.repeat(), 
          validation_steps=val_steps,
          callbacks=[cp_callback,es_Callback])

model.summary()

# model.load_weights('saved_model/my_model.h5')

# epochs = 20
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs,
#   callbacks=[cp_callback,es_Callback]
# )
model.save('saved_model/my_model2.h5') 
print("hello there, general kenobi")