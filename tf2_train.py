import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import numpy as np
import PIL
from natsort import natsorted
from matplotlib import image

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)
data_dir2 = pathlib.Path(os.path.join(os.getcwd(),'TFWP_training'))
# image_count = len(list(data_dir2.glob('*/*.png')))
# print(image_count)

batch_size = 32
img_height = 192
img_width = 192

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

class_names = ['no_worms', 'worms']
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image)) 

num_classes = 2

num_classes = len(class_names)


data_augmentation = keras.Sequential(
  [
	layers.experimental.preprocessing.RandomFlip("horizontal", 
												 input_shape=(img_height, 
															  img_width,
															  3)),
	layers.experimental.preprocessing.RandomRotation(0.1),
	layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


model = Sequential([
	data_augmentation,
	layers.experimental.preprocessing.Rescaling(1./255),
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
#   layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(128, activation='tanh'),
	layers.Dense(num_classes)
])

model.compile(optimizer='adam',
			  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_path, 
	verbose=1, 
	save_weights_only=True,
	period=5)

es_Callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',patience=25,restore_best_weights=True)

model.summary()

# model.load_weights('saved_model/my_model.h5')

epochs = 200
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs,
	callbacks=[es_Callback]
	# callbacks=[cp_callback,es_Callback]
	# callbacks=[cp_callback]
)
model.save('saved_model/my_model5.h5') 

model.load_weights('saved_model/my_model5.h5')
 
all_files = natsorted(os.listdir(os.path.join(os.getcwd(),'TFWP_dev','worms')))
all_images_array = []
for each_file in all_files:
	if(each_file.endswith(".jpg") or each_file.endswith(".png")):
		all_images_array.append(os.path.join(os.path.join(os.getcwd(),'TFWP_dev','worms'),each_file))

all_files2 = natsorted(os.listdir(os.path.join(os.getcwd(),'TFWP_dev','no_worms')))
all_images_array2 = []
for each_file in all_files2:
	if(each_file.endswith(".jpg") or each_file.endswith(".png")):
		all_images_array2.append(os.path.join(os.path.join(os.getcwd(),'TFWP_dev','no_worms'),each_file))

# for the worms
predictions_array = []
w_predicted = []
for each_file in all_images_array:
	this_img = keras.preprocessing.image.load_img(each_file, target_size=(img_height, img_width))
	this_img_array = tf.expand_dims(keras.preprocessing.image.img_to_array(this_img), 0)
	this_prediction = model.predict(this_img_array)
	predictions_array.append(this_prediction)
	score = tf.nn.softmax(this_prediction[0])
	w_predicted.append(np.array(score[1]))

# for the no_worms
predictions_array2 = []
nw_predicted = []
for each_file in all_images_array2:
	this_img = keras.preprocessing.image.load_img(each_file, target_size=(img_height, img_width))
	this_img_array = tf.expand_dims(keras.preprocessing.image.img_to_array(this_img), 0)
	this_prediction = model.predict(this_img_array)
	predictions_array2.append(this_prediction)
	score = tf.nn.softmax(this_prediction[0])
	nw_predicted.append(np.array(score[1]))


num_worms_correctly_predicted=0
for i in range(len(w_predicted)):
	if w_predicted[i] > 0.80:
		num_worms_correctly_predicted+=1
print('num_worms_correctly_predicted: ',num_worms_correctly_predicted,'/100')

num_no_worms_correctly_predicted=0
for i in range(len(nw_predicted)):
	if nw_predicted[i] < 0.80:
		num_no_worms_correctly_predicted+=1
print('num_no_worms_correctly_predicted: ',num_no_worms_correctly_predicted,'/100')


Rw = (num_worms_correctly_predicted/100) 
Rnw = (num_no_worms_correctly_predicted/100)

R = (Rw+Rnw)/2

Pw = num_worms_correctly_predicted/( (100-num_no_worms_correctly_predicted) + num_worms_correctly_predicted)
Pnw = num_no_worms_correctly_predicted/( (100-num_worms_correctly_predicted) + num_no_worms_correctly_predicted)

P = (Pw+Pnw)/2

F1 = 2*(P*R)/(P+R)

print('F1 score is:  ',round(F1,5))
print('Precision is: ',round(P,5))
print('Recall is:    ',round(R,5))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print('goodbye')
# print(predictions_array)
# model.save('saved_model/my_model.h5') 
# new_model = tf.keras.models.load_model('saved_model/my_model')