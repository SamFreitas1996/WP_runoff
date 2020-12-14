import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import numpy as np
import PIL
from natsort import natsorted
from matplotlib import image

import tensorflow as tf
import kerastuner as kt

import IPython

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# data for model
data_dir2 = pathlib.Path(os.path.join(os.getcwd(),'TFWP_training'))

# input variables
batch_size = 32
img_height = 192
img_width = 192

# segmenting the training and validation datasets
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

# class names 
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

# autotune data
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model and model parameters

data_augmentation = keras.Sequential(
	[
	layers.experimental.preprocessing.RandomFlip("horizontal", 
												 input_shape=(img_height, 
															  img_width,
															  3)),
	layers.experimental.preprocessing.RandomRotation(0.2),
	layers.experimental.preprocessing.RandomZoom(0.2),
	]
)

# model builder 
def model_builder(hp):

	# these are standards 
	model = keras.Sequential()
	model.add(data_augmentation)
	model.add(layers.experimental.preprocessing.Rescaling(1./255))

	# hyperparameters 
	hp_filter1 = hp.Int('filters', min_value = 2, max_value = 64, step = 4)
	hp_filter2 = hp.Int('filters', min_value = 2, max_value = 64, step = 4)
	hp_filter3 = hp.Int('filters', min_value = 2, max_value = 64, step = 4)
	hp_kernal_size = hp.Int('kernel_size', min_value = 1, max_value = 5, step = 1)

	# layers that use the hyperparameters
	model.add(layers.Conv2D(hp_filter1, hp_kernal_size, padding='same', activation='relu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(hp_filter2,hp_kernal_size, padding='same', activation='relu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(hp_filter3,hp_kernal_size, padding='same', activation='relu'))
	model.add(layers.MaxPooling2D())

	# output layers
	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(num_classes,activation='sigmoid'))

	hp_lr = hp.Choice('learning_rate',values = [1e-2,1e-3,1e-4])

	model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_lr),
					loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])

	return(model)

# create the tuner object
tuner = kt.Hyperband(model_builder,
					objective = 'val_accuracy',
					max_epochs=25,
					factor=3,
					directory='tuner_dir',
					project_name='WP_runoff')

# create early stop callback 
es_Callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',patience=3,restore_best_weights=True)

# clear any previous output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
	def on_train_end(*args, **kwargs):
		IPython.display.clear_output(wait = True)

# tune the model with tuner.search()
tuner.search(train_ds,
			validation_data = val_ds, 
			epochs = 3, 
			callbacks = [ClearTrainingOutput(),es_Callback])


# get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# print the best parameters
print(best_hps)

# reload in the model 
model = tuner.hypermodel.build(best_hps)

# show the summary 
model.summary()

# train the new model completely 
epochs = 25
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs,
	callbacks=[es_Callback]
	# callbacks=[cp_callback]
)
# save and load back in the model weights 
model.save('saved_model/my_model_hp.h5') 
model.load_weights('saved_model/my_model_hp.h5')

# load in the dev data 
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

# create predictions on the dev data for worms
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

# calculate the number of correctly predicted worms and no_worms
num_worms_correctly_predicted=0
for i in range(len(w_predicted)):
	if w_predicted[i] > 0.70:
		num_worms_correctly_predicted+=1
print('num_worms_correctly_predicted: ',num_worms_correctly_predicted,'/100')

num_no_worms_correctly_predicted=0
for i in range(len(nw_predicted)):
	if nw_predicted[i] < 0.80:
		num_no_worms_correctly_predicted+=1
print('num_no_worms_correctly_predicted: ',num_no_worms_correctly_predicted,'/100')

# calcualte the F1 value from the recalled and precision values
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

print('end')
