#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Santiago Sánchez Muñoz
@date: 1 de noviembre de 2020
"""

####################################
###   Redes Neuronales - MNIST   ###
####################################

""" Librerías """

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

""" Variables globales """

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

""" Uso la notación Snake Case la cual es habitual en Python """

def read_data():
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	return (x_train, y_train), (x_test, y_test)

def show_data(x_train):
	# plot first few images
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# plot raw pixel data
		pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
	# show the figure
	pyplot.show()

def preprocess_data(x_train, y_train, x_test, y_test):
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

def summarize_dataset(x_train, y_train, x_test, y_test):
	print('Dimensión de x_train:', x_train.shape, 'y_train:', y_train.shape)
	print('Dimensión de x_test:', x_test.shape, 'y_test:', y_test.shape)
	print(x_train.shape[0], 'ejemplos de entrenamiento')
	print(x_test.shape[0], 'ejemplos de test\n')

def construc_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

def train_model(model, x_train, y_train, x_test, y_test):
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))

def evaluate(model, x_train, y_train, x_test, y_test):
	return model.evaluate(x_train, y_train, verbose=0), model.evaluate(x_test, y_test, verbose=0)

def print_score(score_train, score_test):
	print('Train loss:', score_train[0])
	print('Train accuracy:', score_train[1])
	print('Test loss:', score_test[0])
	print('Test accuracy:', score_test[1])


########################
#####     MAIN     #####
########################


""" Programa principal. """
def main():
	print("\n###############################################")
	print("###  PRÁCTICA 1 IC - Redes Neuronales: MNIST  ###")
	print("#################################################\n")

	# Leyendo datos de entrenamiento y test
	(x_train, y_train), (x_test, y_test) = read_data()
	summarize_dataset(x_train, y_train, x_test, y_test)

	#show_data(x_train)

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	preprocess_data(x_train, y_train, x_test, y_test)
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = construc_model(input_shape)
	train_model(model, x_train, y_train, x_test, y_test)
	score_train, score_test = evaluate(model, x_train, y_train, x_test, y_test)
	print_score(score_train, score_test)
	y_pred = model.predict(x_test)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	plot_confusion_matrix(model, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
	pyplot.show()
	plot_confusion_matrix(model, x_test, y_test,
                                 cmap=plt.cm.Blues)
	pyplot.show()

if __name__ == "__main__":
	main()
