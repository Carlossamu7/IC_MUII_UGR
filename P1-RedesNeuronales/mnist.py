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
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tabulate import tabulate
import numpy as np

""" Variables globales """

BATCH_SIZE = 128			# tamaño del batch
N_CLASSES = 10				# número de clases
EPOCHS = 12					# épocas
IMG_ROWS, IMG_COLS = 28, 28	# Dimensiones de las imágenes
SHOW_IMGS = False			# Indica si se quiere imprimir algunas imágenes
SHOW_CONFUSSION = False		# Indica si se quiere imprimir algunas imágenes

""" Uso la notación Snake Case la cual es habitual en Python """

def read_data(fashion=False):
	# divide los datos en train y test
	if(fashion):
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	return (x_train, y_train), (x_test, y_test)

def show_data(x, set_init=0):
	for i in range(9):	# sólo imprimo las primeras
		plt.subplot(330 + 1 + i)
		plt.imshow(x[i+set_init], cmap=plt.get_cmap('gray'))
	# pintamos la imagen
	plt.title("MNIST")
	plt.gcf().canvas.set_window_title('IC - Práctica 1')
	plt.show()

def preprocess_data(x_train, y_train, x_test, y_test):
	x_train = x_train.astype('float32')	# conversión a float32
	x_test = x_test.astype('float32')	# conversión a float32
	x_train /= 255						# normalización a [0,1]
	x_test /= 255						# normalización a [0,1]

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
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

def construc_model2(input_shape):
	model = Sequential()

	model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(64))

	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	return model

def construc_model3(input_shape):
	model = Sequential()

	model.add(Conv2D(32, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

def train_model(model, x_train, y_train, x_test, y_test):
	model.fit(x_train, y_train,
	          batch_size=BATCH_SIZE,
	          epochs=EPOCHS,
	          verbose=1,
	          validation_data=(x_test, y_test))

def evaluate(model, x_train, y_train, x_test, y_test):
	return model.evaluate(x_train, y_train, verbose=0), model.evaluate(x_test, y_test, verbose=0)

def print_score(score_train, score_test):
	print("\n----------  RESULTADOS  ----------")
	# Sobre el conjunto de entrenamiento
	print("Train loss: {:.6f}".format(score_train[0]))
	print("Train accuracy: {:.6f}".format(score_train[1]))
	# Sobre el conjunto de test
	print("Test loss: {:.6f}".format(score_test[0]))
	print("Test accuracy: {:.6f}".format(score_test[1]))
	print("")
	table = [["Train", "{:.6f} %".format(100*score_train[1]), "{:.6f} %".format(100*score_train[0])],
			 ["Test", "{:.6f} %".format(100*score_test[1]), "{:.6f} %".format(100*score_test[0])]]
	print(tabulate(table, headers=["Conjunto", "Accuracy", "Loss"], tablefmt='fancy_grid'))
	print("")

def inverse_categorical(y_categorical):
	y = []
	for i in range(len(y_categorical)):
		y.append(np.argmax(np.round(y_categorical[i])))
	return y

""" Muestra matriz de confusión.
- y_real: etiquetas reales.
- y_pred: etiquetas predichas.
- message: mensaje que complementa la matriz de confusión.
- norm (op): indica si normalizar (dar en %) la matriz de confusión. Por defecto 'True'.
"""
def show_confussion_matrix(y_real, y_pred, message="", norm=True):
	mat = confusion_matrix(y_real, y_pred)
	if(norm):
		mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	ax.matshow(mat, cmap="GnBu")
	ax.set(title="Matriz de confusión {}".format(message),
		   xticks=np.arange(10), yticks=np.arange(10),
		   xlabel="Etiqueta", ylabel="Predicción")

	for i in range(10):
		for j in range(10):
			if(norm):
				ax.text(j, i, "{:.0f}%".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")
			else:
				ax.text(j, i, "{:.0f}".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#################################################")
	print("###  PRÁCTICA 1 IC - Redes Neuronales: MNIST  ###")
	print("#################################################\n")

	# Lectura de datos de entrenamiento y test
	(x_train, y_train), (x_test, y_test) = read_data()
	summarize_dataset(x_train, y_train, x_test, y_test)
	x_test_orig = x_test.copy()
	if(SHOW_IMGS):
		show_data(x_train)

	# Preprocesamiento
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
	    x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
	    input_shape = (1, IMG_ROWS, IMG_COLS)
	else:
	    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
	    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
	    input_shape = (IMG_ROWS, IMG_COLS, 1)

	preprocess_data(x_train, y_train, x_test, y_test)

	# Convirtiendo el vector de etiquetas a una matriz binaria
	y_train_categorical = keras.utils.to_categorical(y_train, N_CLASSES)
	y_test_categorical = keras.utils.to_categorical(y_test, N_CLASSES)

	# Construcción del modelo e información de las capas
	model = construc_model3(input_shape)
	model.summary()

	# Entrenamiento del modelo
	train_model(model, x_train, y_train_categorical, x_test, y_test_categorical)

	# Evaluación del modelo
	score_train, score_test = evaluate(model, x_train, y_train_categorical, x_test, y_test_categorical)
	print_score(score_train, score_test)

	# Imprimiendo algunos datos
	if(SHOW_IMGS):
		show_data(x_test_orig)

	# Predicciones
	y_pred_categorical = model.predict(x_test)
	y_pred = inverse_categorical(y_pred_categorical)
	print("Etiquetas predecidas: ", np.array(y_pred))

	# Matriz de confusión
	print(confusion_matrix(y_test, y_pred))
	if(SHOW_CONFUSSION):
		show_confussion_matrix(y_test, y_pred, "sin normalizar", False)
		show_confussion_matrix(y_test, y_pred, "normalizada")

if __name__ == "__main__":
	main()
