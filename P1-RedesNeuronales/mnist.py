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
from keras.layers import Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import numpy as np
from time import time

""" Variables globales """

BATCH_SIZE = 128			# tamaño del batch
N_CLASSES = 10				# número de clases
EPOCHS = 15					# épocas
IMG_ROWS, IMG_COLS = 28, 28	# Dimensiones de las imágenes
SHOW_IMGS = True			# Indica si se quiere imprimir algunas imágenes
SHOW_CONFUSSION = True		# Indica si se quiere imprimir algunas imágenes

""" Uso la notación Snake Case la cual es habitual en Python """

""" Lectura de datos
- fashion: indica si se debe leer mnist (False) o fashion_mnist (True).
"""
def read_data(fashion=False):
	# divide los datos en train y test
	if(fashion):
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	return (x_train, y_train), (x_test, y_test)

""" Muestra 15 imágenes y su etiqueta
- x: imagen.
- y: etiqueta.
- set_init: primera imagen a mostrar.
"""
def show_data(x, y, set_init=0):
	fig = plt.figure(figsize=(15, 10))
	for id in np.arange(15):	# sólo imprimo 15
	    ax = fig.add_subplot(3, 5, id+1, xticks=[], yticks=[])
	    ax.imshow(x[id + set_init], cmap=plt.cm.binary)
	    ax.set_title(str(y[id + set_init]))
	plt.gcf().canvas.set_window_title('IC - Práctica 1')
	plt.show()	# pintamos la imagen

""" Preprocesa los datos de entrada
- x_train: entrada del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- x_test: entrada del conjunto de test.
- y_test: etiquetas del conjunto de test.
"""
def summarize_dataset(x_train, y_train, x_test, y_test):
	print('Dimensión de x_train:', x_train.shape, 'y_train:', y_train.shape)
	print('Dimensión de x_test:', x_test.shape, 'y_test:', y_test.shape)
	print(x_train.shape[0], 'ejemplos de entrenamiento')
	print(x_test.shape[0], 'ejemplos de test\n')

""" Preprocesa los datos de entrada
- x_train: entrada del conjunto de entrenamiento.
- x_test: entrada del conjunto de test.
"""
def preprocess_data(x_train, x_test):
	x_train = x_train.astype('float32')	# conversión a float32
	x_test = x_test.astype('float32')	# conversión a float32
	x_train /= 255						# normalización a [0,1]
	x_test /= 255						# normalización a [0,1]

""" Construcción del modelo. Devuelve el modelo.
- input_shape: tamaño del input.
"""
def construc_model1(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.SGD(lr=0.01),
	              metrics=['accuracy'])
	return model

""" Construcción del modelo. Devuelve el modelo.
- input_shape: tamaño del input.
"""
def construc_model2(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

""" Construcción del modelo. Devuelve el modelo.
- input_shape: tamaño del input.
"""
def construc_model3(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

""" Construcción del modelo. Devuelve el modelo.
- input_shape: tamaño del input.
"""
def construc_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
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

""" Construcción del modelo. Devuelve el modelo.
- input_shape: tamaño del input.
"""
def construc_model_LeakyReLu(input_shape):
	model = Sequential()

	model.add(Conv2D(32, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

""" Entrena el modelo. Devuelve el history.
- model: modelo.
- x_train: entrada del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- x_test: entrada del conjunto de test.
- y_test: etiquetas del conjunto de test.
"""
def train_model(model, x_train, y_train, x_test, y_test):
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
	return model.fit(x_train, y_train,
	          batch_size=BATCH_SIZE,
	          epochs=EPOCHS,
			  validation_data = (x_val, y_val),
	          verbose=1)

""" Muestra la historia del entrenamiento (acc y loss).
- history: historia.
"""
def show_history(history):
	# Accuracy
	plt.plot(history.history['accuracy'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.title('model accuracy')
	plt.gcf().canvas.set_window_title('IC - Práctica 1')
	plt.show()
	# Loss
	plt.plot(history.history['loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.title('model loss')
	plt.gcf().canvas.set_window_title('IC - Práctica 1')
	plt.show()

""" Evalúa el modelo. Devuelve los scores.
- model: modelo.
- x_train: entrada del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- x_test: entrada del conjunto de test.
- y_test: etiquetas del conjunto de test.
"""
def evaluate(model, x_train, y_train, x_test, y_test):
	return model.evaluate(x_train, y_train, verbose=0), model.evaluate(x_test, y_test, verbose=0)

""" Imprime los scores.
- score_train: score del conjunto de entrenamiento.
- score_test: score del conjunto de test.
"""
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

""" Invierte la función to_categorical
- y_categorical: matriz binaria a revertir.
"""
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

""" Escribe las predicciones en un archivo externo "preds.txt".
- y_pred: etiquetas predichas.
"""
def write_predictions(y_pred):
	f=open("preds.txt","w")	# w borra el contenido si lo hay
	for i in y_pred:
		f.write(str(i))
	f.close()	# cerramos el fichero

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
		show_data(x_train, y_train)

	# Preprocesamiento
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
	    x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
	    input_shape = (1, IMG_ROWS, IMG_COLS)
	else:
	    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
	    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
	    input_shape = (IMG_ROWS, IMG_COLS, 1)

	preprocess_data(x_train, x_test)

	# Convirtiendo el vector de etiquetas a una matriz binaria
	y_train_categorical = keras.utils.to_categorical(y_train, N_CLASSES)
	y_test_categorical = keras.utils.to_categorical(y_test, N_CLASSES)

	# Construcción del modelo e información de las capas
	model = construc_model(input_shape)
	model.summary()

	# Entrenamiento del modelo
	start_time = time()
	history = train_model(model, x_train, y_train_categorical, x_test, y_test_categorical)
	elapsed_time = time() - start_time
	print("\nTiempo de entenamiento: {:.2f} s".format(elapsed_time))
	show_history(history)

	# Evaluación del modelo
	score_train, score_test = evaluate(model, x_train, y_train_categorical, x_test, y_test_categorical)
	print_score(score_train, score_test)

	# Imprimiendo algunos datos
	if(SHOW_IMGS):
		show_data(x_test_orig, y_test)

	# Predicciones
	y_pred_categorical = model.predict(x_test)
	y_pred = inverse_categorical(y_pred_categorical)
	print("Etiquetas predecidas: ", np.array(y_pred))
	write_predictions(y_pred)

	# Matriz de confusión
	print(confusion_matrix(y_test, y_pred))
	if(SHOW_CONFUSSION):
		show_confussion_matrix(y_test, y_pred, "sin normalizar", False)
		show_confussion_matrix(y_test, y_pred, "normalizada")


if __name__ == "__main__":
	main()
