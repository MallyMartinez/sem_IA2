"""
Practica 1: Perceptrón Multicapa
Mally Samira Hernandez Martinez | Código: 220286113 
Ingenieria de computacion (INCO) | Seccion D05 I7041
Seminario de Solucion de Problemas de Inteligencia Artificial II
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Cargar los datos
train = pd.read_csv('MINIST/train.csv')
test = pd.read_csv('MINIST/test.csv')

# Separar las etiquetas de las imágenes
X_train = train.iloc[:, 1:].values  # Las imágenes (características)
y_train = train.iloc[:, 0].values   # Las etiquetas

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Normalizar los datos (escala de 0 a 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convertir las etiquetas a una representación categórica (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir el modelo MLP
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Aplanar las imágenes de 28x28
model.add(Dense(128, activation='relu'))  # Capa oculta con 128 neuronas
model.add(Dense(10, activation='softmax'))  # Capa de salida (10 clases)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en el set de prueba: {test_acc:.4f}")
