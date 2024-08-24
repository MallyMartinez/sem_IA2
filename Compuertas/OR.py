"""
Tarea 1: Compuerta AND, OR y XOR
Mally Samira Hernandez Martinez | Código: 220286113 
Ingenieria de computacion (INCO) | Seccion D05 I7041
Seminario de Solucion de Problemas de Inteligencia Artificial II
"""

import numpy as np                                                                              # Importamos las librerias NumPy para 
                                                                                                # trabajar con arrays y funciones matemáticas

class Perceptron:                                                                               # Define la clase Perceptron
    def __init__(self, input_size, learning_rate=0.1, epochs=10000):                            # Establecemos un learning rate y el numero de epocas,hperparametros
        self.W = np.zeros(input_size + 1)                                                       # Creamos el vector de pesos
        self.learning_rate = learning_rate                                                      # Define tasa de aprendizaje
        self.epochs = epochs                                                                    # Define número de épocas

    def activation_function(self,x):                                                            # Funcion de activacion
        return 1 if x >= 0 else 0                                                               # Si x es mayor o igual a 0, retorna 1; de lo contrario, retorna 0

    def predict(self, x):                                                                       # Método para realizar predicciones
        x = np.insert(x, 0, 1)                                                                  # Se establecen los valores de bias para las entradas
        z = self.W.T.dot(x)                                                                     # Se realiza el producto punto entradas y pesos
        return self.activation_function(z)                                                      # Se prueba en la funcion de activacion

    def train(self, X, y):                                                                      # Método para entrenar el perceptrón
        for _ in range(self.epochs):                                                            # Itera sobre épocas
            for inputs, label in zip(X,y):                                                      # Itera sobre datos
                prediction = self.predict(inputs)                                               # Predice
                self.W += self.learning_rate * (label - prediction) * np.insert(inputs, 0, 1)   # Actualizacion de los pesos 

    def print_weights(self):                                                                    # Método para imprimir los pesos finales
        print(f"Pesos despues del entrenamiento: {self.W}")                                     # Se imprimen los pesos
                                                                        
                                                                                                # Definimos las entradas para la compuerta lógica
X= np.array([                                                                                   # Ingresamos los datos de entrada 
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y=np.array([0, 1, 1, 1])                                                                        # Ingresamos la salida esperada para la Compuerta OR

perceptron = Perceptron(input_size=2)                                                           # Creamos una instancia del perceptrón
perceptron.train(X,y)                                                                           # Ingresamos los valores a las funciones

perceptron.print_weights()                                                                      # Imprimimos los pesos finales después del entrenamiento

for inputs in X:                                                                                # Itera sobre las entradas para hacer predicciones
    print(f"Entrada: {inputs} -> Salida predicha: {perceptron.predict(inputs)}")                # Imprimimos los resultados
