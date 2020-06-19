from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pathlib

import datetime


class NeuralNetwork:

    def __init__(self,activationFunction,optimiser,size,name,patience = 5):

        self.activationFunction = activationFunction
        self.optimiser = optimiser
        self.size = size
        self.name = name
        self.patience = patience

        self.model = None
  # For tensorboard

        self.log_dir = 'logs\\fit\\'+ self.name + "\\" \
            + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        self.early_stop  = keras.callbacks.EarlyStopping(monitor='loss',patience = self.patience)
        

    def buildModel(self):

        self.compile()

        return self.model

    def compile(self):

        if self.optimiser == 'sgd' :
            optimiser = keras.optimizers.SGD()
        elif self.optimiser == 'rmsprop' :
            optimiser = tf.keras.optimizers.RMSprop(0.001)
        else : 
            optimiser = 'adam'

        self.model.compile(optimizer=optimiser,
                           loss= 'mse',
                           metrics=['mae', 'mse'])

    def train(self,trainingData,trainingLabels,testingData,testingLabels,epochs):
        
       
        early_history = self.model.fit(
            trainingData,
            trainingLabels,
            epochs=epochs,
            validation_data = (testingData,testingLabels),
            verbose=0,
            callbacks=[ self.tensorboard_callback,self.early_stop],
            )
        return (early_history.history["loss"],early_history.history["mae"],early_history.history["mse"])

    def evaluate(self,testingData,testingLabels):
      
        (test_loss, test_mae, test_mse) = self.model.evaluate(testingData,testingLabels, verbose=2)

        return (test_loss, test_mae, test_mse)

    def predict(self,testingData):

        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

        predictions = probability_model.predict(testingData)

        return predictions

        
#Neural network variants

class LSTM(NeuralNetwork) :
    pass

    # def buildModel(self) :
    #     self.model = [
    #         tf.keras.layers.LSTMCell
    #     ]

class CNN(NeuralNetwork) :
    pass

    def buildModel(self):
        # The inputs are 6-length vectors with 121 timesteps, and the batch size of None
        inputShape = (121,6)

        self.model = keras.Sequential([
            keras.layers.Conv1D(input_shape=inputShape, filters=18, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),
            # second layer
            keras.layers.Conv1D(filters=36, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),

            # third layer
            keras.layers.Conv1D(filters=72, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),

            # second layer
            keras.layers.Conv1D(filters=144, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME')
            
        ])
        return super().buildModel()

    

class FeedForwardNN(NeuralNetwork) :
    pass

    def buildModel(self):
        inputShape = (1,6)

        self.model = keras.Sequential(
                [
                    keras.layers.Dense(24,activation=self.activationFunction, input_shape=[inputShape],kernel_regularizer=keras.regularizers.l2(0.0001)),
                    #keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),

                    keras.layers.Dense(24, activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.2),

                    keras.layers.Dense(12,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.05),

                    keras.layers.Dense(1,activation=self.activationFunction)
                ])
        return super().buildModel()