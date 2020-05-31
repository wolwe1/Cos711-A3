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

        self.model = self.buildModel()
    
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

    def train(self,trainingData,trainingLabels,split):
        
        early_history = self.model.fit(
            trainingData,
            trainingLabels,
            epochs=self.epochs,
            validation_split=split,
            verbose=0,
            callbacks=[ self.tensorboard_callback,self.early_stop],
            )

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

class FeedForwardNN(NeuralNetwork) :
    pass