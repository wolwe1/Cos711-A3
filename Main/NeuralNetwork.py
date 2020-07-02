from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pathlib
import time
import datetime


class NeuralNetwork:

    def __init__(self,activationFunction,optimiser,size,name,patience = 10):

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
        
        #Time it takes for training
        startTime = time.time()

        early_history = self.model.fit(
            trainingData,
            trainingLabels,
            epochs=epochs,
            validation_data = (testingData,testingLabels),
            verbose=0,
            callbacks=[ self.tensorboard_callback,self.early_stop],
            )
        
        endTime = time.time()
        timeTaken = endTime - startTime

        return (early_history.history["mae"],early_history.history["mse"],timeTaken)

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

    def buildModel(self) :
        inputShape = (121,6)

        self.model = keras.Sequential([
            keras.layers.LSTM(100, input_shape=inputShape, return_sequences=False),
            #keras.layers.LSTM(4, return_sequences=True),
            #keras.layers.LSTM(4, return_sequences=True),
            #keras.layers.LSTM(4),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()

class StackedLSTM(NeuralNetwork) :
    pass

    def buildModel(self) :
        inputShape = (121,6)

        self.model = keras.Sequential([
            keras.layers.LSTM(30, input_shape=inputShape, return_sequences=True),
            keras.layers.LSTM(20,return_sequences=True),
            keras.layers.LSTM(10),
            keras.layers.Dense(20,activation=self.activationFunction),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()
    
class GatedRecurrentUnitNetwork(NeuralNetwork) :
    pass

    def buildModel(self) :
        inputShape = (121,6)

        self.model = keras.Sequential([
            keras.layers.GRU(30, input_shape=inputShape, return_sequences=True,activation=self.activationFunction),
            keras.layers.GRU(20, return_sequences=True,activation=self.activationFunction),
            keras.layers.GRU(10,activation=self.activationFunction),
            keras.layers.Dense(20,activation=self.activationFunction),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()

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

            # fourth layer
            keras.layers.Conv1D(filters=144, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),
            
            keras.layers.Flatten(),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()

class LargerFilterCNN(NeuralNetwork) :
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

            # fourth layer
            keras.layers.Conv1D(filters=144, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),
            
            keras.layers.Flatten(),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()

class LargeCNN(NeuralNetwork) : 
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

            # fourth layer
            keras.layers.Conv1D(filters=144, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),
            
            keras.layers.Conv1D(filters=288, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),

            keras.layers.Flatten(),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()
   
class LargestCNN(NeuralNetwork) : 
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

            # fourth layer
            keras.layers.Conv1D(filters=144, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),
            
            keras.layers.Conv1D(filters=288, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),

            keras.layers.Conv1D(filters=572, kernel_size=2,strides=1, padding='SAME', activation=tf.nn.relu),
            keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='SAME'),

            keras.layers.Flatten(),
            keras.layers.Dense(1,activation=self.activationFunction)
        ])
        return super().buildModel()

class FeedForwardNN(NeuralNetwork) :
    pass

    #Overload to reshape the data
    def train(self,trainingData,trainingLabels,testingData,testingLabels,epochs):
        #Datasets are (n,121,6)
        flattenedTrainingData = trainingData.reshape((len(trainingData),726),order = 'F')
        flattenedTestingData = testingData.reshape((len(testingData),726),order = 'F')
        return super().train(flattenedTrainingData,trainingLabels,flattenedTestingData,testingLabels,epochs)

        


    def buildModel(self):
        #Flat version of the 2-D arrays
        inputShape = (726,)

        self.model = keras.Sequential(
                [
                    keras.layers.Dense(448,activation=self.activationFunction, input_shape=inputShape,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.3),

                    keras.layers.Dense(224, activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.2),

                    keras.layers.Dense(112,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(64,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(32,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(1,activation=self.activationFunction)
                ])
        return super().buildModel()