from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pathlib

import datetime

import DataContainer as DC

import PrintUtils

Plotter = PrintUtils.PrintUtils()

import matplotlib.pyplot as plt


#import tensorflow_model_optimization as to
#from tensorflow_model_optimization.sparsity import keras as sparsity

_dataContainer = DC.DataContainer()


class NeuralNetwork:

    def __init__(self,activationFunction,optimiser,size):

        self.dataContainer = _dataContainer
        self.activationFunction = activationFunction
        self.optimiser = optimiser
        self.size = size

        self.testing_data = self.dataContainer.getTestingData()
        self.testing_labels = self.dataContainer.getTestingLabels().values
        self.training_data = self.dataContainer.getTrainingData()
        self.training_labels = self.dataContainer.getTrainingLabels().values

        self.epochs = 100

        self.model = self.buildModel()

#     def __init__(self,model):

#         self.dataContainer = model.dataContainer
#         self.activationFunction = model.activationFunction
#         self.optimiser = model.optimiser
#         self.size = model.size

#         self.testing_data = self.dataContainer.getTestingData()
#         self.testing_labels = self.dataContainer.getTestingLabels().values
#         self.training_data = self.dataContainer.getTrainingData()
#         self.training_labels = self.dataContainer.getTrainingLabels().values

#         self.epochs = 100

#         self.model = model.model
        

  # For tensorboard

        self.log_dir = 'logs\\fit\\' \
            + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        self.early_stop  = keras.callbacks.EarlyStopping(monitor='loss',patience=5)
        

        

    def buildModel(self):
        inputDataShape = len(self.dataContainer.getColumnNames())-2 #-2 for quality and 

        if self.size == 'Medium' :
            self.model = keras.Sequential(
                [
                    keras.layers.Dense(24,activation=self.activationFunction, input_shape=[inputDataShape],kernel_regularizer=keras.regularizers.l2(0.0001)),
                    #keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),

                    keras.layers.Dense(24, activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.2),

                    keras.layers.Dense(12,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.05),

                    keras.layers.Dense(1,activation=self.activationFunction)
                ])
        elif self.size == 'Large' :
            self.model = keras.Sequential(
               [
                    keras.layers.Dense(512,activation=self.activationFunction, input_shape=[inputDataShape],kernel_regularizer=keras.regularizers.l2(0.0001)),
                    #keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),

                    keras.layers.Dense(256, activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.2),

                    keras.layers.Dense(128,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.05),

                    keras.layers.Dense(1,activation=self.activationFunction)
                ])
        elif self.size == "Best" :
            self.model = keras.Sequential(
                [
                    keras.layers.Dense(168,activation=self.activationFunction, input_shape=[inputDataShape],kernel_regularizer=keras.regularizers.l2(0.0001)),
                    #keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),

                    keras.layers.Dense(74, activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.2),

                    keras.layers.Dense(46,activation=self.activationFunction,kernel_regularizer=keras.regularizers.l2(0.0001)),
                    keras.layers.Dropout(0.05),

                    keras.layers.Dense(1,activation=self.activationFunction)
                ]
            )
            

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

    def train(self):
        
        early_history = self.model.fit(
            self.training_data,
            self.training_labels,
            epochs=self.epochs,
            validation_data=(self.testing_data,self.testing_labels),
            #validation_split=0.2,
            verbose=0,
            callbacks=[ self.tensorboard_callback,self.early_stop],
            )

    def evaluate(self):
      
        (test_loss, test_mae, test_mse) = self.model.evaluate(self.testing_data,
                                                                self.testing_labels, verbose=2)

        print('Testing set accuracy: {:5.2f} points'.format(test_mae))

        print('\nTest loss:', test_loss)

        return (test_loss, test_mae, test_mse)

    def predict(self):
        print('''Making predictions''')

  # probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

  # predictions = probability_model.predict(self.testing_data)

        test_predictions = self.model.predict(self.testing_data).flatten()

  # Plotter.plot_value_array(i=1, predictions_array=predictions[0], true_label =self.testing_labels)
  # _ = plt.xticks(range(10), range(10), rotation=45)
  # plt.show()

        # a = plt.axes(aspect='equal')
        # plt.scatter(self.testing_labels, test_predictions)
        # plt.xlabel('True Values [Point]')
        # plt.ylabel('Predictions [Points]')
        # lims = [0, 10]
        # plt.xlim(lims)
        # plt.ylim(lims)
        # _ = plt.plot(lims, lims)

        #plt.show()

        # error = test_predictions - self.testing_labels
        # plt.hist(error, bins=25)
        # plt.xlabel('Prediction Error [Points]')
        # _ = plt.ylabel('Count')

        #plt.show()

        return test_predictions

    def prune(self) :
        batch_size = 32
        num_train_samples = self.training_data.shape[0]
        end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * self.epochs
        print(end_step)

        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                        final_sparsity=0.90,
                                                        begin_step=0,
                                                        end_step=end_step,
                                                        frequency=100)
        }

                
        new_pruned_model = sparsity.prune_low_magnitude(self.model, **pruning_params)


        optimiser = keras.optimizers.SGD(learning_rate=0.03,momentum=0.01, nesterov=False)

        new_pruned_model.compile(optimizer=optimiser,
                    loss= 'mse',
                    metrics=['mae', 'mse'])

        #retrain & evaluate pruned model

        new_pruned_model.fit(self.training_data, self.training_labels,
                batch_size=batch_size,
                epochs=self.epochs,
                verbose=0,
                validation_data=(self.testing_data, self.testing_labels),
                callbacks = [sparsity.UpdatePruningStep(),self.early_stop])

        return new_pruned_model

        
