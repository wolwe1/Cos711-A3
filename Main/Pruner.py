import NeuralNetwork as NN
import DataContainer as DC
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
#import tensorflow_model_optimization as to
#from tensorflow_model_optimization.sparsity import keras as sparsity
from kerassurgeon import Surgeon
#from kerassurgeon.operations import delete_channels
from tfkerassurgeon import identify
from tfkerassurgeon.operations import delete_channels
import ModelPruner as MP
import matplotlib.pyplot as plt


def evaluateNet(evaluations,predictions) :
    losses = []
    MAEs = []
    MSEs = []

    for i in range( len(evaluations) ):
        (loss,MAE,MSE) = evaluations[i]
        losses.append(loss)
        MAEs.append(MAE)
        MSEs.append(MSE)

    print("Averages")
    print("Loss:",np.average(losses))
    print("Mean absolute error(MAE):",np.average(MAEs))
    print("Mean squared error (MSE):",np.average(MSEs))

net = NN.NeuralNetwork
data = DC.DataContainer()
neuralNet = net('relu','sgd','Large')
#neuralNetToBePruned = net('relu','sgd','Large')

evaluations = []
predictions = []

evaluationsPrune = []
predictionsPrune = []

count = 1

for i in range(count) :
    neuralNet.train()
    #neuralNetToBePruned.train()

    evaluations.append( neuralNet.evaluate() )
    #evaluationsPrune.append( neuralNetToBePruned.evaluate())

    predictions.append(neuralNet.predict() )
   # predictionsPrune.append(neuralNeTt.predict())

# print("\n\nPerformed ",count," test runs\n")

# print("\nAverage of evaluations\n")
# evaluateNet(evaluationsMedium,predictionsMedium)

#_, keras_file = tempfile.mkstemp('.h5')
#print('Saving model to: ', keras_file)
#tf.keras.models.save_model(neuralNet.model, keras_file, include_optimizer=False)
#C:\Users\jarro\AppData\Local\Temp\tmpdx9g_ub9.h5



#Prune
#modelPruner = MP.ModelPruner(neuralNet)
#modelPruner.prune()


#create pruned model
# print("OG:",evaluations)
# print("Copy:",evaluationsPrune)
# prunedNN = neuralNetToBePruned.prune()


# (original_test_loss, original_test_mae, original_test_mse) = evaluations[0]
# (test_loss, test_mae, test_mse) = prunedNN.evaluate(neuralNet.dataContainer.getTestingData(), neuralNet.dataContainer.getTestingLabels(), verbose=0)

# lossDiff = test_loss - original_test_loss
# MAELoss = test_mae - original_test_mae
# MSELoss = test_mse - original_test_mse
# print("\nPruned model performance:")
# print("Loss:",test_loss," - Diff ",lossDiff)
# print("MAE:",test_mae," - Diff ",MAELoss)
# print("MSE:",test_mse," - Diff ",MSELoss)

# #strip model
# final_model = sparsity.strip_pruning(prunedNN)
# neuralNet.model.summary()
# final_model.summary()

pruneEvaluations = []
pruneEvaluations.append( evaluations[0])
# print("Original ",neuralNet.model.layers[0].get_weights()[0])
# print("Final ",final_model.layers[0].get_weights()[0])
def pruneLayer(layerName):
    layer = neuralNet.model.get_layer(layerName)

    apoz = identify.get_apoz(neuralNet.model,layer,data.getTrainingData().values)
    high_apoz_channels = identify.high_apoz(apoz)
    prunedNN = delete_channels(neuralNet.model,layer,high_apoz_channels)

    return prunedNN
# New style
def pruneModel():
    try:
        neuralNet.model = pruneLayer("dense")
        neuralNet.model = pruneLayer("dense_1")
        neuralNet.model = pruneLayer("dense_2")
    except :
        print("layer couldnt be pruned")
    
    neuralNet.model.summary()

    (original_test_loss, original_test_mae, original_test_mse) = evaluations[0]
    neuralNet.compile()
    neuralNet.train()
    print("\nPost training")
    (test_loss, test_mae, test_mse) = neuralNet.evaluate()
    pruneEvaluations.append((test_loss, test_mae, test_mse))
    
    lossDiff = test_loss - original_test_loss
    MAELoss = test_mae - original_test_mae
    MSELoss = test_mse - original_test_mse
    print("\nPruned model performance:")
    print("Loss:",test_loss," - Diff ",lossDiff)
    print("MAE:",test_mae," - Diff ",MAELoss)
    print("MSE:",test_mse," - Diff ",MSELoss)
    
print("\nOriginal Model:\n")
neuralNet.model.summary()

for i in range(15) :
    pruneModel()
    neuralNet.model.save("savedModels/prune_model_{}".format(i))

MSEValues = []
for i in range( len( pruneEvaluations) ) :
    (test_loss, test_mae, test_mse) = pruneEvaluations[i]
    MSEValues.append(test_mse)

plt.plot( list(range(0,len(pruneEvaluations)) ),MSEValues)
plt.ylabel('Prune model MSE')
plt.xlabel("Prune Number")
plt.show()