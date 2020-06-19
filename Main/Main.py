from DataReader import DataReader
from NN_Manager import NN_Manager
#Get the dataSets
#tf.random.set_seed(13)
dataReader = DataReader()

(trainingDataSet,testingDataSet) = dataReader.GetDataSets()
(trainingLabels,testingLabels) = dataReader.GetLabels()

record = trainingDataSet[0]
feature = record[0]
entry = feature[0]

#Create chosen networks
networkManager = NN_Manager(trainingDataSet,testingDataSet,trainingLabels,testingLabels)
networkManager.addNetwork("CNN")
#networkManager.addNetwork("LSTM")

# #Train networks
networkManager.trainNetworks()
#networkManager.evaluateNetworks()
performance = networkManager.GetNetworkPerformance()

plotter = DataPlotter()
plotter.plot(performance[0])
plotter.show()
# #Graph performance
# plotter = DataPlotter()
# plotter.plot(performance)