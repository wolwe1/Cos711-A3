from DataPlotting.DataPlotter import DataPlotter
from DataManagement.DataReader import DataReader
#Get the dataSets
#tf.random.set_seed(13)
dataReader = DataReader()

(trainingDataContainer,testingDataContainer) = dataReader.getContainers()

print(trainingDataContainer.getRecord(1))
# trainValues = trainingDataContainer.getRecordValues(1)
# print(trainingDataContainer.getTrainingLabels())

#Create chosen networks
networkManager = NN_Manager()
networkManager.AddNetwork("CNN")
networkManager.AddNetwork("LSTM")

#Train networks
networkManager.TrainNetworks(epochs = 100)
performance = networkManager.GetNetworkPerformance()

#Graph performance
plotter = DataPlotter()
plotter.plot(performance)