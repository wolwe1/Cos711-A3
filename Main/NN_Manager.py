import NeuralNetwork as NN
class NN_Manager :

    def __init__(self,trainingData,testingData,trainingLabels,testingLabels):
        self.networks = []
        self.trainingData = trainingData
        self.testingData = testingData

        self.trainingLabels = trainingLabels
        self.testingLabels = testingLabels

        #Store a list of each networks performance as performance summaries
        self.networkTrainingPerformanceHistory = []

        self.networkTestingPerformanceHistory = []

    def trainNetworks(self) :
        for i in range( len(self.networks) ) :
            (mae,mse,timeToTrain) = self.networks[i].train(self.trainingData,self.trainingLabels,
                        self.testingData,self.testingLabels,100)

            self.networkTrainingPerformanceHistory.append(
                self.createNetworkPerformanceSummary(mae,mse,self.networks[i].name,timeToTrain)
            )

    def evaluateNetworks(self) :

        for i in range( len(self.networks)) :
            
            self.networkTrainingPerformanceHistory.append(
                self.networks[i].evaluate(self.testingData,self.testingLabels)
            )

    def createNetworkPerformanceSummary(self,mae,mse,networkName,timeToTrain) :
        
        networkSummary = {
            "name" : networkName,
            "timeToTrain" : timeToTrain,
            "epochs" : list(range(len(mse))),
            "evaluations" : []
            }
        networkSummaries = []
        #Create MAE summary
        summary1 = {
            "metric":"mae",
            "performance": mae
        }
        
        networkSummaries.append(summary1)

        #Create MSE summary
        summary2 = {
            "metric" : "mse",
            "performance": mse
        }
        
        networkSummaries.append(summary2)

        networkSummary["evaluations"] = networkSummaries

        return networkSummary

        

    def testNetworks(self) :
        for i in range( len(self.networks) ) :
            self.networkTestingPerformanceHistory.append( self.networks[i].test())


    def GetNetworkPerformance(self) :
        return self.networkTrainingPerformanceHistory

    def addNetwork(self,networkName) :

        if networkName == "CNN" :
            network = NN.CNN("relu","adam","medium","CNN")
            network.buildModel()
            self.networks.append(network)

        elif  networkName == "LargeCNN" :
            network = NN.LargeCNN("relu","adam","medium","Larger CNN")
            network.buildModel()
            self.networks.append(network)

        elif  networkName == "LargerFilterCNN" :
            network = NN.LargerFilterCNN("relu","adam","medium","Larger filter CNN")
            network.buildModel()
            self.networks.append(network)

        elif  networkName == "LSTM" :
            network = NN.LSTM("relu","adam","medium","LSTM")
            network.buildModel()
            self.networks.append(network)

        elif  networkName == "StackedLSTM" :
            network = NN.StackedLSTM("relu","adam","medium","Stacked LSTM")
            network.buildModel()
            self.networks.append(network)

        elif  networkName == "GRU" :
            network = NN.GatedRecurrentUnitNetwork("relu","adam","medium","GRU")
            network.buildModel()
            self.networks.append(network)

        else :
            network = NN.FeedForwardNN("relu","adam","medium","FFNN")
            network.buildModel()
            self.networks.append(network)

        