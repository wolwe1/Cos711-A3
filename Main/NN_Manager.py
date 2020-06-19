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
            (loss,mae,mse) = self.networks[i].train(self.trainingData,self.trainingLabels,
                        self.testingData,self.testingLabels,50)
            self.networkTrainingPerformanceHistory.append(
                self.createNetworkPerformanceSummary(loss,mae,mse,self.networks[i].name)
            )

    def evaluateNetworks(self) :

        for i in range( len(self.networks)) :
            
            self.networkTrainingPerformanceHistory.append(
                self.networks[i].evaluate(self.testingData,self.testingLabels)
            )

    def createNetworkPerformanceSummary(self,loss,mae,mse,networkName) :
        
        #Create loss summary
        lossSummary = NPS("loss")
        summary = {
            "name": networkName,
            "epochs": range(len(loss)),
            "performance": loss
        }
        
        lossSummary.addSummary(summary)

        #Create MAE summary
        maeSummary = NPS("mae")
        summary1 = {
            "name": networkName,
            "epochs": range(len(mae)),
            "performance": mae
        }
        
        maeSummary.addSummary(summary1)

        #Create MSE summary
        mseSummary = NPS("mse")
        summary2 = {
            "name": networkName,
            "epochs": range(len(mse)),
            "performance": mse
        }
        
        mseSummary.addSummary(summary2)

        return [lossSummary,maeSummary,mseSummary]

        

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
        elif  networkName == "LSTM" :
            network = NN.LSTM("relu","adam","medium","LSTM")
            network.buildModel()
            self.networks.append(network)
        else :
            network = NN.FeedForwardNN("relu","adam","medium","FFNN")
            network.buildModel()
            self.networks.append(network)

        