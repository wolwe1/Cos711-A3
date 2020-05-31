class NN_Manager :

    def __init__(self,trainingData,testingData):
        self.networks = []
        self.trainingData = trainingData
        self.testingData = testingData

        #Store a list of each networks performance as performance summaries
        self.networkTrainingPerformanceHistory = []
        self.networkTestingPerformanceHistory = []

    def trainNetworks(self) :

        for i in range( len(self.networks) ) :
            self.networkTrainingPerformanceHistory.append(self.networks[i].train())

    def testNetworks(self) :
        for i in range( len(self.networks) ) :
            self.networkTestingPerformanceHistory.append( self.networks[i].test())

        