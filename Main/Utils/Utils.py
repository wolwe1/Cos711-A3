
class NetworkPerformanceSummary:
    
    def __init__(self,measureUnit):
        self.currentIndex = 0
        self.measureUnit = measureUnit
        self.networkNames = []
        self.epochs = []
        self.networkPerformances = []

    def addSummary(self,summary):
        self.networkNames.append(summary.get('name'))
        self.epochs.append(summary.get('epochs'))
        self.networkPerformances.append(summary.get('performance'))

    def getNext(self):

        if(self.currentIndex >= len(self.networkNames) ) :
            return None

        summary  = {
            'networkName' : self.networkNames[self.currentIndex],
            'epochs' : self.epochs[self.currentIndex],
            'performance' : self.networkPerformances[self.currentIndex]
        }
        self.currentIndex = self.currentIndex + 1

        return summary

    def checkRangesEqual(self) :
        datarange = self.epochs[0]

        for i in range(len(self.epochs)) :
            if len(datarange) != len(self.epochs[i]) :
                return False

        return True

    def getCount(self) :
        return len(self.networkNames)




        