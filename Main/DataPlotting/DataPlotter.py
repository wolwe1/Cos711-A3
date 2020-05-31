import matplotlib.pyplot as plt
import numpy as np
from Utils.Utils import NetworkPerformanceSummary as NS

class DataPlotter:

    def __init__(self):
        self.plt = plt

    def plot(self,networkPerformanceSummaryList) :

        fig,axis = self.plt.subplots(len(networkPerformanceSummaryList))

        #Fix stupid type choice of matplotlib
        try:
            test = axis[0]
        except:
            axis = [axis]
        finally:
            for i in range(len(networkPerformanceSummaryList)) :
                networkPerformanceSummary = networkPerformanceSummaryList[i]
                title = networkPerformanceSummary.measureUnit + " of"

                for x in range(networkPerformanceSummary.getCount()) :
                    summary = networkPerformanceSummary.getNext()
                    title = title + " " +summary.get('networkName')
                    if x != networkPerformanceSummary.getCount() -1 :
                        title = title + " vs"
                    x_values = summary.get('epochs')
                    y_values = summary.get('performance')
                    
                    axis[i].plot(x_values,y_values)


            self.plt.subplots_adjust(hspace=0.8)

            legend = tuple(networkPerformanceSummary.networkNames)
            self.plt.legend(legend, loc='upper left')
            self.plt.title(title)


    def show(self) :
        self.plt.show()


#Test data
# epochs = range(0,10)
# performance1 = [1,2,3,4,5,6,7,8,9,10]
# performance2 = [2,3,4,5,6,7,7,9,10,11]
# performance3 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# network1 = "CNN"
# network2 = "LSTM"
# network3 = "FFNN"
# performanceSummary = NS("MSE")
# performanceSummary.addSummary({"name" : network1,"epochs": epochs,"performance":performance1})
# performanceSummary.addSummary({"name" : network2,"epochs": epochs,"performance":performance2})
# performanceSummary.addSummary({"name" : network3,"epochs": range(0,14),"performance":performance3})

# performanceSummary2 = NS("Accuracy")
# performanceSummary2.addSummary({"name" : network1,"epochs": epochs,"performance":performance1})
# performanceSummary2.addSummary({"name" : network2,"epochs": epochs,"performance":performance2})
# performanceSummary2.addSummary({"name" : network3,"epochs": range(0,14),"performance":performance3})

# plotter = DataPlotter()
# plotter.plot([performanceSummary])
# plotter.show()