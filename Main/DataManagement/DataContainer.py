
class DataContainer :

  def __init__(self,Data):
    self.OriginalData = Data
    self.columnnames = Data.columns

  def getColumnNames(self) :
    return self.columnNames

  def getRecord(self,index) :
    return self.OriginalData.iloc[index]

  def getRecordValues(self,index) :
    #return temp - atmos-press
    return self.OriginalData.iloc[index,2:8].to_numpy()#.iloc[index]#.values

  def normaliseData(self) :

    trainingSubset = self.trainingData[self.trainingData.columns[~self.trainingData.columns.isin(['target'])]]
    trainingCopy = trainingSubset.describe().transpose()
    print("Described",trainingCopy)
    #Training data is normed only on itself
    normedTrainingData = self.norm(self.trainingData, trainingCopy,trainingCopy)
    #Testing data is normed with training data
    normedTestingData = self.norm(self.testingData, trainingCopy,trainingCopy)


    return (normedTrainingData,normedTestingData)

  def norm(self,x,x_datasetDescribed,y_datasetDescribed) :
    return (x - x_datasetDescribed['mean']) /  y_datasetDescribed['std']

  def scaleInput(self,x,dataset) :
    A = dataset['min']
    B = dataset['max']
    a = -1 #tanh
    b = 1 #tanh
    x = ( 
      ( (x - A)
      /
      (B - A) ) * (b - a) + a
      )
    return x

  def scaleDataSet(self,dataSet) :
    dataCopy = dataSet.describe().transpose()

    scaledData = self.scaleInput(dataSet,dataCopy)

    return scaledData

  def removeOutliers(self,dataSet) :
    #remove ouliers but donot take quality or type into account
    subset = dataSet[dataSet.columns[~dataSet.columns.isin(['quality','type'])]]

    noOutlierList = dataSet[(np.abs(stats.zscore(subset ) ) < 3).all(axis=1)]
    return noOutlierList

class TestingDataContainer(DataContainer) :
  def __init__(self,data):
    DataContainer.__init__(self, data)


class TrainingDataContainer(DataContainer) :
  def __init__(self,data):
    DataContainer.__init__(self, data)


  def getTrainingLabels(self) :
    return self.OriginalData["target"].to_numpy()

  

