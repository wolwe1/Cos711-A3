import numpy as np
import pandas as pd 
import os
from os.path import dirname, abspath

import scipy
from scipy import stats

from .DataContainer import TestingDataContainer
from .DataContainer import TrainingDataContainer

DATAFILE = "A3-Data"
TEST_DATA_FILE = "Test.csv"
TRAIN_DATA_FILE = "Train.csv"

ROOTDIRECTORY = dirname(dirname(abspath(__file__))) 
FILEDIRECTORY = ROOTDIRECTORY + "\\" + DATAFILE + "\\"

# This class is responsible for reading in the data, scaling it using Approach 3 from data prep slides
# and dispatching a data container for the testing and training sets.
class DataReader :
  def __init__(self):

    self.trainingData = pd.read_csv(FILEDIRECTORY + TRAIN_DATA_FILE)
    self.testingData =  pd.read_csv(FILEDIRECTORY + TEST_DATA_FILE)
    self.trainingRecords = []
    self.testingRecords = []
    
    for index, row in self.trainingData.iterrows():
      self.trainingRecords.append(ReadingSet(row["ID"],row["location"],row["temp":"atmos_press"],row["target"]))

    for index, row in self.testingData.iterrows():
      self.testingRecords.append(ReadingSet(row["ID"],row["location"],row["temp":"atmos_press"]))
    


  def getContainers(self) :
    TestContainer = TestingDataContainer(self.testingData)
    TrainContainer = TrainingDataContainer(self.trainingData)
    return (TrainContainer,TestContainer)


#Input data structure
class ReadingSet :
  def __init__(self,title,location,recordsData,target = None):
    self.title = title
    self.location = location
    self.target = target
    self.data = self.createRecords(recordsData)


  def createRecords(self,recordsData) :
    data = []

    for col in recordsData["temp":"atmos_press"] :
      data.append(col.split(","))
    
    for i in range(len(data)) :
      data[i] = self.preprocessTimeSeries(data[i])
    return data

    
  def preprocessTimeSeries(self,array) :

    arrSize = len(array)

    #Replace string nan with useable equivalent, 0
    for i in range(arrSize) :
      if array[i] == "nan" :
        array[i] = 0
    return array
    

    




    

class Record :
  def __init__(self):
    self.temp
    self.precip
    self.rel_humidity
    self.wind_dir
    self.wind_spd
    self.atmos_press