import numpy as np
import pandas as pd
import csv as csv

class DataReader(object):
    """Responsible for reading in the datasets and creating dataframes"""
    def __init__(self):
        self.trainingSet = []
        self.trainingLabels = []

        self.testingSet = []
        self.testingLabels = []
        self.ReadCSV()
        #Convert to numpy
        self.trainingSet = np.array(self.trainingSet)
        self.trainingLabels = np.array(self.trainingLabels)

        self.testingSet = np.array(self.testingSet)
        self.testingLabels = np.array(self.testingLabels)


    def GetDataSets(self) :
      
         return (self.trainingSet,self.testingSet)

    def GetLabels(self) :

         return (self.trainingLabels,self.testingLabels)

    def ReadCSV(self) :
       with open("C:\\Users\\jarro\\Desktop\\University\\Cos711\\Assignments\\A3\\Code\\Main\\A3-Data\\Prepared-TrainingSet.csv") as trainingCsv:
        reader = csv.reader(trainingCsv, delimiter=',', quotechar='"')
        #skip header
        next(reader, None)
        for row in reader:
            record = []
            target = -1;
            #Loop through features
            for i in range(2,8) :
                feature = row[i].split(",")
                floatFeature = []
                #Convert to float
                for i in range(len(feature)) :
                    floatFeature.append(float(feature[i]))
                record.append(np.array(floatFeature) )
                #Add to record
            self.trainingSet.append(np.array(record)) 
            self.trainingLabels.append(float(row[8]))

        with open("C:\\Users\\jarro\\Desktop\\University\\Cos711\\Assignments\\A3\\Code\\Main\\A3-Data\\Prepared-TestingSet.csv") as trainingCsv:
            reader = csv.reader(trainingCsv, delimiter=',', quotechar='"')
            #skip header
            next(reader, None)
            for row in reader:
                record = []
                target = -1;
                #Loop through features
                for i in range(2,8) :
                    feature = row[i].split(",")
                    floatFeature = []
                    #Convert to float
                    for i in range(len(feature)) :
                        floatFeature.append(float(feature[i]))
                    record.append(np.array(floatFeature) )
                    #Add to record
                self.testingSet.append(np.array(record) )
                self.testingLabels.append(float(row[8]))
               
