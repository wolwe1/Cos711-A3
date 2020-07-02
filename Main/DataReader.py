import numpy as np
import pandas as pd
import csv as csv
import json

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

            #Convert features to useable format
            featuresForRecord = self.ConvertFeaturesToFloatList(row)
           
            #Create timestamps of 1 time stamp being an array of all features at that timestamp
            record = self.GetFeatureTimeStamps(featuresForRecord)
               
            #Add to record
            self.trainingSet.append(record) 
            self.trainingLabels.append(float(row[8]))

        with open("C:\\Users\\jarro\\Desktop\\University\\Cos711\\Assignments\\A3\\Code\\Main\\A3-Data\\Prepared-TestingSet.csv") as trainingCsv:
            reader = csv.reader(trainingCsv, delimiter=',', quotechar='"')
            #skip header
            next(reader, None)
            for row in reader:

                #Convert features to useable format
                featuresForRecord = self.ConvertFeaturesToFloatList(row)
           
                #Create timestamps of 1 time stamp being an array of all features at that timestamp
                record = self.GetFeatureTimeStamps(featuresForRecord)
               
                #Add to record
                self.testingSet.append(record) 
                self.testingLabels.append(float(row[8]))
               

    def ConvertFeaturesToFloatList(self,row) :
        """Takes string features from record and converts them each to an array of floats"""
        features = []
        for i in range(2,8) :
            feature = row[i].split(",")
            floatFeature = []
            #Convert to float
            for x in range(len(feature)) :
                floatFeature.append(float(feature[x]))
            features.append(np.array(floatFeature) )

        return np.array(features)

    def GetFeatureTimeStamps(self,featureListForRecord) :
        """Creates a txf array where t is each time stamp and f is a feature at point t"""
        timeStamps = []
        for i in range(121) :
            timeStamp = []
            for x in range( len(featureListForRecord)) :
                timeStamp.append(featureListForRecord[x,i])

            timeStamps.append(np.array(timeStamp))
        return np.array(timeStamps)


    def WriteSummaryToFile(self,networkPerformanceSummary) :
       
        networkSummary = self.ConvertFloatListToStringList(networkPerformanceSummary)

        with open('NetworkSummary.json', 'w', encoding='utf-8') as f:
            json.dump(networkSummary, f, ensure_ascii=False, indent=4)

    def ConvertFloatListToStringList(self,networkPerformanceSummaries) :
        """Takes string features from record and converts them each to an array of floats"""
        result = []

        #For each network...
        for i in range(len(networkPerformanceSummaries)) :
            summary = networkPerformanceSummaries[i]
            convertedEpochs = summary["epochs"]
            convertedTimeToTrain = str(summary["timeToTrain"])

            #Convert epochs to string
            for x in range(len(convertedEpochs)) :
                convertedEpochs[x] = str(convertedEpochs[x])


            #For each metric
            for x in range(len(summary["evaluations"])) :
                currentEvaluation = summary["evaluations"][x]

                #For each entry for the models performance in that metric
                for j in range( len(currentEvaluation["performance"]) ) :
                    currentEvaluation["performance"][j] = str(currentEvaluation["performance"][j])

                summary["evaluations"][x]["performance"] = currentEvaluation["performance"]

            summary["epochs"] = convertedEpochs
            result.append(summary)

        return result