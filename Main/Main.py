# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from DataReader import DataReader
from NN_Manager import NN_Manager
#Get the dataSets
#tf.random.set_seed(13)
dataReader = DataReader()

(trainingDataSet,testingDataSet) = dataReader.GetDataSets()
(trainingLabels,testingLabels) = dataReader.GetLabels()

#Create chosen networks
networkManager = NN_Manager(trainingDataSet,testingDataSet,trainingLabels,testingLabels)
#networkManager.addNetwork("CNN")
#networkManager.addNetwork("LargerFilterCNN")
#networkManager.addNetwork("LargeCNN")
networkManager.addNetwork("StackedLSTM")
#networkManager.addNetwork("LSTM")
#networkManager.addNetwork("FFNN")

#networkManager.addNetwork("GRU")
#networkManager.addNetwork("StackedLSTM")

# #Train networks
networkManager.trainNetworks()
#networkManager.evaluateNetworks()
performance = networkManager.GetNetworkPerformance()
dataReader.WriteSummaryToFile(performance)

