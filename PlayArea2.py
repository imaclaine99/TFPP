import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.initializers import RandomUniform
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers import LSTM
from keras.initializers import RandomNormal
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import argparse
import MyFunctions as myf

num_samples = 250
myf.EPOCHS = 200
myf.model_description = 'PlayArea2 Test - Removed second dropout, 512 then 256 nodes'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#### PlayPenCode
myf.parse_file("DAX4ML.csv")
myf.parse_file("^GDAXI.csv")


model = Sequential()
# Add flatten - may help?
model.add(Flatten(input_shape=(num_samples, 4)))
model.add(Dense(512, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(.25))
model.add(Dense(256, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(.25))
model.add(Dense(1))       # remove softmax, given we have multi-value output

#MyFunctions.parse_process_plot("DAX_Prices_WithMLPercentages_Trimmed.csv", "BuyWeightingRule", model, "Model1_Percent_NewRandom_Trimmed")
#MyFunctions.parse_process_plot("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule", model, "Model1_Percent_NewRandom_Orig")
#MyFunctions.parse_process_plot(".\parsed_data\DAX4ML.csv_parsed.csv", "BuyWeightingRule", model, "Model1_Percent_NewRandom_moredata")
#MyFunctions.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model, "Model1_Percent_NewRandom_moredata_25dropout")



model2 = Sequential()
# Add flatten - may help?
model2.add(Flatten(input_shape=(num_samples, 4)))
model2.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(.4))
model2.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(.4))
model2.add(Dense(1))       # remove softmax, given we have multi-value output

#MyFunctions.parse_process_plot("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule", model2, "Model1_Relu_Percent_NewRandom")

#model3 = Sequential()
#model3.add(Flatten(input_shape=(num_samples, 4)))
#model3.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#model3.add(Dropout(.5))
#model3.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#model3.add(Dropout(.5))
#model3.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#model3.add(Dense(1))       # remove softmax, given we have multi-value output

#MyFunctions.parse_process_plot("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule", model3, "Model1_ExtraLayer_Relu_Percent_NewRandom")

#model2 = Sequential()
# Add flatten - may help?
#model2.add(Flatten(input_shape=(num_samples, 4)))
#model2.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
#model2.add(Dropout(.4))
#model2.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
#model2.add(Dropout(.4))
#model2.add(Dense(1))       # remove softmax, given we have multi-value output

#MyFunctions.parse_process_plot("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule", model2, "Model1_Relu_Percent_NewRandom_001L2Reg")


model_arr = []
for i in range (1, 32, 2) :

    print (i)
    myf.batch_size = i
    model_new =Sequential()

    model_new.add(Flatten(input_shape=(num_samples, 4)))
#    model_new.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001 + i/1000)))
#    model_new.add(Dense(512, activation="relu", kernel_initializer=RandomUniform(minval=-0.5, maxval=0.5, seed=None), kernel_regularizer=regularizers.l2(0.001 + i/1000)))
    model_new.add(Dense(512, activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.1, seed=None), kernel_regularizer=regularizers.l2(0.01 )))
#    model_new.add(Dense(52, activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.1, seed=None), kernel_regularizer=regularizers.l2(0.01 )))
    model_new.add(Dropout(.2))
    model_new.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#    model_new.add(Dense(26, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#    model_new.add(Dropout(.2))
    model_new.add(Dense(1))       # remove softmax, given we have multi-value output
#    myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model_new, "Model1_Relu_Percent_L2_MoreData_25Dropout_RandomNormal_Batch" + str(i))
    myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "SellWeightingRule", model_new, ".\output_images\Remove2ndDropout\Model1_Relu_Percent_L2_MoreData_25Dropout_RandomNormal_Batch_NewSellRule" + str(i))
    del model_new
