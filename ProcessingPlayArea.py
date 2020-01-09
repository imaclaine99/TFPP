import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers import LSTM

from keras.optimizers import SGD
import matplotlib.pyplot as plt
import argparse
import MyFunctions

# Lets define some parameters
value_range = 250           # we process this many sets of records at once.  250 is a bit more than a year, so seems a good balance between not enough and too many
output_values = 1           # Should be 2, but using 1 now for debug

infile = r"DAX_Prices_WithMLPercentages.csv"
infile_rnd = "DAX_Prices_WithMLPercentages_RandomTarget.csv"
output_grahic_prefix = "DaxPercentages_BasicModel_Dropout_MSE"

# Load and parse a CSV file into Train and TEst XY data.  This creates the correct number of slices of the data set
(trainX, testX, trainY, testY) = MyFunctions.parse_data_to_trainXY(infile, "BuyWeightingRule", value_range)

(trainX_rnd, testX_rnd, trainY_rnd, testY_rnd) = MyFunctions.parse_data_to_trainXY(
    infile_rnd, "BuyWeightingRule", value_range)


# Let's try some Keras!
model = Sequential()
# Add flatten - may help?
model.add(Flatten( input_shape=(value_range, 4)))
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(.5))
model.add(Dense(256, activation="sigmoid"))
model.add(Dropout(.5))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model.add(Dense(output_values))       # remove softmax, given we have multi-value output


model_rnd = Sequential()
# Add flatten - may help?
model_rnd.add(Flatten( input_shape=(value_range, 4)))
model_rnd.add(Dense(512, activation="sigmoid"))
model_rnd.add(Dropout(.5))
model_rnd.add(Dense(256, activation="sigmoid"))
model_rnd.add(Dropout(.5))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model_rnd.add(Dense(output_values))       # remove softmax, given we have multi-value output



# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="mean_squared_error", optimizer='Adam', metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
MyFunctions.plot_and_save_history(H, EPOCHS, output_grahic_prefix + "base")


# train the neural network
model_rnd.compile(loss="mean_squared_error", optimizer='Adam', metrics=["accuracy"])
H_rnd = model.fit(trainX_rnd, trainY_rnd, validation_data=(testX_rnd, testY_rnd),
	epochs=EPOCHS, batch_size=32)
MyFunctions.plot_and_save_history(H_rnd, EPOCHS, output_grahic_prefix + "random")

MyFunctions.plot_and_save_history_with_rand_baseline(H, H_rnd, EPOCHS, output_grahic_prefix + "comparison")



# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(


# evaluate the network
print("[INFO] evaluating RANDOM network...")
predictions = model_rnd.predict(testX_rnd, batch_size=32)
print(classification_report(testY_rnd.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(




# Would be good to be able to copy a model, to be able to run it against random data...
# E.g. pass in a file, a model, and output filename
# parse the data
# copy the model
# run the model with both real data, and randomised data
# compare the two to show the deviation


