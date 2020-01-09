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


#### PlayPenCode
MyFunctions.parse_file("DAX4ML.csv")
MyFunctions.parse_file("^GDAXI.csv")


# Load and parse a CSV file into Train and TEst XY data.  This creates the correct number of slices of the data set
#(trainX, testX, trainY, testY) = MyFunctions.parse_data_to_trainXY(r".\\parsed_data\\^GDAXI.csv_parsed.csv", "BuyWeightingRule", value_range)

(trainX_rnd, testX_rnd, trainY_rnd, testY_rnd) = MyFunctions.parse_data_to_trainXY(
    r".\DAX_Prices_WithMLLabelsRandomTargets.csv", "BuyWeightingRule", value_range)


# Let's try some Keras!
model = Sequential()
# Add flatten - may help?
model.add(Flatten( input_shape=(value_range, 4)))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(256, activation="sigmoid"))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model.add(Dense(output_values))       # remove softmax, given we have multi-value output

# Let's try some Keras!
model_rnd = Sequential()
# Add flatten - may help?
model_rnd.add(Flatten( input_shape=(value_range, 4)))
model_rnd.add(Dense(512, activation="sigmoid"))
model_rnd.add(Dense(256, activation="sigmoid"))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model_rnd.add(Dense(output_values))       # remove softmax, given we have multi-value output






model1a = Sequential()
# Add flatten - may help?
model1a.add(Flatten( input_shape=(value_range, 4)))
model1a.add(Dense(512, activation="sigmoid"))
model1a.add(Dense(256, activation="sigmoid"))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model1a.add(Dense(output_values))       # remove softmax, given we have multi-value output


# Let's try some Keras!
model2 = Sequential()
# Add flatten - may help?
model2.add(Flatten( input_shape=(value_range, 4)))
model2.add(Dense(512, activation="sigmoid"))
model2.add(Dropout(0.5))
model2.add(Dense(256, activation="sigmoid"))
model2.add(Dropout(0.5))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model2.add(Dense(output_values))       # remove softmax, given we have multi-value output




model3 = Sequential()
# Add flatten - may help?
model3.add(Flatten( input_shape=(value_range, 4)))
model3.add(Dense(512, activation="sigmoid"))
model3.add(Dropout(0.5))
model3.add(Dense(256, activation="sigmoid"))
model3.add(Dropout(0.5))
model3.add(Dense(128, activation="sigmoid"))
model3.add(Dropout(0.5))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
model3.add(Dense(output_values))       # remove softmax, given we have multi-value output

#model4 = Sequential()
# Add flatten - may help?   - should flatten AFTER the LSTM, not before, dummy!
#model4.add(Flatten( input_shape=(value_range, 4)))
#model4.add(LSTM(1024, return_sequences=True, input_shape=(value_range, 4)))
#model4.add(Dropout(0.5))
#model4.add(Dense(512, activation="sigmoid"))
#model4.add(Dropout(0.5))
#model4.add(Dense(256, activation="sigmoid"))
#model4.add(Dropout(0.5))
#model4.add(Dense(128, activation="sigmoid"))
#model4.add(Dropout(0.5))
#model4.add(Flatten())
#model.add(Dense(2, activation="softmax"))       # 2 output
#model4.add(Dense(output_values))       # remove softmax, given we have multi-value output




# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="mean_absolute_error", optimizer='Adam', metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
MyFunctions.plot_and_save_history(H, EPOCHS, "model1_mean_abs_error_Adam")


# train the neural network
H_rnd = model.fit(trainX_rnd, trainY_rnd, validation_data=(testX_rnd, testY_rnd),
	epochs=EPOCHS, batch_size=32)
MyFunctions.plot_and_save_history(H_rnd, EPOCHS, "model1_mean_abs_error_Adam_Random")

MyFunctions.plot_and_save_history_with_rand_baseline(H, H_rnd, EPOCHS, "model1_mean_abs_error_Adam_compareToBaseline")



model1a.compile(loss="mean_squared_error", optimizer='Adam', metrics=["accuracy"])
H1a = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
MyFunctions.plot_and_save_history(H1a, EPOCHS, "model1a_mean_square_error_Adam")


model2.compile(loss="mean_absolute_error", optimizer='Adam', metrics=["accuracy"])
model3.compile(loss="mean_absolute_error", optimizer='Adam', metrics=["accuracy"])
#model4.compile(loss="mean_absolute_error", optimizer='Adam', metrics=["accuracy"])




H2 = model2.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

MyFunctions.plot_and_save_history(H2, EPOCHS, "model2_mean_abs_error_Adam")


H3 = model3.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)


#H4 = model4.fit(trainX, trainY, validation_data=(testX, testY),
#	epochs=EPOCHS, batch_size=32)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(








# evaluate the network
print("[INFO] evaluating network 2...")
predictions = model2.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(



# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H2.history["loss"], label="train_loss")
plt.plot(N, H2.history["val_loss"], label="val_loss")
plt.plot(N, H2.history["accuracy"], label="train_acc")
plt.plot(N, H2.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN 2)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("training_img2_2")



# evaluate the network
print("[INFO] evaluating network 3...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(



# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H3.history["loss"], label="train_loss")
plt.plot(N, H3.history["val_loss"], label="val_loss")
plt.plot(N, H3.history["accuracy"], label="train_acc")
plt.plot(N, H3.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("training_img2_3")




# evaluate the network
print("[INFO] evaluating network 4...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))       # labels=['BuyWeight', 'SellTomorrow'] - didn't work :(



