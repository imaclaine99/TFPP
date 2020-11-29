from math import trunc

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from clr_callback import CyclicLR
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from numpy import mean
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import math
from learningratefinder import LearningRateFinder
import csv          # I use this for writing csv logs
import mysql.connector
import time
import datetime
import yfinance
#from alpha_vantage.timeseries import TimeSeries
from sqlalchemy import create_engine


# Set some default values, which can be overridden if wanted
EPOCHS = 50
batch_size = 8                  # Testing suggests a batch of 4 or 2 gives better results than 8 - gets to a better loss reduction AND more quickly.  Let's use 4 moving forward for now...
processing_range = 250          #  Default number of rows to process.  Can set to something else if needed (and useful for testing!)
shuffle_data = True             # For LSTM, may not want to shuffle data.    This applies to both the training/test data split, as well as the fit function when training.
is_dupe_data = False            # If True, duplicates certain y train (not Test) Values.  ONLY USE IF shuffle_data is true
dupe_vals = (1,2,5,10)
output_summary_logfile  = ".\outputsummary.csv"
model_description = ""          # This can be useful to set for logging purposes
# Set some global metrics to track last execution, to make it easier to grab later.  A new model should reset these.
last_exec_duration = 0
model_best_acc = 0
model_best_loss = 999
model_best_val_acc = 0
model_best_val_loss= 999
model_best_combined_ave_loss = 999
best_guess_acc = 0
best_guess_loss = 999
best_guess = 0
standardised_loss = 999
processing_rule = 'BuyV1'     # Only used for logging.  Currently  BuyV1 - 0, 1, 2, 4 ,5.   SellV1.   To do - a range - i.e. prediction of the actual range?  Different loss functions?
default_model_loss_func = 'mean_squared_error'
model_loss_func = default_model_loss_func        # Define this, to allow other options to be tried.

model_last_epochs = 0
model_executions = 0        # Use this to track how many executions.  Will clear session before creating a new model if > 0  ## Not yet implemented
clear_session = True        # Define whether or not we clear Keras session after fit and plot
use_lrf = False             # Do we use the LR Finder or not?  If we do, we then need to populate LR_MIN and LR_MAX
lrf_batch = 64              # Needs to be large enough not to get 'noise', especially with data with limited entries
clr_mode = 'None'           #  'triangular'
lr_min = 1e-15 #2.5e-5             # 0.00001            # Used with CLR - set by LR Finder if that is used
lr_max = 0.01               # Used with CLR - set by LR Finder if that is used
atr_rule = 1                # 1 is the original max of TR over 20 days.  3 is a beta decay based calc.  2 is not yet implemented :)
atr_beta_decay = 0.98       # Gone with 0.98 to start with.  Would be interested to try different values - aim is to not have this too low, as this smooths out spikes in volatility.  Too high though, and it may skew early data.  Currently used only on rule 3
default_optimizer = 'Nadam' # Not used if explicity set when called in compile and fit, otherwise this is used.  Can be anything that is supported, but keep in mind that different optimizers will work with LR very differently!
                            # Other options:   SGD,   SGD+CLR, Adam, ????.    Need to decide best way to manage tunables for the optimser...
                            # SGD+NM  (Nesterov Momentum), AdamDelta (?)
                            # Do I also want to specific optimizer tunables - e.g. opt_lr, opt_??
custom_loss_override = False    # Set this to true to turn 'off' custom loss functions, to make them back to MSE.  This is useful to run a final evaluate with MSE for like for like comparisons with different loss functions on test data.
output_images_path = r'./output_images/'
models_path =  r'./models/'
read_backwards  = False     # Default is to read from start to end.
callbacks = []          # This feels cludgy, but prevents me having to pass everything around...
                        # Can be used for both CLR as well as LRF callbacks.  Just need to make sure this is handled properly
                        # Note:  Once set (e.g. SGD+CLR) CLR can not currently be removed, as the Callback remains.  Need to review this in the future.
                        # Not a big prority, as CLR is not being used...
early_stop_callback = True
early_stopping_min_delta = 0.01
early_stopping_patience = 10
debug_with_date = False         # Set this to true, to keep the DATE column from input files up until the train/predict point.
                                # This is a computational overhead, but helps check what is actually going on.
                                # IF TRUE, 5 columns are used rather than 4, and at the time of compile_and_fit, the DATE is then removed.
db_host = '127.0.0.1'
db_username = 'tfpp'
db_pwd = 'tfpp'
key = 'B7WOWYWCP22UNBW6'        # AlphaVantage Key - should move to OS env variable to remove from code
# pk_816e897106fd485da3c89ac4ae06dfc7 IEX


def parsefile(filename, output_column_name, strip_last_row=True):
    """ Simple function to take a file with OHLC data, as well as an output / target column.  Returns two arrays - one of the OHLC, one of the target
        Strips the last row, as this can't have a target (will be undefined)
        May want to strip the first row in some cases = e.g. percantage based input
    """
    infile = pd.read_csv(Path(filename), engine="python")

    # So, we need to build up an array that has # examples of 250 x 4 for XTrain - 250 rows of O H L C (can ignore date)
    # Need to 'slice' the data somehow :)

    if strip_last_row == True:
        # We drop the last row - its no use.  We now have the four columns of data we need.  Need to now create this into len-249 samples ready for Keras to process
        # Need to ignore the last row when training, as doesn't have a valid indicator.
        end_row = len(infile) - 2
    else:
        end_row = len(infile)

    if debug_with_date:
        olhc_data = infile.loc[0:end_row, 'Date':'Close']
        # Convert Date String to Date INTs to make numpy work better (can't figure out how to manage strings in Numpy!!)
        olhc_data['Date'] = olhc_data['Date'].map (lambda date : int(''.join(c for c in str(date) if c.isdigit())))
    else:
        olhc_data = infile.loc[0:end_row, 'Open':'Close']
    output_data = infile[output_column_name]
    output_data = output_data[0:len(olhc_data)]               # Was len(output_data), but changed to end_row to ensure these two keep in sync
    dates_data = infile.loc[0:end_row, 'Date']                # Optional - and useful

    return olhc_data, output_data, dates_data


def parse_data(olhc_array, results_array, num_samples):
    """
        Function that takes two arrays, and returns them as xtrain / ytrain data, ready for Keras.  This involves a lot of duplication of data, given we slice each n samples
        # Now need to create this into a large Xtrain and yTrain Numpy.  Let's just create the Numpy for now, and worry about the Xtrain / Xtest bit later!

    :param olhc_array:
    :param results_array:
    :return: xtrain, ytrain
    """
    # Now need to create this into a large Xtrain and yTrain Numpy.  Let's just create the Numpy for now, and worry about the Xtrain / Xtest bit later!

    if debug_with_date:
        train_cols = 5
        x_train = np.zeros((len(olhc_array) - num_samples + 1, num_samples, train_cols))  # O H L C
    else:
        train_cols = 4
        x_train = np.zeros((len(olhc_array) - num_samples + 1, num_samples, train_cols))  # O H L C

#    x_train = np.zeros((len(olhc_array) - num_samples + 1, num_samples, train_cols))  # O H L C
    y_train = np.zeros((len(results_array) - num_samples + 1, 1))  # Only 1 output column for now

    for i in range(len(olhc_array) - num_samples + 1):
        # Now, i is the oldest data.  i+value_range is the latest record - this is the one we want for y.
        # I want to set x_train (1, :, :) to new_method_data(i:i+249, :) ...  Shouldn't be too hard :)
        # Not quite sure why the +1 is needed...  I'm tired, but it was a sample short...

        x_train[i, :, :] = olhc_array[i:i + num_samples]

        ## Old code - may have value in the future
        # if output_values == 2 :
        #   y_train[i,:] = new_method_results[i+value_range-1:i+value_range]
        # elif output_values == 1 :
        y_train[i] = results_array.loc[i + num_samples - 1]

    return x_train, y_train


def parse_data_to_trainXY(filename, output_column_name, num_samples=250, strip_first_row=False, test_size=0.25,
                          random_state=42):
    """"
        Function to read in a CSV, process it, and return train and test X Y data

        baseline_to_random_target means we will also randomise the output/target data, re-run, and compare against that.
        THe reason for doing this is to compare our model to random data of the EXACT same distribution to validate its effectiveness against what is effectively guessing
        return: trainX, testX, trainY, testY
    """
    new_method_data, new_method_results, dates = parsefile(filename, output_column_name)
    x_train, y_train = parse_data(new_method_data, new_method_results, num_samples)

    # I am mixxing my variables here - xtrain and ytrain above and actually data and outcomes - need to fix this later
    (trainX, testX, trainY, testY) = train_test_split(x_train,
                                                      y_train, test_size=test_size, random_state=random_state, shuffle = shuffle_data)

    # dupe data
    # Note - we do this AFTER we split test and train, to stop data sets being in both test and train, which would artificially increase the validation accuracy
    if is_dupe_data:
        print ("[INFO] Duping data with values" + str(dupe_vals))
        for i in dupe_vals:
            trainX, trainY = dupe_data(trainX, trainY, i)

    return trainX, testX, trainY, testY

def parse_data_to_trainXY_plusRandY(filename: object, output_column_name: object, num_samples: object = 250, strip_first_row: object = False,
                                    test_size: object = 0.2,
                                    random_state: object = 42) -> object:
    """"
        Function to read in a CSV, process it, and return train and test X Y data

        baseline_to_random_target means we will also randomise the output/target data, re-run, and compare against that.
        THe reason for doing this is to compare our model to random data of the EXACT same distribution to validate its effectiveness against what is effectively guessing
        return: trainX, testX, trainY, testY, trainYRnd, testYRnd
    """
    new_method_data, new_method_results = parsefile(filename, output_column_name)
    x_data, y_data = parse_data(new_method_data, new_method_results, num_samples)

    # Copy ytrain, then shuffle it
    y_data_rnd = np.copy(y_data)
    np.random.shuffle(y_data_rnd)


    # I am mixxing my variables here - xtrain and ytrain above and actually data and outcomes - need to fix this later
    (trainX, testX, trainY, testY) = train_test_split(x_data,
                                                      y_data, test_size=test_size, random_state=random_state)

    # Repeat with Randomised YData
    (trainX_rnd, testX_rnd, trainY_rnd, testY_rnd) = train_test_split(x_data,
                                                      y_data_rnd, test_size=test_size, random_state=random_state)

    return trainX, testX, trainY, testY, trainY_rnd, testY_rnd



def plot_and_save_history(history, epochs, filename):
    """
    Function to plot the loss and accuracy and save to a file, with a date_time stamp coming when I get around to it
    :param history:
    :param filename:
    :return:
    """
    # plot the training loss and accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"]*10, label="train_acc*10")
    plt.plot(N, history.history["val_accuracy"]*10, label="val_acc*10")
    plt.title("Training Loss and Accuracy (" + filename + ")")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(Path(output_images_path + filename))
    plt.close()

def plot_and_save_history_with_rand_baseline(history, history_rand_baseline, epochs, filename):
    """
    Function to plot the loss and accuracy and save to a file, with a date_time stamp coming when I get around to it
    :param history:
    :param filename:
    :return:
    """
    # plot the training loss and accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, np.array(history.history["accuracy"])*10, label="train_acc*10")
    plt.plot(N, np.array(history.history["val_accuracy"])*10, label="val_acc*10")
    plt.plot(N, np.array(history_rand_baseline.history["val_accuracy"])*10, label="val_acc_rand*10")
    plt.plot(N, history_rand_baseline.history["val_loss"], label="val_loss_rand")
    plt.title("Training Loss and Accuracy (" + filename + ")")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.axes().set_ylim([0, 10])        # Set max Y axis to 10.   This may need to change, but works for now.
    plt.legend()
    plt.savefig(Path(filename))
    plt.close()


def plot_and_save_history_with_baseline(history, epochs, filename, metadatafilename):
    """
    Function to plot the loss and accuracy and save to a file, with a date_time stamp coming when I get around to it
    This also reads the 'metadata' from a parsed file, which shows what the best guess of 0 to 5 would be for accuracy and loss, and plots them as well - this is much quicker and cleaner
    than also doing a randomised Y values
    :param history:
    :param filename:
    :return:
    """

    # Globals so we can read them later
    global model_best_combined_ave_loss
    global model_best_acc
    global model_best_loss
    global model_best_val_acc
    global model_best_val_loss
    global model_name_file
    global best_guess_acc
    global best_guess_loss
    global best_guess

    # read meta_data from file
    if (metadatafilename != ''):
        meta_np = np.recfromcsv(metadatafilename)      # best guess, best_guess_loss, best_guess_accuracy
        best_guess = meta_np[0]
        best_guess_loss = meta_np[1]
        best_guess_acc = meta_np[2]
    else:
        best_guess = 0
        best_guess_loss = 0
        best_guess_acc = 0
        meta_np = [0,0,0]

    max_y_axis = 10     # may want to be smarter with this later



    # Some weird type translation happening here...  Check if its a record 64, and if so, use index 0 to get to a value.
    if meta_np[1].dtype.name == 'record64':
        best_guess_loss = best_guess_loss[0]
        best_guess = best_guess[0]
        best_guess_acc = best_guess_acc[0]

 #       loss_multiplier = max_y_axis / meta_np[1][0] /2
 #   else :

    # Test - let's get the 10th percentile of loss - use that for loss multiplier rather than just 'best guess', given meta data is now very hard
    print ("[INFO]: Val Loss 5th Percentile is: " + str(np.percentile(history.history["val_loss"], 5)))
    print ("[INFO]: Val Loss 95th Percentile is: " + str(np.percentile(history.history["val_loss"], 95)))
    print ("[INFO]: Val Loss 99th Percentile is: " + str(np.percentile(history.history["val_loss"], 99)))
    print ("[INFO]: Val Loss 99.5th Percentile is: " + str(np.percentile(history.history["val_loss"], 99.5)))

    # Test code - let's check the 99th percentile against the 'best_guess_loss' and adjust accordingly
    if np.percentile(history.history["val_loss"], 99) > best_guess_loss:
        best_guess_loss = np.percentile(history.history["val_loss"], 99)
        max_y_axis = max_y_axis * 2

    # Test code - let's check the 99th percentile against the 'best_guess_loss' and adjust accordingly
    if np.percentile(history.history["val_loss"], 99) > best_guess_loss:
        best_guess_loss = np.percentile(history.history["val_loss"], 99)
        max_y_axis = max_y_axis * 2

    # Check if the best_guess_loss is too 'high' - if so, lower it.
    if np.percentile(history.history["val_loss"], 99) < (best_guess_loss / 4):
        best_guess_loss = best_guess_loss / 2

    if np.percentile(history.history["val_loss"], 99) < (best_guess_loss / 4):
        best_guess_loss = best_guess_loss / 2

    if np.percentile(history.history["val_loss"], 99) < (best_guess_loss / 4):
        best_guess_loss = best_guess_loss / 2


    loss_multiplier = max_y_axis / best_guess_loss /2
    loss_multiplier = trunc(loss_multiplier/5) * 5

    if loss_multiplier < 1:
        loss_multiplier = 1


    accuracy_multiplier = max_y_axis / max(max(history.history["accuracy"]), max(history.history["val_accuracy"]), max(best_guess_acc, 0)) / 2        # Normalise
    accuracy_multiplier = trunc(accuracy_multiplier/5) * 5

    if accuracy_multiplier < 1:
        accuracy_multiplier = 1

    # No longer needed, as the path is set elsewhere
    plot_filename = filename.split('/')
    plot_filename = plot_filename[len(plot_filename)-1]

    # Due to early stopping, epochs may be less than intended.  Use the history to calculate the epochs
    epochs = len(history.history["loss"])


    # plot the training loss and accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
#    plt.plot(N, history.history["loss"], label="train_loss")
#    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, np.array(history.history["loss"])*loss_multiplier, label="train_loss*" + str(loss_multiplier))
    plt.plot(N, np.array(history.history["val_loss"])*loss_multiplier, label="val_loss*" + str(loss_multiplier))
    plt.plot(N, np.array(history.history["accuracy"])*accuracy_multiplier, label="train_acc*" + str(accuracy_multiplier))
    plt.plot(N, np.array(history.history["val_accuracy"])*accuracy_multiplier, label="val_acc*" + str(accuracy_multiplier))
#    plt.plot(N, np.array(meta_np[1]).tolist()*epochs, label= 'best_guess_loss')
    plt.plot(N, np.full_like(N, best_guess_loss*loss_multiplier, np.float32), label= 'best_guess_loss*' + str(loss_multiplier))
    plt.plot(N, np.full_like(N, best_guess_acc*accuracy_multiplier, np.float32), label= 'best_guess_accuracy*' + str(accuracy_multiplier))
    plt.title("Training Loss and Accuracy (" + filename + ")")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.axes().set_ylim([0, max_y_axis])        # Set max Y axis to 10.   This may need to change, but works for now.
    plt.legend()
    plt.savefig(Path(output_images_path + filename + ".png"))             # 10/2/2020 - added .png, as this is now needed?
    plt.close()

    # Write Summary Data
    # DateTime, ModelNameFile, GuessAccuracy, GuessLoss, EPOCHS, BATCHES, BestLoss, BestAccuracy, ModelDescription
    current_date_time = datetime.datetime.now()
    #model_name_file = __file__.split('\\')



    model_name_file = sys.argv[0].split('/')
    model_name_file = model_name_file[len(model_name_file)-1].split('/')            # Split for other form of directory delimiter
#    model_name_file = model_name_file.split('\\//')
    model_name_file = model_name_file[len(model_name_file)-1]
    # best_guess_accuracy, best_guess,loss, EPOCHS, BATCHES
    model_best_loss = min(history.history["loss"][int(epochs/2):])              # Get the best from the 2nd half of the data.  Use the 2nd half to avoid any early noise
    model_best_acc = max(history.history["accuracy"][int(epochs/2):])
    model_best_val_loss = min(history.history["val_loss"][int(epochs/2):])
    model_best_val_acc = max(history.history["val_accuracy"][int(epochs/2):])

    # Get the best val+train loss divided by two - this is a short cut to help show if there is divergence on the 'good' reults
    model_best_combined_ave_loss = min(np.array(history.history["loss"][int(epochs/2):]) + np.array(history.history["val_loss"][int(epochs/2):]))/2

    import platform

    with open(output_summary_logfile,'a') as fd:
        fd.write("%s,%s,%f,%f,%d,%d,%d,%f,%f,%f,%f,%f, %d,%s,%s\n" % (current_date_time, model_name_file, best_guess_acc, best_guess_loss, best_guess, EPOCHS, batch_size, model_best_acc, model_best_loss, model_best_val_acc, model_best_val_loss, model_best_combined_ave_loss,
                                                                  last_exec_duration, platform.node(), model_description))

def compile_model (model: object, loss: object, optimizer: object = default_optimizer, metrics: object = 'accuracy'):
    """
    Does the same as code in compile_and_fit, but having the ability to call this separately *may* be useful, e.g. for TPU usuage
    :param model:
    :param loss:
    :param optimizer:
    :param metrics:
    :return:
    """

    global use_lrf
    global lr_max
    global lr_min
    global clr_mode

    # TO DO: CHECK IF debug_with_dates is true, and if so, removed DATE from trainX and testX
    if debug_with_date:
        print('UNHANDLED - DATE COLUMN IN TRAIN DATA - NEED TO REMOVE BEFORE COMPILE AND FIT')
        exit(-1);

    if optimizer == 'SGD+CLR':
        optimizer = 'SGD'
        if clr_mode == 'None':
            clr_mode = 'triangular'
        if use_lrf == False:
            use_lrf = True

    if optimizer == 'SGD+NM':
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # if optimizer=='AdamDelta'
    #     optimizer = optimizers.Adadelta()

    print("[INFO] About to start with optimizer " + str(optimizer))
    print("[INFO] Model Description: " + model_description)

    # This allows us to use this function iteratively, but only compile the first time
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

def compile_and_fit(model: object, trainX: object, trainY: object, testX: object, testY: object, loss: object, optimizer: object = default_optimizer, metrics: object = 'accuracy', compile = True) -> object:
    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.01
    #    EPOCHS = 75
    global callbacks
    global use_lrf
    global lr_max
    global lr_min
    global clr_mode

    # TO DO: CHECK IF debug_with_dates is true, and if so, removed DATE from trainX and testX
    if debug_with_date:
        print ('UNHANDLED - DATE COLUMN IN TRAIN DATA - NEED TO REMOVE BEFORE COMPILE AND FIT')
        exit(-1);

    if optimizer == 'SGD+CLR':
        optimizer = 'SGD'
        if clr_mode == 'None':
            clr_mode = 'triangular'
        if use_lrf == False:
            use_lrf = True

    if optimizer=='SGD+NM':
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

   # if optimizer=='AdamDelta'
   #     optimizer = optimizers.Adadelta()

    print ("[INFO] About to start with optimizer " + str(optimizer))
    print ("[INFO] Model Description: " + model_description)

    # This allows us to use this function iteratively, but only compile the first time
    if (compile):
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])


    # Check if LR Finder is to be used, which will then set min / max LR rates  -
    # TODO: Should move this to another function later...
    if use_lrf == True:
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find((trainX, trainY), 1e-15, 1e1,                 # Was e-11 or e-12 starting - want to ensure we start at a low enough value
                 stepsPerEpoch=np.ceil((len(trainX) / float(batch_size))),
                 batchSize=lrf_batch, sampleSize=8192)         #  Increasd to 4096 from 2048 - does this help find better values?          # Using LRF batch size of 48, given data skews - want to see if this helps with early loss osciliation, which may be due to lack of uniformity in data.   Alternative will be to create more Sell Rulle data (clone)
        lrf.plot_loss()

        # Risk here if there are two minimums - this will pick the lowest, not highest.
        lr_start_index = lrf.losses.index(min(lrf.losses))
        min_loss = min(lrf.losses)
        min_loss_lr = lrf.lrs[lrf.losses.index(min(lrf.losses))]

        print("[INFO] - lowest loss rate is " + str(min_loss) + " at LR of " + str(min_loss_lr))


        segment_LRs = []
        # Experimental
        # Split the landscape into 20, and assess min/max per segment
        step = math.floor(len(lrf.losses)/20)
        for i in range(0, len(lrf.losses), step):
            # Loop through in 20 'segments
            current_min = min(lrf.losses[i:i+step])
            current_max = max(lrf.losses[i:i+step])
            current_ave = np.average(lrf.losses[i:i+step])
            segment_LRs.append((current_min,current_max,current_ave))


        # Work backwards to find where to 'start'
        step = math.floor(lrf.losses.index(min(lrf.losses)) / 10)
        max_loss = min_loss
        max_loss_lr = min_loss_lr

        if step > 0:

            for i in range(lr_start_index - step, 0, -step):
                # Iterate backwards by 10% at a time.  Stop when the loss increase is small
                current_loss = lrf.losses[i]
                if current_loss > (max_loss * 1.05):  # 5% better
                    max_loss = current_loss
                    max_loss_lr = lrf.lrs[i]
                else:
                    break
                    # We're done!  This is our min LR
        else:
            print('[INFO] Step is 0 on LRF.  This won''t work well - will set constant LR to avoid erroring')

        print("[INFO] min loss LR " + str(min_loss_lr) + " max loss LR " + str(max_loss_lr))
        # This looks backwards, as the min_loss_lr is at the max_lr (highest LR value), and the max_loss_lr is when we stop, moving right to left, so therefore the lowest LR.
        lr_max = min_loss_lr
        lr_min = max_loss_lr
        # End LRF CODE

    # Enable CLR Mode, if to be used
    if clr_mode == 'triangular':
        clr = CyclicLR(base_lr=lr_min, max_lr=lr_max,
                       step_size=800.)         # Using all defaults except Step Size - changed to 800, which should work well with a batch of 16
        callbacks.append(clr)


    if early_stop_callback == True:
        es = EarlyStopping('loss', min_delta=early_stopping_min_delta, patience=early_stopping_patience, restore_best_weights=True)
        callbacks.append(es)

    # train the neural network

    if len(callbacks) == 0:
        # No Callbacks
        H = model.fit(trainX, trainY, validation_data=(testX, testY),
                      epochs=EPOCHS, batch_size=batch_size, verbose = 2, shuffle = shuffle_data)
    else:
        H = model.fit(trainX, trainY, validation_data=(testX, testY),
                      epochs=EPOCHS, batch_size=batch_size, verbose = 2, shuffle = shuffle_data, callbacks = callbacks)

    return H


def dupe_data(xtrain, ytrain, y_val_to_dupe):
    """
    :param xtrain:
    :param ytrain:
    :param y_val_to_dupe:
    doubles the occurence of y_val_to_dupe in both the x_train and y_train, by adding them at the end
    :return:
    """
    for i in range(0, len(xtrain)):
        if ytrain[i] == y_val_to_dupe:
            # This will perform badly...  Must be a better way??
            xtrain = np.append(xtrain, [xtrain[i]], 0)
            # xtrain.append(xtrain[i])
            #ytrain.append(ytrain[i])
            ytrain = np.append(ytrain, [ytrain[i]], 0)

    # only problem with this is that the data is no longer shuffled at the end...
    xtrain, ytrain = shuffle(xtrain, ytrain)
    return xtrain, ytrain

def parse_process_plot(infile, output_col, model, output_prefix, num_samples=250):
    """
    # Would be good to be able to copy a model, to be able to run it against random data...
        # E.g. pass in a file, a model, and output filename
        # parse the data
        # copy the model
        # run the model with both real data, and randomised data
        # compare the two to show the deviation
    """
    #olhc_data, output_data = parsefile(infile, output_col)
    #xtrain, ytrain = parse_data(olhc_data, output_data, num_samples)

    # TODO: Don't need the random anymore - should change that to reduce TECH DEBT
   # xtrain, xtest, ytrain, ytest, ytrain_rnd, ytest_rnd = parse_data_to_trainXY_plusRandY (infile, output_col)

    xtrain, xtest, ytrain, ytest = parse_data_to_trainXY (infile, output_col)

    # We know how two test sets - the actual one, as well as randomised version, so we can compare the two to better understand deviation from guessing

#    model_rnd = keras.models.clone_model(model)

    meta_filename = infile.split("/")    # This feels very hacky...
    if output_col == 'SellWeightingRule' :
        resolved_meta_filename = "./" + meta_filename[1] + "/meta/" + meta_filename[2] + "_sell_meta.csv"
    elif output_col == 'BuyWeightingRule':
        resolved_meta_filename = "./" + meta_filename[1] + "/meta/" + meta_filename[2] + "_meta.csv"
    else:
        # Not yet implemented - use Sell, as it has higher axis, so avoid chance of missing data
        resolved_meta_filename = "./" + meta_filename[1] + "/meta/" + meta_filename[2] + "_sell_meta.csv"

    import time
    start = time.time()
    H = compile_and_fit(model, xtrain, ytrain, xtest, ytest, loss=model_loss_func, optimizer=default_optimizer)
#    H_rnd = compile_and_fit(model_rnd, xtrain, ytrain_rnd, xtest, ytest_rnd, loss='mean_squared_error')
    end = time.time()

    global last_exec_duration, model_last_epochs
    last_exec_duration = end - start
    model_last_epochs = len(H.history["loss"])

    # TO DO - STANDARDISED LOSS CALC - NOT A HIGH PRIORITY

 #   plot_and_save_history_with_rand_baseline(H, H_rnd, 75, output_prefix)

    plot_and_save_history_with_baseline(H, EPOCHS, output_prefix, resolved_meta_filename)

    if clear_session == True:
    # Reset Keras Session, to avoid overheads on multiple iterations
        K.clear_session()

def parse_process_plot_multi_source(infile_array, output_col, model, output_prefix, num_samples=250, version=1, pre_compiled=False):
    """
    # Same as parse_process_plot, BUT, takes multiple data sources and runs them one EPOCH at a time per file
    # No idea if this will work well or not, but seems worth trying!
    # Version 1 - keep the files data separate, and train one epoch at a time across each
    # Version 2 - merge all the TRAIN data into a single file.  Only test with the TEST data of the first file - this allows us to use the meta data
    """
    #olhc_data, output_data = parsefile(infile, output_col)
    #xtrain, ytrain = parse_data(olhc_data, output_data, num_samples)

    # TODO: Don't need the random anymore - should change that to reduce TECH DEBT
   # xtrain, xtest, ytrain, ytest, ytrain_rnd, ytest_rnd = parse_data_to_trainXY_plusRandY (infile, output_col)

    global custom_loss_override         # May temporarily change this

    xtrain = []
    xtest = []
    ytrain = []
    ytest = []
    index = 0

    for file in infile_array:
        xtrain_new, xtest_new, ytrain_new, ytest_new = parse_data_to_trainXY (file, output_col)
        index+=1
        xtrain.append(xtrain_new)
        xtest.append(xtest_new)
        ytrain.append(ytrain_new)
        ytest.append(ytest_new)

    if version == 2:
        # We need to combine our TRAIN data
        V2_xtrain = np.concatenate(xtrain)              # These are lists of xtrains, which we're now concatenating.  Pretty simple really!
        V2_ytrain = np.concatenate(ytrain)
        V2_xtrain, V2_ytrain = shuffle(V2_xtrain, V2_ytrain)

    # No meta data - multiple files!  However, take it from the 0th file, just to ensure don't break the plot function
    # 1st April 2020 - this actually works well - we TEST on the first file only, but TRAIN on all 4 files

    meta_filename = infile_array[0].split("/")    # This feels very hacky...
    if output_col == 'SellWeightingRule' :
        resolved_meta_filename = "./" + meta_filename[1] + "/meta/" + meta_filename[2] + "_sell_meta.csv"
    else:
        resolved_meta_filename = "./" + meta_filename[1] + "/meta/" + meta_filename[2] + "_meta.csv"

    global last_exec_duration
    if version == 1:
        import time
        start = time.time()

        global EPOCHS
        global model_last_epochs
        # This is a bit messy, but I now need to set EPOCHS to 1 to iterate, and then set it back later...
        EPOCHS_orig = EPOCHS
        EPOCHS = 1

        iteration = 0
        for epoch in range (0, EPOCHS_orig):
            for file in range (0, len(xtrain)):
                print ('[INFO] Iteration ' + str(epoch) + ' on file number ' + str(file))
                if iteration == 0:
                    compile = False if pre_compiled else True
                    H1 = compile_and_fit(model, xtrain[file], ytrain[file], xtest[file], ytest[file], loss=model_loss_func, optimizer=default_optimizer, compile = compile)
                else:
                    H1 = compile_and_fit(model, xtrain[file], ytrain[file], xtest[file], ytest[file],
                                         loss=model_loss_func, optimizer=default_optimizer, compile=False)
                if iteration == 0:
                    My_H = H1
                else:
                    My_H.history["loss"].append(H1.history["loss"][0])           # De-reference to turn list into float
                    My_H.history["val_loss"].append(H1.history["val_loss"][0])
                    My_H.history["accuracy"].append(H1.history["accuracy"][0])
                    My_H.history["val_accuracy"].append(H1.history["val_accuracy"][0])

                iteration+=1
    #    H_rnd = compile_and_fit(model_rnd, xtrain, ytrain_rnd, xtest, ytest_rnd, loss='mean_squared_error')

        EPOCHS = EPOCHS_orig

        end = time.time()

        last_exec_duration = end - start
        model_last_epochs = len(My_H.history["loss"])

     #   plot_and_save_history_with_rand_baseline(H, H_rnd, 75, output_prefix)

        plot_and_save_history_with_baseline(My_H, EPOCHS * len(xtrain), output_prefix, resolved_meta_filename)      # Multiple EPOCHS by the number of input files.

    elif version == 2:
        # Version 2 means V2 xtrain and ytrain contain the concatenated data.  We are TESTING though on xtest[0] / ytest[0] ONLY - i.e. just the first data file.
        # This allows us to compare the results with single file only, which is VERY USEFUL, given different files could have different distributions, even after normalising (this would be interesting to test...)
        import time
        start = time.time()
        H = compile_and_fit(model, V2_xtrain, V2_ytrain, xtest[0], ytest[0], loss=model_loss_func, optimizer=default_optimizer, compile = False if pre_compiled else True )
        #    H_rnd = compile_and_fit(model_rnd, xtrain, ytrain_rnd, xtest, ytest_rnd, loss='mean_squared_error')
        end = time.time()

        last_exec_duration = end - start
        model_last_epochs = len(H.history["loss"])

        #   plot_and_save_history_with_rand_baseline(H, H_rnd, 75, output_prefix)

        # Due to early learning stopping, need to check

        plot_and_save_history_with_baseline(H, EPOCHS, output_prefix, resolved_meta_filename)

    # Do an extra evaluate here with mse as loss function - can be useful and low overhead.
    global standardised_loss

    if model_loss_func != default_model_loss_func :
        print ("[INFO] Attempting model loss override bit with re-valuate")
        prior_loss_override = custom_loss_override
        custom_loss_override = True     # Override and use MSE, so we can compare

        # Recompile the model, to make sure the new loss setting takes effect.  It may be easier to just reset the loss function here to mse, rather than the complex logic of the 'custom loss override'
        # I will try this and think about the change.  This would reduce the need for the extra global variable, as well as needing code in each loss function to manage...
        # Seems like a good idea!
        model.compile(loss=default_model_loss_func, optimizer=default_optimizer, metrics=['accuracy'])      # This feels a bit hacky...
        results = model.evaluate(xtest[0], ytest[0], batch_size=batch_size)
        print(results)

        standardised_loss = results[0]          # Is this correct?  Seems like it should be

        # Change it back - it may not actually be set.  This is a bit of a hack to allow changing of loss function for final eval.
        custom_loss_override = prior_loss_override

        model.compile(loss=model_loss_func, optimizer=default_optimizer, metrics=['accuracy'])      # Recompile with the setting reversed, just in case it is used again elsewhere.
    else:
        # Using default loss function, but do another evaluate (with compile) to ensure a true like for like comparison is possible
        model.compile(loss=default_model_loss_func, optimizer=default_optimizer, metrics=['accuracy'])      # This feels a bit hacky...
        results = model.evaluate(xtest[0], ytest[0], batch_size=batch_size)
        standardised_loss = results[0]          # Is this correct?  Seems like it should be


    if clear_session == True:
    # Reset Keras Session, to avoid overheads on multiple iterations
        K.clear_session()

###  File Functions
def parse_file (infilename, purpose='Train', prefix=''):
    """
    Reads a CSV file in format of date,open,high,low,close, with optional volume
    Parses this based on defined rules (see below) and writes to the ..\parsed_data\<filename>_parsed.csv

    :param infile: filename, stored in ..\input_data   OR  a pandas DataFrame
    ;      purpose: is this for training or eval?  If eval, process ALL rows to the end, and use a different path to avoid issues
    ;       prefix: useful when parsing in a Pandas - add the prefix to the filename to save it
    :return: true, unless an error
    """

    # read it in IF its a string - meaning its a filename
    if isinstance(infilename, str):
        infile = pd.read_csv(Path("./input_data/" + infilename), engine="python")
    else:
        infile = infilename
        infile.to_csv(Path("./input_data/" + prefix+'tempfile.csv'), index=True)     # Write it out for reference - no plans to use currently, but may be useful to check
        infilename = prefix+'tempfile.csv'     # Set a new (temp) filename to write parsed output to

    # Need to ignore the last row, as doesn't have a valid indicator.
    # So, we need to build up an array that has # examples of 250 x 4 for XTrain - 250 rows of O H L C (can ignore date)
    # Need to 'slice' the data somehow :)

    # Drop Null or other invalid data
    #    infile = infile.apply(pd.to_numeric, errors='coerce')
    # try to convert only the columns we want - Open, High, Low, Close - leave date alone
    infile['Open'] = pd.to_numeric(infile['Open'], errors='coerce')
    infile['High'] = pd.to_numeric(infile['High'], errors='coerce')
    infile['Low'] = pd.to_numeric(infile['Low'], errors='coerce')
    infile['Close'] = pd.to_numeric(infile['Close'], errors='coerce')
    infile = infile.dropna(subset=['Open', 'Close'])
    infile = infile.reset_index(drop=True)


    # COMMENT - 10th JUNE 2020 - This looks to be dropping the last row TOO EARLY.  It is correct to remove it - there cannot be a prediction - but is okay to keep until the end (same as the other 20 rows taht get dropped)
    # Not a top priority - it just reduces training by one sample...
    if (purpose == 'Train'):
        # We drop the last row - its no use.  We now have the four columns of data we need.  Need to now create this into len-249 samples ready for Keras to process
        # ohlc_data = infile.loc[0:(len(infile) - 2), 'Date':'Close']
        ohlc_data = infile.loc[0:, 'Date':'Close']      # 10th June - no need to filter last row here, is there??
    else:
        ohlc_data = infile.loc[0:(len(infile)), 'Date':'Close']
    ohlc_data.insert(5, "BuyWeightingRule", 0)   # insert 0's.
    ohlc_data.insert(6, "TrueRange", 0)
    ohlc_data.insert(7, "SellWeightRule", 0)    # insert 0's
    ohlc_data.insert(8, "ATR", 0)               # Adding this to calc and maintain ATR.  There are now different rules, so this is the best way to deal with this
    ohlc_data.insert(9, "P2L_Ratio", 0)
                    # Add two more - this allows testing of the buy and sell parts separately
    ohlc_data.insert(10, "P2L_RatioPositive", 0)        # Removes negatives (sets to 0)  - effectively a new buy rule
    ohlc_data.insert(11, "P2L_RatioNegative", 0)         # Removes positives, and then makes -ve positive - effectively a new sell rule
    ohlc_data.insert(12, "P2L_RatioNeutral", 0)          # 1 if between 0.5 and 2, else 0
    ohlc_data.insert(13, "RangeVariance", 0)             # WIP - currently compares the future range vs the ATR.  Is that useful?
                                                         # May be better to compare the future 20D range with the past 20D range?

    # Convert to a numpy
    ohlc_np: numpy = ohlc_data.to_numpy()

    del ohlc_data           # avoid any mistakes!

    # WE NOW HAVE AN OHLC Numpy array that is created
    # Next step is to calculate all output values based on the future, and then to normalise based on the ATR value


    if purpose == 'Train':
        end_row =  len(ohlc_np)-20+2   # Not quite sure why + 2 rather than + 1, but it works...
    else:
        end_row = len(ohlc_np)

    for i in range (end_row):
        # Define some useful values - Future max / min over 20 periods - used to calc our Buy and Sell Rules
        #       near_future_max = max(ohlc_data['High'][i+1:i+20]-ohlc_data['Close'][i])/ohlc_data['Close'][i]
        #       near_future_mix = min(ohlc_data['High'][i+1:i+20]-ohlc_data['Close'][i])/ohlc_data['Close'][i]
        #numpy is row, col

        if (i < len(ohlc_np)-20+1):    # remove the -2

            near_future_max = (max(ohlc_np[i+1:i+21, 2])-ohlc_np[i,4])/ohlc_np[i,4]
            near_future_min = (min(ohlc_np[i+1:i+21, 3])-ohlc_np[i,4])/ohlc_np[i,4]
            #        near_future_min = (min(ohlc_np[i+1:i+21, 2])-ohlc_np[i,4])/ohlc_np[i,4]    #Orig rule - but used high, rather than low.  Correct now
        else:
            near_future_max = 0
            near_future_min = 0

        # True Range and ATR20 - We need this for later processing into percentages.
        if i == 0:
            #   ohlc_np[i, 6] = 0
            ohlc_np[i, 6] = ohlc_np[i, 2] - ohlc_np[
                i, 3]  # 22nd March - don't use 0 - 0 is less valid than at least using this!!
            ohlc_np[i, 8] = ohlc_np[i, 6]  # ATR
        else:
            #        ohlc_data[i, 6] = max (ohlc_data.iloc[i,2] - ohlc_data.iloc[i,3], abs(ohlc_data.iloc[i,2] - ohlc_data.iloc[i-1,3]), abs(ohlc_data.iloc[i,3] - ohlc_data.iloc[i-1,4]))
            ohlc_np[i, 6] = max(ohlc_np[i, 2] - ohlc_np[i, 3], abs(ohlc_np[i, 2] - ohlc_np[i - 1, 4]),
                                abs(ohlc_np[i, 3] - ohlc_np[i - 1, 4]))
            ohlc_np[i, 8] = ohlc_np[i - 1, 8] * atr_beta_decay + (1 - atr_beta_decay) * ohlc_np[i, 6]
        # Max between H - L, H - Yesterdays Close, Low - Yesterdays Close

        # Calc the max - min as a ratio of the ATR20, noting that the first are percentages at this point
        ohlc_np[i, 13] = (near_future_max - near_future_min) / (ohlc_np[i, 8] / ohlc_np[i, 4])


        # TODO:  Define a BuyWeightingRuleFunction.  Pass ohlc numpy array, and current row number - this provides maximum flexibility.
            # Question is - what would I pass in?  Probably the np array from i, and let the function figure the rest out...
            # Would be a good idea to try that with the SELL Rulr, and the back fit it when ready.
            # Lazy option would be to pass in near
        # Needs the ATR20 before the buy/sell rule though....  do it later :)  Maybe practice a lamda function?

    # # VERSION 1 BUY RULE
    #     ### BUY RULE PROCESSING  ####
    #     if near_future_min < -0.03:  # Changed from 2% to 3%.   Will change to ATR% X 3 or something like that later.  Need to change the order though
    #         #           ohlc_data.iloc[i,5] = 0          # Don't want to buy if a big drop coming up.  Is this too conservative?
    #         ohlc_np[i, 5] = 0
    #     elif near_future_max > 0.10:
    #         #           ohlc_data[i,5] = 5
    #         ohlc_np[i, 5] = 5
    #     elif near_future_max > 0.075:
    #         #          ohlc_data[i,5] = 4
    #         ohlc_np[i, 5] = 4
    #     elif near_future_max > 0.05:
    #         #        ohlc_data[i,5] = 3
    #         ohlc_np[i, 5] = 3
    #     elif near_future_max > 0.02 and near_future_max > abs(
    #             2 * near_future_min):  # 10th June - added ABS in - was missing before.  This may have hurt training!
    #         ohlc_np[i, 5] = 2  # May change later - keep same now for consistency and bug checking
    #     #       ohlc_np[i,5] = 2
    #     else:
    #         #            ohlc_data[i,5] = 0
    #         ohlc_np[i, 5] = 1


        atr20 = ohlc_np[i, 8]/ohlc_np[i,4]      # Divide by close to get to a percentage
        ### BUY RULE PROCESSING  ####
        if near_future_min < -3*atr20 or -near_future_min>near_future_max :          # If min is 3x ATR, don't buy. Also, don't buy if future loss potential is more than future profit potential
            ohlc_np[i,5] = 0
        elif near_future_max > 8*atr20 and near_future_max > -1 * 8 * near_future_min:
 #           ohlc_data[i,5] = 5
            ohlc_np[i, 5] = 5
        elif near_future_max > 4*atr20 and near_future_max > -1 * 4 * near_future_min:
  #          ohlc_data[i,5] = 4
            ohlc_np[i, 5] = 4
        elif near_future_max > 3*atr20 and near_future_max > -1 * 3 * near_future_min:
    #        ohlc_data[i,5] = 3
             ohlc_np[i, 5] = 3
        elif near_future_max > 2*atr20 and near_future_max > abs(2 * near_future_min) :        # 10th June - added ABS in - was missing before.  This may have hurt training!
            ohlc_np[i,5] = 2      # May change later - keep same now for consistency and bug checking
     #       ohlc_np[i,5] = 2
        elif near_future_max > -1 * near_future_min :
#            ohlc_data[i,5] = 0
            ohlc_np[i,5] = 1        # Don't trade off a 1 - but may help with training, in that its better than a 0
        else:
            ohlc_np[i, 5] = 0
        #### TODO :   CAN PROFIT

        #### Sell Rule ###
        if (i < len(ohlc_np) - 20 + 1):   # remove the -2
            sell_weighting_value, p2l_ratio = sell_weighting_rule(ohlc_np, i)
            ohlc_np[i,7] = sell_weighting_value
            # Need to add new column for P2L_Ratio
            ohlc_np[i,9] = math.log(p2l_ratio+0.000001)         # Normalises the value - 0 means P = L.  Log stops outliers.  Ideally should cap, say at +/- 5...
            ohlc_np[i,10] = ohlc_np[i,9] if (p2l_ratio >= 1) else 0    # Basically 0 if potential for loss greater than profit
            ohlc_np[i,11] = -ohlc_np[i,9] if (p2l_ratio < 1) else 0    # Basically 0 if potential for loss greater than profit
            ohlc_np[i,12] = 0 if (p2l_ratio > 2 or p2l_ratio < 0.5) else 1    # Basically 0 if potential for loss greater than profit

    # END OF FIRST LOOP #############

    if purpose == 'Train':
        # Need to calc the "ATR20" for rows missed in the above...
        # WHAT IS THIS NEEDED FOR??  I think these rows are dropped later anyway??
        for i in range(end_row, end_row+20-2):
            ohlc_np[i, 6] = max(ohlc_np[i, 2] - ohlc_np[i, 3], abs(ohlc_np[i, 2] - ohlc_np[i - 1, 4]),
                                abs(ohlc_np[i, 3] - ohlc_np[i - 1, 4]))
            ohlc_np[i, 8] = ohlc_np[i - 1, 8] * atr_beta_decay + (1 - atr_beta_decay) * ohlc_np[i, 6]



    # We have to lose the first and last 20 rows - we don't have valid data for them for Y values or ATR values
#    for i in range (len(ohlc_np)-20 +1, 20-1-1, -1):

    # 15th March - the above is true for training, but for prediction, need to calculate to the last rows.  Do that now, and strip the data depending on the use case.
    # NB: This comment seems old??

    # Loop to convert the OHLC values to ratios based on the ATR20.  ATR20 is already in ohlc[,8]
    for i in range(len(ohlc_np)-1 , 20 - 1 - 1, -1):
        # Max between H - L, H - Yesterdays Close, Low - Yesterdays Close

        #Work backwards, so we don't overwrite'
        if i == 0:
            ohlc_np[0,1] = 0
            ohlc_np[0,2] = 0
            ohlc_np[0,3] = 0
            ohlc_np[0,4] = 0
        else:
            #            atr20 = mean(ohlc_np[i-19:i+1,6])           # This is basically today working back 20...
            # 22nd March.  The ATR20 has potentially large impacts.  One the positive, it helps normalise the data in a way that should be invariant to its range and normal volatility.
            # On the negative, the current approach risks filtering significant movements from even appearing in the data at all.
            # The experience to date is that the current rule certainly is okay for training - some models get effectively 100% accuracy and 0 loss. HOWEVER, the validation is not as good, and even worse,
            # when used on extreme events such as Feb/Mar 2020, it was doing badly - giving BUY signals during a massive drop.  It is plausible that the ATR as currently calculated reduced the volatility appearing in the data, which is not helpful.
            # There are potentially two key problems here:
            # Firstly - the use of MAX vs AVERAGE.  This leads to big changes in the calculation as new data enters, having the large value effectively immediately impact the normalisation in a dramatic way.
            # Secondly, the duration over which is is calculated - a longer range would reduce the impacts here.
            # So, to retain backwards compatibility, but also try some new things, here are the options:
            # Rule 1 - As is ATR20 based on max over the past 20 days, using MAX Values
            # Rule 2 - Use an average of some kind over this period (20 days).  Need to work out how best to do this - probably just averaging ohlc_np[i, 6] ??
            # Rule 3 - An alpha/betay decay - need to look this up, but a value of say .95, which takes the existing value at 0.95 and adds the current.  This is used in many functions, and seems to have the same effect as an EMA.  The
            #       other good positive is that it doesn't need n days of history - it builds itself up.
            # Moving this code - the original atr20 that was here doesn't seem to be used...

            if atr_rule == 1:
                # Calculate the percentages instead of absolute values, based on HISTORIC values up and INCLUDING today
                #        atr20 = max (ohlc_data.iloc[i,2] - ohlc_data.iloc[i,3], abs(ohlc_data.iloc[i,2] - ohlc_data.iloc[i-1,3]), abs(ohlc_data.iloc[i,3] - ohlc_data.iloc[i-1,4]))
                #atr20 = max(ohlc_np[i, 2] - ohlc_np[i, 3], abs(ohlc_np[i, 2] - ohlc_np[i - 1, 3]),
                #            abs(ohlc_np[i, 3] - ohlc_np[i - 1, 4]))
                atr20 = mean(ohlc_np[i - 19:i + 1, 6])  # This is basically today working back 20...
            elif atr_rule == 3:
                # Already calculated - simply re-use the value.
                atr20 = ohlc_np[i,8]

            ohlc_np[i,1] = (ohlc_np[i,1] - ohlc_np[i-1,4])/atr20   #  Subtract yesterdays' close
            ohlc_np[i,2] = (ohlc_np[i,2] - ohlc_np[i-1,4])/atr20    #  Subtract yesterdays' close
            ohlc_np[i,3] = (ohlc_np[i,3] - ohlc_np[i-1,4])/atr20    #  Subtract yesterdays' close
            ohlc_np[i,4] = (ohlc_np[i,4] - ohlc_np[i-1,4])/atr20    #  Subtract yesterdays' close

    #return olhc_data, output_data
    import numpy

    # Capture last ATR20 value before dropping column
    atr20 = ohlc_np[-1, 8]

    # Need to drop coulumn 6 and 8 - the TR and ATR_Beta.
#    ohlc_np = ohlc_np[:,(0,1,2,3,4,5,7)]            # This effectivly drops column 6, the ATR.  This feels clumsy, but works.
    ohlc_np = ohlc_np[:,(0,1,2,3,4,5,7,9,10,11,12, 13)]            # This effectivly drops column 6, the ATR.  This feels clumsy, but works.  Keeping 9 - P2L Ratio

    if atr_rule > 1:
        infilename = 'Rule' + str(atr_rule) + "_B" + str (atr_beta_decay) + infilename

    ## WHERE TO SAVE IF NO FILENAME????   A TEMP FILE??

#    numpy.savetxt(".\\parsed_data\\" + infilename , ohlc_np[19:len(ohlc_np)-20 +1,0:6], delimiter=',', header='Date,Open,High,Low,Close,BuyWeightingRule', fmt=['%s','%f','%f','%f','%f','%f'], comments='')
    if purpose == 'Train':
        numpy.savetxt(".\\parsed_data\\" + infilename , ohlc_np[19:len(ohlc_np)-20 +1,0:12], delimiter=',', header='Date,Open,High,Low,Close,BuyWeightingRule,SellWeightingRule,P2LRatio,P2LRatioPositive,P2LRatioNegative,P2LRatioNeutral,RangeVariance', fmt=['%s','%f','%f','%f','%f','%f', '%f', '%f','%f', '%f', '%f', '%f' ], comments='')
    else:
        numpy.savetxt(".\\parsed_data_full\\" + infilename, ohlc_np[19:, 0:12], delimiter=',',
                      header='Date,Open,High,Low,Close,BuyWeightingRule,SellWeightingRule,P2L_Ratio,P2L_RatioPositive,P2L_RatioNegative,P2L_RatioNeutral,RangeVariance',
                      fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'], comments='')
#    print (ohlc_np           )
    #pd.DataFrame(ohlc_np[19:,0:6]).to_csv(".\\parsed_data\\" + infilename)

    # Could also output an 'easy' min loss and corresponding accuracy, by calculating the loss / accuracy of just using each number in the target, and taking the lower loss.
    # This would be a useful number to compare against - it should be very similar to what 'random' produces over time.  Using this instead of random would be easier, and save time
    # To do this, I need to calculate:
        # Sum of all errors if 0, 1, 2, 3, 4, 5

    # Little hack - return ATR20 of the last value, allowing it to be used in predict.  Messy, but effective!

    if purpose != 'Train':
        return ohlc_np[-1,0], atr20     # Return last date and atr20 of last date

    buy_best_ave_loss = 999
    buy_best_accuracy = 0
    sell_best_ave_loss = 999
    sell_best_accuracy = 0
    i: int
    for i in range(int(max(ohlc_np[:,5])+1)):
        buy_current_ave_loss = np.sum((ohlc_np[:,5] - i) * (ohlc_np[:,5] - i))  / (len(ohlc_np))
        buy_current_accuracy = np.bincount(ohlc_np[:,5].astype(np.int))[i] / len(ohlc_np)       # Count the number of i's via bincount, and then get as a percentage
        sell_current_ave_loss = np.sum((ohlc_np[:,6] - i) * (ohlc_np[:,6] - i))  / (len(ohlc_np))
        if i >= len(np.bincount(ohlc_np[:,6].astype(np.int))):
            # Deal with the fact that we may have different ranges for buy and sell
            sell_current_accuracy = 0
        else:
            sell_current_accuracy = np.bincount(ohlc_np[:,6].astype(np.int))[i] / len(ohlc_np)       # Count the number of i's via bincount, and then get as a percentage

        if buy_current_ave_loss < buy_best_ave_loss :
            buy_best_ave_loss = buy_current_ave_loss
            buy_best_accuracy = buy_current_accuracy
            buy_best_guess = i

        if sell_current_ave_loss < sell_best_ave_loss :
            sell_best_ave_loss = sell_current_ave_loss
            sell_best_accuracy = sell_current_accuracy
            sell_best_guess = i

    print ('Buy Rule Best guess is ' + str(buy_best_guess) + ' with loss of ' + str(buy_best_ave_loss) + ' and accuracy of ' + str(buy_best_accuracy))
    print ('Sell Rule Best guess is ' + str(sell_best_guess) + ' with loss of ' + str(sell_best_ave_loss) + ' and accuracy of ' + str(sell_best_accuracy))
    numpy.savetxt(".\\parsed_data\\meta\\" + infilename + "_meta.csv", np.array([buy_best_guess, buy_best_ave_loss, buy_best_accuracy]), header='Data')
    numpy.savetxt(".\\parsed_data\\meta\\" + infilename + "_selL_meta.csv", np.array([sell_best_guess, sell_best_ave_loss, sell_best_accuracy]), header='Data')

def buy_weighting_rule (ohlc_np, index):
    """

    :param ohlc_np: numpy array of d,o,h,l,c, buyrule,atr20
    :param index: row number we're processing
    :return:
    """

def sell_weighting_rule (ohlc_np, index, rule_version = 2):
    """

    :param ohlc_np: Numpy array of OHLC
    :param index: What index number we're up to
    :return: Sell Rule Value

    """
    near_future_max = (max(ohlc_np[index + 2:index + 21, 2]) )          # Highest price after tomorrow for next 20 days

    # Sell Rule:
    # If tomorrow's Low (i+1, 3) is higher than the next 20 days High (i+1:i+21, 2 - near_future_max), then we should sell tomorrow
    # Also sell if average of tomorrows open, close, high is higher than the next 20 days close
    # =IF(D3>MAX(C4:C23),2,IF(AVERAGE(B3,C3,E3)>MAX(E4:E23),1,0))

    # TODO:  Use a proper Average function, to make the code cleaner to read

    if rule_version == 1:
        # Original Rule - not overly good, which may limit ability to train for it
        if ohlc_np[index+1, 3] > near_future_max :                  # if tomorrow's Low is higher than the next NN days High - DEFINITE SELL!!!
            return 3
        elif ((sum(ohlc_np[index+1, 1:3]) + ohlc_np[index+1, 4])/3) > max(ohlc_np[index+2:index+21, 4]) :       # Average of O, H, C (not, low, which is 3)
            return 2
        elif ((sum(ohlc_np[index + 1, 1:3]) + ohlc_np[index + 1, 4]) / 3) > max(
            ohlc_np[index + 2:index + 11, 4]):  # Average of O, H, C (not, low, which is 3)
            return 1
        else :
            return 0
    elif rule_version == 2 :
        # Better rule - uses a near term likely sell price to determine whether or not to sell.  A better rule may make it easier to train, which is obviously preferable
        # Need to create a new numpy column, for likely_sell_price_tomorrow

        # For each day, this is the average of tomorrow's open and low - i.e. we're being pessimistic, noting we'll likely exceed this.
        # Should ideally move this to the outer loop, so this is done once only.  For now, doing it here
        #likely_sell_price_tomorrow_np = np.fromfunction (lambda i, j, ohlc_np : (ohlc_np[index+1+i, 3] + ohlc_np[index+1+i, 1]) / 2, (22, 1))
        likely_sell_price_tomorrow_np = np.zeros(20)
        for i in range(20):
          #  print(str(index) + " index and i:" + str(i))
            if (index+i+2)>len(ohlc_np):
                likely_sell_price_tomorrow_np[i] = likely_sell_price_tomorrow_np[i-1]
                print ("Sell Rule overrun i: "+ str(i) + " index: " + str (index) )
            else:
                likely_sell_price_tomorrow_np[i] = (ohlc_np[index+1+i, 3] + ohlc_np[index+1+i, 1]) / 2

        # index 0 in the new NP is TOMORROW
        max_loss_20d = likely_sell_price_tomorrow_np[0] - min(likely_sell_price_tomorrow_np[1:])
        max_profit_20d = max(likely_sell_price_tomorrow_np[1:]) - likely_sell_price_tomorrow_np[0]
        # remove negatives - they stuff things up!
        if max_loss_20d <= 0:           # Changed to < to avoid divide by zero and infinity issues
            max_loss_20d = 0.01   # Add the 0.1 to avoid divide by zero errors. 
        if max_profit_20d < 0:
            max_profit_20d = 0.01
        p2l_ratio = (max_profit_20d)/(max_loss_20d)

        # Want to keep the p2l_ratio - may be useful.  Options - return a tuple OR, set the p2l_ratio as a global...
        # Let's return a tuple, as this is only used in one place.

# Original V2 Ratios - Found these VERY hard to train for - is this even useful??
#        if p2l_ratio < 0.0001 :
#            return 10, p2l_ratio
#        elif p2l_ratio < 0.001 :
#            return 5, p2l_ratio
#        elif p2l_ratio < 0.01 :
#            return 2, p2l_ratio
#        elif p2l_ratio < 0.1 :
#            return 1, p2l_ratio
#        else:
#            return 0, p2l_ratio

        if p2l_ratio < 0.0001:
            return 15, p2l_ratio
        if p2l_ratio < 0.001 :
            return 10, p2l_ratio
        elif p2l_ratio < 0.01 :
            return 7.5, p2l_ratio
        elif p2l_ratio < 0.1 :
            return 5, p2l_ratio
        elif p2l_ratio < 0.25 :
            return 2, p2l_ratio
        elif p2l_ratio < 0.5 :
            return 1, p2l_ratio
        else:
            return 0, p2l_ratio


def get_db_connection ():
    return mysql.connector.connect(user=db_username, password=db_pwd,
                                  host=db_host,
                                  database='tfpp')

def read_from_from_db(sort='None', unique_id=None):     # Sort can be None, Random, NodesAsc, NodesDesc, StdLossDescByXXX
    cnx = get_db_connection()

    cursor = cnx.cursor(dictionary=True)

    if unique_id != None:
        query = "SELECT * FROM testmodels where unique_id = %s "
        cursor.execute(query, (unique_id,))
    else:
        if sort == 'None':
            query = ("SELECT * FROM testmodels "
                     "where started <> 'True' " )
        elif sort == 'NodesAsc':
            if processing_rule in ('Buy', 'BuyV1'):
                query = ("SELECT * FROM testmodels "
                     "where started <> 'True' "
                     "order by TotalNodes asc, priority desc")
            else:
                query = ("SELECT * FROM testmodels "
                         "where unique_id not in "
                         "(select unique_id from tfpp.executionlog el "
                         "where el.Rule = '" + str (processing_rule) +"') "
                         "order by TotalNodes asc, priority desc")
        elif sort == 'NodesDesc':
            query = ("SELECT * FROM testmodels "
                     "where started <> 'True' "
                     "order by TotalNodes desc, priority desc")
        elif sort == 'Random':
            if processing_rule in ('Buy', 'BuyV1'):
                query = ("SELECT * FROM testmodels "
                         "where started <> 'True' "
                         "order by priority desc, RAND()")
            else:
                query = ("SELECT * FROM testmodels "
                        "where unique_id not in "
                        "(select unique_id from tfpp.executionlog el "
                        "where el.Rule = '" + str (processing_rule) +"') "
                        "order by priority desc, RAND()")
        else:
            # Must mean its StdLossDescByXXX
            # This means we need to get the XXX (Rule) and then we want to find the lowest Standardised Loss from that rule not already processed
            source_rule = sort[13:]
            query = ("SELECT * FROM testmodels "
                        "where unique_id in "
                        "(select * from "
                        "(select unique_id from executionlog el1 "
                        "where rule = '" + source_rule + "' "
                        "and unique_id not in "
                        "(select unique_id from executionlog el2 where rule = '" + str (processing_rule) +"') "
                        "order by standardised_loss asc limit 1) inner_t )")
        query = query + " LIMIT 1"
        cursor.execute(query)

    rowDict = cursor.fetchone()
    print(rowDict)

    # Now update it to started
    uniqueID = rowDict['unique_id']
    cursor.execute("""
        UPDATE testmodels SET Started='True'
        WHERE unique_id = %s
    """, (uniqueID,))

    # for i in (cursor): print (i)
    cnx.commit()
    cnx.close()

    # Convert strings to Dicts
    for layer in range(1, rowDict['Layers']+1):
        rowDict['Layer' + str(layer)] = eval(rowDict['Layer' + str(layer)])

    return rowDict

def db_update_row (rowDict, success=True):
    global model_best_acc, model_best_combined_ave_loss, model_best_loss, model_best_val_acc, model_best_val_loss

    if success == False:
        rowDict['Finished'] = False
        model_best_acc = 0
        model_best_loss = 9999999
        model_best_val_acc = 0
        model_best_val_loss = 9999999
        model_best_combined_ave_loss = 9999999

    for db_connect_loop in range(1,24*12):
        try:
            cnx = get_db_connection()  #mysql.connector.connect(user=db_username, password=db_pwd,
                                     #     host=db_host,
                                     #     database='tfpp')
            break
        except:
            print ('[INFO: DB Connect Error on try number ' + str(db_connect_loop))
            time.sleep (5*60)

    uniqueID = rowDict['unique_id']

    cursor = cnx.cursor(dictionary=True)

    query = ("SELECT 1")
    cursor.execute(query)
    cursor.fetchone()

    # Legacy - only use if BuyRule1
    if processing_rule in ('Buy', 'BuyV1'):

        rowcount = cursor.execute("""
            UPDATE testmodels SET Started='True'
            WHERE unique_id = %s
        """, (uniqueID,))



        cursor.execute("""
            UPDATE testmodels SET Finished='True'
            where unique_id = %s
        """, (uniqueID,))


        query = ("UPDATE testmodels SET Finished = %s "
                     "WHERE unique_id = %s")
        cursor.execute(query, ('True', uniqueID))

        print ("[INFO] Update DB for unique id " + str(uniqueID) + "with Error Details (if any) : " + str(rowDict['ErrorDetails']))


        query = ("UPDATE testmodels SET Finished = %s, "
                     "model_best_acc = %s, "
                     "model_best_loss = %s, "
                     "model_best_val_acc = %s, "
                     "model_best_val_loss = %s, "
                     "model_best_combined_ave_loss = %s, "
                     "Epochs = %s,"
                     "ErrorDetails = %s "
                     "WHERE unique_id = %s")

        rowcount = cursor.execute(query, ('True', float(model_best_acc),
                      float(model_best_loss),
                      float(model_best_val_acc),
                      float(model_best_val_loss),
                      float(model_best_combined_ave_loss),
                      int(model_last_epochs),
                      str(rowDict['ErrorDetails']),
                      uniqueID))

        if rowcount != 1:
            print ("[ERROR]: Update Row DB has returned row count of " + str (rowcount))
            print ('Full ModelDict follows')
            print (rowDict)
    # End Legacy

    # Now, we don't actually update the testmodels table at all - all the data will be updated to execution log, which can be sliced and diced as required to get up to date info per rule, etc, etc.



    # Before we copy the Dict, update the new values that may be used elsewhere
    rowDict.update(model_best_acc=float(model_best_acc))
    rowDict.update(model_best_loss=float(model_best_loss))
    rowDict.update(model_best_val_acc=float(model_best_val_acc))
    rowDict.update(model_best_val_loss=float(model_best_val_loss))
    rowDict.update(model_best_combined_ave_loss=float(model_best_combined_ave_loss))

    # Update our ExecLog Table with INSERT
    insert_dict = rowDict.copy()
    #add extra pieces
    # Labels NOT in rowDict
    # DateTime,FileName,BestGuessAcc,BestGuessLoss,BatchSize,ExecDuration,ExecMachine,Description,LayersCombined
    # Differences
    # EPOCHS, Epochs_Actual
    # ColumnsToRemove
    # Started,Finished,LastUpdate

    # Inconsistent results with / and \ characters - just replace for reliability!
    model_name_file = sys.argv[0].replace('/', '\\').split('\\')
    model_name_file = model_name_file[len(model_name_file) - 1]

    import platform

    insert_dict.update(DateTime = datetime.datetime.now())
    insert_dict.update(FileName = model_name_file)
    insert_dict.update(BestGuess = best_guess)
    insert_dict.update(BestGuessAcc = best_guess_acc)
    insert_dict.update(BestGuessLoss = best_guess_loss)
    insert_dict.update(BatchSize = batch_size)
    insert_dict.update(ExecDuration = last_exec_duration)
    insert_dict.update(ExecMachine = platform.node())
    insert_dict.update(Description = model_description)
    insert_dict.update(LayersCombined = str(insert_dict['Layer1']) + str(insert_dict['Layer2']) + str(insert_dict['Layer3']) + str(insert_dict['Layer4']) + str(insert_dict['Layer5']) )
    insert_dict.update(Rule = processing_rule)
    insert_dict.update(standardised_loss = standardised_loss)           # Used to compare between different loss functions that may be used on training

    # Update newly calc'ed values- these have NOT yet been set in the Dict.
    # Should they??

    # EPOCHS, Epochs_Actual
    # Keep Epochs - its the target.
    insert_dict.update(Epochs= EPOCHS)
    #insert_dict.pop('Epochs', None)

    insert_dict.update(Epochs_Actual = model_last_epochs)

    # Started,Finished,LastUpdate
    insert_dict.pop('Started', None)
    insert_dict.pop('Finished', None)
    insert_dict.pop('LastUpdate', None)
    insert_dict.pop('priority', None)

    # Parse Dict to String for Layers
    insert_dict.update(Layer1 = str(insert_dict['Layer1']))
    insert_dict.update(Layer2 = str(insert_dict['Layer2']))
    insert_dict.update(Layer3 = str(insert_dict['Layer3']))
    insert_dict.update(Layer4 = str(insert_dict['Layer4']))
    insert_dict.update(Layer5 = str(insert_dict['Layer5']))

    placeholder = ", ".join(["%s"] * len(insert_dict))
    insert_stmt = "insert into executionlog ({columns}) values ({values});".format(columns=",".join(insert_dict.keys()), values=placeholder)
    cursor.execute(insert_stmt, list(insert_dict.values()))

    cnx.commit()
    cnx.close()

def read_row (datafile):
    """
    Read the next NON-STARTED row and update the flag to STARTED
    :return: 
    """
# Read in the Models file
    tempfilename = os.path.splitext(datafile)[0] + '.bak'
    try:
        os.remove(tempfilename)  # delete any existing temp file
    except OSError:
        pass

    os.rename(datafile, tempfilename)

    # create a temporary dictionary from the input file
    with open(tempfilename, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        #header = next(reader)  # skip and save header
        outfile = open(datafile, mode='w', newline='')
        fieldnames = ['Layers', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Started', 'Finished',
                      'model_best_acc',
                      'model_best_loss', 'model_best_val_acc', 'model_best_val_loss',
                      'model_best_combined_ave_loss', 'ErrorDetails', 'TotalNodes']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        return_row = None           # Default - If None, we're done or broken

        for row in reader:
            ## Any Updates??
            if return_row == None and (row['Started'] == False or row['Started'] == 'False' or row['Started']=='FALSE'):
                row['Started'] = True
                return_row = row
            writer.writerow(row)

        outfile.close()

        # Convert the layers to Dicts
        if return_row != None :
            return_row['Layer1'] = eval(return_row['Layer1'])
            if len(return_row['Layer2']) > 0:
                return_row['Layer2'] = eval(return_row['Layer2'])
            if len(return_row['Layer3']) > 0:
                return_row['Layer3'] = eval(return_row['Layer3'])
            if len(return_row['Layer4']) > 0:
                return_row['Layer4'] = eval(return_row['Layer4'])
            if len(return_row['Layer5']) > 0:
                return_row['Layer5'] = eval(return_row['Layer5'])

        return return_row

def finish_update_row (datafile, row_to_update, success=True):
    """
        Updates a current row with Finished and all relevant metrics
            :return:
    """
    # Read in the Models file
    tempfilename = os.path.splitext(datafile)[0] + '.bak'
    try:
        os.remove(tempfilename)  # delete any existing temp file
    except OSError:
        pass
    os.rename(datafile, tempfilename)

    if success == False:
        row_to_update['Finished'] = False
        row_to_update['model_best_acc'] = 0
        row_to_update['model_best_loss'] = 9999     #'N/A'
        row_to_update['model_best_val_acc'] = 0
        row_to_update['model_best_val_loss'] = 9999  #'N/A'
        row_to_update['model_best_combined_ave_loss'] = 9999   #'N/A'

    # create a temporary dictionary from the input file
    with open(tempfilename, mode='r') as infile:
        reader = csv.DictReader(infile, skipinitialspace=True)
        # header = next(reader)  # skip and save header
        outfile = open(datafile, mode='w', newline='')
        fieldnames = ['Layers', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Started', 'Finished',
                      'model_best_acc',
                      'model_best_loss', 'model_best_val_acc', 'model_best_val_loss',
                      'model_best_combined_ave_loss', 'ErrorDetails', 'TotalNodes']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        found_row = False
        for row in reader:
            ## Any Updates??

            # Convert the layers to Dicts
            row['Layer1'] = eval(row['Layer1'])
            if len(row['Layer2']) > 0:
                row['Layer2'] = eval(row['Layer2'])
            if len(row['Layer3']) > 0:
                row['Layer3'] = eval(row['Layer3'])
            if len(row['Layer4']) > 0:
                row['Layer4'] = eval(row['Layer4'])
            if len(row['Layer5']) > 0:
                row['Layer5'] = eval(row['Layer5'])
            row['Started'] = eval(row['Started'].title())

            if row == row_to_update:
                found_row = True
                row['Finished'] = True
                row['model_best_acc'] = float(model_best_acc)
                row['model_best_loss'] = float(model_best_loss)
                row['model_best_val_acc'] = float(model_best_val_acc)
                row['model_best_val_loss'] = float(model_best_val_loss)
                row['model_best_combined_ave_loss'] = float(model_best_combined_ave_loss)

            writer.writerow(row)
        return found_row


def save_model(model, model_filename):
    model.save(Path(models_path + processing_rule +'_' + str(processing_range) + '_' + model_filename))     # Added processing rule - help reduce clutter and confusion, although its not yet used consistently...

def dict_to_description(modelDict):
    layers = int(modelDict['Layers'])
    model_description = str(layers) + 'Layers_' + modelDict['Layer1']['LayerType'] + str(
        modelDict['Layer1']['Nodes'])
    if layers >= 2:
        model_description += '_' + modelDict['Layer2']['LayerType'] + str(modelDict['Layer2']['Nodes'])
    if layers >= 3:
        model_description += '_' + modelDict['Layer3']['LayerType'] + str(modelDict['Layer3']['Nodes'])
    if layers >= 4:
        model_description += '_' + modelDict['Layer4']['LayerType'] + str(modelDict['Layer4']['Nodes'])
    if layers >= 5:
        model_description += '_' + modelDict['Layer5']['LayerType'] + str(modelDict['Layer5']['Nodes'])

    return model_description


def load_model (model_filename):
    return keras.models.load_model (Path(models_path + model_filename))

def biased_squared_mean(y_true, y_pred):
    # The aim of this is to punish FALSE NEGATIVES more than FALSE POSITIVES.
    # The hypothesis is that this may help in training the SELL rule.  Selling early is less harmful than not selling, so we want to ensure that missing a SELL signal has a stronger weighting
    # This is not simple, as the y_true and y_pred are Keras tensors (matrices).  Therefore, need to define the formula in a single equation to avoid a complex (and slow) loop...

    # E.g.
    # DEFAULT MSE Loss = (pred - true) ^ 2
    # I want
    #        Loss = if pred > true then (pred - true)^2 / 2
    #               else (pred - true) ^2

    # Basically - pred greater than true is NOT as bad as true greater than pred...  HOW TO DO THIS??
    # pred - true     -  positive if pred > true, 0 if same, negative if not
    # abs of this -
    # square ot

    diff =  y_true  - y_pred    # Will be positive IF true is ahead of prediction  TRUE.   Negative means the prediction was ahead of true (not as bad).  Positive means true was ahead of prediction (bad)

    square_diff = K.square(diff)

    # Turning this off - no longer use this.  Instead, have a mandatory evalute on 'default' loss function post calcluation.  Much cleaner.
    #if custom_loss_override:
    #    final_diff = square_diff
    #else:
    final_diff = square_diff + diff         # This is basically RMS, but with extra penalty for being too conservative.


    return K.mean(final_diff, axis=-1)  # Note the `axis=-1`

def biased_squared_mean2(y_true, y_pred):
    # The aim of this is to punish FALSE NEGATIVES more than FALSE POSITIVES.
    # The hypothesis is that this may help in training the SELL rule.  Selling early is less harmful than not selling, so we want to ensure that missing a SELL signal has a stronger weighting
    # This is not simple, as the y_true and y_pred are Keras tensors (matrices).  Therefore, need to define the formula in a single equation to avoid a complex (and slow) loop...

    # E.g.
    # DEFAULT MSE Loss = (pred - true) ^ 2
    # I want
    #        Loss = if pred > true then (pred - true)^2 / 2
    #               else (pred - true) ^2

    # Basically - pred greater than true is NOT as bad as true greater than pred...  HOW TO DO THIS??
    # pred - true     -  positive if pred > true, 0 if same, negative if not
    # abs of this -
    # square ot

    const = 2

    diff = y_true - y_pred
    # if custom_loss_override:
    #     mask = K.less(y_pred, y_true) #i.e. y_pred - y_true < 0
    #     mask = K.cast(mask, tf.float32)
    #
    #     print("[INFO] Loss Override has worked - doing basic mean of square")
    #     print('[INFO] MSE' + str(K.mean(K.square(diff))) + ' vs new way of : ' + str(K.mean((const - 1) * mask * K.square(diff) + K.square(diff))))
    #     return K.mean(K.square(diff))
    # else:
    print("[INFO] Loss Override not required.  Doing funky calc!")

    mask = K.less(y_pred, y_true) #i.e. y_pred - y_true < 0
    mask = K.cast(mask, tf.float32)
    return K.mean((const - 1) * mask * K.square(diff) + K.square(diff))
    # This is basically 2 x the loss if pred is less than true - i.e. missing a signal, we want to punish.

def download_to_db (symbol, period='2y'):
    """
        Download data from YFinance and store in DB.
        Also checks for inconsistencies with existing history (excluding volume), ignoring decial places
    :param symbol: Ticker code
    :param period: YFinance compatible period - 2 years default
    :return:
    """


    sqlEngine = create_engine('mysql+mysqlconnector://' + db_username +  ':'+ db_pwd + '@' + db_host + '/tfpp',   pool_recycle=3600)

    ## Download from Yahoo
    gdaxi = yfinance.Ticker(symbol)
    #hist = gdaxi.history(period="max")
    hist = gdaxi.history(period=period)

    # Check if we have today's data - if so, we remove
    from datetime import date

    if hist.index[-1] == pd.Timestamp(date.today()):
        # This means we have today's data in the download - either this is COB in the current TZ (okay), but let's assume its new partial data and remove it
        hist = hist.iloc[0:-1,]


    # Upload to temp table
    hist.to_sql("temp_price_table", con=sqlEngine, if_exists='append')

    # execute SQL to upate main table from temp table
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)

    qry = ("insert ignore into pricedatadaily (Symbol, Date, Open, High, Low, Close, Volume) "     
           "select '^GDAXI', Date, Open, High, Low, Close, Volume from temp_price_table; " )
    rowcount = cursor.execute(qry);

    print("[INFO] " + str(cursor.rowcount) + " rows were inserted from temp table")

    # Check that there are no anomalies - i.e. differences between data
    qry = ("SELECT Date, Open, High, Low, Close, Volume from temp_price_table " 
            "where not exists "
            "(select Date from pricedatadaily "
            "where symbol = '" + symbol + "' " 
            "and date = temp_price_table.date "
            " and round(pricedatadaily.open, 0) =  round(temp_price_table.open, 0) "
            " and round(close, 0) =  round(temp_price_table.close, 0) "
            " and round(high, 0) =  round(temp_price_table.high, 0) "
            " and round(low, 0) =  round(temp_price_table.low, 0) "
            #"  -- and volume =  temp_price_table.volume
            " ); ")

    cursor.execute(qry);
    rows = cursor.fetchall()
    cnx.commit()
    cnx.close()

    if len(rows) > 0 and rows is not None:
        print('[ERROR] rowcount of ' + str(rows) + ' when checking for value differences - NEED TO INVESTIGATE')
        print('[ERROR] E.g.:')
        print(rows[0])
        exit(-1)

def parse_from_db (symbol, records=500):
    """
        Extracts records from pricedataaily table, saves as tmp_csv, and then parses ready for processing

    :param symbol:
    :param records:
    :return: atr20 of the last record.
    """

    # Step 1 - Run SQL to extract and store in a PD
    sqlEngine = create_engine('mysql+mysqlconnector://' + db_username + ':' + db_pwd + '@' + db_host + '/tfpp',
                              pool_recycle=3600)
    records = pd.read_sql("""
            select * from 
            (SELECT * FROM tfpp.pricedatadaily
            where symbol = '^GDAXI'
            order by date desc
            limit 500
            ) t
            order by date asc""", sqlEngine)

    print (records)
    # Step 2 - call parse_file
    global debug_with_date
    old_val = debug_with_date
    debug_with_date = True
    atr20 = parse_file(records,  purpose='Predict', prefix=symbol+'_')
    debug_with_date = old_val       # Review this in future, but this forces to keep the Date field in X_Train, which has its uses...  Just need to remove before processing elsewhere
    return atr20

def download_and_parse (symbol, samples = 500):
    """

    :param symbol: Stock code or index code to download
    :return: TBC - numpy or PD array.  Or just save parsed data?
    """

    ts = TimeSeries(key, output_format='pandas')            # key is a global - should be stored / fetched externally

    # Use the ADJUSTED DATA Series.  The unadjusted has strange issues - e.g. Open can equal previous close, and periods where it doesn't align to the longer data series
    # Using it would likely create significantly weird predictions
    #daily_data, meta = ts.get_daily(symbol='^GDAXI', outputsize = 'full')
    daily_data, meta2 = ts.get_daily_adjusted(symbol=symbol, outputsize = 'full')

    daily_data = daily_data[0:samples]    # Pair it back a bit

    # Now reverse it
    daily_data = daily_data[::-1]

    # Convert the date index to a column for consistency
    daily_data.reset_index(level=0, inplace=True)

    # Rename to max compatible with current syntax
    daily_data.rename(columns = {'date': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low','4. close': 'Close','5. volume': 'Volume'}, inplace = True)

    #debug_with_date = True
    # Does this need to be enabled??  May want to overwrite and reset this??

    global debug_with_date
    old_val = debug_with_date
    debug_with_date = True
    parse_file(daily_data,  purpose='Predict', prefix=symbol+'_')
    debug_with_date = old_val       # Review this in future, but this forces to keep the Date field in X_Train, which has its uses...  Just need to remove before processing elsewhere

def predict_code_model (parsed_filename, model_file, predictions=1, expected_col = 'BuyWeightingRule'):
    """

    :param symbol: Symbol to use (e.g. ^GDAXI).  NOTE - data must have been downloaded and CURRENT - not checked (yet) - may add optional date check
    :param model_file: Filename of model to use - MUST EXIST, and have been trained accordingly.
    :param predictions: How many values to predict - default is one (the latest).  0 means do ALL
    :return: an array of some kind based on DATES, MODEL, EXPECTED and PREDICTION  (EXPECTED MAY NOT EXIST, but is useful for back testing on new data)
    """

    #filename = symbol + '_tempfile.csv'
    global debug_with_date, shuffle_data
    old_val = debug_with_date
    debug_with_date = True
    new_method_data, new_method_results, date_labels = parsefile(r'.\\parsed_data_full\\' + parsed_filename,
                                                                     expected_col, strip_last_row=False)      # Not actually using the Rule Column - just use a dummy val
    shuffle_data = False  # This is real - no shuffling!
    x_train, y_train = parse_data(new_method_data, new_method_results, processing_range)
    debug_with_date = old_val

    # Load a model, and then use it against current data - let's see what it predicts!
    model = load_model(model_file)



    # Check the date is the correct one!
    last_date = x_train[-1, processing_range - 1, 0]

    # If the last date is TODAY'S DATE - we need to remove it.  IF TODAY, IT MEANS THE MARKET IS OPEN AND WE HAVE A PARTIAL RECORD
    if (last_date == datetime.datetime.now().strftime("%Y%m%d000000")):
        x_train = x_train[0:-1]
        y_train = y_train[0:-1]
        last_date = x_train[-1, processing_range - 1, 0]

    print('[INFO]: Last Date that we are processing for is: ' + str(last_date))

    if predictions == 0:
        predictions = len(x_train)       # Do ALL.  Need to check if there's any filtering to do, but that should have happened in the parseing already??

    results = []
    for i in range(len(x_train) - 1, len(x_train) - predictions - 1, -1):
        new_predictions = model.predict(x_train[i:i+1, 0:, 1:])         # i:i+1 gives one sample, but ensures its in 3 dimensions, as needed by predict
                                                                        # 0 is all rows in the sample, and 1: filters out the Date which we're carrying
        expected_value = y_train[i][0]                                     # Keep this - may be useful
        #        results.append (DATE, MODEL, EXPECTED, PREDICTIONS)
        results.append ([x_train[i, processing_range - 1, 0], model_file, expected_value, new_predictions[0,0]])        # In future, may want to preserve an array here - but for now, keep single value

    return results

def store_atr20_to_db (symbol, date, atr20):
    """
        Stores the ATR20 for a given date and symbol.
        For now, I'm storing this in 'PREDICTIONS' - even though it is NOT a prediction.  This, however, is convenient given it will be used with predictions
    :param symbol:
    :param atr20:
    :return:
    """

def disable_GPU ():
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

## Some test code
#model = Sequential()
# Add flatten - may help?
#model.add(Flatten( input_shape=(250, 4)))
#model.add(Dense(512, activation="sigmoid"))
#model.add(Dropout(.5))
#model.add(Dense(256, activation="sigmoid"))
#model.add(Dropout(.5))
#model.add(Flatten())  - no longer needed
#model.add(Dense(2, activation="softmax"))       # 2 output
#model.add(Dense(1))       # remove softmax, given we have multi-value output

#parse_process_plot("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule", model, "Model1_Percent_NewRandom")

#### PlayPenCode
#myf.parse_file("DAX4ML.csv")
#parse_file("^GDAXI.csv")

