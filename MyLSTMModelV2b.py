from clr_callback import CyclicLR
from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, LSTM
from keras.layers import GaussianNoise
from keras.regularizers import L1L2
import ModelV2Config as ModelConfig
import csv
import MyFunctions
import sys

# Global config
MyFunctions.db_host = ModelConfig.db_host
MyFunctions.db_username = ModelConfig.db_username
MyFunctions.db_pwd = ModelConfig.db_pwd

# Variant on Model V2 with the following changes
# Use Rule 3 of ATR with Beta of TBC
# Use multi file processing but down to 150 epochs
# Use more selective Permutation Creator
# To Do -ATR Rule, Multi File, Epochs

infile_array = (".\parsed_data\\" + 'Rule3_B0.98^GDAXI.csv', ".\parsed_data\\" + 'Rule3^STOXX50E_OHLC.csv',
                ".\parsed_data\\" + 'Rule3_B0.98^FTSE_2019OHLC.csv', ".\parsed_data\\" + 'Rule3_B0.98^GSPC.csv')


# Should move this to MyFunctions later
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Disable GPU
if ModelConfig.disableGPU == True:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      # Disable all GPUS
      tf.config.set_visible_devices([], 'GPU')
      visible_devices = tf.config.get_visible_devices()
      for device in visible_devices:
        assert device.device_type != 'GPU'
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass

#config = tf.config
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#set_session(session)


# Enhancesments over Model V1b (WIP):
# 1. Move config outside of this file.  Makes it easier to maintain code vs config
# 2. The creator will now take a Dict of Dicts, including number of layers.  This is a bit more work to parse, but provides a LOT more flexibility

# Config Dict is based on:
#fieldnames = ['Layers', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Started', 'Finished', 'model_best_acc',
#              'model_best_loss', 'model_best_val_acc', 'model_best_val_loss',
#              'model_best_combined_ave_loss']
#
# Each Layer Consists of:
# fieldnames = ['LayerType', 'Nodes', 'OverFittingHelp', 'ReturnSequences'

class MyLSTMModelV2b (object):
    import MyFunctions as myf
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
#    dropout = 0.2

    def __init__(self, configDict):
        self.num_layers = configDict['Layers']
        self.myf.EPOCHS = ModelConfig.epochs
        self.myf.processing_rule = ModelConfig.buy_or_sell
        self.myf.batch_size = 64 #ModelConfig.batch_size
        self.myf.use_lrf = ModelConfig.use_lrf
        self.myf.is_dupe_data = ModelConfig.is_dupe_data
        self.myf.early_stopping_min_delta = ModelConfig.early_stopping_min_delta
#        self.myf.read_backwards = ModelConfig.read_backwards
#        self.db_read_sort = ModelConfig.db_read_sort
#        MyFunctions.read_backwards = ModelConfig.read_backwards     # This is messy....
        self.myf.dupe_vals = (1,2,2,5,10)       # Can't hurt to have 2 in twice can it?
     #   self.myf.default_optimizer = 'SGD+CLR'
        self.flattened = False      # Check if we've flattened, and ensure we do by the last layer
        if 'clr_mode' in locals():
            self.myf.clr_mode = ModelConfig.clr_mode
        self.myf.lr_min = 1.75e-5  # Used with CLR - set by LR Finder if that is used
        self.myf.lr_max = .009  # Used with CLR - set by LR Finder if that is used
        self.error = False
        self._model_def(configDict)



        # Haven't yet compiled, and run LRF.  Need to move this   -  MOVED TO MYFUNCTIONS
        #if clr_mode == 'triangular':
        #    clr = CyclicLR(base_lr=self.myf.lr_min, max_lr=self.myf.lr_max,
        #                   step_size=2000.)         # Using all defaults
        #    self.myf.callbacks.append(clr)

    def _model_def(self, configDict):
        self.model = Sequential()

        if ModelConfig.is_input_noise:
            self.model.add(GaussianNoise(ModelConfig.input_noise, input_shape=(ModelConfig.num_samples, 4)))
        # Layer 1
        current_layer = configDict['Layer1']

        # if last layer and DENSE, add flatten

        if current_layer['LayerType'] == 'Dense':
            # Layer1, so flatten it if layer 1 is Dense (kind of implies dense at the start is dumb...)
            if int(configDict['Layers']) == 1:
                # 1 Layer Dense - Flatten
                self.model.add(Flatten(input_shape=(ModelConfig.num_samples, 4)))
                self.flattened = True
                if int(current_layer['Nodes']) <= 1:
                    layer_dropout = 0
                else:
                     layer_dropout = ModelConfig.dense_dropout
                self.model.add(Dense(2 ** int(current_layer['Nodes']), activation="selu", kernel_initializer=ModelConfig.dense_kernel_initialiser, kernel_regularizer=ModelConfig.dense_regulariser))
                self.model.add(Dropout(layer_dropout))
            else:
                if self.flattened == False:
                    # Check if we have any future LSTMs or not.  If we do NOT, flatten now
                    future_lstm = False
                    for look_ahead_layer in range(2, int(configDict['Layers'])+1):
                        if configDict['Layer' + str(look_ahead_layer)]['LayerType'] == 'LSTM':
                            future_lstm = True
                if future_lstm == False:
                    self.flattened = True
                    self.model.add(Flatten(input_shape=(ModelConfig.num_samples, 4)))
                    layer_dropout = ModelConfig.dense_dropout
                else:
                    layer_dropout = 0
                self.model.add(Dense(2 ** int(current_layer['Nodes']), activation="selu", input_shape=(ModelConfig.num_samples, 4), kernel_initializer=ModelConfig.dense_kernel_initialiser, kernel_regularizer=ModelConfig.dense_regulariser))
                self.model.add(Dropout(layer_dropout))
        elif current_layer['LayerType'] == 'LSTM':
            if current_layer['ReturnSequences'] == 'True' or current_layer['ReturnSequences'] == True:
                return_sequences = True
            else:
                return_sequences = False
                self.flattened = True           # Return sequences of False implies this - prevent adding another FLatten
            self.model.add(LSTM(2 ** int(current_layer['Nodes']), input_shape=(ModelConfig.num_samples, 4), return_sequences=return_sequences, dropout=ModelConfig.dropout, bias_regularizer=ModelConfig.bias_regulariser))
            # Not yet using the OverFitting Helper Variable - will consider that later.
        else:
            self.error = 'Unknown Layer Type on Layer 1'

        if int(configDict['Layers']) > 1 :
            for layer in range(2, int(configDict['Layers'])+1):
               if layer == 2:
                   current_layer = configDict['Layer2']
               elif layer == 3:
                   current_layer = configDict['Layer3']
               elif layer == 4:
                   current_layer = configDict['Layer4']
               elif layer == 5:
                   current_layer = configDict['Layer5']

               if current_layer['LayerType'] == 'Dense':
                # Check if last layer - if so, flatten
                    if layer == int(configDict['Layers']) and self.flattened == False:
                        self.model.add(Flatten())
                        self.flattened = True
                    if layer < int(configDict['Layers']):
                        activation = 'selu'
                    else:
                        activation = None       # Linear activation on last layer - helps stop dead nodes
                    if self.flattened == False:
                        # Check if we have any future LSTMs or not.  If we do NOT, flatten now
                        future_lstm = False
                        for look_ahead_layer in range(layer+1, int(configDict['Layers']) + 1):
                            if configDict['Layer' + str(look_ahead_layer)]['LayerType'] == 'LSTM':
                                 future_lstm = True
                        if future_lstm == False:
                            self.flattened = True
                            self.model.add(Flatten())
                            self.model.add(Dense(2 ** int(current_layer['Nodes']), activation=activation, kernel_initializer=ModelConfig.dense_kernel_initialiser, kernel_regularizer=ModelConfig.dense_regulariser))
                            if int(current_layer['Nodes']) > 2:       # No dropout if under 2 ^^ 2 nodes
                                self.model.add(Dropout( ModelConfig.dense_dropout))
                        else:
                            # Add without flattening.  No dropout if future LSTM - doesn't make much sense
                            self.model.add(Dense(2 ** int(current_layer['Nodes']), activation=activation, kernel_initializer=ModelConfig.dense_kernel_initialiser, kernel_regularizer=ModelConfig.dense_regulariser))
                    else:
                        # Add without flattening
                        self.model.add(Dense(2 ** int(current_layer['Nodes']), activation=activation, kernel_regularizer=ModelConfig.dense_regulariser))
                        if int(current_layer['Nodes']) > 2:       # No dropout if under 2 ^^ 2 nodes
                            self.model.add(Dropout(ModelConfig.dense_dropout))

               elif current_layer['LayerType'] == 'LSTM':
                    if current_layer['ReturnSequences'] == 'True' or current_layer['ReturnSequences'] == True:
                        return_sequences = True
                    else:
                        return_sequences = False
                        self.flattened = True       # LSTM without return sequences effectively flattens
                    self.model.add(LSTM(2 ** int(current_layer['Nodes']),  return_sequences=return_sequences, dropout=ModelConfig.dropout, bias_regularizer=ModelConfig.bias_regulariser))              # Removed input_shape=(ModelConfig.num_samples, 4),  # that should only be used in the first layer
                    # Not yet using the OverFitting Helper Variable - will consider that later.
               else:
                    self.error = 'Unknown Layer Type on Layer # ' + str(layer) + ' ' + str(current_layer)
       # self.model.add(Dense(1))               # add a one node dense, no activation - may help learning by having this on last node?
       # Check if last layer and LSTM and Return Sequences is true - if so, flatten.  Could argue that this is not needed, as would be picked up by a config that ends in Dense anyway, so is an illegal permutation...
        last_layer = 'Layer' + str(configDict['Layers'])
        if configDict[last_layer]['LayerType'] == 'LSTM' and (configDict[last_layer]['ReturnSequences'] == 'True' or configDict[last_layer]['ReturnSequences'] == True  ) and self.flattened == False:
            self.model.add(Flatten())

if __name__ == "__main__":
    import os

    MyFunctions.processing_rule = ModelConfig.buy_or_sell  # Yes - this is very messy...

    while True:
        #modelDict = MyFunctions.read_row(ModelConfig.datafile)
        modelDict = MyFunctions.read_from_from_db(ModelConfig.db_read_sort)
        if modelDict == None:
            break;

        try:
            model = MyLSTMModelV2b(modelDict)

            model.myf.model_description = 'LSTMMModelV2b_ATR3_BetaTBC ' + ModelConfig.buy_or_sell + model.myf.dict_to_description(modelDict) + ModelConfig.opt
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()

            if ModelConfig.buy_or_sell[:3] == 'Buy':
                rule_column = 'Buy'
            else:
                rule_column = 'Sell'

            model.myf.parse_process_plot_multi_source(infile_array, rule_column + "WeightingRule", model.model,
                                                      model.myf.model_description, version=2)

            #model.myf.finish_update_row(ModelConfig.datafile, modelDict)
            model.myf.db_update_row(modelDict)      # Use the myf from the model to make sure we get the relevant global values.  This may explain some strange behaviour with updates not working...
            if model.myf.model_best_loss < 1.5:
                model.myf.save_model(model.model, str(modelDict['unique_id']) +'_'+str(modelDict['Layers'])+'_Layers_' +ModelConfig.buy_or_sell + '.h5')
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print('Occurred with configDict:')
            print(modelDict)
            modelDict['ErrorDetails'] = sys.exc_info()[0]
#            MyFunctions.finish_update_row(ModelConfig.datafile, modelDict, False)
            MyFunctions.db_update_row(modelDict, False)


