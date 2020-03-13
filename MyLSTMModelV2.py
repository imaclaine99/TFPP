from clr_callback import CyclicLR
from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, LSTM
from keras.regularizers import L1L2
import ModelV2Config as ModelConfig
import csv
import MyFunctions
import sys

# Global config
MyFunctions.db_host = ModelConfig.db_host
MyFunctions.db_username = ModelConfig.db_username
MyFunctions.db_pwd = ModelConfig.db_pwd


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

class MyLSTMModelV2 (object):
    import MyFunctions as myf
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
#    dropout = 0.2

    def __init__(self, configDict):
        self.num_layers = configDict['Layers']
        self.myf.EPOCHS = ModelConfig.epochs
        self.myf.batch_size = ModelConfig.batch_size
        self.myf.use_lrf = ModelConfig.use_lrf
        self.myf.is_dupe_data = ModelConfig.is_dupe_data
#        self.myf.read_backwards = ModelConfig.read_backwards
#        self.db_read_sort = ModelConfig.db_read_sort
#        MyFunctions.read_backwards = ModelConfig.read_backwards     # This is messy....
        self.myf.dupe_vals = (1,2,2,5,10)       # Can't hurt to have 2 in twice can it?
        self.myf.default_optimizer = 'SGD+CLR'
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

        # Layer 1
        current_layer = configDict['Layer1']

        # if last layer and DENSE, add flatten

        if current_layer['LayerType'] == 'Dense':
            # Layer1, so flatten it if layer 1 is Dense (kind of implies dense at the start is dumb...)
            if int(configDict['Layers']) == 1:
                # 1 Layer Dense - Flatten
                self.model.add(Flatten(input_shape=(ModelConfig.num_samples, 4)))
                self.flattened = True
                self.model.add(Dense(2 ** int(current_layer['Nodes']), activation="selu", kernel_initializer=ModelConfig.dense_kernel_initialiser))
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
                self.model.add(Dense(2 ** int(current_layer['Nodes']), activation="selu", input_shape=(ModelConfig.num_samples, 4), kernel_initializer=ModelConfig.dense_kernel_initialiser))
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
                            self.model.add(Dense(2 ** int(current_layer['Nodes']), activation=activation, kernel_initializer=ModelConfig.dense_kernel_initialiser))
                        else:
                            # Add without flattening
                            self.model.add(Dense(2 ** int(current_layer['Nodes']), activation=activation, kernel_initializer=ModelConfig.dense_kernel_initialiser))
                    else:
                        # Add without flattening
                        self.model.add(Dense(2 ** int(current_layer['Nodes']), activation="selu"))
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


    while True:
        #modelDict = MyFunctions.read_row(ModelConfig.datafile)
        modelDict = MyFunctions.read_from_from_db(ModelConfig.db_read_sort)
        if modelDict == None:
            break;

        try:
            model = MyLSTMModelV2(modelDict)

            model.myf.model_description = 'LSTMMModelV2 ' + ModelConfig.buy_or_sell + model.myf.dict_to_description(modelDict) + ModelConfig.opt
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule",
                                    model.model,
                                    model.myf.model_description)

            #model.myf.finish_update_row(ModelConfig.datafile, modelDict)
            MyFunctions.db_update_row(modelDict)
            if model.myf.model_best_loss < 1.5:
                model.myf.save_model(model.model, str(modelDict['unique_id']) +'_'+str(modelDict['Layers'])+'_Layers.h5')
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print('Occurred with configDict:')
            print(modelDict)
            modelDict['ErrorDetails'] = sys.exc_info()[0]
#            MyFunctions.finish_update_row(ModelConfig.datafile, modelDict, False)
            MyFunctions.db_update_row(modelDict, False)


    start_layer = 1             #  Same as no sequences if only 1 layer
    start_layer1_nodes = 7


    for layers in (1,2, 3, 4, 5)      :
        if layers < start_layer :
            continue
        for nodes1 in range(1, node_iterations)    :
            if (layers == start_layer) and (nodes1 < start_layer1_nodes) :
                continue
            for nodes2 in range(1, node_iterations):
                if layers < 2 and nodes2 > 1 or nodes2 > nodes1:
                    continue
                for nodes3 in range(1, node_iterations):
                    if layers < 3 and nodes3 > 1 or nodes3 > nodes2 :
                        continue
                    for nodes4 in range(1, node_iterations):
                        if layers < 4 and nodes4 > 1 or nodes4 > nodes3:
                            continue
                        for nodes5 in range(1, node_iterations):
                            if layers < 5 and nodes5 > 1 or nodes5 > nodes4:
                                continue
                            for opt in ( 'Adadelta',  'Adamax'):          # Removed , 'SGD+CLR' - it's not working properly - not worth the time right now to figure out what/why
                                                                                             # Also removed 'Adam', 'SGD+NM', 'Nadam'
                                model = MyLSTMModelV1b(layers, [2 ** nodes1, 2 ** nodes2, 2 ** nodes3, 2 ** nodes4, 2 ** nodes5])

                                print (layers)
                                print ([2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                                model.model.summary()

                                if buy_or_sell == 'Buy':
                                    model.myf.model_description = 'LSTMMModelV1b ' + buy_or_sell + ' Rule - 0.0 Dropout 0.25 TestRatio NoDupeData' + opt
                                    model.myf.default_optimizer = opt
                                    model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model.model,
                                                           "LTSM_Model1bBuy_ NoDrop_NoDupeData" + opt + "_" + str(layers) + " layers and " + str(
                                                               nodes1) + ", " + str(nodes2) + ", " + str(
                                                               nodes3) + ", " + str(nodes4) + ", " + str(nodes5) + ", ")


                                else:
                                    model.myf.model_description = 'LSTMMModelV1b Sell Rule - No Dropout No CLR+LRF 0.25 TestRatio NoDupeData' + opt
                                    model.myf.default_optimizer = opt
                                    model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "SellWeightingRule", model.model,
                                                           "LTSM_Model1bSell_ NoDrop_NoDupeData" + opt + "_" + str(layers) + " layers and " + str(nodes1) + ", "+ str(nodes2) + ", "+ str(nodes3) + ", "+ str(nodes4) + ", "+ str(nodes5) + ", ")

