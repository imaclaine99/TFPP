from clr_callback import CyclicLR
from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, LSTM
from keras.regularizers import L1L2


# Enhances the first model, by ensuring that the LAST layer has no sequences returned, but ALL OTHER LAYERS DO
# still running STATELESS

num_samples = 250
epochs = 250
node_iterations = 9
#clr_mode = 'triangular'            # Default is none.  May not be a good idea to use with Nadam?
use_lrf = False #  True
batch_size = 16
dropout = 0.0           #   0.4
buy_or_sell = 'Buy'     # Buy  Sell
is_dupe_data = False #True   # Experimental
bias_regulariser = L1L2(l1= 0, l2=0)

class MyLSTMModelV1b (object):
    import MyFunctions as myf
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
#    dropout = 0.2

    def __init__(self, num_layers, nodes_per_layer):
        self.num_layers = num_layers
        self._nodes_per_layer = nodes_per_layer
        self.l2_reg = 0.001
        self.myf.EPOCHS = epochs
        self.myf.batch_size = batch_size
        self.myf.use_lrf = use_lrf
        self.myf.is_dupe_data = is_dupe_data
        self.myf.dupe_vals = (1,2,2,5,10)       # Can't hurt to have 2 in twice can it?
        self.myf.default_optimizer = 'SGD+CLR'
        if 'clr_mode' in locals():
            self.myf.clr_mode = clr_mode
        self.myf.lr_min = 1.75e-5  # Used with CLR - set by LR Finder if that is used
        self.myf.lr_max = .009  # Used with CLR - set by LR Finder if that is used

        self._model_def()

        # Haven't yet compiled, and run LRF.  Need to move this   -  MOVED TO MYFUNCTIONS
        #if clr_mode == 'triangular':
        #    clr = CyclicLR(base_lr=self.myf.lr_min, max_lr=self.myf.lr_max,
        #                   step_size=2000.)         # Using all defaults
        #    self.myf.callbacks.append(clr)

    def _model_def(self):
        self.model = Sequential()

        for i in range(0, self.num_layers):
            # BTW - the original looks buggy for more than one layer?

            if i == 0 and self.num_layers == 1:
                self.model.add(LSTM(self._nodes_per_layer[i], input_shape=(num_samples, 4), dropout=dropout, bias_regularizer=bias_regulariser))
            elif i == 0:
                # Means we're on first layer, but not last - there we want to define input shape AND return sequences
                self.model.add(LSTM(self._nodes_per_layer[i], input_shape=(num_samples, 4), return_sequences=True, dropout=dropout, bias_regularizer=bias_regulariser))
            elif i == self.num_layers-1 :
                # We're on the last layer - don't return sequences, don't have dropout
                self.model.add(LSTM(self._nodes_per_layer[i]), bias_regularizer=bias_regulariser)
            else :
                # Return sequences, but no input shape - not sure if this matters?
                self.model.add(LSTM(self._nodes_per_layer[i], return_sequences=True, bias_regularizer=bias_regulariser))
        #                                   activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.1, seed=None),
 #                                         kernel_regularizer=regularizers.l2(self.l2_reg)))
##            self.model.add(Dense(self._nodes_per_layer[i], activation="relu", kernel_initializer=RandomNormal(mean=0, stddev=0.1, seed=None),
        # ##                               kernel_regularizer=regularizers.l2(self.l2_reg)))
 ##           self.model.add(Dropout(self.dropout))
#        self.model.add(Flatten())
        self.model.add(Dense(1))  # remove softmax, given we have multi-value output

if __name__ == "__main__":
    import os



#    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # comment out if CUDA is not to be used
    start_layer = 1             #  Same as no sequences if only 1 layer
    start_layer1_nodes = 8


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

