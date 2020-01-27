from clr_callback import CyclicLR
from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, LSTM

import MyFunctions as myf

# Enhances the first model, by ensuring that the LAST layer has no sequences returned, but ALL OTHER LAYERS DO
# still running STATELESS

num_samples = 250
node_iterations = 5
clr_mode = 'triangular'

class MyLSTMModelV1b (object):
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
    dropout = 0.2

    def __init__(self, num_layers, nodes_per_layer):
        self.num_layers = num_layers
        self._nodes_per_layer = nodes_per_layer
        self.l2_reg = 0.001
        self._model_def()

        if clr_mode == 'triangular':
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                           step_size=2000.)         # Using all defaults
            myf.callbacks.append(clr)

    def _model_def(self):
        self.model = Sequential()

        for i in range(0, self.num_layers):
            # BTW - the original looks buggy for more than one layer?
            if i == 0 and self.num_layers == 1:
                self.model.add(LSTM(self._nodes_per_layer[i], input_shape=(num_samples, 4)))
            elif i == 0:
                # Means we're on first layer, but not last - there we want to define input shape AND return sequences
                self.model.add(LSTM(self._nodes_per_layer[i], input_shape=(num_samples, 4), return_sequences=True))
            elif i == self.num_layers-1 :
                # We're on the last layer - don't retrun sequences
                self.model.add(LSTM(self._nodes_per_layer[i]))
            else :
                # Return sequences, but no input shape
                self.model.add(LSTM(self._nodes_per_layer[i], return_sequences=True))
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
    start_layer1_nodes = 1


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
                            model = MyLSTMModelV1b(layers, [2 ** nodes1, 2 ** nodes2, 2 ** nodes3, 2 ** nodes4, 2 ** nodes5])
                            print (layers)
                            print ([2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                            model.model.summary()
                            myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model.model,
                                                   "LTSM 1b_CLR_" + str(layers) + " layers and " + str(nodes1) + ", "+ str(nodes2) + ", "+ str(nodes3) + ", "+ str(nodes4) + ", "+ str(nodes5) + ", ")

