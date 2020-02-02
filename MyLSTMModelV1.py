from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, LSTM

num_samples = 250
epochs = 100
batch_size = 8
buy_or_sell = 'Sell'

class MyLSTMModelV1 (object):
    import MyFunctions as myf
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
    dropout = 0.2

    def __init__(self, num_layers, nodes_per_layer):
        self.num_layers = num_layers
        self._nodes_per_layer = nodes_per_layer
        self.l2_reg = 0.001
        self.myf.EPOCHS = epochs
        self.myf.batch_size = batch_size
        self._model_def()

    def _model_def(self):
        self.model = Sequential()

        for i in range(0, self.num_layers):
            if i == 0 and self.num_layers == 1:
                self.model.add(LSTM(self._nodes_per_layer[i], input_shape=(num_samples, 4)))
            else:
                self.model.add(LSTM(self._nodes_per_layer[i]))
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
    start_layer = 1
    start_layer1_nodes = 1

    max_nodes = 9       #  Was 9, but 10 doesn't really make sense (too many), and is very time consuming


    for layers in (1,2, 3, 4, 5)      :
        if layers < start_layer :
            continue
        for nodes1 in range(1, max_nodes+1)    :
            if (layers == start_layer) and (nodes1 < start_layer1_nodes) :
                continue
            for nodes2 in range(1, max_nodes+1):
                if layers < 2 and nodes2 > 1 or nodes2 > nodes1:
                    continue
                for nodes3 in range(1, max_nodes+1):
                    if layers < 3 and nodes3 > 1 or nodes3 > nodes2 :
                        continue
                    for nodes4 in range(1, max_nodes+1):
                        if layers < 4 and nodes4 > 1 or nodes4 > nodes3:
                            continue
                        for nodes5 in range(1, max_nodes+1):
                            if layers < 5 and nodes5 > 1 or nodes5 > nodes4:
                                continue
                            model = MyLSTMModelV1(layers, [2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                            print (layers)
                            print ([2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                            model.model.summary()
                            if buy_or_sell == 'Buy':
                                model.myf.model_description = 'LSTMMModelV1 Buy Rule'
                                model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model.model,
                                                       ".\output_images\LTSM 1st_Model_" + str(layers) + " layers and " + str(nodes1) + ", "+ str(nodes2) + ", "+ str(nodes3) + ", "+ str(nodes4) + ", "+ str(nodes5) + ", ")

                            else:
                                model.myf.model_description = 'LSTMMModelV1 Sell Rule - No Dropout - No CLR'
                                model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "SellWeightingRule", model.model,
                                                       "LTSM_Model1Sell_" + str(layers) + " layers and " + str(nodes1) + ", "+ str(nodes2) + ", "+ str(nodes3) + ", "+ str(nodes4) + ", "+ str(nodes5) + ", ")


### THIS IS BUGGY - DO NOT USE - USE 1B INSTEAD!!