from keras import Sequential, regularizers
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout

import MyFunctions

num_samples = 250

class MyModelV1 (object):
    lr_reg = 0.001      # Not sure of the difference of this vs the init?
    dropout = 0.2
    activation = None
#    kernel_initialiser_random = RandomNormal(mean=0, stddev=0.1, seed=None)
    kernel_initialiser_random = 'glorot_uniform'

    def __init__(self, num_layers, nodes_per_layer):
        self.num_layers = num_layers
        self._nodes_per_layer = nodes_per_layer
        self.l2_reg = 0.001
        self._model_def()

    def _model_def(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(num_samples, 4)))

        for i in range(0, self.num_layers):
            self.model.add(Dense(self._nodes_per_layer[i], activation="relu", kernel_initializer=self.kernel_initialiser_random,
                                kernel_regularizer=regularizers.l2(self.l2_reg)))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1))  # remove softmax, given we have multi-value output

if __name__ == "__main__":
    import os
    import tensorflow as tf

    # Disable GPU
    if True == True:
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

#    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # comment out if CUDA is not to be used
    start_layer = 1
    start_layer1_nodes = 1


    for layers in (1,2, 3, 4, 5)      :
        if layers < start_layer :
            continue
        for nodes1 in range(1, 11)    :
            if (layers == start_layer) and (nodes1 < start_layer1_nodes) :
                continue
            for nodes2 in range(1, 11):
                if layers < 2 and nodes2 > 1 or nodes2 > nodes1:
                    continue
                for nodes3 in range(1, 11):
                    if layers < 3 and nodes3 > 1 or nodes3 > nodes2 :
                        continue
                    for nodes4 in range(1, 11):
                        if layers < 4 and nodes4 > 1 or nodes4 > nodes3:
                            continue
                        for nodes5 in range(1, 11):
                            if layers < 5 and nodes5 > 1 or nodes5 > nodes4:
                                continue
                            for activation in (None, 'relu', 'softplus', 'tanh'):
                                model = MyModelV1(layers, [2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                                model.activation= activation
                                print (layers)
                                print ([2**nodes1, 2**nodes2, 2**nodes3, 2**nodes4, 2**nodes5])
                                model.model.summary()
                                MyFunctions.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule", model.model,
                                                       "ModelV1Dense_Activation_"+str(model.kernel_initialiser_random) + "_" + str(model.activation) +"_"+ str(layers) + " layers and " + str(nodes1) + ", "+ str(nodes2) + ", "+ str(nodes3) + ", "+ str(nodes4) + ", "+ str(nodes5) + ", ")

