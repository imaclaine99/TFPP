from keras.regularizers import L1L2
from keras.initializers import RandomNormal

num_samples = 250
epochs = 250
node_iterations = 9
#clr_mode = 'triangular'            # Default is none.  May not be a good idea to use with Nadam?
use_lrf = False #  True
batch_size = 16
dropout = 0.0           #   0.4
dense_dropout = 0.0     # Not used on the last layer (obviously!)
buy_or_sell = 'Buy'     # Buy  Sell
is_dupe_data = False #True   # Experimental
bias_regulariser = L1L2(l1= 0, l2=0)
dense_regulariser = L1L2(l1 = 0, l2=0)
opt = 'Adamax'
is_input_noise = False
input_noise = 0.0
early_stopping_min_delta = 0.005        # More than MyFuncs Default

datafile = r'.\models_to_test2.csv'
dense_kernel_initialiser = RandomNormal(mean=0, stddev=0.1, seed=None)
#dense_kernel_initialiser = 'glorot_uniform'
db_host = '192.168.0.109'
db_username = 'tfpp'
db_pwd = 'tfpp'



gpu = True
if (gpu):
    db_read_sort = 'Random'       # None, Random, NodesAsc, NodesDesc  - Use Random for GPU
    disableGPU = False
else:
    db_read_sort = 'Random'       # None, Random, NodesAsc, NodesDesc  - Use Random for GPU
    disableGPU = True
