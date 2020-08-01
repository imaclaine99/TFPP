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
buy_or_sell = 'BuyV3b'     # Buy or Sell  - Can be a variant - first three letters are used to determine if Buy or Sell column is to be used.
is_dupe_data = False #True   # Experimental
bias_regulariser = L1L2(l1= 0, l2=0)            # Used on LSTM layers
kernel_regulariser = L1L2(l1 = 0, l2=0)         # Used on LSTM layers
dense_regulariser = L1L2(l1 = 0, l2=0)
opt = 'Adamax'
is_input_noise = False
input_noise = 0.0
early_stopping_min_delta = 0.005        # More than MyFuncs Default
early_stopping_patience = 20

datafile = r'.\models_to_test2.csv'
#dense_kernel_initialiser = RandomNormal(mean=0, stddev=0.1, seed=None)
#dense_kernel_initialiser = 'glorot_uniform'
#dense_kernel_initialiser = 'glorot_normal'
dense_kernel_initialiser = 'lecun_normal'
#db_host = '199.244.51.253' #'192.168.0.109'
db_host = '192.168.0.109'
db_username = 'tfpp'
#db_pwd = 'qQwWeErRtT123123!@'
db_pwd = 'tfpp'

gpu = True
if (gpu):
    db_read_sort = 'Random'       # None, Random, NodesAsc, NodesDesc  - Use Random for GPU
    disableGPU = False
else:
    db_read_sort = 'Random'       # None, Random, NodesAsc, NodesDesc  - Use Random for GPU
    disableGPU = True
