from keras.regularizers import L1L2

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
opt = 'Adamax'
datafile = r'.\models_to_test2.csv'