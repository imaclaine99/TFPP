import MyLSTMModelV2b
import ModelV2Config as ModelV2Config
import MyFunctions as myf

# Load a model
model_for_prediction = '31204_5_Layers.h5'
model = myf.load_model(model_for_prediction)
print('')
print('[INFO] Model File: ' + model_for_prediction)


# Load a dataset(ideally unseen)
parsed_filename = 'Rule3^GDAXI_Now.csv'

#myf.debug_with_date = True          # We want the date - we're using it
#new_method_data, new_method_results, date_labels = myf.parsefile(r'.\\parsed_data\\' + parsed_filename, 'BuyWeightingRule', strip_last_row=False)
#myf.shuffle_data = False            # This is real - no shuffling!
#x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)


# Run against ALL data
predictions = myf.predict_code_model (parsed_filename, model_for_prediction + '.h5', predictions=0)

print(predictions)

# Get validation_loss
# Function should NOT care about model or data being correct - that is up to the caller.
# Maybe output both the actual loss as well as some form of box plot of the loss - would be useful to see not just the 'average' but the spread.