import MyLSTMModelV2b
import MyFunctions as myf
import ModelV2Config


# The sequence here needs to be:
#
###
# 1. Download / get the latest data.  This needs to be the correct size and shape (i.e. Date, Open, High, Low, Close, with 220)
#   1. a - Where to best get this from?  Automated would obviously be preferable, just need to be aware of the date that is has data for, given google data seems to be a bit late
#    1. b - Make this flexible - don't want to be bound to anything too rigid' \
#
# 2. Have a loop to process through new data, and run predictions with different model / rule combos  (note - a saved model effectively has the rule built in)
# 3. This is somewhat separate, but a means to do more training on new data.  This doesn't need to be run daily, but needs to be able to be run.


# Comment the below out - AV seems to be having an issue - weekend??
#myf.download_and_parse('^GDAXI')

# Now what do I want?
# Function that:
# Takes a CODE (e.g. ^GDAXI)
# Takes a number of days (e.g. 1, 10, 100)
# Takes a MODEL_FILENAME
# Runs N predictions
# Returns this is an array

#predictions = myf.predict_code_model ('^GDAXI', '31204_5_Layers.h5', predictions=1)

#parsed_filename = '^GDAXI_tempfile.csv'
parsed_filename = '^GDAXI_Now.csv'
myf.parse_file(parsed_filename, purpose='Predict')


predictions = myf.predict_code_model (parsed_filename, 'BuyV2_29856_5_Layers.h5', predictions=0)
print (predictions)

#predictions = []
#for model_file in ('31204_5_Layers.h5', 'RangeVariance_29840_Iteration1.h5', 'RangeVariance_29840_Iteration0.h5', '29840_5_Layers.h5', '37618_5_Layers.h5'):
#    predictions = predictions + myf.predict_code_model('^GDAXI', model_file, predictions=10)

for i in predictions:
    print (i)

# Write output somewhere
import numpy
predictions_np = numpy.asarray(predictions, dtype='O')          # dtype O allows savetxt to work better for some reason...

numpy.savetxt(".\\predictions_data\\" + parsed_filename, predictions_np, delimiter=',',
              header='Date,Model,Expected,Predicated',
              fmt=['%f', '%s', '%f', '%f'],
              comments='')

exit(0);

# Now - what if we want to write the prediction out?
# Probably don't want to write to the current file - that could be a pain.  How about a new file - Date, TargetValue, PredictedValue?
# Only problem here is that at this level, I don't have the x/y train data, so would need to reparse it.  That's not too hard, and would have the dates to X Ref.
x_train, y_train = myf.parsefile(r'.\\input_data\\' + '^GDAXI_Now.csv',strip_last_row=False)




# OLD CODE HERE - THIS IS WORKING SPACE

filename = '^GDAXI_tempfile.csv'
myf.debug_with_date = True          # We want the date - we're using it
new_method_data, new_method_results, date_labels = myf.parsefile(r'.\\parsed_data_full\\' + filename, 'BuyWeightingRule', strip_last_row=False)
myf.shuffle_data = False            # This is real - no shuffling!
x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)

# Load a model, and then use it against current data - let's see what it predicts!

# Check the date is the correct one!
last_date = x_train[-1,ModelV2Config.num_samples-1,0]
print ('[INFO]: Last Date that we are processing for is: ' + str(last_date))

for model_file in ('31204_5_Layers.h5', 'RangeVariance_29840_Iteration1.h5', 'RangeVariance_29840_Iteration0.h5', '29840_5_Layers.h5', '37618_5_Layers.h5'):
    model = myf.load_model(model_file)
    print('')
    print('[INFO] Model File: ' + model_file)

    for i in range (len(x_train)-1, 0, -1):
        new_predictions = model.predict(x_train[i:i+1, 0:, 1:])         # i:i+1 gives one sample, but ensures its in 3 dimensions, as needed by predict
                                                                        # 0 is all rows in the sample, and 1: filters out the Date which we're carrying
        print ('Date:' + str(date_labels[i +(len(date_labels) - len(x_train))]) + ' Prediction:  ' + str(float(new_predictions)))