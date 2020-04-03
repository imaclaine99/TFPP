import MyLSTMModelV2
import MyFunctions as myf
import ModelV2Config

# Load a model, and then use it against current data - let's see what it predicts!
model = myf.load_model('31204_5_Layers.h5')

# Need data...
# Need to get 250 records of OHLC, and then normalise.  Will likely need more than 250 rows to normalise
# Let's keep it simple - load and parse exactly the same way as we train.  Then use parameters to tell where to start and stop.

myf.shuffle_data = False            # This is real - no shuffling!
filename = '^GDAXI_Now.csv'
myf.parse_file(filename,  purpose='Predict')


new_method_data, new_method_results = myf.parsefile(r'.\\parsed_data_full\\' + filename, 'BuyWeightingRule', strip_last_row=False)
x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)


# We now have X and Y data in appropriate arrays.
# Let's start num_samples in, and then run the rest!

#for i in range (0, len(x_train)):
#    new_predictions = model.model.predict(x_train[i:i+1])
#    print ('Prediction:  ' + str(float(new_predictions)))
#    print ('Based on data set ending on :')
  #  print (x_train[i])

# Let's load STOXX data (parsed) and see how that goes with a given model
filename = '^STOXX50E_OHLC.csv'
myf.parse_file(filename)

infile = ".\parsed_data\\" + filename
new_method_data, new_method_results = myf.parsefile(infile, 'BuyWeightingRule', strip_last_row=True)
x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)

#H = model.model.evaluate(x_train, y_train, verbose=2)
#print (H)
for i in range (0, len(x_train)):
    new_predictions = model.model.predict(x_train[i:i+1])
    print ('Prediction:  ' + str(float(new_predictions)))
  #  print ('Based on data set ending on :')
  #  print (x_train[i])



xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY (infile, 'BuyWeightingRule')



H = model.model.fit(xtrain, ytrain, validation_data=(xtest, ytest))
print (H)

#H = model.model.fit()

meta_filename = infile.split("\\")  # This feels very hacky...
resolved_meta_filename = r'.\\parsed_data\\meta\\' + filename + "_meta.csv"

#myf.plot_and_save_history_with_baseline(H, myf.EPOCHS, filename + "STOXX Check", resolved_meta_filename)

H = myf.parse_process_plot(infile , "BuyWeightingRule",
                            model.model,
                             'Model_31204_STOXX_Replay_OHLC')


