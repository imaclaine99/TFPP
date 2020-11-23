import MyLSTMModelV2b
import ModelV2Config as ModelV2Config
import MyFunctions as myf
import numpy as np


myf.clear_session = False   # Set this to False, so we can keep the model for re-use.

output_col = 'BuyWeightingRule'

# Load a model
model_for_prediction = 'BuyV3b_250_22776_2_Layers_BuyV3b.h5'    #'31204_5_Layers.h5'
model = myf.load_model(model_for_prediction)
print('')
print('[INFO] Model File: ' + model_for_prediction)


# Load a dataset(ideally unseen)
#parsed_filename = 'Rule3^GDAXI_Now.csv'
parsed_filename = './parsed_data/Rule3_B0.98^GDAXI.csv'


#myf.debug_with_date = True          # We want the date - we're using it
#new_method_data, new_method_results, date_labels = myf.parsefile(r'.\\parsed_data\\' + parsed_filename, 'BuyWeightingRule', strip_last_row=False)
#myf.shuffle_data = False            # This is real - no shuffling!
#x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)


# Run against ALL data
#predictions = myf.predict_code_model (parsed_filename, model_for_prediction + '.h5', predictions=0)

#print(predictions)

# Standardised Loss, as per Standard Process
xtrain_new, xtest_new, ytrain_new, ytest_new = myf.parse_data_to_trainXY (parsed_filename, output_col, test_size=0.001)

# Test size of 0.1% means that Train is effectively all the data (close enough!!)

model.compile(loss=myf.default_model_loss_func, optimizer=myf.default_optimizer,
              metrics=['accuracy'])  # This feels a bit hacky...
results = model.evaluate(xtest_new, ytest_new, batch_size=myf.batch_size)
print(results)

standardised_loss = results[0]  # Is this correct?  Seems like it should be

# Get validation_loss
# Function should NOT care about model or data being correct - that is up to the caller.
# Maybe output both the actual loss as well as some form of box plot of the loss - would be useful to see not just the 'average' but the spread.

## Best Guess Logic on xtrain, ytrain
# Let's assume one ytrain value
if len(ytrain_new[0]) != 1:
    print('[ERROR] Multile output values are not yet handled')


min_target = min(ytrain_new)[0]
max_target = max(ytrain_new)[0]

best_loss = 999999
for i in np.arange(min_target, max_target, 0.01):
    # Do something
    loss = np.sum(np.square(ytrain_new-i))/len(ytrain_new)
    if loss < best_loss:
        best_loss = loss
        best_value = i

print (f"Best Guess Loss was: {best_loss} at a value of {best_value}")

standardised_loss = []

for i in range(0, 5):
    # Let's run another fit cycle - see if it gets better / worse
    myf.parse_process_plot(parsed_filename, output_col, model.model, 'Reprocess Testing' )

    # Standardised Loss, as per Standard Process
    xtrain_new, xtest_new, ytrain_new, ytest_new = myf.parse_data_to_trainXY (parsed_filename, output_col, test_size=0.001)

    # Test size of 0.1% means that Train is effectively all the data (close enough!!)

    model.compile(loss=myf.default_model_loss_func, optimizer=myf.default_optimizer,
                  metrics=['accuracy'])  # This feels a bit hacky...
    results = model.evaluate(xtest_new, ytest_new, batch_size=myf.batch_size)
    print(results)

    standardised_loss.append(results[0])  # Is this correct?  Seems like it should be


print (f'Model {model_for_prediction} and data file {parsed_filename} has been processed')
print (f'This file has a best guess loss of {best_loss} with a guess value of {best_value}')
print ('Additional processing gives the following loss values:')
for i in standardised_loss:
    print ('\t' + str(i))