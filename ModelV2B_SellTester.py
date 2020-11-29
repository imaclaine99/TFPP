import MyLSTMModelV2b
import MyFunctions as myf
from keras.regularizers import L1L2

# Sell Rule is not training as well - or more correctly, is training, but not validating well
# THis will test some ideas to address that



import ModelV2Config as ModelConfig

ModelConfig.buy_or_sell = 'Sell'         # Override Config, otherwise reporting is wrong!

model_id = 27070 #29840 # 31940 # # # #  #  # # # 31940  #,  #13535   #4857#   21910   # 16937    $14046??

modelDict = myf.read_from_from_db(
    unique_id=model_id)  # 52042   - Very good training loss, but bad validation loss - will be interesting to see if dupe data helps

Baseline = False
DupeDataTest = False
DupeAndNewLossTest = False
dropout_test = False
NoiseTest = False
ExtraDupeAndNewLossTest = False
NewLossTest = False
New2LossTest = False
new_p2l_rule = False
new_p2l_rule_multifile =False
new_p2l_rule_multifile_RandomModel=True

if Baseline:
    for i in range (0,3):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = False
            model.myf.EPOCHS = 250
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellBaseline_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)

if DupeDataTest:
    for i in range (0,5):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = True
            model.myf.EPOCHS = 250
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellDupeDataData_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)

if DupeAndNewLossTest:
    for i in range (0,5):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = True
            model.myf.EPOCHS = 250
            model.myf.model_loss_func = myf.biased_squared_mean
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellDupeDataAndNewLoss_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)

if ExtraDupeAndNewLossTest:
    for i in range (0,5):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = True
            model.myf.dupe_vals = (1, 2, 5, 5, 10, 10)

            model.myf.EPOCHS = 250
            model.myf.model_loss_func = myf.biased_squared_mean
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellExtraDupeDataAndNewLoss_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)



if NewLossTest:
    for i in range (0,5):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = False
            model.myf.EPOCHS = 250
            model.myf.model_loss_func = myf.biased_squared_mean
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellNewLoss_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)

            # Now evaluate one more time with the loss function overrided
            model.myf.custom_loss_override = True
#            model.model.evaluate()
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)



if New2LossTest:
    for i in range (0,5):
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.is_dupe_data = False
            model.myf.EPOCHS = 250
            model.myf.model_loss_func = myf.biased_squared_mean2
            model.model.summary()
            model.myf.model_description = str(model_id) + 'ModelV2b SellNew2Loss_Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "SellWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)

if dropout_test:
    for i in range (0,5):
        for dropout in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            ModelConfig.dense_dropout = dropout
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.EPOCHS = 250

            model.model.summary()
            model.myf.model_description = str(model_id) + ' DropoutTest_Exec_'+str(dropout) + 'Iteration' + str(i)
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "BuyWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)


if NoiseTest:
    for i in range (0,5):
        for noise in (1.0, 0.75, 0.5, 0.4,0.333, 0.25, 0.175, 0.1, 0.06666, 0.0333, 0.01, 0.00333, 0.001,0.000333, .0001, .0000333, .00001, 0.0000333, 0.00001, 0.00000333, 0):
            ModelConfig.input_noise = noise
            ModelConfig.is_input_noise = True
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.EPOCHS = 250
            model.myf.model_description = str(model_id) + ' NoiseTestExec_'+str(noise) + 'Iteration' + str(i)
            model.model.summary()
            print('[INFO]' + model.myf.model_description)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "BuyWeightingRule", model.model,
                                                      model.myf.model_description, version=2)
            #if model.myf.model_best_loss < 1.5:
            #    myf.save_model(model.model, model.myf.model_description + '.h5')
            model.myf.db_update_row(modelDict)



if new_p2l_rule == True:
    print ('NEW ATR RULE TESTING')
    filename = '^GDAXI.csv'
    myf.atr_rule = 3
    myf.parse_file(filename, purpose='Train')
    infile = "./parsed_data/Rule3_B0.98^GDAXI.csv"
    for rule in ( 'RangeVariance', 'P2LRatioPositive', 'P2LRatioNegative', 'P2LRatioNeutral','P2LRatio'):
        ModelConfig.buy_or_sell = rule + 'V3b'     # Used for reporting purposes
        xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY(infile, rule)
        for i in range(0, 8):
            print ('[INFO] Rule: ' + rule + ' Iteration: ' + str(i))
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.EPOCHS = 1200
            model.myf.early_stopping_patience = 50
            model.myf.is_dupe_data = False
            model.model.summary()
            model.myf.model_description =  str(model_id) + ' P2L Target Test' + rule
            model.myf.default_optimizer = ModelConfig.opt
            model.myf.parse_process_plot(infile, rule, model.model, model.myf.model_description+"Iteration" + str(i))
            if model.myf.model_best_loss < 1.5:
                myf.save_model(model.model, rule+'_' +str(model_id) + '_Iteration' + str(i) + '.h5')

            model.myf.db_update_row(modelDict)      # Should move this into parse_process_plot...  Can't, because that doesn't take the modelDict...

if new_p2l_rule_multifile == True:
    print ('NEW ATR RULE TESTING')
    #filename = '^GDAXI.csv'
    #myf.atr_rule = 3
    #myf.parse_file(filename, purpose='Train')
    #infile = "./parsed_data/Rule3_B0.98^GDAXI.csv"
    for rule in ( 'RangeVariance', 'P2LRatioPositive', 'P2LRatioNegative', 'P2LRatioNeutral','P2LRatio'):
        ModelConfig.buy_or_sell = rule + 'V3b_multi'     # Used for reporting purposes
        #xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY(infile, rule)
        for i in range(0, 8):
            print ('[INFO] Rule: ' + rule + ' Iteration: ' + str(i))
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.EPOCHS = 1200
            model.myf.early_stopping_patience = 50
            model.myf.is_dupe_data = False
            model.model.summary()
            model.myf.model_description =  str(model_id) + ' P2L Target Test' + rule
            model.myf.default_optimizer = ModelConfig.opt
            #model.myf.parse_process_plot(infile, rule, model.model, model.myf.model_description+"Iteration" + str(i))
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, rule, model.model,
                                                      model.myf.model_description+"Iteration" + str(i), version=2)
            if model.myf.model_best_loss < 1.5:
                myf.save_model(model.model, rule+'_V3BMulti_' +str(model_id) + '_Iteration' + str(i) + '.h5')

            model.myf.db_update_row(modelDict)      # Should move this into parse_process_plot...  Can't, because that doesn't take the modelDict...


if new_p2l_rule_multifile_RandomModel == True:
    print ('NEW ATR RULE TESTING')
    #filename = '^GDAXI.csv'
    #myf.atr_rule = 3
    #myf.parse_file(filename, purpose='Train')
    #infile = "./parsed_data/Rule3_B0.98^GDAXI.csv"
    for rule in ('P2LRatioPositive', 'P2LRatioNegative', 'P2LRatioNeutral','P2LRatio',  'RangeVariance'):
        ModelConfig.buy_or_sell = rule + 'V3b_multi'     # Used for reporting purposes
        #xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY(infile, rule)
        for i in range(0, 8):
            print ('[INFO] Rule: ' + rule + ' Iteration: ' + str(i))
            modelDict = myf.read_from_from_db('Random')
            model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model_id = modelDict['unique_id']
            #model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
            model.myf.EPOCHS = 1200
            model.myf.early_stopping_patience = 50
            model.myf.is_dupe_data = False
            model.model.summary()
            model.myf.model_description =  str(model_id) + ' P2L Target Test' + rule
            model.myf.default_optimizer = ModelConfig.opt
            #model.myf.parse_process_plot(infile, rule, model.model, model.myf.model_description+"Iteration" + str(i))
            model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, rule, model.model,
                                                      model.myf.model_description+"Iteration" + str(i), version=2)
            if model.myf.model_best_loss < 1.5:
                myf.save_model(model.model, rule+'_V3BMulti_' +str(model_id) + '_Iteration' + str(i) + '.h5')

            model.myf.db_update_row(modelDict)      # Should move this into parse_process_plot...  Can't, because that doesn't take the modelDict...


### END HERE

if single_epoch_at_a_time == True:
    infile = ".\parsed_data\\" + '^GDAXI.csv'
    xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY(infile, 'BuyWeightingRule')
    modelDict = myf.read_from_from_db(unique_id=45044)  #
    for i in range(0, 1):
        model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
        model.myf.EPOCHS = 1
        model.myf.is_dupe_data = False
        model.model.summary()
        model.myf.model_description = '45044 IterateEPOCHTEST'
        model.myf.default_optimizer = ModelConfig.opt
        for epoch in range(0,100):
            if epoch == 0:
                model_compile = True
            else:
                model_compile= False
            H = model.myf.compile_and_fit(model.model, xtrain, ytrain, xtest, ytest, loss='mean_squared_error', optimizer=ModelConfig.opt, compile = model_compile)
            print(H)

if multiple_data_files == True:
    modelDict = myf.read_from_from_db(unique_id=45044)  #
    for rule in (3,):        # Already done rule 1
        myf.atr_rule = rule
        if rule == 1:
            infile_array = (".\parsed_data\\" + '^GDAXI.csv', ".\parsed_data\\" + '^STOXX50E_OHLC.csv')
        elif rule == 3:
            infile_array = (".\parsed_data\\" + 'Rule3^GDAXI.csv', ".\parsed_data\\" + 'Rule3^STOXX50E_OHLC.csv')

        for i in range(0, 5):
            model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
            model.myf.model_description = '45044 MultiFileTest_Exec_ATR_Rule' + str(rule)
            model.myf.EPOCHS = 250
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot_multi_source(infile_array, "BuyWeightingRule", model.model, model.myf.model_description + "Iteration" + str(i))
            if model.myf.model_best_loss < 1.5:
                myf.save_model(model.model, 'ATRRule' + str(myf.atr_rule) + '_45044_Iteration' + str(i) + '.h5')

if new_atr_rule_beta_compare == True:
    multi_file_version = 2
    batch = 64
    print ('NEW ATR RULE BETA TESTING - Version' + str(multi_file_version) + 'ShuffledBatch' + str(batch))

    myf.atr_rule = 3
    for filename in ('^FTSE_2019OHLC.csv', '^GSPC.csv', '^GDAXI.csv', '^STOXX50E_OHLC.csv'):
        for beta_decay in (0.95, 0.98, 0.99):
           myf.atr_beta_decay = beta_decay
           myf.parse_file(filename, purpose='Train')

    for beta_decay in (0.95, 0.98, 0.99):
        myf.atr_beta_decay = beta_decay
       # myf.parse_file(filename, purpose='Train')

        infile_array = (".\parsed_data\\" + 'Rule3_B' + str(beta_decay) +'^GDAXI.csv', ".\parsed_data\\" + 'Rule3_B' + str(beta_decay) +'^STOXX50E_OHLC.csv',
                            ".\parsed_data\\" + 'Rule3_B' + str(beta_decay) +'^FTSE_2019OHLC.csv', ".\parsed_data\\" +'Rule3_B' + str(beta_decay) +'^GSPC.csv')

        model_id = 73767 #73767,   45044, 40358
        modelDict = myf.read_from_from_db(unique_id=model_id)  #73767,   45044, 40358

        for i in range(0, 3):
            model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
            model.myf.batch_size = batch
            model.myf.EPOCHS = 100  # * 4          # X 4 as I've just increased the batch size
            model.myf.is_dupe_data = False
            model.model.summary()
            model.myf.model_description = str(model_id) + ' MultiFile ATR Rule 3 Beta' + str(beta_decay) + ' Shuffled Multi File Version LargerAgainBatch ' + str(multi_file_version)
            model.myf.default_optimizer = ModelConfig.opt
            model.myf.parse_process_plot_multi_source(infile_array, "BuyWeightingRule", model.model, model.myf.model_description+"Iteration" + str(i), version=multi_file_version)
            if model.myf.model_best_loss < 1.5:
                myf.save_model(model.model, str(model_id) + ' MultiFile ATR Rule 3 Beta' + str(beta_decay) + 'Shuffled'+ 'Iteration' + str(i) + '.h5')


if rule3_reprocess_and_predict == True:
    # Step 1 - Re-process the in scope models
    print('[INFO] Rule 3 Reprocess and Predict on two models')
    for model_id in (73767, 40358):
        modelDict = myf.read_from_from_db(unique_id=model_id)
        model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
        model.myf.EPOCHS = 250
        model.myf.is_dupe_data = False
        model.myf.model_description = str(model_id)+'Rule3_Reprocess250Epochs'
        model.myf.default_optimizer = ModelConfig.opt
        infile = ".\parsed_data\Rule3^GDAXI.csv"
        model.myf.parse_process_plot(infile, "BuyWeightingRule", model.model, model.myf.model_description)

        if model.myf.model_best_loss < 1.5:
            myf.save_model(model.model, model.myf.model_description + '.h5')



    # Step 2 - Reparse the Dax_Now data with Rule 3
    # Parse Rule 3
    filename = '^GDAXI_Now.csv'
    myf.atr_rule = 3
    myf.parse_file(filename, purpose='Predict')

    # Step 3 - Predict with these two models
    for model_id in (73767, 40358):
        model = myf.load_model( str(model_id) + 'Rule3_Reprocess250Epochs.h5')
        model.summary()
        new_method_data, new_method_results = myf.parsefile(r'.\\parsed_data_full\\' + filename, 'BuyWeightingRule', strip_last_row=False)
        x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)
        for i in range (0, len(x_train)):
            new_predictions = model.model.predict(x_train[i:i+1])
            print ('Prediction:  ' + str(float(new_predictions)))
            print ('Based on data set ending on :')
            print (x_train[i])




if multifile_rule3_reprocess_and_predict == True:
    # Step 1 - Re-process the in scope models
    print('[INFO] MultiFile Rule 3 Reprocess and Predict on two models - 4 Files and 99 Beta')

    # Parse Rule 3
    myf.atr_rule = 3
    myf.atr_beta_decay = 0.99
    for filename in ('^FTSE_2019OHLC.csv', '^GSPC.csv'):
        myf.parse_file(filename, purpose='Train')

    # temporary remove - already processed - just need to run Step 3
    for model_id in []:            # (73767, 40358):
        modelDict = myf.read_from_from_db(unique_id=model_id)
        model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
        model.myf.EPOCHS = 250
        model.myf.is_dupe_data = False
        model.myf.model_description = str(model_id)+'Multi4File_Rule3_99_Reprocess250Epochs'
        model.myf.default_optimizer = ModelConfig.opt

#        infile_array = (".\parsed_data\\" + 'Rule3^GDAXI.csv', ".\parsed_data\\" + 'Rule3^STOXX50E_OHLC.csv')
        infile_array = (".\parsed_data\\" + 'Rule3^GDAXI.csv', ".\parsed_data\\" + 'Rule3^STOXX50E_OHLC.csv',
                        ".\parsed_data\\" + 'Rule3^FTSE_2019OHLC.csv', ".\parsed_data\\" +'Rule3^GSPC.csv')
        model.myf.parse_process_plot_multi_source(infile_array, "BuyWeightingRule", model.model, model.myf.model_description)

        if model.myf.model_best_loss < 1.5:
            myf.save_model(model.model, model.myf.model_description + '.h5')



    # Step 2 - Reparse the Dax_Now data with Rule 3
    # Parse Rule 3
    filename = '^GDAXI_Now.csv'
    #myf.atr_rule = 3
    #myf.parse_file(filename, purpose='Predict')

    # Step 3 - Predict with these two models
    filename = '^GDAXI_Now.csv'
    for model_id in (73767, 40358):
        print ('[INFO] Model ID:' + str (model_id))
        model = myf.load_model( str(model_id) + 'Multi4File_Rule3_99_Reprocess250Epochs.h5')
        model.summary()
        new_method_data, new_method_results = myf.parsefile(r'.\\parsed_data_full\\' + filename, 'BuyWeightingRule', strip_last_row=False)
        x_train, y_train = myf.parse_data(new_method_data, new_method_results, ModelV2Config.num_samples)
        for i in range (0, len(x_train)):
            new_predictions = model.model.predict(x_train[i:i+1])
            print ('Prediction:  ' + str(float(new_predictions)))
#            print ('Based on data set ending on :')
#            print (x_train[i])

if stateful_off_test:
    print('[INFO] LSTM Stateful Off Test')

    infile = ".\parsed_data\\" + '^GDAXI.csv'
    for model_id in (73767, 40358):
        print ('[INFO] Model ID:' + str (model_id))
        modelDict = myf.read_from_from_db(unique_id=model_id)  #
        for i in range(0, 3):
            model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
            model.myf.EPOCHS = 100
            model.lstm_stateful = False
            model.myf.is_dupe_data = False
            model.myf.model_description = str(model_id) + 'LSTM Stateful Off Test'
            model.myf.default_optimizer = ModelConfig.opt
            model.myf.parse_process_plot(infile, "BuyWeightingRule", model.model, model.myf.model_description + str(i))



# How to compare use of different LOSS FUNCTIONS???
#
#
# Idea:  Train as per normal, and at the end, run once on the test set with STANDARD MSE loss function to get a standard number.
# The idea is that the LOSS function should not affect the model, just how it trains, so this should work okay.
