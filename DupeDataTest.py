import MyLSTMModelV2
import MyLSTMModelV2b
import MyFunctions as myf
import ModelV2Config



# Will use this to test DupeData
# Let's run against a given model 5 times with and then without Dupe Data - duping ALL data - this should have no impact if it all works




import ModelV2Config as ModelConfig


dupe = False
single_epoch_at_a_time = False
multiple_data_files = False
new_atr_rule = False
rule3_reprocess_and_predict = False
multifile_rule3_reprocess_and_predict = False
new_atr_rule_beta_compare = False
stateful_off_test = False
BuyRuleCorrection = True

if dupe == True:
    modelDict = myf.read_from_from_db(
        unique_id=45044)  # 52042   - Very good training loss, but bad validation loss - will be interesting to see if dupe data helps

    for dupe_data in (True,):
        for i in range (0,5):
            model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
            model.myf.EPOCHS = 100
            model.myf.is_dupe_data = dupe_data
            model.myf.dupe_vals = (1,1,3,4,5,5)     # Everything should have no impact!
            model.model.summary()
            model.myf.model_description = '45044 DupeDataTest_113455_Dupe' + str(dupe_data) + '_Exec' + str(i)
            model.myf.default_optimizer = ModelConfig.opt
            model.model.summary()
            model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule",
                                         model.model,
                                         model.myf.model_description)


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


if new_atr_rule == True:
    print ('NEW ATR RULE TESTING')
    filename = '^GDAXI.csv'
    myf.atr_rule = 3
    myf.atr_beta_decay = 0.99
    myf.parse_file(filename, purpose='Train')
    filename = '^GDAXI_Now.csv'
    myf.atr_rule = 3
    myf.parse_file(filename, purpose='Train')
    infile = ".\parsed_data\Rule3^GDAXI.csv"
    xtrain, xtest, ytrain, ytest = myf.parse_data_to_trainXY(infile, 'BuyWeightingRule')
    modelDict = myf.read_from_from_db(unique_id=45044)  #
    for i in range(0, 5):
        model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
        model.myf.EPOCHS = 100
        model.myf.is_dupe_data = False
        model.model.summary()
        model.myf.model_description = '45044 ATR Rule 3 Beta99'
        model.myf.default_optimizer = ModelConfig.opt
        model.myf.parse_process_plot(infile, "BuyWeightingRule", model.model, model.myf.model_description+"Iteration" + str(i))
        if model.myf.model_best_loss < 1.5:
            myf.save_model(model.model, 'NewATRRule3_45044_Iteration' + str(i) + '.h5')

if new_atr_rule_beta_compare == True:
    multi_file_version = 2
    batch = 64
    print ('NEW ATR RULE BETA TESTING - Version' + str(multi_file_version) + 'ShuffledBatch' + str(batch))

    myf.atr_rule = 3
    for filename in ('^FTSE_2019OHLC.csv', '^GSPC.csv', '^GDAXI.csv', '^STOXX50E_OHLC.csv'):
        for beta_decay in ( 0.98, 0.99,0.95,):
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


if BuyRuleCorrection:
    print('[INFO] Buy Rule Correction Test (Correct Value 2)')

    myf.atr_rule = 3
    myf.atr_beta_decay = 0.98

    for filename in ('^FTSE_2019OHLC.csv', '^GSPC.csv', '^GDAXI.csv', '^STOXX50E_OHLC.csv'):
        myf.parse_file(filename, purpose='Train')

    model_id = 14046
    modelDict = myf.read_from_from_db(
        unique_id=model_id)

    for i in range(0,3):
        # Use all 4 files
        model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
        model.myf.EPOCHS = 250
        model.myf.processing_rule='BuyFixed'
        model.model.summary()
        model.myf.model_description = str(model_id) + ' FixedBuyRule3'  + ' Iteration' + str(i)
        print('[INFO]' + model.myf.model_description)
        model.myf.default_optimizer = ModelConfig.opt
        model.model.summary()
        model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "BuyWeightingRule", model.model,
                                                  model.myf.model_description, version=2)
        # if model.myf.model_best_loss < 1.5:
        #    myf.save_model(model.model, model.myf.model_description + '.h5')
        model.myf.db_update_row(modelDict)

        # Use just one  file
        model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
        model.myf.EPOCHS = 250
        model.myf.processing_rule='BuyFixed'
        model.model.summary()
        model.myf.model_description = str(model_id) + ' FixedBuyRule3SingleFile'  + ' Iteration' + str(i)
        print('[INFO]' + model.myf.model_description)
        model.myf.default_optimizer = ModelConfig.opt
        model.model.summary()
        model.myf.parse_process_plot_multi_source((".\parsed_data\\" + 'Rule3_B0.98^GDAXI.csv',), "BuyWeightingRule", model.model,
                                                  model.myf.model_description, version=2)
        # if model.myf.model_best_loss < 1.5:
        #    myf.save_model(model.model, model.myf.model_description + '.h5')
        model.myf.db_update_row(modelDict)

