import MyLSTMModelV2b
import MyFunctions
import ModelV2Config as ModelConfig

modelDict = MyFunctions.read_from_from_db(unique_id=31507)      # 98080
model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
model.model.summary()

model.myf.model_description = 'LSTMMModelV2b ' + ModelConfig.buy_or_sell + model.myf.dict_to_description(
    modelDict) + ModelConfig.opt
model.myf.default_optimizer = ModelConfig.opt
model.model.summary()

model.myf.parse_process_plot_multi_source(MyLSTMModelV2b.infile_array, "BuyWeightingRule", model.model,
                                          model.myf.model_description, version=2)


#model.myf.finish_update_row(ModelConfig.datafile, modelDict)
model.myf.db_update_row(modelDict)

if model.myf.model_best_loss < 1.5:
    MyFunctions.save_model(model.model, str(modelDict['unique_id']) + '_' + str(modelDict['Layers']) + '_Layers.h5')