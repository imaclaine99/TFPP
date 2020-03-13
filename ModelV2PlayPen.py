import MyLSTMModelV2
import MyFunctions
import ModelV2Config as ModelConfig

modelDict = MyFunctions.read_from_from_db(unique_id=98080)
model = MyLSTMModelV2.MyLSTMModelV2(modelDict)
model.model.summary()

model.myf.model_description = 'LSTMMModelV2 ' + ModelConfig.buy_or_sell + model.myf.dict_to_description(
    modelDict) + ModelConfig.opt
model.myf.default_optimizer = ModelConfig.opt
model.model.summary()
model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule",
                             model.model,
                             model.myf.model_description)

# model.myf.finish_update_row(ModelConfig.datafile, modelDict)
#yFunctions.db_update_row(modelDict)
if model.myf.model_best_loss < 1.5:
    model.myf.save_model(model, str(modelDict['unique_id']) + '_' + str(modelDict['Layers']) + '_Layers.h5')