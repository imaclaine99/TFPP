import MyLSTMModelV2b
import ModelV2Config as ModelConfig
import MyFunctions as myf

# Global config
myf.db_host = ModelConfig.db_host
myf.db_username = ModelConfig.db_username
myf.db_pwd = ModelConfig.db_pwd

model_id = 4857   # 16937

modelDict = myf.read_from_from_db(unique_id=4857)      # 98080

new_file_array = []

# Adjust for path
#for  file in MyLSTMModelV2b.infile_array:
#    new_file_array.append('..\\' + file)

modelDict = myf.read_from_from_db(
    unique_id=model_id)
for i in range (0,2):
        model = MyLSTMModelV2b.MyLSTMModelV2b(modelDict)
        model.myf.EPOCHS = 25
        model.model.summary()
        model.myf.model_description = str(model_id) + ' Simple Test Iteration' + str(i)
        model.myf.default_optimizer = ModelConfig.opt
        model.model.summary()
        model.myf.parse_process_plot_multi_source( MyLSTMModelV2b.infile_array, "BuyWeightingRule", model.model,
                                                  model.myf.model_description, version=2)
        model.myf.db_update_row(modelDict)
#        if model.myf.model_best_loss < 1.5:
 #           myf.save_model(model.model, model.myf.model_description + '.h5')
