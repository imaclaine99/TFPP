import csv
import sys
import MyLSTMModelV2
import ModelV2Config
from keras import backend as K

#with open('models_to_test.csv', 'w', newline='') as csvfile:
    #fieldnames = ['LayerType', 'Nodes', 'OverFittingHelp', 'ReturnSequences', 'Started', 'Finished']
#    fieldnames = ['Layer1', 'Layer2', 'Layer3']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#    writer.writeheader()
#
list_all_layer_types = []

def has_NonCompliantLSTM(modelDict):
    """
        Checks if we have non compliant LSTM layers, allowing them to be removed
        This includes an LSTM with False Return sequences followed by any other LSTM
    :param modelDict:
    :return: True of False
    """
    for start_layer in range(1, modelDict['Layers']+1):
        if modelDict['Layer' + str(start_layer)]['LayerType'] == 'LSTM' and modelDict['Layer' + str(start_layer)]['ReturnSequences'] == False:
            # We have an LSTM with False Return Sequences - check if we have any more LSTMs
            for check_layer in range( start_layer+1, modelDict['Layers']):
                if modelDict['Layer' + str(check_layer)]['LayerType'] == 'LSTM':
                    return True

for layer_type in ('Dense', 'LSTM') :
    if layer_type == 'Dense' :
        max_nodes  = 12
        returnSequencesList = {False}       # not applicable
    else:
        max_nodes = 10
        returnSequencesList = {False,True}  # not applicable
    for nodes in range(0,max_nodes):	# 9 if LSTM
        for over_fitting_help in {0}:      # Potential to add more later - not top priority           | (0, L1, L2, L1L2):
            for returnSequences in returnSequencesList:
                configTuple = (layer_type, nodes, over_fitting_help, returnSequences)
                layerDict = {'LayerType': layer_type, 'Nodes': nodes, 'OverFittingHelp': over_fitting_help, 'ReturnSequences': returnSequences}
                multiLayerDict = {"Layer1": layerDict, "Layer2": layerDict}
                print (configTuple)
                print (layerDict)
            #    writer.writerow(layerDict)
                print (multiLayerDict)
                list_all_layer_types.append(layerDict)
           #     writer.writerow(multiLayerDict)



print(list_all_layer_types)
# Now we have the full list, let's create and write the permutations

with open('models_to_test2.csv', 'w', newline='') as csvfile:
    #fieldnames = ['LayerType', 'Nodes', 'OverFittingHelp', 'ReturnSequences', 'Started', 'Finished']
    fieldnames = ['Layers', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Started', 'Finished', 'model_best_acc',
                      'model_best_loss', 'model_best_val_acc', 'model_best_val_loss',
                      'model_best_combined_ave_loss', 'TotalNodes','ErrorDetails']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    # 1 Layer
    #for layer1 in list_all_layer_types:
    #    configDict = {'Layers': 1, 'Layer1': layer1, "Started" : False, 'Finished' : False, 'model_best_acc':'', 'model_best_loss':'', 'model_best_val_acc':'', 'model_best_val_loss':'', 'model_best_combined_ave_loss':''}

    #2 Layer
    for layer1 in list_all_layer_types:
        print ('[INFO] Next Layer 1 iteration')
        for layer2 in [''] + list_all_layer_types:
            for layer3 in [''] + list_all_layer_types:
                for layer4 in [''] + list_all_layer_types:
                    for layer5 in [''] : #+ list_all_layer_types:
                        if layer2 == '':
                            layers = 1
                        elif layer3 == '':
                            layers = 2
                        elif layer4 == '':
                            layers = 3
                        elif layer5 == '':
                            layers = 4
                        else:
                            layers = 5
                        # Check if we are iterating on a future layer - i.e. layerN is not blank but an earlier layer is
                        if layer2 =='' and (layer3 != '' or layer4 != '' or layer5 != ''):
                            # Do noting
                            pass
                        elif layer3 == '' and (layer4 != '' or layer5 != ''):
                            pass
                        elif layer4 == '' and layer5 != '':
                            pass
                        # Check if we're the last layer.  If we are, the we MUST have one node only
                        elif (layers == 5 and int(layer5['Nodes'] > 0)) or (layers == 4 and int(layer4['Nodes']> 0)) or (layers == 3 and int(layer3['Nodes']> 0)) or (layers == 2 and int(layer2['Nodes'])>0) or (layers == 1 and int (layer1['Nodes']> 0)):
                            pass
                        # New Rule:  Don't allow a later layer to have any more than 1 more node than the previous layer
                        elif (layers ==5 and int(layer5['Nodes']) > int(layer4['Nodes'])+1) or (layers >= 4 and int(layer4['Nodes']) > int(layer3['Nodes'])+1) or (layers>=3 and int(layer3['Nodes']) > int(layer2['Nodes'])+1) or (layers>= 2 and int(layer2['Nodes']) > int(layer1['Nodes'])+1) :
                            pass
                        # New Rule:  Don't allow any LSTM to occur AFTER an LSTM that has return_sequences of FALSE
                        else:
                            configDict = {'Layers': layers, 'Layer1': layer1, 'Layer2' : layer2, 'Layer3': layer3, 'Layer4' : layer4, 'Layer5': layer5, "Started": False, 'Finished': False, 'model_best_acc': '',
                                          'model_best_loss': '', 'model_best_val_acc': '', 'model_best_val_loss': '',
                                          'model_best_combined_ave_loss': '', 'ErrorDetails' : ''}
                            #writer.writerow(configDict)

        #                    model = MyLSTMModelV2.MyLSTMModelV2(configDict)
                            if has_NonCompliantLSTM(configDict):
                                # Filter out models where we have False ReturnSequences for LSTM followed by other LSTM - this just doesn't work (or make sense!)
                                pass
                            else:
                                try:
               #                     model.model.summary()
                                    # Need more on this....
                                    model_description = str(layers) + 'Layers_' + configDict['Layer1']['LayerType']+str(configDict['Layer1']['Nodes'])
                                    # Count the total nodes - will be useful for later (maybe).  E.g. use to decide whether to use CPU or GPU
                                    total_nodes = 2 ** int(configDict['Layer1']['Nodes'])
                                    if layers >= 2:
                                        model_description += '_' + configDict['Layer2']['LayerType']+str(configDict['Layer2']['Nodes'])
                                        total_nodes += 2 ** int(configDict['Layer2']['Nodes'])
                                    if layers >= 3:
                                        model_description += '_' + configDict['Layer3']['LayerType']+str(configDict['Layer3']['Nodes'])
                                        total_nodes += 2 ** int(configDict['Layer3']['Nodes'])
                                    if layers >= 4:
                                        model_description += '_' + configDict['Layer4']['LayerType']+str(configDict['Layer4']['Nodes'])
                                        total_nodes += 2 ** int(configDict['Layer4']['Nodes'])
                                    if layers >= 5:
                                        model_description += '_' + configDict['Layer5']['LayerType']+str(configDict['Layer5']['Nodes'])
                                        total_nodes += 2 ** int(configDict['Layer5']['Nodes'])
                  #                  model.myf.model_description = 'LSTMMModelV2 ' + ModelV2Config.buy_or_sell + model_description + ModelV2Config.opt
                  #                  model.myf.default_optimizer = ModelV2Config.opt
                   #                 model.myf.parse_process_plot(".\parsed_data\^GDAXI.csv", "BuyWeightingRule",
                     #                                            model.model,
                    #                                             model.myf.model_description)
                     #               configDict['model_best_loss'] = model.myf.model_best_loss
                    #                configDict['model_best_acc'] = model.myf.model_best_acc
                   #                 configDict['model_best_combined_ave_loss'] = model.myf.model_best_combined_ave_loss
                   #                 configDict['model_best_val_acc'] = model.myf.model_best_val_acc
                   #                 configDict['model_best_val_loss'] = model.myf.model_best_val_loss
                                    configDict['TotalNodes'] = total_nodes
                                    writer.writerow(configDict)

                                except:
                                    print("Oops!", sys.exc_info()[0], "occured.")
                                    print('Occurred with configDict:')
                                    print(configDict)
                                    configDict['ErrorDetails'] = sys.exc_info()[0]
                                    writer.writerow(configDict)

                            K.clear_session()