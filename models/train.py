
import torch 
import pandas as pd 

candidates_list_1rst_turn = ['JADOT', 'DUPONT-AIGNAN', 'LASSALLE', 'ARTHAUD', 'POUTOU', \
                   'ROUSSEL', 'MACRON', 'MÉLENCHON', 'ZEMMOUR', 'PECRESSE', 'LE PEN', 'HIDALGO']


def train_neurals_networks(n_epochs, train_loader, list_of_loss_function, \
                            list_of_models, list_of_optim, print_epochs = 50): 
    """
    input : 
    - n_epochs, number of epochs
    - train_loader : torch loader for train dataset
    - list_of_models : list of models we want to train
    - list_loss_function : one loss_function for each model 
    - list_of_optim : one optimizer for each model
    - print_epochs : we will print loss every print_epochs
    
    output : 
    - train each model simustanously  
    """
    n_models = len(list_of_models) 
    for epochs in range(n_epochs):
        losses =[0] * n_models 
        batch_number = 0
        for x_batch, y_batch in train_loader:
            batch_number += 1
            
            for i in range(n_models):
                
                list_of_optim[i].zero_grad()
                y_pred = list_of_models[i](x_batch)
                loss = list_of_loss_function[i](y_pred, y_batch)
                loss.backward()
                list_of_optim[i].step()
                losses[i] += loss.item()
            
        if epochs % print_epochs == 0 :
            print("epoch n°", epochs)
            for i, loss in enumerate(losses) :     
                print("  model n°", i, "- loss :", loss/batch_number)
            print("----")


def pred_model(x_train, x_test, list_of_models, turn=1):
    """
    input : 
    - x_train : numpy array
    - x_test : numpy array
    - list_of_models : list of models we have trained

    output : 
    - a list of dataframe which has the same size than list_of_models
        each element of the list is a tupple : 
        (pd.dataframe of predictions on train set, pd.dataframe of predictions on test set)
    """
    if turn == 1:
        candidates_list = candidates_list_1rst_turn
    if turn == 2:
        candidates_list = ["LE PEN"]

    list_of_df = []
    x_tensor_train = torch.from_numpy(x_train).float()
    x_tensor_test = torch.from_numpy(x_test).float()

    with torch.no_grad():
        for model in list_of_models :
            y_pred_train = model(x_tensor_train).numpy()
            y_pred_test = model(x_tensor_test).numpy()
            df_train = pd.DataFrame(y_pred_train, columns=candidates_list)
            df_test = pd.DataFrame(y_pred_test, columns=candidates_list)

            list_of_df.append((df_train, df_test))
            
    return list_of_df
