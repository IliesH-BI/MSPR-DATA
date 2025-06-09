from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

candidates_list_1rst_turn = ['JADOT', 'DUPONT-AIGNAN', 'LASSALLE', 'ARTHAUD', 'POUTOU', \
                   'ROUSSEL', 'MACRON', 'MÉLENCHON', 'ZEMMOUR', 'PECRESSE', 'LE PEN', 'HIDALGO']


def load_data(dataset, features_to_keep, batch_size, min_habs, test_size, turn=1, apply_PCA=False, n_components_PCA=10):
    """
    args :
        - dataset : panda dataframe containing INSEE features and results of each candidates
        - features : list of features we want to keep 
        - batch_size : size of the batch for training
        - min_habs : only keep cities for which the population >= min_habs
        - test_size : use to split the dataset
        - candidates_list : list of all candidates we want to keep

    return : 
        - x_train, x_test : numpy array containing only the numerical features 
                            index of array correspond to town_index (see below)
        - y_train, y_test : numpy arrau containing the candidates results 
            columns are in the order of candidates_list
        - train_loader : torch dataloader for x_train and y_train
        - town_index_train, town_index_test : name of city and type of agglomeration
            for each index of x_train, x_test
    """

    if turn == 1:
        candidates_list = candidates_list_1rst_turn
    else:
        candidates_list = ["LE PEN"]

    final_dataset = dataset[features_to_keep + candidates_list + \
                            ["code", "pop", "type_agglo", "commune"]]

    # SELECT A SPECIFIC AMOUNT OF HABITANTS 
    final_dataset = final_dataset[final_dataset["pop"] >= min_habs].drop(columns=["pop"])

    X = final_dataset.drop(columns=(["commune", "code", "type_agglo"] + candidates_list))
    Y = final_dataset[candidates_list]
    Y = Y / 100

    df_x_train, df_x_test, df_y_train, df_y_test = \
        train_test_split(X, Y, test_size=test_size, random_state=42)

    commune_index_train = final_dataset.loc[df_x_train.index, ["commune", "type_agglo"]]
    commune_index_test = final_dataset.loc[df_x_test.index, ["commune", "type_agglo"]]

    nb_features = df_x_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(df_x_train)

    x_train = scaler.transform(df_x_train)
    x_test = scaler.transform(df_x_test)

    if apply_PCA:
        pca = PCA(n_components_PCA)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    y_train = df_y_train.values
    y_test = df_y_test.values

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), \
                                  torch.from_numpy(y_train).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return x_train, x_test, y_train, y_test, train_loader, \
           commune_index_train, commune_index_test
