import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


candidates_list_1rst_turn = ['JADOT', 'DUPONT-AIGNAN', 'LASSALLE', 'ARTHAUD', 'POUTOU', \
                   'ROUSSEL', 'MACRON', 'MÉLENCHON', 'ZEMMOUR', 'PECRESSE', 'LE PEN', 'HIDALGO']

cat_list = ['Village', 'Bourg', 'Petit ville', 'Ville moyenne', 'Grande ville', 'Métropole']

cat_values = {"num_agglo": {'Village': 1, 'Bourg': 2, 'Petit ville': 3, \
                            'Ville moyenne': 4, 'Grande ville': 5, 'Métropole': 6}}


def regression_plot(true_train, true_test, list_of_pred_df, candidat, town_train_index, town_test_index, turn=1):
    """
    input :
    - true_train : numpy array of true result for train_test
    - true_test : numpy array of true result for test_set
    - list_of_pred_df
        if we have learned n model, list_of_pred_df is a list of size n composed of
        tuple (dataframe of prediction for train_set, dataframe of prediction for test_set)

    - candidat : candidate for which we will plot the result
    - town_train/test_index : contain the city name and type_agglo of each lign of train and test set


    For each model, plot two subfigures :
        1) a scatter plot on the left side : true result of candidat on train test vs predict
            result of candidat on train set
        2) a scatter plot of the right side  : true result of candidat on test test vs predict
            result of candidat on test set
    """

    if turn == 1:
        candidates_list = candidates_list_1rst_turn
    else:
        candidates_list = ["LE PEN"]

    df_true_train = pd.DataFrame(true_train, columns=candidates_list)
    df_true_test = pd.DataFrame(true_test, columns=candidates_list)

    nb_of_pred = len(list_of_pred_df)
    i = candidates_list.index(candidat)
    f, axes = plt.subplots(nb_of_pred, 2, figsize=(25, nb_of_pred * 10))

    for j, pred in enumerate(list_of_pred_df):
        df_pred_train, df_pred_test = pred
        candidat_true_train = candidat + '_true_train' + str(j)
        candidat_true_test = candidat + '_true_test' + str(j)
        candidat_pred_train = candidat + '_pred_train' + str(j)
        candidat_pred_test = candidat + '_pred_test' + str(j)

        # Compute a dataframe which will be used to plot regression on train set with seaborn
        plot_df_train = pd.DataFrame(columns=[candidat_true_train, candidat_pred_train])
        plot_df_train[candidat_true_train] = df_true_train[candidat].values
        plot_df_train[candidat_pred_train] = df_pred_train[candidat].values

        # We sort by type_agglo so that the small city will be first plot and the big city will plot above
        plot_df_train["type_agglo"] = town_train_index["type_agglo"].values
        plot_df_train["type_agglo"] = pd.Categorical(plot_df_train["type_agglo"], categories=cat_list, ordered=True)
        plot_df_train.sort_values('type_agglo', inplace=True, ascending=True)

        # We compute a numerical vales for each type_agglo so that the dot for big city are bigger
        plot_df_train["num_agglo"] = plot_df_train["type_agglo"]
        # On remplace les catégories de type_agglo
        plot_df_train["type_agglo"] = plot_df_train["type_agglo"].cat.rename_categories(cat_values)
        
        # On remplace les autres colonnes si nécessaire
        cols_to_replace = [col for col in plot_df_train.columns if col != "type_agglo"]
        for col in cols_to_replace:
            if pd.api.types.is_categorical_dtype(plot_df_train[col]):
                plot_df_train[col] = plot_df_train[col].cat.rename_categories(cat_values)
            elif pd.api.types.is_object_dtype(plot_df_train[col]):  # Si la colonne est de type objet (chaînes de caractères)
                plot_df_train[col] = plot_df_train[col].replace(cat_values)


        # We then do the same thing for test set
        plot_df_test = pd.DataFrame(columns=[candidat_pred_test, candidat_true_test])
        plot_df_test[candidat_pred_test] = df_pred_test[candidat].values
        plot_df_test[candidat_true_test] = df_true_test[candidat].values

        plot_df_test["type_agglo"] = town_test_index["type_agglo"].values
        plot_df_test["type_agglo"] = pd.Categorical(plot_df_test["type_agglo"], categories=cat_list, ordered=True)
        plot_df_test.sort_values('type_agglo', inplace=True)

        plot_df_test["num_agglo"] = plot_df_test["type_agglo"]
        
        # Remplacer les catégories pour type_agglo
        plot_df_test["type_agglo"] = plot_df_test["type_agglo"].cat.rename_categories(cat_values)
        
        # Remplacer les autres colonnes si nécessaire
        cols_to_replace = [col for col in plot_df_test.columns if col != "type_agglo"]
        for col in cols_to_replace:
            if pd.api.types.is_categorical_dtype(plot_df_test[col]):
                plot_df_test[col] = plot_df_test[col].cat.rename_categories(cat_values)
            elif pd.api.types.is_object_dtype(plot_df_test[col]):  # Si la colonne est de type objet (chaînes de caractères)
                plot_df_test[col] = plot_df_test[col].replace(cat_values)


        min_result = plot_df_train[candidat_true_train].min()
        max_result = plot_df_train[candidat_true_train].max()

        R2_train = round(r2_score(plot_df_train[candidat_true_train], plot_df_train[candidat_pred_train]),3)
        R2_test = round(r2_score(plot_df_test[candidat_true_test], plot_df_test[candidat_pred_test]),3)


        if nb_of_pred != 1:
            axes[j,0].set_xlim(min_result, max_result)
            axes[j,0].set_ylim(min_result, max_result)
            axes[j,1].set_xlim(min_result, max_result)
            axes[j,1].set_ylim(min_result, max_result)

            sg = sns.scatterplot(y=candidat_true_train, x=candidat_pred_train, data=plot_df_train, \
                            hue="type_agglo", size="num_agglo", ax=axes[j, 0])
            sd = sns.scatterplot(y=candidat_true_test, x=candidat_pred_test, data=plot_df_test, \
                            hue="type_agglo", size="num_agglo", ax=axes[j, 1])

            sg.text(0.8, 0.2, "R2 = " + str(R2_train), \
                    horizontalalignment='center', verticalalignment='center', \
                    size='x-large', color='white', weight='bold', \
                    bbox=dict(facecolor='black', alpha=0.5), transform=axes[j, 0].transAxes)

            sd.text(0.8, 0.2, "R2 = " + str(R2_test), \
                    horizontalalignment='center', verticalalignment='center', \
                    size='x-large', color='white', weight='bold', \
                    bbox=dict(facecolor='black', alpha=0.5), transform=axes[j, 1].transAxes)
            x = np.arange(0, 1, 0.01)

            axes[j,0].plot(x, x)
            axes[j,1].plot(x, x)

        else:
            axes[0].set_xlim(min_result, max_result)
            axes[0].set_ylim(min_result, max_result)
            axes[1].set_xlim(min_result, max_result)
            axes[1].set_ylim(min_result, max_result)

            sg = sns.scatterplot(y=candidat_true_train, x=candidat_pred_train, data=plot_df_train, \
                            hue="type_agglo", size="num_agglo", ax=axes[0])
            sd = sns.scatterplot(y=candidat_true_test, x=candidat_pred_test, data=plot_df_test, \
                            hue="type_agglo", size="num_agglo", ax=axes[1])

            sg.text(0.8, 0.2, "R2 = " + str(R2_train), \
                    horizontalalignment='center', verticalalignment='center', \
                    size='x-large', color='white', weight='bold', \
                    bbox=dict(facecolor='black', alpha=0.5), transform=axes[0].transAxes)

            sd.text(0.8, 0.2, "R2 = " + str(R2_test), \
                    horizontalalignment='center', verticalalignment='center', \
                    size='x-large', color='white', weight='bold', \
                    bbox=dict(facecolor='black', alpha=0.5), transform=axes[1].transAxes)
            x = np.arange(0, 1, 0.01)
            axes[0].plot(x, x)
            axes[1].plot(x, x)

