import torch
import numpy as np
from sklearn.metrics import r2_score

eps = 0.001


def cross_entropy_loss(y, target):
    """"
    Compute the cross_entropy between two discrete probability distribution
    Return the mean over the batch
    """
    return torch.mean(torch.sum(- target * torch.clamp(torch.log(y+eps), min=-1000, max=1000), 1))


def cross_entropy_loss_1Neuron(y, target):
    """
    Used with one neuron neural network 
    Idea : the neuron predict the result (y) of one candidat. 
           if the true result is t, our loss will be 
           - t * log(y) - (1-t) * log(1-y) 
    args :
        - (batch_size, ) torch of prediction result for one candidate
        - (batch_size, ) torch of true result

    
    return : 
        mean over the batch of soft cross-entropy loss
    """
    return torch.mean(- target * torch.clamp(torch.log(y+eps),min=-10000, max=10000)\
                      - (1-target) * torch.clamp(torch.log(1-y+eps), min=-10000, max=10000))


def accurracy(predict, true):
    """
    Used in the second turn to compute accurary of the model
    (= number of town we predict correctly LE PEN vs number of town we are wrong)

    :arg
        - predict : series of numerical predict result value for LE PEN
        - true : series of numerical true result values for LE PEN
    """

    winner_predict = np.where(predict >= 0.5, 1, 0).ravel()
    true_predict = np.where(true >= 0.5, 1, 0).ravel()

    accuracy = (winner_predict == true_predict).sum() / true_predict.shape[0]
    return accuracy


def compute_R2_df(predict, true, verbose=True):
    """
    :param predict: dataframe of prediction results for each candidats
    :param true: true results
    :return: list of R2 for each candidates
    """
    R2_list = []
    for cand in predict.columns:
        r2 = r2_score(true[cand], predict[cand])
        R2_list.append(r2)

        if verbose:
            print("R2 for", cand, " : ", r2)

    return R2_list


def compute_R2_np(predict, true):
    """
    :param predict: numpy array of prediction results for each candidats
    :param true: true results
    :return: np array of R2 for each candidates
    """
    R2 = np.zeros((1, predict.shape[1]))

    for i in predict.shape[1]:
        R2[0, i] = r2_score(true[:, i], predict[:, i])

    return R2
