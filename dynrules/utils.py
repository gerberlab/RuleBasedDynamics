import torch
import torch.nn.functional as F
import numpy as np
import pickle


def move_to_numpy(obj):
    if type(obj) == list:
        npobj = []
        for object in obj:
            objnp = move_to_numpy(object)
            npobj.append(objnp)
        return npobj
    elif type(obj) == dict:
        npdict = dict()
        for key in obj:
            objnp = move_to_numpy(obj[key])
            npdict[key] = objnp 
        return npdict
    else:
        if type(obj) == torch.Tensor:
            return obj.cpu().detach().numpy()
        else:
            return obj
            

def save_model(model, file):
    torch.save(model, file)


def pickle_save(data, file, move_numpy=True):
    with open(file, "wb") as h:
        if move_numpy is True:
            pickle.dump(move_to_numpy(data), h)
        else:
            pickle.dump(data, h)


def pickle_load(file):
    with open(file, "rb") as h:
        data = pickle.load(h)
    return data


def KL_bernoulli(q_params, prior_probs):
    q_prob = torch.sigmoid(q_params)
    q_logprob = F.logsigmoid(q_params)
    q_invlogprob = F.logsigmoid(-q_params)

    KL = torch.sum(q_prob*(q_logprob - torch.log(prior_probs))) + \
            torch.sum((1-q_prob)*(q_invlogprob - torch.log(1-prior_probs)))
    return KL


def KL_gaussian(mu, var, prior_mean, prior_var):
    return 0.5*torch.sum(var/prior_var + ((mu-prior_mean)**2)/prior_var - 1.0 - torch.log(var/prior_var))
