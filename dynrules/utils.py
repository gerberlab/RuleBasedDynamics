import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.colors as mcolors


_ctemp = mcolors.TABLEAU_COLORS
COLORS = list(_ctemp.keys())*3


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


def get_discrete_gradient(t, x, mask_times):
    # for log, assuming input is xlog
    ntime, nsubj, notu = x.shape
    ygrad = np.zeros((ntime-1, nsubj, notu))

    for i in range(ntime-1):
        dt = t[i+1] - t[i]
        ygrad[i,:,:] = (x[i+1,:,:] - x[i,:,:])/dt

    xclip = x[:-1,:,:]
    tclip = t[:-1]
    if mask_times is not None:
        # zero out gradients for before introduced
        for oidx in range(notu):
            ygrad[tclip<mask_times[oidx],:,oidx] = 0

    return tclip, xclip, ygrad


def get_data(t, xlog, tind_first, device):
    num_time, num_subj, num_taxa = xlog.shape   
    xabs = np.exp(xlog)
    max_range = 1.1*np.nanmax(xabs, axis=(0,1)) # max over TxS; final shape O

    times = torch.from_numpy(t).to(torch.float)
    xstd = torch.from_numpy(np.nanstd(xlog, axis=0)).to(torch.float)
    xlog = torch.from_numpy(xlog).to(torch.float)
    xmean = torch.nanmean(xlog, dim=0)

    tstd, tmean = torch.std_mean(times)
    xnorm = (xlog-xmean)/(xstd + 1e-6)
    xnorm[torch.isnan(xnorm)] = 0
    # TODO: need to think more about norm-representaiton and latent encoding for masked times...
    tnorm = (times-tmean)/(tstd + 1e-6)

    xnormflat = xnorm.reshape((num_time,-1))
    norm_data = torch.cat([tnorm[:,None].T, xnormflat.T]).T
    norm_data = norm_data.reshape((-1))

    data = {'times': times.to(device),  'log_abundance': xlog.to(device), 'norm_data': norm_data.to(device), 'x_norm': xnorm.to(device), \
            'tind_first': tind_first}
    return data, max_range


def get_grad_data(tclip, xlogclip, ygrad, tind_first, device):
    #* get data for grad matching
    x_data = np.exp(xlogclip)
    x_data[np.isnan(xlogclip)] = 0
    times = torch.from_numpy(tclip).to(torch.float)
    xlog_data = torch.from_numpy(xlogclip).to(torch.float)
    x_data = torch.from_numpy(x_data).to(torch.float)
    xlog_grad = torch.from_numpy(ygrad).to(torch.float)

    data = {'times': times.to(device), 'log_abundance': xlog_data.to(device), 
            'abundance': x_data.to(device), 'gradient_log': xlog_grad.to(device),
            'tind_first': tind_first}
    return data


def integrate_log_gradient(tfull, ygrad, log_ic, mask_times):
    ntimes = len(tfull)
    _, nsubj, ntaxa = ygrad.shape

    x_log_int = np.zeros((ntimes, nsubj, ntaxa))
    x_log_int[0,:,:] = log_ic

    #* using mask times; integrate each taxa separately
    for oidx in range(ntaxa):
        for t in range(1,ntimes):
            if tfull[t] >= mask_times[oidx]:
                dt = tfull[t] - tfull[t-1]
                dyn = ygrad[t-1,:,oidx]
                x_log_int[t,:,oidx] = x_log_int[t-1,:,oidx] + dyn*dt
            else:
                x_log_int[t,:,oidx] = x_log_int[t-1,:,oidx]

    x_int = np.exp(x_log_int)
    return x_log_int, x_int


def get_interaction_matrix_MAP(params):
    rules = params['rules']
    dets = params['detectors']
    #! self interactions are off by definition
    num_taxa, num_rules = rules.shape
    inter_mat = np.zeros((num_taxa, num_taxa))

# TODO:  TOPOLOGY MATRIX; get sign of interaction too
    for i in range(num_taxa):
        for k in range(num_rules):
            #* i is target variable, j is effector taxon
            #* loop through rules for target taxon i
            if rules[i,k] > 0.5:
                for j in range(num_taxa):
                    if dets[i,k,j] > 0.5:
                        inter_mat[i,j] = 1

    return inter_mat
