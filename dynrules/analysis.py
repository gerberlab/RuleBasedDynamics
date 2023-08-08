from dynrules.utils import pickle_load, integrate_log_gradient, get_discrete_gradient, get_interaction_matrix_MAP, COLORS, sigmoid
from pathlib import Path
from dynrules.rule_visualization import RuleVisualizer, RuleSet
import matplotlib.pyplot as plt 
import torch 
import numpy as np 
import seaborn as sns
import pandas as pd 

#* common elements to most post processing
#* -----------------------------------------------
#* plot parameter traces if available
#* sample from posterior and plot fit
#* compute rmse (gradient and/or integrated trajectory)
#* ....if ground truth, plot error of predicted parameters
#* ....some depiction of learned interactions between taxa -- interpretable model**
#*** note common naming in data and results***

# TODO: clean up distinction in traces between rules and glv matrix...

#! -------------------
#* common methods
def extract_var_trace(ptrace, varname):
    n = len(ptrace)
    pinit = ptrace[0][varname]
    if type(pinit) == float:
        pinit_shape = (1,)
    else:
        if len(pinit.shape) == 0:
            pinit_shape = (1,)
        else:
            pinit_shape = pinit.shape
    shape = (n,) + pinit_shape
    vtrace = np.zeros(shape)
    for i in range(n):
        val = ptrace[i][varname]
        vtrace[i,:] = val
    return vtrace


def plot_trace(ptrace, varname, savedir, gtruth):
    trace = extract_var_trace(ptrace, varname)
    gt_exists = False
    if gtruth is not None:
        #* if key exists, have ground truth for that parameter...
        if varname in gtruth:
            nepoch = trace.shape[0]
            gt = gtruth[varname]
            gt_trace = np.repeat(np.expand_dims(gt, axis=0), nepoch, axis=0)
            gt_exists = True

    if len(trace.shape) == 1:
        fig, ax = plt.subplots()
        ax.plot(trace, color=COLORS[0])
        if gt_exists:
            ax.plot(gt_trace, '--', color=COLORS[0])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(varname)
        ax.set_title(f"{varname} trace")
        plt.savefig(savedir / f"{varname}_trace.png", bbox_inches="tight")
        plt.close()
    elif len(trace.shape) == 2:
        nepoch, ntaxa = trace.shape
        fig, ax = plt.subplots()
        for i in range(ntaxa):
            ax.plot(trace[:,i], label=f'OTU {i}', color=COLORS[i])
            if gt_exists:
                ax.plot(gt_trace[:,i], '--', color=COLORS[i])
        ax.legend(bbox_to_anchor=(1.01,1.01))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(varname)
        ax.set_title(f"{varname} trace")
        plt.savefig(savedir / f"{varname}_trace.png", bbox_inches="tight")
        plt.close()
    elif len(trace.shape) == 3:
        nepoch, ntaxa, nrules = trace.shape
        for i in range(ntaxa):
            fig, ax = plt.subplots()
            for j in range(nrules):
                ax.plot(trace[:,i,j], label=f'Rule {j}')
                if gt_exists:
                    ax.plot(gt_trace[:,i,j], '--', color=COLORS[j])
            ax.legend(bbox_to_anchor=(1.01,1.01))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(varname)
            ax.set_title(f"{varname} trace (OTU {i})")
            plt.savefig(savedir / f"{varname}_trace_OTU{i}.png", bbox_inches="tight")
            plt.close()
    else:
        print("not implemented")


def plot_all_traces(res, savedir, gtruth):
    ptrace = res['param_trace']
    for vname in ptrace[0].keys():
        print(f"plotting {vname} trace...")
        plot_trace(ptrace, vname, savedir, gtruth)



#! -------------------
#* gradient matching common methods

def plot_gradient_fits_MAP(data, res, mapfitpath):
    # data = {'times': times.to(device), 'gradient': y.to(device), 'abundance': x.to(device)}
    tclip = data['times']
    grad_data = data['gradient']
    x_data = data['abundance']
    num_time, num_subj, num_taxa = x_data.shape
    taxonomy = res['taxonomy']
    prediction = res['prediction']

    #* transform back data
    # TODO: settle this; are we using this?
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']
    # prediction = prediction*data['y_rescale'] + data['y_shift']

    for s in range(1): #num_subj):
        for oidx in range(num_taxa):
            fig, ax = plt.subplots()
            ax.plot(tclip, grad_data[:,s,oidx], '-x', label=f"OTU {oidx} data")
            ax.plot(tclip, prediction[:,s,oidx], '-x', label=f"OTU {oidx} pred")
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Gradient log x")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(mapfitpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()
            
    #* compute rmse for each subject and taxon over time
    sle = (grad_data - prediction)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(mapfitpath / "rmse_gradfits.npy", errors)


# TODO: check if take rmse of median or median of rmse samples???***
# TODO: feel like it should be the latter, but mdsine2 might do the former? -- different when integrating?
def plot_gradient_fits_posterior(data, model, res, postfitpath, device):
    # data = {'times': times.to(device), 'gradient': y.to(device), 'abundance': x.to(device)}
    tclip = data['times']
    grad_data = data['gradient']
    x_data = data['abundance']
    num_time, num_subj, num_taxa = x_data.shape
    taxonomy = res['taxonomy']
    prediction = res['prediction']
    ntime, nsubj, ntaxa = prediction.shape

    #* transform back data
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']

    data_np = data.copy()
    data = {}
    for key in ['times', 'gradient', 'abundance', 'log_abundance']:
        data[key] = torch.from_numpy(data_np[key]).to(dtype=torch.float, device=device)

    #* take model samples
    n_samples = 1000
    pred_samples = np.zeros((n_samples, ntime, nsubj, ntaxa))
    model.train()
    for i in range(n_samples):
        _, pred = model(data)
        pred_samples[i,:] = pred.cpu().detach().clone().numpy() #*data_np['y_rescale'] + data_np['y_shift']
        if i % 100 == 0:
            print(f"{i} of {n_samples}")

    #* plot fits
    med = np.percentile(pred_samples, q=50, axis=0)
    low = np.percentile(pred_samples, q=5, axis=0)
    high = np.percentile(pred_samples, q=95, axis=0)

    for s in range(1): #num_subj):
        for oidx in range(num_taxa):
            fig, ax = plt.subplots()
            ax.plot(tclip, med[:,s,oidx], '-o', label=f"OTU {oidx} model fit", color='tab:blue')
            ax.fill_between(tclip, y1=low[:,s,oidx], y2=high[:,s,oidx], alpha=0.2, color='tab:blue')
            ax.plot(tclip, grad_data[:,s,oidx], '--x', label=f"OTU {oidx} data", color='black')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Gradient log x")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(postfitpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()

    #* compute rmse for each subject and taxon over time
    sle = (grad_data - med)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(postfitpath / "rmse_gradfits.npy", errors)


#! -------------------
#* model specific methods..
def post_process_results_grad_match(modelpath, ground_truth=None):
    device = torch.device("cpu")

    res = pickle_load(modelpath / "results.pkl")
    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")
    
    #* traces
    print("PLOTTING TRACES...")
    tracepath = modelpath / "traces"
    tracepath.mkdir(exist_ok=True, parents=True)
    plot_all_traces(res, tracepath, ground_truth)

    #* gradient predictions
    print("PLOTTING GRADIENT FITS...")
    mapfitpath = modelpath / "map_gradient_fits"
    mapfitpath.mkdir(exist_ok=True, parents=True)
    postfitpath = modelpath / "post_gradient_fits"
    postfitpath.mkdir(exist_ok=True, parents=True)
    plot_gradient_fits_MAP(data, res, mapfitpath) # TODO: spit out dataframe of rmse's of fits
    plot_gradient_fits_posterior(data, model, res, postfitpath, device)  # TODO: spit out dataframe of rmse's of fits

    print("*DONE POST PROCESSING*")


#! forward sims:
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_dense_times(times, dt):
    lintimes = np.arange(times[0], times[-1], dt)
    dense_times = np.sort(np.unique(np.concatenate([times, lintimes]))) #.round(decimals=2)))
    original_inds = np.where(np.in1d(dense_times, times))[0] # TODO: make more robust

    if len(original_inds) != len(times):
        raise ValueError("issue getting dense times...")
    return dense_times, original_inds


def forward_sim_glv(params, log_ic, times, mask_times, dt=0.01):
    # TODO: ideally this isn't separate from model...
    log_beta_self = params['log_beta_self']
    log_alpha_default = params['log_alpha_default']
    beta_matrix = params['beta_matrix']

    num_subj, num_taxa = log_ic.shape
    #* forward sim dynamics
    alpha = np.exp(log_alpha_default)
    nonself_mask = 1.0 - np.diag(np.ones(num_taxa))
    beta = beta_matrix*nonself_mask - np.exp(log_beta_self)
    
    dense_times, original_inds = get_dense_times(times, dt)

    num_time = len(dense_times)

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,:] = log_ic

    # forward integrate
    for i in range(1, num_time):
        dt = dense_times[i] - dense_times[i-1]
        time = dense_times[i]

        #* mask input
        input = np.exp(xlog[i-1,:,:]) #* set this to zero when taxon is masked; shape SxO
        for oidx in range(num_taxa):
            if time < mask_times[oidx]:
                input[:,oidx] = 0
        dyn = alpha + input @ beta.T  # NOTE: interactions transposed since putting x in front; consistent with interaction rows = i, columns = j ***elaborate...
        
        #* mask dynamics
        # dyn shape SxO
        for oidx in range(num_taxa):
            if time < mask_times[oidx]:
                dyn[:,oidx] = 0

        xlog[i,:,:] = xlog[i-1,:,:] + dyn*dt

    #* subsample to original times
    xlog = xlog[original_inds,:,:]
    return xlog


def forward_sim_rules(params, log_ic, times, mask_times, dt=0.01):
    det_select = params['det_sample']
    rule_select = params['rule_sample']
    min_range = params['min_range']
    mrange = params['mrange']
    split_props = params['split_proportions']
    split_temp = params['split_temp']
    beta = np.exp(params['log_beta'])
    alpha_default = np.exp(params['log_alpha_default'])
    alpha_rules = params['alpha_rules']

    num_subj, num_taxa = log_ic.shape
    dense_times, original_inds = get_dense_times(times, dt)
    num_time = len(dense_times)

    # initial conditions
    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,:] = log_ic

    # forward integrate
    for i in range(1, num_time):
        dt = dense_times[i] - dense_times[i-1]
        time = dense_times[i]
        x = np.exp(xlog[i-1,:,:])
        splits = sigmoid((((x-min_range)/mrange)[:,None,None,:] - split_props[None,:,:,:])/split_temp)
        # splits shape:  S x O[output] x R x O[input]
        detectors = splits*(time >= mask_times[None,None,None,:]) #* mask input from masked taxa
        rules = np.prod((1.0-det_select*(1.0-detectors)), axis=-1)
        res = np.sum(alpha_rules*rule_select*rules, axis=-1) + alpha_default

        #* add self-interactions
        res = (res - beta*x)*(time >= mask_times[None,:]) #* mask input
        xlog[i,:,:] = xlog[i-1,:,:] + res*dt
    #* subsample to original times
    xlog = xlog[original_inds,:,:]
    return xlog


def get_glv_gmatch_holdout_error(modelpath, xlog_ho, xgrad_ho, xlog_train):
    get_glv_fwdsim_error(modelpath, xlog_ho, name="test")
    get_glv_fwdsim_error(modelpath, xlog_train, name="train")
    get_glv_grad_holdout_error(modelpath, xgrad_ho, xlog_ho)


def get_grad_pred_glv(params, xlogdata, times, mask_times):
    log_beta_self = params['log_beta_self']
    log_alpha_default = params['log_alpha_default']
    beta_matrix = params['beta_matrix']

    _, _, num_taxa = xlogdata.shape
    
    #* forward sim dynamics
    alpha = np.exp(log_alpha_default)
    nonself_mask = 1.0 - np.diag(np.ones(num_taxa))
    beta = beta_matrix*nonself_mask - np.exp(log_beta_self)

    x = np.exp(xlogdata)
    #* mask input
    x = x*(times[:,None,None] >= mask_times[None,None,:])
    res = alpha + (x @ beta.T)
    #* mask output
    res = res*(times[:,None,None] >= mask_times[None,None,:])
    return res


def get_glv_grad_holdout_error(modelpath, xgrad_ho, xlogdata):
    device = torch.device("cpu")

    #* load model
    res = pickle_load(modelpath / "results.pkl")
    taxonomy = res['taxonomy']

    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")

    outpath = modelpath / "grad_ho_cv"
    outpath.mkdir(exist_ok=True, parents=True)

    times = data['times']
    mask_times = data['mask_times']
    ntime, nsubj, ntaxa = xgrad_ho.shape
    #* forward simulate model taking initial condition from holdout data
    # ic = xlog_ho[0,:] 
    #! get intial condition after mask
    #* get first non-nan value
    ic = np.zeros((nsubj, ntaxa))
    # for s in range(nsubj):
    #     for oidx in range(ntaxa):
    #         temp = xlog_ho[:,s,oidx]
    #         ic[s,oidx] = temp[np.isfinite(temp)][0]
    first_idx = np.zeros(ntaxa, dtype=int)
    for oidx in range(ntaxa):
        first_idx[oidx] = np.where(times > mask_times[oidx])[0][0]
    
    for oidx in range(ntaxa):
        ic[:,oidx] = xgrad_ho[first_idx[oidx],:,oidx]
    #TODO: don't need grad holdout; but can calc thie error too...

    # TODO: take multiple posterior samples (integrate these in parallel!! -- this can be gpu accelerated too)
    
    #* transform back data
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']

    data_np = data.copy()
    data = {}
    for key in ['times', 'gradient', 'abundance', 'log_abundance']:
        data[key] = torch.from_numpy(data_np[key]).to(dtype=torch.float, device=device)

    #* take model samples
    n_samples = 10
    params = []
    model.train()
    for i in range(n_samples):
        _, pred = model(data)
        prms = model.get_params()
        params.append(prms)
  
    #* sim grad
    # TODO: parallelize over samples
    # TODO: consolodate functions...
    pred_samples = np.zeros((n_samples, ntime, nsubj, ntaxa))
    for i in range(n_samples):
        prms = params[i]
        gradpred = get_grad_pred_glv(prms, xlogdata, times, mask_times)
        #fwsim = forward_sim_rules(prms, ic, times, mask_times)
        pred_samples[i,:] = gradpred


    #* plot fits and calculate error
    med = np.percentile(pred_samples, q=50, axis=0)
    low = np.percentile(pred_samples, q=5, axis=0)
    high = np.percentile(pred_samples, q=95, axis=0)

    for s in range(1): #num_subj):
        for oidx in range(ntaxa):
            fig, ax = plt.subplots()
            ax.plot(times, med[:,s,oidx], '-o', label=f"OTU {oidx} model fit", color='tab:blue')
            ax.fill_between(times, y1=low[:,s,oidx], y2=high[:,s,oidx], alpha=0.2, color='tab:blue')
            ax.plot(times, xgrad_ho[:,s,oidx], '--x', label=f"OTU {oidx} data", color='black')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Log abundance")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(outpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()

    #* compute rmse for each subject and taxon over time
    sle = (xgrad_ho - med)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(outpath / "rmse_gradfits.npy", errors)
    print("SAVED GRAD HO SIMS")


def get_glv_fwdsim_error(modelpath, xlog_ho, name="test"):
    device = torch.device("cpu")

    #* load model
    res = pickle_load(modelpath / "results.pkl")
    taxonomy = res['taxonomy']

    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")

    outpath = modelpath / f"fwsim_cv_{name}"
    outpath.mkdir(exist_ok=True, parents=True)

     #* spits out numpy file with errors for each taxon
    times = data['times']
    mask_times = data['mask_times']
    ntime, nsubj, ntaxa = xlog_ho.shape
    #* forward simulate model taking initial condition from holdout data
    # ic = xlog_ho[0,:] 
    #! get intial condition after mask
    #* get first non-nan value
    ic = np.zeros((nsubj, ntaxa))
    # for s in range(nsubj):
    #     for oidx in range(ntaxa):
    #         temp = xlog_ho[:,s,oidx]
    #         ic[s,oidx] = temp[np.isfinite(temp)][0]
    first_idx = np.zeros(ntaxa, dtype=int)
    for oidx in range(ntaxa):
        first_idx[oidx] = np.where(times > mask_times[oidx])[0][0]
    
    for oidx in range(ntaxa):
        ic[:,oidx] = xlog_ho[first_idx[oidx],:,oidx]
    #TODO: don't need grad holdout; but can calc thie error too...

    # TODO: take multiple posterior samples (integrate these in parallel!! -- this can be gpu accelerated too)
    
    #* transform back data
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']

    data_np = data.copy()
    data = {}
    for key in ['times', 'gradient', 'abundance', 'log_abundance']:
        data[key] = torch.from_numpy(data_np[key]).to(dtype=torch.float, device=device)

    #* take model samples
    n_samples = 2
    params = []
    model.train()
    for i in range(n_samples):
        _, pred = model(data)
        prms = model.get_params()
        params.append(prms)

    #* forward sim
    # TODO: parallelize over samples
    pred_samples = np.zeros((n_samples, ntime, nsubj, ntaxa))

    for i in range(n_samples):
        prms = params[i]
        fwsim = forward_sim_glv(prms, ic, times, mask_times)
        pred_samples[i,:] = fwsim

    #* plot fits and calculate error
    med = np.percentile(pred_samples, q=50, axis=0)
    low = np.percentile(pred_samples, q=5, axis=0)
    high = np.percentile(pred_samples, q=95, axis=0)

    for s in range(1): #num_subj):
        for oidx in range(ntaxa):
            fig, ax = plt.subplots()
            ax.plot(times, med[:,s,oidx], '-o', label=f"OTU {oidx} model fit", color='tab:blue')
            ax.fill_between(times, y1=low[:,s,oidx], y2=high[:,s,oidx], alpha=0.2, color='tab:blue')
            ax.plot(times, xlog_ho[:,s,oidx], '--x', label=f"OTU {oidx} data", color='black')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Log abundance")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(outpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()

    #* compute rmse for each subject and taxon over time
    sle = (xlog_ho - med)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(outpath / "rmse_gradfits.npy", errors)
    print("SAVED FORWARD SIMS")


# TODO: most of this function is a copy of the above, modularize and have option to just change the dynamics piece
def get_rule_gmatch_holdout_error(modelpath, xlog_ho, xgrad_ho, xlog_train):
    get_rule_fwdsim_holdout_error(modelpath, xlog_ho, "test") #* modify this to also take in train data
    get_rule_fwdsim_holdout_error(modelpath, xlog_train, "train")
    get_rule_grad_holdout_error(modelpath, xgrad_ho, xlog_ho)


def get_grad_pred_rules(params, xlog, times, mask_times):
    det_select = params['det_sample']
    rule_select = params['rule_sample']
    min_range = params['min_range']
    mrange = params['mrange']
    split_props = params['split_proportions']
    split_temp = params['split_temp']
    beta = np.exp(params['log_beta'])
    alpha_default = np.exp(params['log_alpha_default'])
    alpha_rules = params['alpha_rules']

    x = np.exp(xlog)

    splits = sigmoid((((x-min_range)/mrange)[:,:,None,None,:] - split_props[None,None,:,:,:])/split_temp)
    # splits shape: T x S x O[output] x R x O[input]
    detectors = splits*(times[:,None,None,None,None] >= mask_times[None,None,None,None,:]) #* mask input from masked taxa
    rules = np.prod((1.0-det_select*(1.0-detectors)), axis=-1)
    res = np.sum(alpha_rules*rule_select*rules, axis=-1) + alpha_default

    #* add self-interactions
    res = (res - beta*x)*(times[:,None,None] >= mask_times[None,None,:]) #* mask input
    return res


def get_rule_grad_holdout_error(modelpath, xgrad_ho, xlogdata):
    device = torch.device("cpu")

    #* load model
    res = pickle_load(modelpath / "results.pkl")
    taxonomy = res['taxonomy']

    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")

    outpath = modelpath / "grad_ho_cv"
    outpath.mkdir(exist_ok=True, parents=True)

    times = data['times']
    mask_times = data['mask_times']
    ntime, nsubj, ntaxa = xgrad_ho.shape
    #* forward simulate model taking initial condition from holdout data
    # ic = xlog_ho[0,:] 
    #! get intial condition after mask
    #* get first non-nan value
    ic = np.zeros((nsubj, ntaxa))
    # for s in range(nsubj):
    #     for oidx in range(ntaxa):
    #         temp = xlog_ho[:,s,oidx]
    #         ic[s,oidx] = temp[np.isfinite(temp)][0]
    first_idx = np.zeros(ntaxa, dtype=int)
    for oidx in range(ntaxa):
        first_idx[oidx] = np.where(times > mask_times[oidx])[0][0]
    
    for oidx in range(ntaxa):
        ic[:,oidx] = xgrad_ho[first_idx[oidx],:,oidx]
    #TODO: don't need grad holdout; but can calc thie error too...

    # TODO: take multiple posterior samples (integrate these in parallel!! -- this can be gpu accelerated too)
    
    #* transform back data
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']

    data_np = data.copy()
    data = {}
    for key in ['times', 'gradient', 'abundance', 'log_abundance']:
        data[key] = torch.from_numpy(data_np[key]).to(dtype=torch.float, device=device)

    #* take model samples
    n_samples = 10
    params = []
    model.train()
    for i in range(n_samples):
        _, pred = model(data)
        prms = model.get_params()
        params.append(prms)
  
    #* sim grad
    # TODO: parallelize over samples
    # TODO: consolodate functions...
    pred_samples = np.zeros((n_samples, ntime, nsubj, ntaxa))
    for i in range(n_samples):
        prms = params[i]
        gradpred = get_grad_pred_rules(prms, xlogdata, times, mask_times)
        #fwsim = forward_sim_rules(prms, ic, times, mask_times)
        pred_samples[i,:] = gradpred


    #* plot fits and calculate error
    med = np.percentile(pred_samples, q=50, axis=0)
    low = np.percentile(pred_samples, q=5, axis=0)
    high = np.percentile(pred_samples, q=95, axis=0)

    for s in range(1): #num_subj):
        for oidx in range(ntaxa):
            fig, ax = plt.subplots()
            ax.plot(times, med[:,s,oidx], '-o', label=f"OTU {oidx} model fit", color='tab:blue')
            ax.fill_between(times, y1=low[:,s,oidx], y2=high[:,s,oidx], alpha=0.2, color='tab:blue')
            ax.plot(times, xgrad_ho[:,s,oidx], '--x', label=f"OTU {oidx} data", color='black')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Log abundance")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(outpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()

    #* compute rmse for each subject and taxon over time
    sle = (xgrad_ho - med)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(outpath / "rmse_gradfits.npy", errors)
    print("SAVED GRAD HO SIMS")


def get_rule_fwdsim_holdout_error(modelpath, xlog_ho, name="test"):
    device = torch.device("cpu")

    #* load model
    res = pickle_load(modelpath / "results.pkl")
    taxonomy = res['taxonomy']

    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")

    outpath = modelpath / f"fwsim_cv_{name}"
    outpath.mkdir(exist_ok=True, parents=True)

     #* spits out numpy file with errors for each taxon
    times = data['times']
    mask_times = data['mask_times']
    ntime, nsubj, ntaxa = xlog_ho.shape
    #* forward simulate model taking initial condition from holdout data
    # ic = xlog_ho[0,:] 
    #! get intial condition after mask
    #* get first non-nan value
    ic = np.zeros((nsubj, ntaxa))
    # for s in range(nsubj):
    #     for oidx in range(ntaxa):
    #         temp = xlog_ho[:,s,oidx]
    #         ic[s,oidx] = temp[np.isfinite(temp)][0]
    first_idx = np.zeros(ntaxa, dtype=int)
    for oidx in range(ntaxa):
        first_idx[oidx] = np.where(times > mask_times[oidx])[0][0]
    
    for oidx in range(ntaxa):
        ic[:,oidx] = xlog_ho[first_idx[oidx],:,oidx]
    #TODO: don't need grad holdout; but can calc thie error too...

    # TODO: take multiple posterior samples (integrate these in parallel!! -- this can be gpu accelerated too)
    
    #* transform back data
    # grad_data = grad_data*data['y_rescale'] + data['y_shift']

    data_np = data.copy()
    data = {}
    for key in ['times', 'gradient', 'abundance', 'log_abundance']:
        data[key] = torch.from_numpy(data_np[key]).to(dtype=torch.float, device=device)

    #* take model samples
    n_samples = 10
    params = []
    model.train()
    for i in range(n_samples):
        _, pred = model(data)
        prms = model.get_params()
        params.append(prms)

    #* forward sim
    # TODO: parallelize over samples
    pred_samples = np.zeros((n_samples, ntime, nsubj, ntaxa))

    for i in range(n_samples):
        prms = params[i]
        fwsim = forward_sim_rules(prms, ic, times, mask_times)
        pred_samples[i,:] = fwsim

    #* plot fits and calculate error
    med = np.percentile(pred_samples, q=50, axis=0)
    low = np.percentile(pred_samples, q=5, axis=0)
    high = np.percentile(pred_samples, q=95, axis=0)

    for s in range(1): #num_subj):
        for oidx in range(ntaxa):
            fig, ax = plt.subplots()
            ax.plot(times, med[:,s,oidx], '-o', label=f"OTU {oidx} model fit", color='tab:blue')
            ax.fill_between(times, y1=low[:,s,oidx], y2=high[:,s,oidx], alpha=0.2, color='tab:blue')
            ax.plot(times, xlog_ho[:,s,oidx], '--x', label=f"OTU {oidx} data", color='black')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Log abundance")
            ax.set_title(f"{taxonomy[oidx]} - Subject {s}")
            plt.savefig(outpath / f"grad_{taxonomy[oidx]}_S{s}.png")
            plt.close()

    #* compute rmse for each subject and taxon over time
    sle = (xlog_ho - med)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(outpath / "rmse_gradfits.npy", errors)
    print("SAVED FORWARD SIMS")

