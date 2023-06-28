from dynrules.utils import pickle_load, integrate_log_gradient, get_discrete_gradient, get_interaction_matrix_MAP
from pathlib import Path
from dynrules.rule_visualization import RuleVisualizer, RuleSet
import matplotlib.pyplot as plt 
from dynrules.data import get_cdiff_processed_logdata
import torch 
import numpy as np 
import seaborn as sns
import pandas as pd 


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


def plot_trace(ptrace, varname, savedir):
    trace = extract_var_trace(ptrace, varname)

    if len(trace.shape) == 1:
        fig, ax = plt.subplots()
        ax.plot(trace)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(varname)
        ax.set_title(f"{varname} trace")
        plt.savefig(savedir / f"{varname}_trace.png", bbox_inches="tight")
        plt.close()
    elif len(trace.shape) == 2:
        nepoch, ntaxa = trace.shape
        fig, ax = plt.subplots()
        for i in range(ntaxa):
            ax.plot(trace[:,i], label=f'OTU {i}')
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
            ax.legend(bbox_to_anchor=(1.01,1.01))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(varname)
            ax.set_title(f"{varname} trace (OTU {i})")
            plt.savefig(savedir / f"{varname}_trace_OTU{i}.png", bbox_inches="tight")
            plt.close()
    else:
        print("not implemented")


def plot_all_traces(res, savedir):
    ptrace = res['param_trace']
    for vname in ptrace[0].keys():
        print(f"plotting {vname} trace...")
        plot_trace(ptrace, vname, savedir)


def plot_gradient_fits_MAP(data, res, mapfitpath):
    # data = {'times': times.to(device), 'gradient': y.to(device), 'abundance': x.to(device)}
    tclip = data['times']
    grad_data = data['gradient']
    x_data = data['abundance']
    num_time, num_subj, num_taxa = x_data.shape
    taxonomy = res['taxonomy']
    prediction = res['prediction']

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

    #* compute rmse for each subject and taxon over time
    sle = (grad_data - prediction)**2
    msle = np.nanmean(sle, axis=0)
    errors = np.sqrt(msle)
    np.save(mapfitpath / "rmse_gradfits.npy", errors)


def plot_gradient_fits_posterior(data, model, postfitpath):
    pass


def output_rules(res, rulepath):
    final_params = res['params']
    taxonomy = res['taxonomy']
    rset = RuleSet(final_params, taxonomy)
    rset.print_to_file(rulepath)


def post_process_results(modelpath):
    res = pickle_load(modelpath / "results.pkl")
    data = pickle_load(modelpath / "data.pkl")
    model = torch.load(modelpath / "model.pt")
    
    #* traces
    print("PLOTTING TRACES...")
    tracepath = modelpath / "traces"
    tracepath.mkdir(exist_ok=True, parents=True)
    plot_all_traces(res, tracepath)

    #* gradient predictions
    print("PLOTTING GRADIENT FITS...")
    mapfitpath = modelpath / "map_gradient_fits"
    mapfitpath.mkdir(exist_ok=True, parents=True)
    postfitpath = modelpath / "post_gradient_fits"
    postfitpath.mkdir(exist_ok=True, parents=True)
    plot_gradient_fits_MAP(data, res, mapfitpath) # TODO: spit out dataframe of rmse's of fits
    plot_gradient_fits_posterior(data, model, postfitpath)  # TODO: spit out dataframe of rmse's of fits

    #* rules
    print("OUTPUTTING RULES...")
    rulepath = modelpath / "rules"
    rulepath.mkdir(exist_ok=True, parents=True)
    output_rules(res, rulepath)

    #* integrated predictions
    # TODO: take multiple samples from model and forward sim...
    # need to make sure rescaling is correct and initial conditions, masking done right too...
    print("***ALL DONE***")


# if __name__ == "__main__":
#     basepath = Path("./experiments/gmatch_original/redo_test_1k_NEW")
#     post_process_results(basepath)
