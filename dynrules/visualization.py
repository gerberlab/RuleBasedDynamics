import matplotlib.pyplot as plt
import numpy as np

def plot_taxa(t, x, subj, taxa=None):
    ntime, nsubj, notus = x.shape
    if taxa is None:
        taxa = [f"OTU{i+1}" for i in range(notus)]
    fig, ax = plt.subplots()
    for oidx in range(notus):
        ax.plot(t, x[:,subj,oidx], '-x', label=taxa[oidx])
    ax.legend()
    ax.set_title(f"Subject {subj}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    return fig, ax
