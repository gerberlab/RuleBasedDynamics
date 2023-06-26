import pandas as pd
import numpy as np
from pathlib import Path


class Dataset:
    def __init__(self, counts, mass, meta):
        self.counts = pd.read_csv(counts, sep="\t", index_col=0) 
        self.mass = pd.read_csv(mass, sep="\t")
        self.meta = pd.read_csv(meta, sep="\t", index_col=0) # NOTE: there's an extra column in datafile...

    def get_sampleID(self, t, s):
        # SHOULD BE UNIQUE
        return self.meta.loc[(self.meta.loc[:,"time"] == t) & (self.meta.loc[:,"subjectID"] == s), "sampleID"].values[0]

    def get_count_mass_data(self, notus=13):
        subjs = self.meta.loc[:,"subjectID"].unique()
        times = np.sort(self.meta.loc[:,"time"].unique())
        # notus = 13 #14
        ntime = len(times)
        nsubj = len(subjs)
        nrep = 3 #* qpcr replicates

        # take top N (averaged over all time points and subjects) {doing roughly to compare with mdsine paper, if agree then keep same}
        ra = self.counts.sum(axis=1)/(self.counts.values.sum())
        #* top 14 agree with mdsine, can use top 13 if excluding hiranonis
        index = ra.sort_values(ascending=False)[:notus].index
        topcounts = self.counts.loc[index,:]

        ycounts = np.zeros((ntime, nsubj, notus))
        wmass = np.zeros((nrep, ntime, nsubj))

        for i,t in enumerate(times):
            for j,s in enumerate(subjs):
                for oidx, taxon in enumerate(index):
                    sample = self.get_sampleID(t,s)
                    ycounts[i,j,oidx] = topcounts.loc[taxon,str(sample)] # TODO: convert all sample values to str
                wmass[:,i,j] = self.mass.iloc[sample-1,:].values # TODO: add sample id to masses, make cleaner

        return times, ycounts, wmass, index
    

def get_cdiff_processed_logdata(basepath="./CDIFF_DATA/"):
    #* from criteria established below
    datapath = Path(basepath)
    dataset = Dataset(counts=datapath/"counts.txt",
                    mass=datapath/"biomass.txt",
                    meta=datapath/"meta.tsv")
                    
    times, ycounts, wmass, index = dataset.get_count_mass_data()

    rescale = 1e-9
    wmass_geomean = np.exp(np.mean(np.log(wmass), axis=0))*rescale
    xdata = (ycounts/(ycounts.sum(axis=2, keepdims=True)))*wmass_geomean[:,:,None]
    ntime, nsubj, notu = xdata.shape

    xlogdata = np.log(xdata + 1e-8)

    cdiff = index.get_loc('Clostridium-difficile')
    time_mask = np.zeros((notu,))
    time_mask[cdiff] = 28.9

    #* remove first few time points
    nremove = 3
    times = times[nremove:]
    xlogdata = xlogdata[nremove:,:,:]

    return times, xlogdata, time_mask, index
