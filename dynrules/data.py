import pandas as pd
import numpy as np
from pathlib import Path


class Dataset:
    def __init__(self, counts, qpcr, metadata, taxa_metadata, taxonomy):
        self.counts = pd.read_csv(counts, sep="\t", index_col=0) 
        self.qpcr = pd.read_csv(qpcr, sep="\t", index_col=0)
        self.metadata = pd.read_csv(metadata, sep="\t")
        self.taxmeta = pd.read_csv(taxa_metadata, sep="\t", index_col=0)
        self.taxonomy = pd.read_csv(taxonomy, sep="\t", index_col=0)
    
    def get_sampleID(self, t, s):
        # SHOULD BE UNIQUE
        return self.metadata.loc[(self.metadata.loc[:,"time"] == t) & (self.metadata.loc[:,"subjectID"] == s), "sampleID"].values[0]

    def top_abundance_filter(self, notus):
        """
        filter by total relative abundance, keep 'notus' otus
        this is done 'in place' and modifies internal data
        """
        # take top N (averaged over all time points and subjects)
        ra = self.counts.sum(axis=1)/(self.counts.values.sum())
        index = ra.sort_values(ascending=False)[:notus].index
        
        # use index to subset counts, taxonomy, and taxmeta
        self.counts = self.counts.loc[index,:]
        self.taxonomy = self.taxonomy.loc[index,:]
        self.taxmeta = self.taxmeta.loc[index,:]
         
    # TODO: add consistency filter
    # TODO: add option to remove times here too... currently doing outside of this class
    
    def get_count_mass_data(self):
        """
        get count and mass matrix data
        """
        
        subjs = self.metadata.loc[:,"subjectID"].unique()
        times = np.sort(self.metadata.loc[:,"time"].unique())
        ntime = len(times)
        nsubj = len(subjs)
        notus = self.taxonomy.shape[0]
        taxa = list(self.taxonomy.index)
        nrep = self.qpcr.shape[1]

        ycounts = np.zeros((ntime, nsubj, notus))
        wmass = np.zeros((nrep, ntime, nsubj))

        for i,t in enumerate(times):
            for j,s in enumerate(subjs):
                sample = self.get_sampleID(t,s)
                wmass[:,i,j] = self.qpcr.loc[sample,:].values
                for oidx, taxon in enumerate(taxa):
                    ycounts[i,j,oidx] = self.counts.loc[taxon,str(sample)]
                # TODO: make all strings, not sure why columns strings, but qpcr index ints?
        return times, ycounts, wmass #, taxa_type
                            
    def get_data_for_inference(self, rescale=1):
        """
        return data matrices to use for model inference        
        returns times, abundance data, time mask, and type info
        
        """
        EPS = 1e-8
        times, ycounts, wmass = self.get_count_mass_data()
        
        wmass_geomean = np.exp(np.mean(np.log(wmass), axis=0))*rescale
        xdata = (ycounts/(ycounts.sum(axis=2, keepdims=True)))*wmass_geomean[:,:,None]
        ntime, nsubj, notu = xdata.shape

        xlogdata = np.log(xdata + EPS)
        
        time_mask = self.taxmeta.time.values
        type_names = self.taxmeta.type.values 
        # 0 == bacteria; 1 == phage
        typeinfo = -1*np.ones(len(type_names), dtype=int)
        typeinfo[type_names == 'bacteria'] = 0
        typeinfo[type_names == 'phage'] = 1
        
        if (typeinfo == -1).any():
            raise ValueError("invalid taxa type")
        
        return times, xlogdata, time_mask, typeinfo
    

def get_cdiff_processed_logdata(basepath="./CDIFF_DATA/processed_data", notus=13, n_time_remove=3, rescale = 1e-9):
    cdiff_path = Path(basepath)
    dataset = Dataset(counts=cdiff_path / "reads.tsv",
                 qpcr=cdiff_path / "qpcr.tsv",
                 metadata=cdiff_path / "meta.tsv",
                 taxa_metadata=cdiff_path / "taxa_meta.tsv",
                 taxonomy=cdiff_path / "taxonomy.tsv")
                    
    #* for c diff data top 14 agree with mdsine, can use top 13 if excluding hiranonis
    dataset.top_abundance_filter(notus)
    times, xlogdata, time_mask, type_info = dataset.get_data_for_inference(rescale=1e-9)
    
    #* remove first few time points
    times = times[n_time_remove:]
    xlogdata = xlogdata[n_time_remove:,:,:]

    return times, xlogdata, time_mask, type_info, dataset
