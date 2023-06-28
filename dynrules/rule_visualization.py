import torch 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from dynrules.utils import COLORS
# matplotlib.use('agg')

EFFECTOR_COLOR='gold'
GROWTH_INCREASE_COLOR='lightgreen'
GROWTH_DECREASE_COLOR='salmon'
DATA_COLOR='black'


# TODO: also visualize alpha default and beta ...

class RuleSet:
    def __init__(self, params, taxonomy=None): 
        # todo: want detectors to be clearer...
        # todo... use indices for now
        #* maybe the model should store taxa too
        #* or have overarching class that contains the model and taxa
        self.process_model(params) 

        if taxonomy is None:
            self.taxa = [f"OTU {oidx}" for oidx in range(self.num_taxa)] # TODO: option to input names/datastructure...
        else:
            self.taxa = taxonomy


    #* add access by index
    # todo: also add access by taxon name
    def __getitem__(self, key):
        return self.rules[key]

    def effects(self, key):
        return self.rule_effects[key]

    def process_model(self, params):

        rule_selectors = params['rules']
        det_selectors = params['detectors']
        split_proportions = params['split_proportions']
        alphas = params['alpha_rules']
        full_range = params['mrange']
        min_range = params['min_range']
        
        self.num_taxa, nrules, ndet = det_selectors.shape
        nsets = self.num_taxa 

        self.rules = dict() #* taxa->rule_dict
        self.rule_effects = dict() #* taxa->alpha (change in growth)
        for taxa in range(self.num_taxa):
            trules = dict() #* rule_dict: rule->det_dict
            alpha_rules = dict()
            ruleID = 0
            for rule in range(nrules):
                rule_select = rule_selectors[taxa, rule]
                if rule_select > 0.5:
                    det_dict = dict()
                    ruleID += 1 #* number rules from 1
                    for det in range(ndet):
                        cov_select = det_selectors[taxa, rule, det]
                        if cov_select > 0.5:
                            #* if selecting [use det as the id]
                            threshold = split_proportions[taxa, rule, det]
                            # NOTE: we rescale x, not splits...
                            det_dict[det] = threshold
                    trules[ruleID] = det_dict
                    alpha_rules[ruleID] = alphas[taxa,rule]
            self.rule_effects[taxa] = alpha_rules
            self.rules[taxa] = trules
        #* want dict [taxa -> rules] 
        #* rules[ind] -> dets={taxa; thresholds...}
        #* det[ind==taxon_idx] -> threshold

    def print_to_file(self, savepath):
        for oidx in range(self.num_taxa):
            filename = savepath / f"rules_{self.taxa[oidx]}.txt"
            with open(filename, "w") as f:
                f.write(f"Rules for {self.taxa[oidx]}: \n")
                for ridx in self.rules[oidx].keys():
                    f.write(f"Rule {ridx}: \n")
                    dets = self.rules[oidx][ridx]
                    alpha = self.effects(oidx)[ridx]
                    effector_taxa = list(dets.keys())
                    for effector in effector_taxa:
                        thres = dets[effector]
                        f.write(f"IF {self.taxa[effector]} > {thres}: \n")
                    f.write(f"growth rate shifts by {alpha} \n\n")

        # print(f"Wrote rules to file: {filename}")
    #* methods for more intuitive access
    # def get_rule_set(self, taxon):
    #     pass 

    # def get_detector_set(self, rule):
    #     pass 

    # def get_taxon_threshold(self, detector):
    #     pass 

    #* method for visualization? output rule set to pdf


class RuleVisualizer:
    # TODO: for grad matching; either rescale model params or supply with 
    # TODO: original data we fit with...
    def __init__(self, params, times, x_data_train, x_data_plot, prediction, taxonomy=None):
        #* evaluate model
        self.params = params
        self.rules = RuleSet(params)
        # TODO: add to rule set class?
        self.base_growth_rates = np.exp(params['log_alpha_default']) # TODO: save exp instead??
        self.self_interaction = np.exp(params['log_beta'])
        self.times = times #data['times']
        self.train_data = x_data_train
        self.x_data = x_data_plot #data['abundance']
        self.prediction = prediction
        self.min_range = params['min_range']
        self.mrange = params['mrange']
        if taxonomy is None:
            self.taxa = [f"OTU {oidx}" for oidx in range(self.rules.num_taxa)] # TODO: option to input names/datastructure...
        else:
            self.taxa = taxonomy
            
    def plot_rules(self, taxon_idx, rule_idx, subj):
        # TODO: option to also plot data [need not...]
        subjs = [subj]

        max_n_det = 0 #* number detectors in rule?... [could be 0?]
        # for rule_idx in rule_ids:
        #* usage [taxa_idx][rule_idx][det_idx=taxon] = threshold
        dets = self.rules[taxon_idx][rule_idx] #* is a dict of det->threshold
        max_n_det = max(max_n_det, len(dets))

        #* effector taxa from detector keys
        effector_taxa = list(dets.keys())
        target_taxon = taxon_idx

        alpha = self.rules.effects(taxon_idx)[rule_idx]

        #* just plotting sample 0 for now
        # todo: either plot all samples or [better], add as arg input...
        # s = 0
        fig, ax = plt.subplots(nrows=max_n_det+1, ncols=len(subjs), constrained_layout=True, sharex=True)
        ax = np.reshape(ax, (max_n_det+1, len(subjs)))
        for s_ind, s in enumerate(subjs): #range(self.num_samples):
            #* map detectors to corresponding    
            final_shade = None
            for i, effector_taxon in enumerate(effector_taxa):
                # TODO:
                # thres = self.mu_l[effector_taxon]*dets[effector_taxon] #! NOTE: rescale x, not split***
                # print("THRESHOLD = ", thres)
                x = self.prediction[:,s,effector_taxon] #*SHAPE T, S, O
                x_data = self.x_data[:,s,effector_taxon] #self.mu_l[effector_taxon]*self.x_data_lst[effector_taxon,s,:]
                data_t = self.times # TODO...

                if x_data is not None:
                    min_x = min(np.amin(x), np.amin(x_data)) #* range for plotting
                    max_x = max(np.amax(x), np.amax(x_data))
                else:
                    min_x = np.amin(x)
                    max_x = np.amax(x)
                # min_x = min_x - 0.1*np.abs(min_x)
                # max_x = max_x + 0.1*np.abs(max_x)

                if (i==0) and (s_ind == 0): #* condition first for legend
                    ax[i,s_ind].plot(self.times, x, linewidth=2, label="Model fit")
                    if x_data is not None:
                        ax[i,s_ind].plot(data_t, x_data, 'x', color=DATA_COLOR, label="Data") 
                    ax[i,s_ind].legend()
                else:
                    ax[i,s_ind].plot(self.times, x, linewidth=2)
                    if x_data is not None:
                        ax[i,s_ind].plot(data_t, x_data, 'x', color=DATA_COLOR) 

                x_train = self.train_data[:,s,effector_taxon]
                xstandard = (x_train-self.min_range[effector_taxon])/self.mrange[effector_taxon] # NOTE: x is in log space...
                thres = dets[effector_taxon]
                split_point = thres*self.mrange[effector_taxon] + self.min_range[effector_taxon]
                cond = (xstandard > thres) # TODO: use x-stnadarized instead; NOTE: x is only time dim
                if final_shade is None:
                    final_shade = cond 
                else: 
                    final_shade = final_shade & cond
                ax[i,s_ind].fill_between(self.times[:-1], min_x, max_x, where=cond,
                    facecolor=EFFECTOR_COLOR, alpha=0.5) #, transform=trans)
                ax[i,s_ind].axhline(y=split_point, xmin=self.times[0], xmax=self.times[-1], linestyle='--', color='k')
                ax[i,s_ind].text(1.01, .99, 'threshold: {:.2f}'.format(split_point), ha='left', va='top', transform=ax[i,s_ind].transAxes)
                ax[i,s_ind].text(1.01, .90, 'proportion: {:.2f}'.format(thres), ha='left', va='top', transform=ax[i,s_ind].transAxes)
                ax[i,0].set_title("{}".format(self.taxa[effector_taxon])) # TODO: add taxononmy

            x = self.prediction[:,s,target_taxon]
            x_data = self.x_data[:,s,target_taxon] #self.mu_l[target_taxon]*self.x_data_lst[target_taxon,s,:]
            data_t = self.times
            alpha_default = self.base_growth_rates[target_taxon]
            beta = self.self_interaction[target_taxon]
            
            if x_data is not None:
                min_x = min(np.amin(x), np.amin(x_data)) #* range for plotting
                max_x = max(np.amax(x), np.amax(x_data))
            else:
                min_x = np.amin(x)
                max_x = np.amax(x)
            # min_x = min_x - 0.1*np.abs(min_x)
            # max_x = max_x + 0.1*np.abs(max_x)
            ax[max_n_det,s_ind].plot(self.times, x, linewidth=2)
            if x_data is not None:
                ax[max_n_det,s_ind].plot(data_t, x_data, 'x', color=DATA_COLOR) 
            # if s == 0:
            ax[max_n_det,0].set_title("{}".format(self.taxa[target_taxon]))
            # todo: can color based on up or down?
            # todo: will need to store alpha rules too in rule set
            if alpha >=0:
                color = GROWTH_INCREASE_COLOR
            else:
                color = GROWTH_DECREASE_COLOR
            print("alpha = ", alpha)
            ax[max_n_det,s_ind].fill_between(self.times[:-1], min_x, max_x, where=final_shade,
                    facecolor=color, alpha=0.5) #, transform=trans)
            ax[max_n_det,s_ind].text(1.01, .99, 'base growth rate: {:.2f}'.format(alpha_default), ha='left', va='top', transform=ax[max_n_det,s_ind].transAxes)
            ax[max_n_det,s_ind].text(1.01, .90, 'self interaction: {:.2f}'.format(beta), ha='left', va='top', transform=ax[max_n_det,s_ind].transAxes)
            ax[max_n_det,s_ind].text(1.01, .80, 'alpha: {:.2f}'.format(alpha), ha='left', va='top', transform=ax[max_n_det,s_ind].transAxes)
            fig.supxlabel("Time")
            fig.supylabel("Log abundance")
            plt.suptitle("Rule {}".format(rule_idx))
        return ax

    def plot_all_rules(self, taxon, subj):
        num_taxon_rules = len(self.rules[taxon])
        axs = []
        for rule_id in range(1, num_taxon_rules+1):
            ax = self.plot_rules(taxon, rule_id, subj)
            axs.append(ax)
        return axs

    def plot_all_rules_save(self, taxon, subj, savedir):
        num_taxon_rules = len(self.rules[taxon])
        axs = []
        for rule_id in range(1, num_taxon_rules+1):
            ax = self.plot_rules(taxon, rule_id, subj)
            plt.savefig(savedir / f"OTU{taxon}_R{rule_id}.png")
            axs.append(ax)
        return axs
        #         #* create directory for each taxa's rules
        # outdir = os.path.join(savedir, "{}_rules".format(self.taxa[taxon]))
        # create_dir(outdir)

        # num_taxon_rules = len(self.rules[taxon])

        # for rule_id in range(1, num_taxon_rules+1):
        #     ax = self.plot_rules(taxon, rule_id)
        #     plt.savefig(os.path.join(outdir, "{}_Rule-{}.pdf".format(self.taxa[taxon], rule_id)))
        #     plt.close() 
