import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import numpy as np
from dynrules.utils import KL_bernoulli, KL_gaussian


def normal_logpdf(x, mu, var):
    return torch.distributions.Normal(loc=mu, scale=torch.sqrt(var)).log_prob(x)


class AnnealedParameter(nn.Module):
    def __init__(self, start_temp, end_temp, method="linear"):
        super().__init__()
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.value = None
        self.method = method

    def anneal(self, epoch, num_epochs):
        if self.method == "linear":
            self.linear_anneal(epoch, num_epochs)
        elif (self.method == "hyp") or (self.method == "hyperbolic"):
            self.hyp_anneal(epoch, num_epochs)
        elif self.method == "constant":
            self.value = self.end_temp
        else:
            raise ValueError('incorrect method type')
    
    def linear_anneal(self, epoch, num_epochs):
        percent = epoch/num_epochs
        if percent < 0.1:
            self.value = self.start_temp
            return
        if (percent >= 0.1) & (percent <= 0.9):
            interp = (percent-0.1)/0.8
            self.value = (1.0-interp)*self.start_temp + interp*self.end_temp
            return
        self.value = self.end_temp

    def hyp_anneal(self, epoch, num_epochs):
        percent = epoch/num_epochs
        if percent < 0.1:
            self.value = self.start_temp
            return
        if (percent >= 0.1) & (percent <= 0.9):
            interp = (percent-0.1)/0.8
            hyp = (1.0-interp)*(1.0/self.start_temp) + interp*(1.0/self.end_temp)
            self.value = 1/hyp
            return
        self.value = self.end_temp 
        

class BernoulliVariable(AnnealedParameter):
    def __init__(self, shape, prior_prob, device, start_temp=0.5, end_temp=0.01):
        super().__init__(start_temp, end_temp)

        self.shape = shape
        self.device = device 

        self.selectors = None

        self.p_selectors = torch.tensor(prior_prob, requires_grad=False, dtype=torch.float, device=self.device)
        self.q_params = nn.Parameter(torch.normal(0,1,size=shape), requires_grad=True)

    def manual_set_params(self, value):
        self.q_params.data = value
        
    def init_from_grad_match(self, gmatch_data):
        self.q_params.data = torch.from_numpy(gmatch_data).float()

    def default_init(self):
        self.q_params.data = torch.normal(0,1,size=self.shape)

    def forward(self):
        p_log = F.logsigmoid(torch.stack((self.q_params, -self.q_params)))

        #* sample variables        
        # self.selectors = gumbel_softmax(p_log, hard=True, dim=0, tau=self.value)[0]
        if self.training:
            self.selectors = gumbel_softmax(p_log, hard=True, dim=0, tau=self.value)[0]
        else:
            self.selectors = torch.argmin(p_log, dim=0)

        self.q_prob = torch.sigmoid(self.q_params)
        KL = KL_bernoulli(self.q_params, self.p_selectors)
        return self.selectors, KL
    

#* interface for dynamics classes
class DynamicsBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def init_from_data(self, data):
        pass 

    def set_temps(self, epoch, num_epochs):
        pass 

    def dynamics(self, logx, params):
        pass 

    def forward(self):
        pass 


class RuleDynamics(DynamicsBase):
    def __init__(self, num_taxa, num_subj, num_rules, num_time, rule_prior_prob, detector_prior_prob,
                 split_start_temp, split_end_temp, max_range,
                 device, concrete_start_temp=0.5, concrete_end_temp=0.001):
        super().__init__()
        self.num_taxa = num_taxa 
        self.num_rules = num_rules 
        self.num_covariates = num_taxa
        self.num_detectors = self.num_covariates
        self.num_time = num_time
        self.device = device
        self.input_dim = (num_taxa*num_subj + 1)*(num_time) # for VAE's

        self.max_range = torch.from_numpy(max_range).to(device)
        self.split_temperature = AnnealedParameter(split_start_temp, split_end_temp) #, method="constant") #! @testing ;compare w/linear annealing

        #* stochastic parameters
        self.det_select = BernoulliVariable((self.num_taxa, self.num_rules, self.num_detectors), detector_prior_prob,
                                device, concrete_start_temp, concrete_end_temp)
        self.rule_select = BernoulliVariable((self.num_taxa, self.num_rules), rule_prior_prob, device, concrete_start_temp,
                                concrete_end_temp)

        #* deterministic parameters
        self.splits = nn.Parameter(torch.Tensor(self.num_taxa,self.num_rules,self.num_covariates), requires_grad=True)
        self.log_alpha_default = nn.Parameter(torch.Tensor(self.num_taxa,), requires_grad=True)
        self.log_beta = nn.Parameter(torch.Tensor(self.num_taxa,), requires_grad=True)
        self.alpha_rules = nn.Parameter(torch.Tensor(self.num_taxa,self.num_rules), requires_grad=True)

        #* mask to remove 'self interactions'
        self.nonself_mask = torch.reshape((1.0 - torch.diag(torch.ones((self.num_taxa,), device=self.device, requires_grad=False)) ), (self.num_taxa, 1, self.num_covariates))

    def init_from_data(self, data):
        pass
    
    def init_from_grad_match(self, gmatch):
        self.rule_select.init_from_grad_match(gmatch['rule_prob'])
        self.det_select.init_from_grad_match(gmatch['det_prob'])
        self.splits.data = torch.from_numpy(gmatch['logit_split_proportions']).float()
        self.log_alpha_default.data = torch.from_numpy(gmatch['log_alpha_default']).float()
        self.log_beta.data = torch.from_numpy(gmatch['log_beta']).float()
        self.alpha_rules.data = torch.from_numpy(gmatch['alpha_rules']).float()

    def default_init(self):
        self.rule_select.default_init()
        self.det_select.default_init()
        self.splits.data = torch.normal(-1,0.5, size=(self.num_taxa,self.num_rules,self.num_covariates))
        self.log_alpha_default.data = torch.normal(0,1, size=(self.num_taxa,))
        self.log_beta.data = torch.normal(0,1, size=(self.num_taxa,))
        self.alpha_rules.data = torch.normal(0,1, size=(self.num_taxa,self.num_rules))

    def set_temps(self, epoch, num_epochs):
        self.rule_select.anneal(epoch, num_epochs)
        self.det_select.anneal(epoch, num_epochs)
        self.split_temperature.anneal(epoch, num_epochs)

    @staticmethod
    def dynamics(logx, rule_params, taxa_mask):
        # input x shape is SxO; for single time point
        x = torch.exp(logx)

        rule_select = rule_params['rules'] # KxR
        det_select = rule_params['detectors'] # shape: KxRxD (D=O)
        alpha_rules = rule_params['alpha_rules'] # KxR
        alpha_default = rule_params['alpha_default'] # K
        beta = rule_params['beta'] # K
        split_props = rule_params['split_proportions'] # KxRxO
        max_range = rule_params['max_range']
        split_temp = rule_params['split_temperature']
        
        #* min range and mrange are taxa specific; used to normalize abundance of each taxon...
        #* since we are dealing with actual abundance x; not log space for rule input; don't really need the min range (can just set to 0) 
        xproportion = (x/max_range)[:,None,None,:]  # shape should be SxO -> SxKxRxO;  split shape KxRxO -> SxKxRxO

        sprops = split_props[None,:,:,:]
        splits = torch.sigmoid((xproportion - sprops)/split_temp)
        detectors = splits*taxa_mask[:,None,None,:]
    #! check, is no self implemented here???    
        rules = torch.prod((1.0-det_select*(1.0-detectors)), dim=-1) # resulting shape: SxKxR
        res = torch.sum(alpha_rules*rule_select*rules, dim=-1) + alpha_default  # SxK

        #* add self-interactions
        # ; x here is also not in log space, but actual abundance
        res = (res - beta*x)*taxa_mask
        return res

    def forward(self, input):
        data = input['norm_data'] #* for amortized stochastic params...

        det_select, KL_det = self.det_select()
        rule_select, KL_rule = self.rule_select()
        det_select = det_select*self.nonself_mask
        alpha_rules = self.alpha_rules + 0 # TODO: why does this make a difference?!?!
        alpha_default = torch.exp(self.log_alpha_default)
        beta = torch.exp(self.log_beta)
        split_proportions = torch.sigmoid(self.splits)

        rule_prob = self.rule_select.q_prob
        det_prob = self.det_select.q_prob

        params = {'rules': rule_select,
                  'detectors': det_select,
                  'rule_prob': rule_prob,
                  'det_prob': det_prob,
                  'max_range': self.max_range,
                  'split_temperature': self.split_temperature.value,
                  'alpha_rules': alpha_rules,
                  'alpha_default': alpha_default,
                  'beta': beta,
                  'split_proportions': split_proportions,
                  'KL_rule': KL_rule,
                  'KL_det': KL_det}
        
        self.params = params
        KL = KL_det + KL_rule #+ KL_params
        return params, KL


class LogisticDynamics(DynamicsBase):
    def __init__(self, num_taxa, device):
        super().__init__()
        self.num_taxa = num_taxa
        self.device = device

        self.log_alpha_default = nn.Parameter(torch.Tensor(self.num_taxa,), requires_grad=True)
        self.log_beta = nn.Parameter(torch.Tensor(self.num_taxa,), requires_grad=True) 

    def dynamics(self, logx, params):
        x = torch.exp(logx)
        alpha = params['alpha']
        beta = params['beta']

        return alpha - beta*x 

    def forward(self):
        alpha = torch.exp(self.log_alpha_default)
        beta = torch.exp(self.log_beta)

        params = {'alpha': alpha,
                  'beta': beta}
        self.params = params
        KL = torch.tensor(0)
        return params, KL


class LogXInitial(nn.Module):
    def __init__(self, num_taxa, num_subj, device, prior_mean, prior_var):
        super().__init__()
        self.num_taxa = num_taxa
        self.num_subj = num_subj
        self.prior_mean = torch.from_numpy(prior_mean).to(device)
        self.prior_var = torch.from_numpy(prior_var).to(device)
        self.device = device
        self.q_mu = nn.Parameter(torch.from_numpy(prior_mean), requires_grad=True)
        self.q_var = nn.Parameter(torch.normal(0,1, size=(self.num_subj,self.num_taxa)), requires_grad=True) 
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_subj,self.num_taxa), device=self.device, requires_grad=False)   

    def forward(self):
        self.sample_eps()
        mu = self.q_mu
        var = torch.exp(self.q_var)
        x = mu + torch.sqrt(var)*self.eps
        KL = KL_gaussian(mu, var, self.prior_mean, self.prior_var)
        return x, KL


class ProcessVariance(nn.Module):
    def __init__(self, input_dim, device, prior_mean=0.2**2, prior_var=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 50
        #* transform with inverse softmax to compare with mu
        self.prior_mean = torch.log(torch.exp(torch.tensor(prior_mean)) - 1.0)
        self.prior_var = prior_var
        self.device = device

        self.q_encode = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )

        self.q_mu_params = nn.Linear(self.hidden_dim, 1)
        self.q_var_params = nn.Linear(self.hidden_dim, 1)
        self.eps = None
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(1,), device=self.device, requires_grad=False)

    def forward(self, input):
        self.sample_eps()
        enc = self.q_encode(input)
        mu = self.q_mu_params(enc)
        var = torch.exp(self.q_var_params(enc))
        x = mu + torch.sqrt(var)*self.eps
        x = F.softplus(x)
        KL = KL_gaussian(mu, var, self.prior_mean, self.prior_var)
        self.mu = mu
        self.var = var
        return x, KL


class LatentTrajectory(nn.Module):
    def __init__(self, shape, input_dim, device, tinds_first=None):
        super().__init__()
        self.shape = shape
        _, _, num_taxa = shape
        self.input_dim = input_dim
        self.hidden_dim = 50 #100 #50
        self.full_dim = np.prod(shape)
        self.device = device

        self.time_mask = torch.ones(self.shape)
        if tinds_first is not None:
            #* zero out nan'ed values
            for i in range(num_taxa):
                idx = tinds_first[i]
                if idx > 0:
                    self.time_mask[:(idx),:,i] = 0
        
        self.q_encode = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )

        self.q_mu_params = nn.Linear(self.hidden_dim, self.full_dim)
        self.q_var_params = nn.Linear(self.hidden_dim, self.full_dim)
        self.eps = None
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=self.shape, device=self.device, requires_grad=False)

    def init_from_data(self, data):
        pass 

    def forward(self, input):
        self.sample_eps()
        norm_data = input['norm_data']
        enc = self.q_encode(norm_data)
        mu = self.q_mu_params(enc).reshape(self.shape)
        var = torch.exp(self.q_var_params(enc)).reshape(self.shape)
        x = mu + torch.sqrt(var)*self.eps
        #* does nan*0 = nan though?
        x = x*self.time_mask #+ torch.nan*(1-self.time_mask) #! just zero out or nan?
        return x, mu, var


class LatentTrajectorySepSubjEnc(nn.Module):
    def __init__(self, num_time, num_subj, num_otu, device):
        super().__init__()
        self.num_time = num_time
        self.num_subj = num_subj
        self.num_otu = num_otu
        self.hidden_dim = 5 #100 #50
        self.device = device

        self.input_dim = self.num_time*self.num_otu
        self.output_dim = (self.num_time-1)*self.num_otu

        self.q_encode = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )

        self.q_mu_params = nn.Linear(self.hidden_dim, self.output_dim)
        self.q_var_params = nn.Linear(self.hidden_dim, self.output_dim)

        self.eps = None
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_time-1, self.num_subj, self.num_otu), device=self.device, requires_grad=False)

    def init_from_data(self, data):
        pass 

    def forward(self, input):
        self.sample_eps()
        data = input['x_norm'] # shape TxSxO
        data = torch.reshape(data.transpose(0,1), (self.num_subj, self.num_time*self.num_otu))
        enc = self.q_encode(data) # output SxH
        mu = torch.reshape(self.q_mu_params(enc), (self.num_subj, self.num_time-1, self.num_otu)).transpose(0,1)
        var = torch.reshape(F.softplus(self.q_var_params(enc)), (self.num_subj, self.num_time-1, self.num_otu)).transpose(0,1)
        x = mu + torch.sqrt(var)*self.eps
        return x, mu, var
    

class LatentTrajectorySepSubjTMIX(nn.Module):
    def __init__(self, num_time, num_subj, num_otu, device):
        super().__init__()
        self.num_time = num_time
        self.num_subj = num_subj
        self.num_otu = num_otu
        self.hidden_dim = 2 #50 #100 #50
        self.device = device

        #* initial encoding layers otus -> hidden
        self.m1a = torch.nn.Parameter(torch.normal(0, 1, size=(1, self.num_otu, self.hidden_dim), requires_grad=True))
        self.b1a = torch.nn.Parameter(torch.normal(0, 1, size=(1, 1, self.hidden_dim), requires_grad=True))
        self.m1b = torch.nn.Parameter(torch.normal(0, 1, size=(1, self.hidden_dim, self.hidden_dim), requires_grad=True))
        self.b1b = torch.nn.Parameter(torch.normal(0, 1, size=(1, 1, self.hidden_dim), requires_grad=True))

        #* time connecting layers
        # NOTE: could add in time info or missing time info too (would then have H -> H+1)
        self.m2a = torch.nn.Parameter(
            torch.normal(0, 1, size=(self.num_time*(self.hidden_dim), (self.num_time-1)*(self.hidden_dim)), requires_grad=True))
        self.b2a = torch.nn.Parameter(
            torch.normal(0, 1, size=(1, (self.num_time-1)*(self.hidden_dim)), requires_grad=True))
        self.m2b = torch.nn.Parameter(
            torch.normal(0, 1, size=((self.num_time-1)*(self.hidden_dim), (self.num_time-1)*(self.hidden_dim)), requires_grad=True))
        self.b2b = torch.nn.Parameter(
            torch.normal(0, 1, size=(1, (self.num_time-1)*(self.hidden_dim)), requires_grad=True))
    #NOTE: taking in full time but outputting one less, could maybe do some other way?...

        #* linear layers for mean and variance (hidden -> num_otus)
        self.q_mu_mat = torch.nn.Parameter(torch.normal(0, 1, size=(1, self.hidden_dim, self.num_otu), requires_grad=True))
        self.q_mu_b = torch.nn.Parameter(torch.normal(0, 1, size=(1, 1, self.num_otu), requires_grad=True))
        self.q_var_mat = torch.nn.Parameter(torch.normal(0, 1, size=(1, self.hidden_dim, self.num_otu), requires_grad=True))
        self.q_var_b = torch.nn.Parameter(torch.normal(0, 1, size=(1, 1, self.num_otu), requires_grad=True))

        self.eps = None
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_time-1, self.num_subj, self.num_otu), device=self.device, requires_grad=False)

    def init_from_data(self, data):
        pass 

    def forward(self, input):
        self.sample_eps()
        data = input['x_norm'] # shape TxSxO

        #* pass through initial encoding layers
        l1 = data @ self.m1a + self.b1a
        l1 = F.softplus(l1)
        l2 = l1 @ self.m1b + self.b1b
        l2 = F.softplus(l2)

        #* reshape so all time points connected
        l2 = torch.reshape(l2.transpose(0,1), (self.num_subj, self.num_time*self.hidden_dim))
        # NOTE: option to add time info here...
        l3 = l2 @ self.m2a + self.b2a
        l3 = F.softplus(l3)
        l4 = l3 @ self.m2b + self.b2b
        l4 = F.softplus(l4)

        #* separate out time points
        l4 = torch.reshape(l4, (self.num_subj, self.num_time-1, self.hidden_dim)).transpose(0,1)

        #* pass through layers for mean and variance
        mu = l4 @ self.q_mu_mat + self.q_mu_b
        var = F.softplus(l4 @ self.q_var_mat + self.q_var_b)

        x = mu + torch.sqrt(var)*self.eps
        return x, mu, var
    

class LatentTrajectoryIndependent(nn.Module):
    def __init__(self, shape, input_dim, device):
        super().__init__()
        self.shape = shape
        self.input_dim = input_dim
        self.hidden_dim = 50
        self.full_dim = np.prod(shape)
        self.device = device

        self.q_mu_params = nn.Parameter(torch.Tensor(*self.shape), requires_grad=True)
        self.q_var_params = nn.Parameter(torch.normal(0,1, size=self.shape), requires_grad=True) 
        self.eps = None
    
    def init_from_data(self, data):
        self.q_mu_params.data = data['log_abundance'][1:,:,:]
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=self.shape, device=self.device, requires_grad=False)

    def forward(self, input):
        self.sample_eps()
        mu = self.q_mu_params
        var = torch.exp(self.q_var_params)
        x = mu + torch.sqrt(var)*self.eps
        return x, mu, var


class StochasticDynamics(nn.Module):
    def __init__(self, num_taxa, num_subj, num_time, num_rules, measurement_variance, 
                 max_range, rule_prior_prob, detector_prior_prob,
                 initial_condition, xinitial_var,
                 device, mask_times=None, tinds_first=None, split_start_temp=1.0, split_end_temp=0.01,
                 concrete_start_temp=0.5, concrete_end_temp=0.001, lr=1e-3):
        # TODO: concrete temps should go into rule params init
        super().__init__()
        self.num_taxa = num_taxa
        self.num_subj = num_subj
        self.num_time = num_time
        self.num_rules = num_rules
        self.num_covariates = num_taxa
        self.num_detectors = self.num_covariates
        self.measurement_variance = torch.tensor(measurement_variance).to(device)
        self.tinds_first = tinds_first

        self.device = device
        self.lr = lr
        self.input_dim = (num_taxa*num_subj + 1)*(num_time) # for VAE's

        #* time masks
        if mask_times is None:
            self.mask_times = -1*torch.ones((self.num_taxa,), device=self.device, requires_grad=False)
        else:
            self.mask_times = torch.from_numpy(mask_times).to(self.device)

        self.ic_placement_mask = torch.zeros((num_time, num_subj, num_taxa)).to(self.device)
        if tinds_first is not None:
            for i in range(num_taxa):
                self.ic_placement_mask[tinds_first[i],:,i] = 1
        else:
            self.ic_placement_mask[0,0,:] = 1

        self.taxa_mask = torch.ones((num_time, num_subj, num_taxa)).to(self.device)
        for i in range(self.num_taxa):
            tstart = tinds_first[i]
            self.taxa_mask[:tstart,:,i] = 0

        #* model parameters
        self.dynamics_model = RuleDynamics(num_taxa,num_subj,num_rules,num_time,rule_prior_prob,detector_prior_prob,
                                          split_start_temp, split_end_temp, max_range, device)
        # self.dynamics_model = LogisticDynamics(num_taxa, device)
        
        self.logx_initial = LogXInitial(num_taxa, num_subj, device, initial_condition, xinitial_var)
        self.process_variance = ProcessVariance(self.input_dim, device)
        latent_shape = (num_time-1, num_subj, num_taxa)
        self.latent_trajectory = LatentTrajectory(latent_shape, self.input_dim, device, tinds_first=tinds_first)
        # self.latent_trajectory = LatentTrajectorySepSubjEnc(num_time, num_subj, num_taxa, device)
        # self.latent_trajectory = LatentTrajectorySepSubjTMIX(num_time, num_subj, num_taxa, device)
        
        #* optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_params(self):
        params = {}
        with torch.no_grad():
            params['latent_x'] = self.param_latent_x.cpu().detach().clone().numpy() #* including iniital condition
            params['dyn_params'] = self.dynamics_model.params
            params['process_var'] = self.param_process_var.cpu().detach().clone().numpy()
            params['pv_mean'] = self.process_variance.mu.cpu().detach().clone().numpy()
            params['pv_var'] = self.process_variance.var.cpu().detach().clone().numpy()
            params['KL_dynamics'] = self.KL_dynamics.cpu().detach().clone().numpy()
            params['KL_process_var'] = self.KL_process_var.cpu().detach().clone().numpy()
            params['KL_logx_initial'] = self.KL_logx_initial.cpu().detach().clone().numpy()
            params['KL_latent'] = self.KL_latent.cpu().detach().clone().numpy()
            params['log_likelihood'] = self.log_likelihood.cpu().detach().clone().numpy()
        return params
    
    def init_from_data(self, data):
        #* initialize alpha, beta, splits, and probs...
        self.dynamics_model.init_from_data(data)
        # self.latent_trajectory.init_from_data(data)

    def init_from_grad_match(self, gmatch):
        self.dynamics_model.init_from_grad_match(gmatch)

    def init_latent_from_pretrain(self, latent_params):
        for name, val in self.latent_trajectory.named_parameters():
            val.data = torch.from_numpy(latent_params[name]).float()

    def default_init(self):
        self.dynamics_model.default_init()

    def set_temps(self, epoch, num_epochs):
        self.dynamics_model.set_temps(epoch, num_epochs)

    def forward(self, input):
        times = input['times']
        # x shape TxSxO
        logx_data = input['log_abundance']
        norm_data = input['norm_data']

        #* sample initial condition
        logx_initial, KL_logx_initial = self.logx_initial()
        #* sample rule parameters
        dynamics_parameters, KL_dynamics = self.dynamics_model(input)
        #* sample process variance
        process_var, KL_process_var = self.process_variance(norm_data)

        #* sample latent trajectory, amortized [after initial condition; prior conditional on sample from initial]
        logx_rest, logx_mean, logx_var = self.latent_trajectory(input)

        zero_ic = torch.zeros((self.num_subj,self.num_taxa)).to(self.device)
        logx = torch.cat([zero_ic[None,:,:], logx_rest]) 
        logx = logx + logx_initial[None,:,:]*self.ic_placement_mask
        # TODO: nan-out masked times

        KL_latent = 0
        for i in range(1,self.num_time):
            delta_tm1 = times[i] - times[i-1]
            log_mu = logx[i-1,:] + delta_tm1*self.dynamics_model.dynamics(logx[i-1,:], dynamics_parameters, self.taxa_mask[i-1,:,:])

            logp = normal_logpdf(logx[i,:], log_mu, delta_tm1*process_var)
            logq = normal_logpdf(logx[i,:], logx_mean[i-1,:], logx_var[i-1,:]) #* for each time; for each SxO
            #! note logx_mean and logx_var start at t_k=0 not t_k=1
            # TODO: make this more clear^^
            KL_masked = (logq - logp)*self.taxa_mask[i-1,:,:]
            KL_latent += KL_masked.sum()

        #* compute ELBO
        KL = KL_dynamics + KL_process_var + KL_logx_initial + KL_latent

        data_loglik = 0
        for i in range(self.num_taxa):
            data_loglik += torch.distributions.Normal(loc=logx_data[self.tinds_first[i]:,:,i], scale=torch.sqrt(self.measurement_variance[i])).log_prob(logx[self.tinds_first[i]:,:,i]).sum()
        
        self.ELBO_loss = -(data_loglik - KL)

        self.param_latent_x = logx
        self.param_process_var = process_var

        self.KL_dynamics = KL_dynamics
        self.KL_process_var = KL_process_var
        self.KL_logx_initial = KL_logx_initial
        self.KL_latent = KL_latent
        self.log_likelihood = data_loglik

        return self.ELBO_loss, logx, dynamics_parameters


class GradientMatchingNew(nn.Module):
    def __init__(self, num_taxa, num_subj, num_time, num_rules, empirical_variance, 
                 max_range, rule_prior_prob, detector_prior_prob,
                 device, mask_times=None, tinds_first=None, split_start_temp=1.0, split_end_temp=0.01, anneal_method="linear",
                 concrete_start_temp=0.5, concrete_end_temp=0.001,
                 lr=1e-3):
        super().__init__()
        self.num_taxa = num_taxa
        self.num_subj = num_subj
        self.num_time = num_time
        self.num_rules = num_rules
        self.num_covariates = num_taxa
        self.num_detectors = self.num_covariates
        self.measurement_variance = torch.tensor(empirical_variance).to(device)
        # self.tinds_first = tinds_first

        self.device = device
        self.lr = lr

        self.max_range = torch.from_numpy(max_range).to(device)

        #* time masks
        if mask_times is None:
            self.mask_times = -1*torch.ones((self.num_taxa,), device=self.device, requires_grad=False)
        else:
            self.mask_times = torch.from_numpy(mask_times).to(self.device)
        
        # self.taxa_mask = torch.ones((num_time, num_subj, num_taxa)).to(self.device)
        # for i in range(self.num_taxa):
        #     tstart = tinds_first[i]
        #     self.taxa_mask[:tstart,:,i] = 0

        #* model parameters
        self.split_temperature = AnnealedParameter(split_start_temp, split_end_temp, method=anneal_method)

        #* stochastic parameters
        self.det_select = BernoulliVariable((self.num_taxa, self.num_rules, self.num_detectors), detector_prior_prob,
                                device, concrete_start_temp, concrete_end_temp)
        self.rule_select = BernoulliVariable((self.num_taxa, self.num_rules), rule_prior_prob, device, concrete_start_temp,
                                concrete_end_temp)

        #* deterministic parameters
        self.split_points = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,self.num_rules,self.num_covariates)), requires_grad=True)
        self.log_alpha_default = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
        self.log_beta = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
        self.alpha_rules = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,self.num_rules)), requires_grad=True)

        #* mask to remove 'self interactions'
        self.nonself_mask = torch.reshape((1.0 - torch.diag(torch.ones((self.num_taxa,), device=self.device, requires_grad=False)) ), (self.num_taxa, 1, self.num_covariates))

        #* optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def manual_set_params(self, params):
        self.det_select.manual_set_params(params['det_select'])
        self.rule_select.manual_set_params(params['rule_select'])
        self.split_points.data = params['split_points']
        self.log_alpha_default.data = params['log_alpha_default']
        self.log_beta.data = params['log_beta']
        self.alpha_rules.data = params['alpha_rules']

    def get_params(self):
        params = {}
        with torch.no_grad():
            params['log_likelihood'] = self.log_likelihood.cpu().detach().clone().numpy()
            params['KL_rule'] = self.KL_rule.cpu().detach().clone().numpy()
            params['KL_det'] = self.KL_det.cpu().detach().clone().numpy()
            params['rule_prob'] = self.rule_select.q_prob.cpu().detach().clone().numpy()
            params['det_prob'] = self.det_select.q_prob.cpu().detach().clone().numpy()
            params['max_range'] = self.max_range.cpu().detach().clone().numpy()
            params['split_temperature'] = self.split_temperature.value
            params['alpha_rules'] = self.alpha_rules.cpu().detach().clone().numpy()
            params['log_beta'] = self.log_beta.cpu().detach().clone().numpy()
            params['log_alpha_default'] = self.log_alpha_default.cpu().detach().clone().numpy()
            params['split_proportions'] = torch.sigmoid(self.split_points).cpu().detach().clone().numpy()
            params['logit_split_proportions'] = self.split_points.cpu().detach().clone().numpy()
        return params
    
    def set_temps(self, epoch, num_epochs):
        self.split_temperature.anneal(epoch, num_epochs)
        self.rule_select.anneal(epoch, num_epochs)
        self.det_select.anneal(epoch, num_epochs)

    # TODO: option to initialize this as well... could be logisitc fits... correlations...

    def set_params_from_data(self, data):
        x = data['abundance']
        minval = torch.amin(x, dim=(0,1))
        maxval = torch.amax(x, dim=(0,1))
        self.min_range = (minval).to(self.device)
        max_range = (maxval).to(self.device)
        self.mrange = max_range - self.min_range

    def forward(self, input):
        times = input['times'] #* for masks
        logx_data = input['log_abundance']
        x_data = input['abundance']
        xlog_grad = input['gradient_log']

        # * sample indicators
        det_select, KL_det = self.det_select()
        rule_select, KL_rule = self.rule_select()

        #* evaluate rule function
        det_select = det_select*self.nonself_mask

        # splits = torch.sigmoid((((x_data-self.min_range)/self.mrange)[:,:,None,None,:] - torch.sigmoid(self.split_points)[None,None,:,:,:])/self.split_temperature.value)
        split_temp = self.split_temperature.value
        # x_data is size TxSxO 
        xproportion = ((x_data - self.min_range)/self.mrange)[:,:,None,None,:]
        sprops = torch.sigmoid(self.split_points)[None,None,:,:,:]
        splits = torch.sigmoid((xproportion - sprops)/split_temp)

        # splits shape: T x S x O[output] x R x O[input]
        detectors = splits*(times[:,None,None,None,None] >= self.mask_times[None,None,None,None,:]) #* mask input from masked taxa
        rules = torch.prod((1.0-det_select*(1.0-detectors)), dim=-1)
        res = torch.sum(self.alpha_rules*rule_select*rules, dim=-1) + torch.exp(self.log_alpha_default)

        #* add self-interactions
        res = (res - torch.exp(self.log_beta)*x_data)*(times[:,None,None] >= self.mask_times[None,None,:]) #* mask input

        data_loglik = 0
        for i in range(self.num_taxa):
            data_loglik += torch.distributions.Normal(loc=xlog_grad[times>=self.mask_times[i],:,:], scale=torch.sqrt(self.measurement_variance)).log_prob(res[times>=self.mask_times[i],:,:]).sum()

        KL = KL_det + KL_rule
        self.ELBO_loss = -(data_loglik - KL)

        self.KL_det = KL_det
        self.KL_rule = KL_rule
        self.log_likelihood = data_loglik

        rule_params = {'rules': rule_select,
            'detectors': det_select,
            'max_range': self.max_range,
            'split_temperature': self.split_temperature.value,
            'alpha_rules': self.alpha_rules,
            'alpha_default': torch.exp(self.log_alpha_default),
            'beta': torch.exp(self.log_beta),
            'split_proportions': torch.sigmoid(self.split_points)}
        return self.ELBO_loss, res, rule_params


def train(model, data, num_epochs, save_trace=False):
    if save_trace:
        param_trace = []
    else:
        param_trace = None 
    
    model.train()
    ELBOs = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.set_temps(epoch, num_epochs)
        model.forward(data)
        model.optimizer.zero_grad()
        model.ELBO_loss.backward()
        model.optimizer.step()

        if save_trace:
            param_trace.append(model.get_params())

        if epoch % max(int(num_epochs/10), 1) == 0:
            print("epoch = ", str(epoch))
            print("loss = ", model.ELBO_loss)

        ELBOs[epoch] = model.ELBO_loss.cpu().clone().detach().numpy()
    return ELBOs, param_trace
#! ---------------------------------------------------------------------------------

# class LatentPretrainer(nn.Module):
#     def __init__(self, num_taxa, num_subj, num_time, measurement_variance,
#                  process_var_prior, device, tinds_first=None, lr=1e-3):
#         super().__init__()
#         self.num_taxa = num_taxa
#         self.num_subj = num_subj
#         self.num_time = num_time
#         self.measurement_variance = torch.tensor(measurement_variance).to(device)
#         self.tinds_first = tinds_first

#         self.device = device
#         self.lr = lr
#         self.input_dim = (num_taxa*num_subj + 1)*(num_time) # for VAE's

#         latent_shape = (num_time-1, num_subj, num_taxa)
#         self.latent_trajectory = LatentTrajectory(latent_shape, self.input_dim, device, tinds_first=tinds_first)

#         self.process_var_prior = process_var_prior

#         #* optimizer
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

#     def set_temps(self, epoch, num_epochs):
#         # TODO: modify trainer so this isn't needed
#         pass

#     def get_params(self):
#         params = {}
#         for name, val in self.latent_trajectory.named_parameters():
#             params[name] = val.cpu().detach().clone().numpy()
#         return params
    
#     def forward(self, input):
#         times = input['times']
#         # x shape TxSxO
#         logx_data = input['log_abundance'] # TODO: except for initial condition..
#         norm_data = input['norm_data']

#         logx_data = logx_data[1:,:,:] #* remove initial time point
#         logx, mu, var = self.latent_trajectory(input)
#         KL = KL_gaussian(mu, var, 0, self.process_var_prior)
#         data_loglik = 0
#         for i in range(self.num_taxa):
#             data_loglik += torch.distributions.Normal(loc=logx_data[self.tinds_first[i]:,:,i], scale=torch.sqrt(self.measurement_variance[i])).log_prob(logx[self.tinds_first[i]:,:,i]).sum()
        
#         self.ELBO_loss = -(data_loglik - KL)

#         return self.ELBO_loss, logx
