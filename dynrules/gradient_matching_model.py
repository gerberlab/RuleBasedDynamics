import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import numpy as np 


def KL_bernoulli(q_params, prior_probs):
    q_prob = torch.sigmoid(q_params)
    q_logprob = F.logsigmoid(q_params)
    q_invlogprob = F.logsigmoid(-q_params)

    KL = torch.sum(q_prob*(q_logprob - torch.log(prior_probs))) + \
            torch.sum((1-q_prob)*(q_invlogprob - torch.log(1-prior_probs)))
    return KL


class AnnealedParameter(nn.Module):
    def __init__(self, start_temp, end_temp, method="linear"):
        super().__init__()
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.method = method 

        self.concrete_temperature = None 

    def set_temp(self, epoch, num_epochs):
        if self.method == "linear":
            self.linear_anneal(epoch, num_epochs)
        elif (self.method == "hyp") or (self.method == "hyperbolic"):
            self.hyp_anneal(epoch, num_epochs)
        else:
            raise ValueError('incorrect method type')
        
    def linear_anneal(self, epoch, num_epochs):
        percent = epoch/num_epochs
        if percent < 0.1:
            self.concrete_temperature = self.start_temp
            return
        if (percent >= 0.1) & (percent <= 0.9):
            interp = (percent-0.1)/0.8
            self.concrete_temperature = (1.0-interp)*self.start_temp + interp*self.end_temp
            return
        self.concrete_temperature = self.end_temp

    def hyp_anneal(self, epoch, num_epochs):
        percent = epoch/num_epochs
        if percent < 0.1:
            self.concrete_temperature = self.start_temp
            return
        if (percent >= 0.1) & (percent <= 0.9):
            interp = (percent-0.1)/0.8
            hyp = (1.0-interp)*(1.0/self.start_temp) + interp*(1.0/self.end_temp)
            self.concrete_temperature = 1/hyp
            return
        self.concrete_temperature = self.end_temp 


class BernoulliVariable(AnnealedParameter):
    def __init__(self, shape, prior_prob, device, 
                start_temp=0.5, end_temp=0.01, anneal_method="linear"):
        super().__init__(start_temp, end_temp, method=anneal_method)

        self.shape = shape
        self.device = device 

        self.selectors = None

        self.p_selectors = torch.tensor(prior_prob, requires_grad=False, dtype=torch.float, device=self.device)
        q_params = torch.zeros(self.shape, requires_grad=True, device=self.device)
        torch.nn.init.uniform_(q_params)
        self.q_params = torch.nn.Parameter(q_params)

    def forward(self):
        p_log = F.logsigmoid(torch.stack((self.q_params, -self.q_params)))
        
        if self.training:
            self.selectors = gumbel_softmax(p_log, hard=True, dim=0, tau=self.concrete_temperature)[0]
        else:
            self.selectors = torch.argmin(p_log, dim=0)

        self.q_prob = torch.sigmoid(self.q_params)
        KL = KL_bernoulli(self.q_params, self.p_selectors)
        return self.selectors, KL


class GlobalVariance(nn.Module):
    def __init__(self, num_taxa, num_subj, num_time, prior_mean, prior_variance, device):
        super().__init__()

        self.device = device
        #* get from empirical estimate; transform so as expected after passing through softmax
        prior_mu = torch.log(torch.exp(prior_mean) - 1)
        self.prior_mean = prior_mu.to(self.device) 
        self.prior_var = prior_variance

        self.hidden_dim = 10

        self.q_mu = nn.Parameter(prior_mu, requires_grad=True)
        self.q_var = nn.Parameter(torch.normal(0,1, size=(1,)), requires_grad=True) 

        # self.q_encode = nn.Sequential(
        #     nn.Linear(num_time*num_subj*num_taxa , self.hidden_dim),
        #     nn.Softplus(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.Softplus()
        # )

        # self.q_mu_params = nn.Linear(self.hidden_dim, 1)
        # self.q_var_params = nn.Linear(self.hidden_dim, 1)

        self.eps = None 

    def init_params(self, params):
        pass

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(1,), device=self.device, requires_grad=False)   

    def forward(self, input):
        self.sample_eps()

        # y = input['gradient'].reshape(-1)

        # enc = self.q_encode(y)
        # mu = self.q_mu_params(enc)
        mu = self.q_mu
        var = torch.exp(self.q_var) #self.q_var_params(enc))

        x = mu + torch.sqrt(var)*self.eps 
        x = F.softplus(x)
        KL = 0.5*torch.sum(var/self.prior_var + ((mu-self.prior_mean)**2)/self.prior_var - 1.0 - torch.log(var/self.prior_var))
        return x, KL




class GradientMatchingModel(nn.Module):
    def __init__(self, num_taxa, num_subj, num_time, num_rules, empirical_variance, rule_prior_prob, detector_prior_prob, 
                 device, mask_times=None, split_start_temp=1.0, split_end_temp=0.01, anneal_method="linear",
                 concrete_start_temp=0.5, concrete_end_temp=0.001,
                 lr=1e-3, learn_variance=False, variance_prior_variance=1):
        super().__init__()
        self.num_taxa = num_taxa 
        self.num_rules = num_rules 
        self.num_covariates = num_taxa
        self.num_detectors = self.num_covariates
        self.device = device
        self.lr = lr

        self.learn_variance = learn_variance
        if learn_variance is True:
            self.data_variance = GlobalVariance(num_taxa, num_subj, num_time, prior_mean=empirical_variance, prior_variance=variance_prior_variance, device=device)
        else:
            self.data_variance =  torch.from_numpy(empirical_variance).to(dtype=torch.float, device=self.device)

        self.min_range = None #torch.zeros((self.num_taxa,), dtype=torch.float, device=device, requires_grad=False)
        #* init from data 
        self.mrange = None 

        #* time masks
        if mask_times is None:
            self.mask_times = -1*torch.ones((self.num_taxa,), device=self.device, requires_grad=False)
        else:
            self.mask_times = torch.from_numpy(mask_times).to(self.device)


        #* stochastic discrete parameters
        self.det_select = BernoulliVariable((self.num_taxa, self.num_rules, self.num_detectors), detector_prior_prob,
                                device, concrete_start_temp, concrete_end_temp, anneal_method)
        self.rule_select = BernoulliVariable((self.num_taxa, self.num_rules,), rule_prior_prob, device, concrete_start_temp,
                                concrete_end_temp, anneal_method)
        self.split_temperature = AnnealedParameter(split_start_temp, split_end_temp, method=anneal_method)

        self.split_points = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,self.num_rules,self.num_covariates)), requires_grad=True)
        self.log_alpha_default = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
            # nn.Parameter(torch.Tensor(self.num_taxa), requires_grad=True)
            # torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
        self.log_beta = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
            # nn.Parameter(torch.Tensor(self.num_taxa), requires_grad=True)
            # torch.normal(0,1, size=(self.num_taxa,)), requires_grad=True)
        self.alpha_rules = nn.Parameter(torch.normal(0,1, size=(self.num_taxa,self.num_rules)), requires_grad=True) 

        #* mask to remove 'self interactions'
        self.nonself_mask = torch.reshape((1.0 - torch.diag(torch.ones((self.num_taxa,), device=self.device, requires_grad=False)) ), (self.num_taxa, 1, self.num_covariates))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_params(self):
        with torch.no_grad():
            rule_params = {}
            rule_params['min_range'] = self.min_range.cpu().detach().clone().numpy()
            rule_params['mrange'] = self.mrange.cpu().detach().clone().numpy()
            rule_params['split_temp'] = self.split_temperature.concrete_temperature
            rule_params['split_proportions'] = torch.sigmoid(self.split_points).cpu().detach().clone().numpy()
            rule_params['logit_split_proportions'] = self.split_points.cpu().detach().clone().numpy()
            rule_params['log_beta'] = self.log_beta.cpu().detach().clone().numpy()
            rule_params['log_alpha_default'] = self.log_alpha_default.cpu().detach().clone().numpy()
            rule_params['alpha_rules'] = self.alpha_rules.cpu().detach().clone().numpy()
            rule_params['rule_temp'] = self.rule_select.concrete_temperature
            rule_params['det_temp'] = self.det_select.concrete_temperature
            rule_params['rules'] = self.rule_select.q_prob.cpu().detach().clone().numpy()
            rule_params['detectors'] = self.det_select.q_prob.cpu().detach().clone().numpy()
            rule_params['data_variance'] = self.learned_variance.cpu().detach().clone().numpy()
            return rule_params

    def set_params_from_data(self, data):
        x = data['abundance']
        minval = torch.amin(x, dim=(0,1))
        maxval = torch.amax(x, dim=(0,1))
        self.min_range = (minval).to(self.device)
        max_range = (maxval).to(self.device)
        self.mrange = max_range - self.min_range

    def set_temps(self, epoch, num_epochs):
        self.split_temperature.set_temp(epoch, num_epochs)
        self.rule_select.set_temp(epoch, num_epochs)
        self.det_select.set_temp(epoch, num_epochs)

    def init_params(self, params):
        self.log_alpha_default.data = torch.from_numpy(params['log_alpha_default']).float()
        self.log_beta.data = torch.from_numpy(params['log_beta']).float()
        if self.learn_variance is True:
            self.data_variance.init_params(params)

    def forward(self, input):
        times = input['times'] #* for masks
        y = input['gradient'] 
        x = input['abundance']

        # * sample indicators
        det_select, KL_det = self.det_select()
        rule_select, KL_rule = self.rule_select()
        if self.learn_variance is True:
            data_variance, KL_var = self.data_variance(input)
        else:
            data_variance = self.data_variance
            KL_var = 0
        self.learned_variance = data_variance #* for saving to output

        det_select = det_select*self.nonself_mask
        splits = torch.sigmoid((((x-self.min_range)/self.mrange)[:,:,None,None,:] - torch.sigmoid(self.split_points)[None,None,:,:,:])/self.split_temperature.concrete_temperature)
        # splits shape: T x S x O[output] x R x O[input]
        detectors = splits*(times[:,None,None,None,None] >= self.mask_times[None,None,None,None,:]) #* mask input from masked taxa
        rules = torch.prod((1.0-det_select*(1.0-detectors)), dim=-1)
        res = torch.sum(self.alpha_rules*rule_select*rules, dim=-1) + torch.exp(self.log_alpha_default)

        #* add self-interactions
        res = (res - torch.exp(self.log_beta)*x)*(times[:,None,None] >= self.mask_times[None,None,:]) #* mask input

        data_loglik = 0
        for i in range(self.num_taxa):
            # data_loglik += -((xdata[self.t_span>self.mask_times[i],:,:]-res[self.t_span>self.mask_times[i],:,:])**2).sum()
            data_loglik += torch.distributions.Normal(loc=y[times>=self.mask_times[i],:,:], scale=torch.sqrt(data_variance)).log_prob(res[times>=self.mask_times[i],:,:]).sum()

        KL = KL_det + KL_rule + KL_var
        self.ELBO_loss = -(data_loglik - KL)
        return self.ELBO_loss, res


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

            # print("rule probs = ", model.rule_select.q_prob)
            # print("det probs = ", model.det_select.q_prob)

        ELBOs[epoch] = model.ELBO_loss.cpu().clone().detach().numpy()
    return ELBOs, param_trace

