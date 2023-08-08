import numpy as np 


def subsample_times(t,x,st):
    #* find time indices closest to st
    isub = np.sort(np.abs(np.subtract.outer(t, st)).argmin(axis=0))
    tsub = t[isub]
    xsub = x[isub,:,:]
    return tsub, xsub


def get_two_taxa_exp_dynamics_test_logspace():
    mask_times = np.array([0, 0])

    num_time, num_subj, num_taxa = 100, 3, 2
    t =  np.linspace(0,5,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.1, 0.5, 1.1]))
    xlog[0,:,1] = np.log(np.array([0.8, 0.5, 0.4]))

    a0 = 2.0
    a1 = 1.3 
    b0 = 0 #1.0
    b1 = 0 #0.5

    for i in range(1,num_time):
        dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt

        dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt
    return t,xlog, mask_times


# TODO: add process noise and subsample times**
def get_logistic_test():
    mask_times = np.array([0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2
    t =  np.linspace(0,5,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.1, 0.5, 1.1]))
    xlog[0,:,1] = np.log(np.array([0.8, 0.5, 0.4]))

    a0 = 2.0
    a1 = 1.3 
    b0 = 1.0
    b1 = 0.5

    for i in range(1,num_time):
        dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt

        dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75,\
              3.0, 3.5, 4.0, 4.5, 4.8
    ])
    t, xlog = subsample_times(t,xlog, sample_times)
    return t,xlog, mask_times


def get_test_one_rule():
    mask_times = np.array([0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2
    t =  np.linspace(0,5,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.1, 0.5, 0.7]))
    xlog[0,:,1] = np.log(np.array([0.8, 0.5, 1.0]))

    a0 = 2.0
    a1 = 1.3 
    b0 = 1.0
    b1 = 0.5
    alpha = -1.5
    threshold = 1.7

    for i in range(1,num_time):
        dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt

        # dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        # xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt
        agrowth1 = a1*(np.exp(xlog[i-1,:,0])<threshold) + (a1+alpha)*(np.exp(xlog[i-1,:,0])>=threshold)
        dyn1 = agrowth1 - b1*np.exp(xlog[i-1,:,1])
        xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75,\
              3.0, 3.5, 4.0, 4.5, 4.8
    ])
    t, xlog = subsample_times(t,xlog, sample_times)

    #* get first non-nan index:
    tind_first = np.zeros(num_taxa, dtype=int)

    return t,xlog, mask_times, tind_first



def get_one_rule_masked():
    mask_times = np.array([1.0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2
    t =  np.linspace(0,6,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.1, 0.5, 0.7]))
    xlog[0,:,1] = np.log(np.array([0.8, 0.5, 1.0]))

    a0 = 2.0
    a1 = 1.3 
    b0 = 1.0
    b1 = 0.5
    alpha = -1.5
    threshold = 1.7

    for i in range(1,num_time):
        dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt*(t[i] > mask_times[0])

        # dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        # xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt
        agrowth1 = a1*(np.exp(xlog[i-1,:,0])<threshold) + (a1+alpha)*(np.exp(xlog[i-1,:,0])>=threshold)
        dyn1 = agrowth1 - b1*np.exp(xlog[i-1,:,1])
        xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75,\
              3.0, 3.25, 3.5, 4.0, 4.5, 4.8, 5.3, 5.7, 5.9
    ])
    t, xlog = subsample_times(t,xlog, sample_times)

    for i in range(num_taxa):
        xlog[t<mask_times[i],:,i] = np.nan

    # #* get first non-nan index:
    # tind_first = np.zeros(num_taxa, dtype=int)
    # for i in range(num_taxa):
    #     temp = xlog[:,0,i] #* assuming all subjects same
    #     tind_first[i] = int(np.argmax(~np.isnan(temp)))

    typeinfo = np.zeros(num_taxa, dtype=int)
    return t, xlog, mask_times, typeinfo #, tind_first


def get_one_rule_masked_rescale(scale=1):
    mask_times = np.array([1.0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2
    t =  np.linspace(0,6,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.1, 0.5, 0.7]))
    xlog[0,:,1] = np.log(np.array([0.8, 0.5, 1.0]))

    a0 = 2.0*scale
    a1 = 1.3*scale 
    b0 = 1.0*scale
    b1 = 0.5*scale
    alpha = -1.5*scale
    threshold = 1.7

    for i in range(1,num_time):
        dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt*(t[i] > mask_times[0])

        # dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        # xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt
        agrowth1 = a1*(np.exp(xlog[i-1,:,0])<threshold) + (a1+alpha)*(np.exp(xlog[i-1,:,0])>=threshold)
        dyn1 = agrowth1 - b1*np.exp(xlog[i-1,:,1])
        xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75,\
              3.0, 3.25, 3.5, 4.0, 4.5, 4.8, 5.3, 5.7, 5.9
    ])
    t, xlog = subsample_times(t,xlog, sample_times)

    for i in range(num_taxa):
        xlog[t<mask_times[i],:,i] = np.nan

    #* get first non-nan index:
    tind_first = np.zeros(num_taxa, dtype=int)
    for i in range(num_taxa):
        temp = xlog[:,0,i] #* assuming all subjects same
        tind_first[i] = int(np.argmax(~np.isnan(temp)))

    return t,xlog, mask_times, tind_first


def glv_test_example_two_taxa_no_mask():
    mask_times = np.array([0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2
    t =  np.linspace(0,4,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    xlog[0,:,0] = np.log(np.array([0.05, 0.01, 0.1]))
    xlog[0,:,1] = np.log(np.array([0.3, 0.5, 1.2]))

    a0 = 3.0
    a1 = 1.3
    a = np.array([a0, a1])

    b0 = -0.7
    b1 = -0.5
    b01 = -0.5
    b10 = 0.3
    interactions = np.array([[b0, b01], [b10, b1]])

    for i in range(1,num_time):
        dyn = a + np.exp(xlog[i-1,:,:]) @ interactions  # TODO: should things be 'transposed'??
        xlog[i,:,:] = xlog[i-1,:,:] + dyn*dt
        # dyn0 = a0 - b0*np.exp(xlog[i-1,:,0])
        # xlog[i,:,0] = xlog[i-1,:,0] + dyn0*dt*(t[i] > mask_times[0])

        # # dyn1 = a1 - b1*np.exp(xlog[i-1,:,1])
        # # xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt
        # agrowth1 = a1*(np.exp(xlog[i-1,:,0])<threshold) + (a1+alpha)*(np.exp(xlog[i-1,:,0])>=threshold)
        # dyn1 = agrowth1 - b1*np.exp(xlog[i-1,:,1])
        # xlog[i,:,1] = xlog[i-1,:,1] + dyn1*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5,\
              3.0, 3.5, 3.9 #, 4.5, 4.8, 5.3, 5.7, 5.9
    ])
    t, xlog = subsample_times(t,xlog, sample_times)

    for i in range(num_taxa):
        xlog[t<mask_times[i],:,i] = np.nan

    #* get first non-nan index:
    tind_first = np.zeros(num_taxa, dtype=int)
    for i in range(num_taxa):
        temp = xlog[:,0,i] #* assuming all subjects same
        tind_first[i] = int(np.argmax(~np.isnan(temp)))

    return t,xlog, mask_times, tind_first

# TODO: create class to generate data from different models, then have methods for fixed examples
def glv_test_example_two_taxa_masked():
    mask_times = np.array([1.0, 0])

    num_time, num_subj, num_taxa = 1000, 3, 2 # dense time for forward simulation, subsample times after integration
    t =  np.linspace(0,6,num_time)
    dt = t[1]-t[0] 

    xlog = np.zeros((num_time, num_subj, num_taxa))
    # initial condition
    xlog[0,:,0] = np.log(np.array([0.05, 0.01, 0.1]))
    xlog[0,:,1] = np.log(np.array([0.05, 0.1, 0.02]))

    # growth rates
    a0 = 3.0
    a1 = 1.3
    a = np.array([a0, a1])

    # interactions
    b0 = -0.7
    b1 = -0.5
    b01 = -0.5
    b10 = 0.6
    interactions = np.array([[b0, b01], [b10, b1]])

    # forward simulation
    for i in range(1,num_time): #* want actual time as well for mask
        time = t[i]
        #* mask input
        input = np.exp(xlog[i-1,:,:]) #* set this to zero when taxon is masked; shape SxO
        for oidx in range(num_taxa):
            if time < mask_times[oidx]:
                input[:,oidx] = 0

        dyn = a + input @ interactions.T  # NOTE: interactions transposed since putting x in front; consistent with interaction rows = i, columns = j ***elaborate...
        
        #* mask dynamics
        # dyn shape SxO
        for oidx in range(num_taxa):
            if time < mask_times[oidx]:
                dyn[:,oidx] = 0

        xlog[i,:,:] = xlog[i-1,:,:] + dyn*dt

    #* subsample times
    sample_times = np.array([
        0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01, 1.2,\
        1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75,\
              3.0, 3.25, 3.5, 4.0, 4.5, 4.8, 5.3, 5.7, 5.9
    ])
    t, xlog = subsample_times(t,xlog, sample_times)

    #* mask output
    for i in range(num_taxa):
        xlog[t<mask_times[i],:,i] = np.nan

    type_info = np.array([0,0])

    gtruth = {'log_alpha_default': np.log(a),
              'log_beta_self': np.log(np.abs(np.diagonal(interactions))),
              'beta_matrix': interactions
              }

    return t ,xlog, mask_times, type_info, gtruth


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # t,x,m, typeinfo = get_one_rule_masked() #glv_test_example_two_taxa_masked()
    t,x,m, typeinfo, gtruth = glv_test_example_two_taxa_masked()

    # added_noise = 0.01
    # x = x + np.random.normal(loc=0, scale=np.sqrt(added_noise), size=x.shape)
    #* convert to log space?

    num_time, num_subj, num_taxa = x.shape
    for subj in range(num_subj):
        fig, ax = plt.subplots()
        for oidx in range(num_taxa):
            ax.plot(t, x[:,subj,oidx], '-x', label=f'OTU {oidx}')
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Abundance")
    plt.show()
