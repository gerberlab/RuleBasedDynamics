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

    #* get first non-nan index:
    tind_first = np.zeros(num_taxa, dtype=int)
    for i in range(num_taxa):
        temp = xlog[:,0,i] #* assuming all subjects same
        tind_first[i] = int(np.argmax(~np.isnan(temp)))

    return t,xlog, mask_times, tind_first


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t,x,m, ifirst = get_one_rule_masked()

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
