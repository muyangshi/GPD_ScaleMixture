"""
April 23, 2024
Simulate data from a stationary model for the GPD Scale Mixture Model:
    (spatially constant phi, rho, sigma, ksi, St)
"""
if __name__ == "__main__":
    # %%
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% imports
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy
    from mpi4py import MPI
    from utilities import *
    import gstools as gs
    import rpy2.robjects as robjects
    from rpy2.robjects import r 
    from rpy2.robjects.numpy2ri import numpy2rpy
    from rpy2.robjects.packages import importr

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    random_generator = np.random.RandomState((rank+1)*7)
    
    try:
        data_seed
    except:
        data_seed = 2345
    finally:
        if rank == 0: print('data_seed:', data_seed)

    if rank == 0: print('Pareto:', norm_pareto)

    # %% Simulation Setup ---------------------------------------------------------------------------------------------

    # Numbers - Ns, Nt ------------------------------------------------------------------------------------------------   
    np.random.seed(data_seed)
    Nt = 32 # number of time replicates
    Ns = 500 # number of sites/stations
    Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

    # missing indicator matrix ----------------------------------------------------------------------------------------
    miss_proportion = 0.0
    miss_matrix = np.full(shape = (Ns, Nt), fill_value = 0)
    for t in range(Nt):
        miss_matrix[:,t] = np.random.choice([0, 1], size = (Ns,), p = [1-miss_proportion, miss_proportion])
    miss_matrix = miss_matrix.astype(bool) # matrix of True/False indicating missing, True means missing


    # Sites - random unifromly (x,y) generate site locations ----------------------------------------------------------
    sites_xy = np.random.random((Ns, 2)) * 10
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # Elevation Function ----------------------------------------------------------------------------------------------
    # Note: the simple elevation function 1/5(|x-5| + |y-5|) is way too similar to the first basis
    #       this might cause identifiability issue
    # def elevation_func(x,y):
        # return(np.abs(x-5)/5 + np.abs(y-5)/5)
    # elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2), seed=data_seed)
    # elevations = elev_surf_generator((sites_x, sites_y))

    # spatiall constant
    elevations = np.full(shape = (sites_xy.shape[0],), fill_value = 0.0)

    # Knots - isometric grid of 9 + 4 = 13 knots ----------------------------------------------------------------------

    # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

    # isometric knot grid
    N_outer_grid = 16
    h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
    v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
    x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    x_outer_pos              = x_pos[0::2]
    x_inner_pos              = x_pos[1::2]
    y_outer_pos              = y_pos[0::2]
    y_inner_pos              = y_pos[1::2]
    X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
    X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
    knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
    knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
    knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
    knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
    knots_xy                 = knots_xy[knots_id_in_domain]
    knots_x                  = knots_xy[:,0]
    knots_y                  = knots_xy[:,1]
    k                        = len(knots_id_in_domain)  

    # Copula Splines --------------------------------------------------------------------------------------------------

    radius    = 2 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
    bandwidth = radius**2/6 # so that the effective range of gaussian kernel is radius
    radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?
    assert k == len(knots_xy)
    
    # Weight matrix generated using Gaussian Smoothing Kernel
    gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix[site_id, :] = weight_from_knots

    # Weight matrix generated using wendland basis
    wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
        wendland_weight_matrix[site_id, :] = weight_from_knots
    
    # -----------------------------------------------------------------------------------------------------------------
    # Setup For the Model 
    # -----------------------------------------------------------------------------------------------------------------

    # Marginal Model - GP(sigma, ksi) threshold u ---------------------------------------------------------------------
    
    """
    We no longer need a spline fit surface for mu or any of the like, right?
    Because ut(s), the threshold for a site s at time t, will be empiricially
    estimated from the actual data.
    """

    # Scale logsigma(s) ----------------------------------------------------------------------------------------------
    
    Beta_logsigma_m   = 2 # just intercept and elevation
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
    C_logsigma[0,:,:] = 1.0 
    C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Shape ksi(s) ----------------------------------------------------------------------------------------------------
    
    Beta_ksi_m   = 2 # just intercept and elevation
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Threshold u(t,s) ------------------------------------------------------------------------------------------------
    u_matrix = np.full(shape = (Ns, Nt), fill_value = 20.0)


    # Setup For the Copula/Data Model - X = e + X_star = R^phi * g(Z) -------------------------------------------------

    # Nugget Variance
    tau = 10.0

    # Covariance K for Gaussian Field g(Z) ----------------------------------------------------------------------------
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi ---------------------------------------------------------------------------------------------
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots

    # -----------------------------------------------------------------------------------------------------------------
    # Parameter Configuration For the Model 
    # -----------------------------------------------------------------------------------------------------------------

    # Censoring probability
    p = 0.9

    # Marginal Parameters - GP(sigma, ksi) ----------------------------------------------------------------------------
    Beta_logsigma       = np.array([0.0, 0.0])
    Beta_ksi            = np.array([0.25, 0.0])
    sigma_Beta_logsigma = 1
    sigma_Beta_ksi      = 1

    sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
    ksi_matrix   = (C_ksi.T @ Beta_ksi).T

    # Data Model Parameters - X_star = R^phi * g(Z) -------------------------------------------------------------------

    # range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

    # ### scenario 1
    # # phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
    # ### scenario 2
    # phi_at_knots = 0.65 - np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    # ### scenario 3
    # # phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
    # #                          scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))

    range_at_knots = np.array([1.0] * k)
    phi_at_knots   = np.array([0.7] * k)

    # %% Generate Simulation Data ------------------------------------------------------------------------------------
    # Generate Simulation Data
    np.random.seed(data_seed)
    # W = g(Z), Z ~ MVN(0, K)
    range_vec = gaussian_weight_matrix @ range_at_knots
    K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                        coords = sites_xy, kappa = nu, cov_model = "matern")
    Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W         = g(Z) 

    # R^phi Scaling Factor
    phi_vec    = gaussian_weight_matrix @ phi_at_knots
    S_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        # S_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
        S_at_knots[:,t] = rlevy(n = 1, m = delta, s = gamma)   # stationary model
    R_at_sites = wendland_weight_matrix @ S_at_knots
    R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)
    
    # Nuggets
    nuggets = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)),
                                                  cov  = tau**2,
                                                  size = Nt).T

    # Generate Observed Process Y
    X_star       = R_phi * W
    X            = X_star + nuggets

    pX           = np.full(shape = (Ns, Nt), fill_value = np.nan)
    Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)
    
    for t in np.arange(Nt): # single core version
        # CDF of the generated X
        pX[:,t] = pRW(X[:,t], phi_vec, gamma_vec, tau)
        censored_idx = np.where(pX[:,t] <= p)[0]
        exceed_idx   = np.where(pX[:,t] > p)[0]

        # censored below
        Y[censored_idx,t]  = u_matrix[censored_idx,t]

        # threshold exceeded
        Y[exceed_idx,t]  = qCGP(pX[exceed_idx, t], p, 
                                u_matrix[exceed_idx, t], 
                                sigma_matrix[exceed_idx, t],
                                ksi_matrix[exceed_idx, t])
    
    # %% saving data
    # Saving data -----------------------------------------------------------------------------------------------------
    if rank == 0:
        folder = '../data/stationary_seed'+str(data_seed)+'_t'+str(Nt)+'_s'+str(Ns)+'/'
        np.save(folder+'Y', Y)
        np.save(folder+'X', X)
        np.save(folder+'Z', Z)
        np.save(folder+'X_star', X_star)
        np.save(folder+'S_at_knots', S_at_knots)
        np.save(folder+'nuggets', nuggets)
        np.save(folder+'miss_matrix_bool', miss_matrix)

        np.save(folder+'sites_xy',sites_xy)
        np.save(folder+'elevations', elevations)
        np.save(folder+'logsigma_matrix', np.log(sigma_matrix))
        np.save(folder+'ksi_matrix', ksi_matrix)
    for t in range(Nt):
        Y[:,t][miss_matrix[:,t]] = np.nan
    
    # %% Check Data Generation ----------------------------------------------------------------------------------------

    # checking stable variables S -------------------------------------------------------------------------------------

    # levy.cdf(R_at_knots, loc = 0, scale = gamma) should look uniform
    site_i = int(np.floor(scipy.stats.uniform(0, k).rvs()))
    print(site_i)
    emp_p = np.linspace(1/Nt, 1-1/Nt, num=Nt)
    emp_q = scipy.stats.uniform().ppf(emp_p)
    plt.plot(emp_q, np.sort(scipy.stats.levy.cdf(S_at_knots[site_i,:], scale=gamma)),
             c='blue',marker='o',linestyle='None')
    plt.xlabel('Uniform')
    plt.ylabel('Observed')
    plt.axline((0,0), slope = 1, color = 'black')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title('QQ-Plot site {}'.format(site_i))
    plt.show()
    plt.close()
    # scipy.stats.probplot(scipy.stats.levy.cdf(S_at_knots[i,:], scale=gamma), dist='uniform', fit=False, plot=plt)

        
    # checking Pareto distribution ------------------------------------------------------------------------------------
    if norm_pareto == 'standard':
        site_i = int(np.floor(scipy.stats.uniform(0, Ns).rvs()))
        print(site_i)
        emp_p = np.linspace(1/Nt, 1-1/Nt, num=Nt)
        emp_q = scipy.stats.uniform().ppf(emp_p)
        plt.plot(emp_q, np.sort(scipy.stats.pareto.cdf(W[site_i,:], b = 1)),
                c='blue',marker='o',linestyle='None')
        plt.xlabel('Uniform')
        plt.ylabel('Observed')
        plt.axline((0,0), slope = 1, color = 'black')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.title('QQ-Plot site {}'.format(site_i))
        plt.show()
        plt.close()
        # scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)

    # checking model X_star -------------------------------------------------------------------------------------------
    site_i = int(np.floor(scipy.stats.uniform(0, Ns).rvs()))
    print(site_i)
    emp_p = np.linspace(1/Nt, 1-1/Nt, num=Nt)
    emp_q = scipy.stats.uniform().ppf(emp_p)
    plt.plot(emp_q, np.sort(pRW(X[site_i,:], phi_vec[site_i], gamma_vec[site_i], tau)),
                c='blue',marker='o',linestyle='None')
    plt.xlabel('Uniform')
    plt.ylabel('Observed')
    plt.axline((0,0), slope = 1, color = 'black')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title('QQ-Plot site {}'.format(site_i))
    plt.show()
    plt.close()

    # checking marginal exceedance ------------------------------------------------------------------------------------
    
    ## all pooled together
    pY = np.array([])
    for t in range(Nt):
        exceed_idx = np.where(Y[:,t] > u_matrix[:,t])[0]
        pY = np.append(pY, pGP(Y[exceed_idx,t],u_matrix[exceed_idx,t],sigma_matrix[exceed_idx,t],ksi_matrix[exceed_idx,t]))
    nquant = len(pY)
    emp_p = np.linspace(1/nquant, 1-1/nquant, num=nquant)
    emp_q = scipy.stats.uniform().ppf(emp_p)
    plt.plot(emp_q, np.sort(pY),
             c='blue',marker='o',linestyle='None')
    plt.xlabel('Uniform')
    plt.ylabel('Observed')
    plt.axline((0,0), slope = 1, color = 'black')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title('QQ-Plot of all exceedance')
    plt.show()
    plt.close()

    # %% Plot Generated Surfaces --------------------------------------------------------------------------------------

    # 0. Grids for plots
    plotgrid_res_x = 150
    plotgrid_res_y = 175
    plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
    plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
    plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
    plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
    plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

    gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

    wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
        wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots


    # 1. Station, Knots 
    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)
    ax.set_aspect('equal', 'box')
    for i in range(k):
        circle_i = plt.Circle((knots_xy[i,0], knots_xy[i,1]), radius_from_knots[i],
                                color='r', fill=True, fc='grey', ec='None', alpha = 0.2)
        ax.add_patch(circle_i)
    ax.scatter(sites_x, sites_y, marker = '.', c = 'blue', label='sites')
    ax.scatter(knots_x, knots_y, marker = '+', c = 'red', label = 'knot', s = 300)
    space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                    fill = False, color = 'black')
    ax.add_patch(space_rectangle)
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=5))
    box = ax.get_position()
    legend_elements = [matplotlib.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                    matplotlib.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=20),
                    matplotlib.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'grey', markersize = 20, alpha = 0.2),
                    matplotlib.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=20, label='Spatial Domain')]
    plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel('longitude', fontsize = 20)
    plt.ylabel('latitude', fontsize = 20)
    plt.subplots_adjust(right=0.6)
    plt.show()
    plt.savefig(folder+'stations.pdf',bbox_inches="tight")
    plt.close()


    # 2. Elevation
    fig, ax = plt.subplots()
    elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                                cmap = 'bwr')
    ax.set_aspect('equal', 'box')
    plt.colorbar(elev_scatter)
    plt.title('elevation')
    plt.show()
    plt.savefig(folder+'station_elevation.pdf')
    plt.close()       


    # 3. phi surface
    # heatplot of phi surface
    phi_vec_for_plot = (gaussian_weight_matrix_for_plot @ phi_at_knots).round(3)
    graph, ax = plt.subplots()
    heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                        cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.title(r'true $\phi$ surface')
    plt.show()
    plt.savefig(folder+'true_phi_surface.pdf')
    plt.close()


    # 4. Plot range surface
    # heatplot of range surface
    range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
    graph, ax = plt.subplots()
    heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                        cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.title(r'true $\rho$ surface')
    plt.show()
    plt.savefig(folder+'true_range_surface.pdf')
    plt.close()
# %%
