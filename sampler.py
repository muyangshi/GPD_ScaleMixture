"""
March 26, 2024
MCMC Sampler for GPD Scale Mixture Model
"""
if __name__ == "__main__":
    # %%
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% imports
    import os
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
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
    import pickle
    from time import strftime, localtime
    import time

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
    np.random.seed(data_seed)

    if rank == 0: print('Pareto:', norm_pareto)

    try:
        with open('iter.pkl','rb') as file:
            start_iter = pickle.load(file) + 1
            if rank == 0: print('start_iter loaded from pickle, set to be:', start_iter)
    except Exception as e:
        if rank == 0: 
            print('Exception loading iter.pkl:', e)
            print('Setting start_iter to 1')
        start_iter = 1
    
    if norm_pareto == 'shifted': n_iters = 5000
    if norm_pareto == 'standard': n_iters = 5000

    # %% Load Simulated Dataset ---------------------------------------------------------------------------------------

    Y                  = np.load('Y_sc2_t24_s100_truth.npy')
    logsigma_estimates = np.load('logsigma_matrix.npy')[:,0]
    ksi_estimates      = np.load('ksi_matrix.npy')[:,0]
    stations           = np.load('sites_xy.npy')
    elevations         = np.load('elevations.npy')

    # %% Load Real Dataset --------------------------------------------------------------------------------------------


    # %% Setup --------------------------------------------------------------------------------------------------------
    # Setup

    Ns = Y.shape[0] # number of sites/stations
    Nt = Y.shape[1] # number of time replicates

    # Censoring -------------------------------------------------------------------------------------------------------

    # threshold probability and quantile
    p        = 0.9
    u_matrix = np.full(shape = (Ns, Nt), fill_value = np.nanquantile(Y, p)) # threshold u on Y, i.e. p = Pr(Y <= u)
    u_vec    = u_matrix[:,rank]

    # missing indicator -----------------------------------------------------------------------------------------------
    miss_matrix = np.isnan(Y)
    miss_idx_1t = np.where(np.isnan(Y[:,rank]) == True)[0]
    obs_idx_1t  = np.where(np.isnan(Y[:,rank]) == False)[0]
    # Note:
    #   miss_idx_1t and obs_idx_1t stays the same throughout the entire MCMC
    #   they are part of the "dataset's attribute"

    # Sites -----------------------------------------------------------------------------------------------------------
    
    sites_xy = stations
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # Knots - isometric grid of 9 + 4 = 13 knots ----------------------------------------------------------------------

    # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))
    # isometric knot grid
    N_outer_grid = 9
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
    
    # Basis Parameters - for the Gaussian and Wendland Basis
    radius = 4 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
    # bandwidth = 4 # range for the gaussian kernel
    effective_range = radius # effective range for gaussian kernel: exp(-3) = 0.05
    bandwidth = effective_range**2/6
    radius_from_knots = np.repeat(radius, k) # influence radius from a knot

    # Generate the weight matrices
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
    
    # Marginal Model - GP(sigma, ksi) threshold u ---------------------------------------------------------------------
    
    # Scale logsigma(s)
    Beta_logsigma_m   = 2 # just intercept and elevation
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
    C_logsigma[0,:,:] = 1.0 
    C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Shape ksi(s)
    Beta_ksi_m   = 2 # just intercept and elevation
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Setup For the Copula/Data Model - X = e + X_star = R^phi * g(Z) -------------------------------------------------

    # Covariance K for Gaussian Field g(Z) 
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots
    
    # %% Estimate Parameter -------------------------------------------------------------------------------------------
    # Estimate Parameters

    if start_iter == 1:
        # We estimate parameter's initial values to start the chains

        # Marginal Parameters - GP(sigma, ksi) ------------------------------------------------------------------------
        
        # scale
        Beta_logsigma = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
        sigma_vec     = np.exp((C_logsigma.T @ Beta_logsigma).T)[:,rank]

        # shape
        Beta_ksi = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]
        ksi_vec  = ((C_ksi.T @ Beta_ksi).T)[:,rank]

        # regularization
        sigma_Beta_logsigma = 1
        sigma_Beta_ksi      = 1

        # Data Model Parameters - X = e + R^phi * g(Z) ----------------------------------------------------------------

        # Nugget Variance (var = tau^2, stdev = tau)
        tau = 10

        # rho - covariance K
        range_at_knots = np.array([])
        distance_matrix = np.full(shape=(Ns, k), fill_value=np.nan)
        # distance from knots
        for site_id in np.arange(Ns):
            d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), XB = knots_xy)
            distance_matrix[site_id,:] = d_from_knots
        # each knot's "own" sites
        sites_within_knots = {}
        for knot_id in np.arange(k):
            knot_name = 'knot_' + str(knot_id)
            sites_within_knots[knot_name] = np.where(distance_matrix[:,knot_id] <= radius_from_knots[knot_id])[0]
        # empirical variogram estimates
        for key in sites_within_knots.keys():
            selected_sites           = sites_within_knots[key]
            demeaned_Y               = (Y.T - np.nanmean(Y, axis = 1)).T
            bin_center, gamma_variog = gs.vario_estimate((sites_x[selected_sites], sites_y[selected_sites]), 
                                                        np.nanmean(demeaned_Y[selected_sites], axis=1))
            fit_model = gs.Exponential(dim=2)
            fit_model.fit_variogram(bin_center, gamma_variog, nugget=False)
            # ax = fit_model.plot(x_max = 4)
            # ax.scatter(bin_center, gamma_variog)
            range_at_knots = np.append(range_at_knots, fit_model.len_scale)
        if rank == 0:
            print('estimated range:',range_at_knots)
        # check for unreasonably large values, intialize at some smaller ones
        range_upper_bound = 4
        if len(np.where(range_at_knots > range_upper_bound)[0]) > 0:
            if rank == 0: print('estimated range >', range_upper_bound, ' at:', np.where(range_at_knots > range_upper_bound)[0])
            if rank == 0: print('range at those knots set to be at', range_upper_bound)
            range_at_knots[np.where(range_at_knots > range_upper_bound)[0]] = range_upper_bound
        # check for unreasonably small values, initialize at some larger ones
        range_lower_bound = 0.01
        if len(np.where(range_at_knots < range_lower_bound)[0]) > 0:
            if rank == 0: print('estimated range <', range_lower_bound, ' at:', np.where(range_at_knots < range_lower_bound)[0])
            if rank == 0: print('range at those knots set to be at', range_lower_bound)
            range_at_knots[np.where(range_at_knots < range_lower_bound)[0]] = range_lower_bound    
        
        # g(Z)
        range_vec = gaussian_weight_matrix @ range_at_knots
        K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')
        Z         = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)), 
                                                        cov  = K,
                                                        size = Nt).T
        W         = g(Z)

        # phi
        phi_at_knots = np.array([0.5] * k)
        phi_vec = gaussian_weight_matrix @ phi_at_knots

        # S ~ Stable
        if size == 1:
            S_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
            for t in np.arange(Nt):
                obs_idx_1t  = np.where(miss_matrix[:,t] == False)[0]

                pY_1t = pCGP(Y[obs_idx_1t, t], p,
                             u_vec[obs_idx_1t], sigma_vec[obs_idx_1t], ksi_vec[obs_idx_1t])

                # S_at_knots[:,t] = np.min(qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_vec[obs_idx_1t], tau
                #                                 ) / 2)**(1/phi_at_knots)

                S_at_knots[:,t] = np.median(qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_vec[obs_idx_1t], tau
                                                ) / W[obs_idx_1t, t])**(1/phi_at_knots)
        if size > 1:
            comm.Barrier()
            obs_idx_1t  = np.where(miss_matrix[:,rank] == False)[0]
            pY_1t = pCGP(Y[obs_idx_1t, rank], p,
                         u_vec[obs_idx_1t], sigma_vec[obs_idx_1t], ksi_vec[obs_idx_1t])
            X_1t  = qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_vec[obs_idx_1t], tau)
            # S_1t  = np.min(X_1t/2) ** (1/phi_at_knots)
            S_1t  = np.median(X_1t / W[obs_idx_1t, rank]) ** (1/phi_at_knots)

            S_gathered = comm.gather(S_1t, root = 0)
            S_at_knots = np.array(S_gathered).T if rank == 0 else None
            S_at_knots = comm.bcast(S_at_knots, root = 0)
        
        # X_star = R^phi * g(Z)
        X_star = ((wendland_weight_matrix @ S_at_knots).T ** phi_vec).T * W
    else:
        # We will continue with the last iteration from the traceplot
        pass

    # %% Load/Hardcode parameters -------------------------------------------------------------------------------------
    # Load/Hardcode parameters

    # True parameter values as with the simulation
    np.random.seed(data_seed)

    u_matrix = np.full(shape = (Ns, Nt), fill_value = 20.0)
    u_vec    = u_matrix[:,rank]

    Beta_logsigma       = np.array([0.0, 0.25])
    Beta_ksi            = np.array([0.0, 0.1])
    sigma_Beta_logsigma = 1
    sigma_Beta_ksi      = 1

    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z
    phi_at_knots = 0.65 - np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    range_vec = gaussian_weight_matrix @ range_at_knots
    K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                        coords = sites_xy, kappa = nu, cov_model = "matern")
    Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W         = g(Z) 

    phi_vec    = gaussian_weight_matrix @ phi_at_knots
    S_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        S_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
    R_at_sites = wendland_weight_matrix @ S_at_knots
    R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)
    
    # Nuggets
    tau = 10
    nuggets = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)),
                                                  cov  = tau**2,
                                                  size = Nt).T
    X_star       = R_phi * W
    X_truth      = X_star + nuggets

    # %% Plot Parameter Surfaces --------------------------------------------------------------------------------------
    # Plot Parameter Surface
    if rank == 0 and start_iter == 1:
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
        
        # weight from knot plots --------------------------------------------------------------------------------------
        import geopandas as gpd
        state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')

        # Define the colors for the colormap (white to red)
        # Create a LinearSegmentedColormap
        colors = ["#ffffff", "#ff0000"]
        n_bins = 50  # Number of discrete color bins
        cmap_name = "white_to_red"
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        min_w = 0
        max_w = 1
        n_ticks = 11  # (total) number of ticks
        ticks = np.linspace(min_w, max_w, n_ticks).round(3)

        idx = 5
        gaussian_weights_for_plot = gaussian_weight_matrix_for_plot[:,idx]
        wendland_weights_for_plot = wendland_weight_matrix_for_plot[:,idx]

        fig, axes = plt.subplots(1,2)
        state_map.boundary.plot(ax=axes[0], color = 'black', linewidth = 0.5)
        heatmap = axes[0].imshow(wendland_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap = colormap, vmin = min_w, vmax = max_w,
                            interpolation='nearest', 
                            extent = [minX, maxX, maxY, minY])
        axes[0].invert_yaxis()                    
        # axes[0].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        axes[0].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        axes[0].set_xlim(minX, maxX)
        axes[0].set_ylim(minY, maxY)
        axes[0].set_aspect('equal', 'box')
        axes[0].title.set_text('wendland weights knot ' + str(idx))

        state_map.boundary.plot(ax=axes[1], color = 'black', linewidth = 0.5)
        heatmap = axes[1].imshow(gaussian_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap = colormap, vmin = min_w, vmax = max_w,
                            interpolation='nearest', 
                            extent = [minX, maxX, maxY, minY])
        axes[1].invert_yaxis()                    
        # axes[1].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        axes[1].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        axes[1].set_xlim(minX, maxX)
        axes[1].set_ylim(minY, maxY)
        axes[1].set_aspect('equal', 'box')
        axes[1].title.set_text('gaussian weights knot ' + str(idx))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
        plt.savefig('weights.pdf')
        plt.show()
        plt.close()
        # -------------------------------------------------------------------------------------------------------------
        
    
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
        plt.savefig('stations.pdf',bbox_inches="tight")
        plt.close()


        # 2. Elevation
        fig, ax = plt.subplots()
        elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                                  cmap = 'bwr')
        ax.set_aspect('equal', 'box')
        plt.colorbar(elev_scatter)
        # plt.show()
        plt.savefig('station_elevation.pdf')
        plt.close()       
    
    
        # 3. phi surface
        # heatplot of phi surface
        phi_vec_for_plot = (gaussian_weight_matrix_for_plot @ phi_at_knots).round(3)
        graph, ax = plt.subplots()
        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('heatmap phi surface.pdf')
        plt.close()

    
        # 4. Plot range surface
        # heatplot of range surface
        range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('heatmap range surface.pdf')
        plt.close()
    
    
        # 5. GP Surfaces
        logsigma_vec = ((C_logsigma.T @ Beta_logsigma).T)[:,rank]
        ksi_vec      = ((C_ksi.T @ Beta_ksi).T)[:,rank]

        def my_ceil(a, precision=0):
            return np.true_divide(np.ceil(a * 10**precision), 10**precision)

        def my_floor(a, precision=0):
            return np.true_divide(np.floor(a * 10**precision), 10**precision)

        # Scale # -------------------------------------------------------------------------------------
        ## logsigma(s) plot stations
        vmin = min(my_floor(min(logsigma_estimates), 1), my_floor(min(logsigma_vec), 1))
        vmax = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max(logsigma_vec), 1))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        logsigma_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV logsigma estimates')
        logsigma_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_vec, norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline logsigma fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(logsigma_est_scatter, cax = cbar_ax)
        plt.savefig('initial_logsigma_estimates.pdf')
        plt.close()

        # Shape # -------------------------------------------------------------------------------------
        # ksi(s) plot stations
        vmin = min(my_floor(min(ksi_estimates), 1), my_floor(min(ksi_vec), 1))
        vmax = max(my_ceil(max(ksi_estimates), 1), my_ceil(max(ksi_vec), 1))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        ksi_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV ksi estimates')
        ksi_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_vec, norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline ksi fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ksi_est_scatter, cax = cbar_ax)
        plt.savefig('initial_ksi_estimates.pdf')
        plt.close()

    # %% Adaptive Update & Block Update Setup -------------------------------------------------------------------------
    # Adaptive Update & Block Update Setup
        
    # Block Update Specification --------------------------------------------------------------------------------------
    
    if norm_pareto == 'standard':
        phi_block_idx_size   = 4
        range_block_idx_size = 4
    
    if norm_pareto == 'shifted':
        phi_block_idx_size = 4
        range_block_idx_size = 4

    # Create Coefficient Index Blocks - each block size does not exceed size specified above

    ## phi
    phi_block_idx_dict = {}
    lst = list(range(k))
    for i in range(0, k, phi_block_idx_size):
        start_index = i
        end_index   = i + phi_block_idx_size
        key         = 'phi_block_idx_'+str(i//phi_block_idx_size+1)
        phi_block_idx_dict[key] = lst[start_index:end_index]

    ## range
    range_block_idx_dict = {}
    lst = list(range(k))
    for i in range(0, k, range_block_idx_size):
        start_index = i
        end_index   = i + range_block_idx_size
        key         = 'range_block_idx_'+str(i//range_block_idx_size+1)
        range_block_idx_dict[key] = lst[start_index:end_index]

    # Adaptive Update: tuning constants -------------------------------------------------------------------------------

    c_0 = 1
    c_1 = 0.8
    offset = 3 # the iteration offset: trick the updater thinking chain is longer
    # r_opt_1d = .41
    # r_opt_2d = .35
    # r_opt = 0.234 # asymptotically
    r_opt = .35
    adapt_size = 10

    # Adaptive Update: Proposal Variance Scalar and Covariance Matrix -------------------------------------------------

    if start_iter == 1: # initialize the proposal scalar variance and covariance
        import proposal_cov as pc
        
        # sigma_m: proposal scalar variance for St, Zt, phi, range, tau, marginal Y, and regularization terms ---------
        S_log_cov               = pc.S_log_cov               if pc.S_log_cov               is not None else np.tile(0.05*np.eye(k)[:,:,None], reps = (1,1,Nt))
        Z_cov                   = pc.Z_cov                   if pc.Z_cov                   is not None else np.tile(np.eye(Ns)[:,:,None],reps = (1,1,Nt))
        tau_var                 = pc.tau_var                 if pc.tau_var                 is not None else 1
        sigma_Beta_logsigma_var = pc.sigma_Beta_logsigma_var if pc.sigma_Beta_logsigma_var is not None else 1
        sigma_Beta_ksi_var      = pc.sigma_Beta_ksi_var      if pc.sigma_Beta_ksi_var      is not None else 1
               
        # St
        sigma_m_sq_St_list = [np.mean(np.diag(S_log_cov[:,:,t])) for t in range(Nt)] if rank == 0 and norm_pareto == 'shifted' else None
        sigma_m_sq_St_list = [(np.diag(S_log_cov[:,:,t])) for t in range(Nt)]        if rank == 0 and norm_pareto == 'standard' else None
        sigma_m_sq_St      = comm.scatter(sigma_m_sq_St_list, root = 0) if size>1 else sigma_m_sq_St_list[0]
        
        # Zt
        sigma_m_sq_Zt_list = [(np.diag(Z_cov[:,:,t])) for t in range(Nt)] if rank == 0 else None
        sigma_m_sq_Zt      = comm.scatter(sigma_m_sq_Zt_list, root = 0) if size>1 else sigma_m_sq_Zt_list[0]
        
        if rank == 0:
            sigma_m_sq = {}

            # phi
            for key in phi_block_idx_dict.keys(): sigma_m_sq[key] = (2.4**2)/len(phi_block_idx_dict[key])
            
            # range
            for key in range_block_idx_dict.keys(): sigma_m_sq[key] = (2.4**2)/len(range_block_idx_dict[key])
            
            # tau
            sigma_m_sq['tau'] = tau_var

            # marginal Y
            sigma_m_sq['Beta_logsigma'] = (2.4**2)/Beta_logsigma_m
            sigma_m_sq['Beta_ksi']      = (2.4**2)/Beta_ksi_m
            
            # regularization
            sigma_m_sq['sigma_Beta_logsigma'] = sigma_Beta_logsigma_var
            sigma_m_sq['sigma_Beta_ksi']      = sigma_Beta_ksi_var

        # Sigma0: proposal covariance matrix for phi, range, and marginal Y -------------------------------------------
        phi_cov                 = pc.phi_cov           if pc.phi_cov           is not None else 1e-2 * np.identity(k)
        range_cov               = pc.range_cov         if pc.range_cov         is not None else 0.5  * np.identity(k)
        Beta_logsigma_cov       = pc.Beta_logsigma_cov if pc.Beta_logsigma_cov is not None else 1e-6 * np.identity(Beta_logsigma_m)
        Beta_ksi_cov            = pc.Beta_ksi_cov      if pc.Beta_ksi_cov      is not None else 1e-7 * np.identity(Beta_ksi_m)
        
        if rank == 0:
            Sigma_0 = {}

            # phi
            phi_block_cov_dict = {}
            for key in phi_block_idx_dict.keys():
                start_idx                    = phi_block_idx_dict[key][0]
                end_idx                      = phi_block_idx_dict[key][-1]+1
                phi_block_cov_dict[key] = phi_cov[start_idx:end_idx, start_idx:end_idx]
            Sigma_0.update(phi_block_cov_dict)

            # range
            range_block_cov_dict = {}
            for key in range_block_idx_dict.keys():
                start_idx                      = range_block_idx_dict[key][0]
                end_idx                        = range_block_idx_dict[key][-1]+1
                range_block_cov_dict[key] = range_cov[start_idx:end_idx, start_idx:end_idx]
            Sigma_0.update(range_block_cov_dict)

            # marginal Y
            Sigma_0['Beta_logsigma'] = Beta_logsigma_cov
            Sigma_0['Beta_ksi']      = Beta_ksi_cov

        # Checking dimensions -----------------------------------------------------------------------------------------
        assert k               == phi_cov.shape[0]
        assert k               == range_cov.shape[0]
        assert k               == S_log_cov.shape[0]
        assert Nt              == S_log_cov.shape[2]
        assert Ns              == Z_cov.shape[0]
        assert Beta_logsigma_m == Beta_logsigma_cov.shape[0]
        assert Beta_ksi_m      == Beta_ksi_cov.shape[0]
    else: # pickle load the Proposal Variance Scalar, Covariance Matrix
        
        # sigma_m: proposal scalar variance for St, Zt, phi, range, tau, marginal Y, and regularization terms ---------
        ## St
        if rank == 0:
            with open('sigma_m_sq_St_list.pkl', 'rb') as file: sigma_m_sq_St_list = pickle.load(file)
        else:
            sigma_m_sq_St_list = None
        if size != 1: sigma_m_sq_St = comm.scatter(sigma_m_sq_St_list, root = 0)

        ## Zt
        if rank == 0: 
            with open('sigma_m_sq_Zt_list.pkl', 'rb') as file: sigma_m_sq_Zt_list = pickle.load(file)
        else:
            sigma_m_sq_Zt_list = None
        sigma_m_sq_Zt = comm.scatter(sigma_m_sq_Zt_list, root = 0) if size>1 else sigma_m_sq_Zt_list[0]

        ## phi, range, tau, marginal Y, regularizations
        if rank == 0: 
            with open('sigma_m_sq.pkl','rb') as file: sigma_m_sq = pickle.load(file)
        
        # Sigma0: proposal covariance matrix for phi, range, and marginal Y -------------------------------------------
        if rank == 0:
            with open('Sigma_0.pkl', 'rb') as file:   Sigma_0    = pickle.load(file)

    # Adaptive Update: Counter ----------------------------------------------------------------------------------------
    
    ## St   
    if norm_pareto == 'shifted':     
        num_accepted_St_list = [0] * size if rank == 0 else None
        if size != 1: num_accepted_St = comm.scatter(num_accepted_St_list, root = 0)

    if norm_pareto == 'standard':
        num_accepted_St_list = [[0] * k] * size if rank == 0 else None
        num_accepted_St      = comm.scatter(num_accepted_St_list, root= 0) if size>1 else num_accepted_St_list[0]
    
    ## Zt
    num_accepted_Zt_list = [[0] * Ns] * size if rank == 0 else None
    num_accepted_Zt      = comm.scatter(num_accepted_Zt_list, root = 0) if size>1 else num_accepted_Zt_list[0]

    ## Other variables: phi, range, tau, marginal Y, regularizaiton
    if rank == 0:
        num_accepted = {}
        # phi
        for key in phi_block_idx_dict.keys(): num_accepted[key] = 0
        # range
        for key in range_block_idx_dict.keys(): num_accepted[key] = 0
        # tau
        num_accepted['tau'] = 0
        # marginal Y
        num_accepted['Beta_logsigma'] = 0
        num_accepted['Beta_ksi']      = 0
        # regularization
        num_accepted['sigma_Beta_mu0']      = 0
        num_accepted['sigma_Beta_mu1']      = 0
        num_accepted['sigma_Beta_logsigma'] = 0
        num_accepted['sigma_Beta_ksi']      = 0
        
    # %% Storage and Initialize ---------------------------------------------------------------------------------------
    # Storage and Initialize
            
    # Storage for traceplots
    if start_iter == 1:
        loglik_trace              = np.full(shape = (n_iters, 1), fill_value = np.nan)               if rank == 0 else None # overall likelihood
        loglik_detail_trace       = np.full(shape = (n_iters, 5), fill_value = np.nan)               if rank == 0 else None # detail likelihood
        S_trace_log               = np.full(shape = (n_iters, k, Nt), fill_value = np.nan)           if rank == 0 else None # log(S)
        phi_knots_trace           = np.full(shape = (n_iters, k), fill_value = np.nan)               if rank == 0 else None # phi_at_knots
        range_knots_trace         = np.full(shape = (n_iters, k), fill_value = np.nan)               if rank == 0 else None # range_at_knots
        Beta_logsigma_trace       = np.full(shape = (n_iters, Beta_logsigma_m), fill_value = np.nan) if rank == 0 else None # logsigma Covariate Coefficients
        Beta_ksi_trace            = np.full(shape = (n_iters, Beta_ksi_m), fill_value = np.nan)      if rank == 0 else None # ksi Covariate Coefficients
        sigma_Beta_logsigma_trace = np.full(shape = (n_iters, 1), fill_value = np.nan)               if rank == 0 else None # prior sd for beta_logsigma's
        sigma_Beta_ksi_trace      = np.full(shape = (n_iters, 1), fill_value = np.nan)               if rank == 0 else None # prior sd for beta_ksi's
        Y_trace                   = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
        tau_trace                 = np.full(shape = (n_iters, 1), fill_value = np.nan)               if rank == 0 else None
        Z_trace                   = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
        # X_star_trace              = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
        # X_trace                   = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
    else:
        loglik_trace              = np.load('loglik_trace.npy')              if rank == 0 else None
        loglik_detail_trace       = np.load('loglik_detail_trace.npy')       if rank == 0 else None
        S_trace_log               = np.load('S_trace_log.npy')               if rank == 0 else None
        phi_knots_trace           = np.load('phi_knots_trace.npy')           if rank == 0 else None
        range_knots_trace         = np.load('range_knots_trace.npy')         if rank == 0 else None
        Beta_logsigma_trace       = np.load('Beta_logsigma_trace.npy')       if rank == 0 else None
        Beta_ksi_trace            = np.load('Beta_ksi_trace.npy')            if rank == 0 else None
        sigma_Beta_logsigma_trace = np.load('sigma_Beta_logsigma_trace.npy') if rank == 0 else None
        sigma_Beta_ksi_trace      = np.load('sigma_Beta_ksi_trace.npy')      if rank == 0 else None
        Y_trace                   = np.load('Y_trace.npy')                   if rank == 0 else None
        tau_trace                 = np.load('tau_trace.npy')                 if rank == 0 else None
        Z_trace                   = np.load('Z_trace.npy')                   if rank == 0 else None
        # X_star_trace              = np.load('X_star_trace.npy')
        # X_trace                   = np.load('X_trace.npy')

    # Initialize Parameters
    if start_iter == 1:
        # Initialize at the truth/at other values
        S_matrix_init_log        = np.log(S_at_knots)  if rank == 0 else None
        phi_knots_init           = phi_at_knots        if rank == 0 else None
        range_knots_init         = range_at_knots      if rank == 0 else None
        Beta_logsigma_init       = Beta_logsigma       if rank == 0 else None
        Beta_ksi_init            = Beta_ksi            if rank == 0 else None
        sigma_Beta_logsigma_init = sigma_Beta_logsigma if rank == 0 else None
        sigma_Beta_ksi_init      = sigma_Beta_ksi      if rank == 0 else None
        Y_matrix_init            = Y                   if rank == 0 else None
        tau_init                 = tau                 if rank == 0 else None
        Z_init                   = Z                   if rank == 0 else None
        # X_star_init              = X_star              if rank == 0 else None
        # X_init                   = X                   if rank == 0 else None
        if rank == 0: # store initial value into first row of traceplot
            S_trace_log[0,:,:]             = S_matrix_init_log # matrix (k, Nt)
            phi_knots_trace[0,:]           = phi_knots_init
            range_knots_trace[0,:]         = range_knots_init
            Beta_logsigma_trace[0,:]       = Beta_logsigma_init
            Beta_ksi_trace[0,:]            = Beta_ksi_init
            sigma_Beta_logsigma_trace[0,:] = sigma_Beta_logsigma_init
            sigma_Beta_ksi_trace[0,:]      = sigma_Beta_ksi_init
            Y_trace[0,:,:]                 = Y_matrix_init
            tau_trace[0,:]                 = tau_init
            Z_trace[0,:,:]                 = Z_init
            # X_star_trace[0,:,:]            = X_star_init
            # X_trace[0,:,:]                 = X_init
    else:
        last_iter = start_iter - 1
        S_matrix_init_log        = S_trace_log[last_iter,:,:]             if rank == 0 else None
        phi_knots_init           = phi_knots_trace[last_iter,:]           if rank == 0 else None
        range_knots_init         = range_knots_trace[last_iter,:]         if rank == 0 else None
        Beta_logsigma_init       = Beta_logsigma_trace[last_iter,:]       if rank == 0 else None
        Beta_ksi_init            = Beta_ksi_trace[last_iter,:]            if rank == 0 else None
        sigma_Beta_logsigma_init = sigma_Beta_logsigma_trace[last_iter,0] if rank == 0 else None # must be value, can't be array([value])
        sigma_Beta_ksi_init      = sigma_Beta_ksi_trace[last_iter,0]      if rank == 0 else None # must be value, can't be array([value])
        Y_matrix_init            = Y_trace[last_iter,:,:]                 if rank == 0 else None
        tau_init                 = tau_trace[last_iter,:]                 if rank == 0 else None
        Z_init                   = Z_trace[last_iter,:,:]                 if rank == 0 else None
        # X_star_init              = X_star_trace[last_iter,:,:]            if rank == 0 else None
        # X_init                   = X_trace[last_iter,:,:]                 if rank == 0 else None
    
    # Set Current Values
    ## Marginal Model -------------------------------------------------------------------------------------------------
    ## ---- GPD covariate coefficients --> GPD surface ----
    Beta_logsigma_current = comm.bcast(Beta_logsigma_init, root = 0)
    Beta_ksi_current      = comm.bcast(Beta_ksi_init, root = 0)
    Scale_vec_current     = np.exp((C_logsigma.T @ Beta_logsigma_current).T)[:,rank]
    Shape_vec_current     = ((C_ksi.T @ Beta_ksi_current).T)[:,rank]

    ## ---- GPD covariate coefficients prior variance ----
    sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_init, root = 0)
    sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_init, root = 0)

    ## Copula Model ---------------------------------------------------------------------------------------------------
    ## ---- S Stable ----
    # note: directly comm.scatter an numpy nd array along an axis is tricky,
    #       hence we first "redundantly" broadcast an entire S_matrix then split
    S_matrix_init_log = comm.bcast(S_matrix_init_log, root = 0) # matrix (k, Nt)
    S_current_log     = np.array(S_matrix_init_log[:,rank]) # vector (k,)
    R_vec_current     = wendland_weight_matrix @ np.exp(S_current_log)

    ## ---- Z ----
    Z_matrix_init = comm.bcast(Z_init, root = 0)    # matrix (Ns, Nt)
    Z_1t_current = np.array(Z_matrix_init[:,rank]) # vector (Ns,)

    ## ---- phi ----
    phi_knots_current = comm.bcast(phi_knots_init, root = 0)
    phi_vec_current   = gaussian_weight_matrix @ phi_knots_current

    ## ---- range_vec (length_scale) ----
    range_knots_current = comm.bcast(range_knots_init, root = 0)
    range_vec_current   = gaussian_weight_matrix @ range_knots_current
    K_current           = ns_cov(range_vec = range_vec_current,
                                 sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
    cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

    ## ---- Nugget standard deviation: tau ----
    tau_current = comm.bcast(tau_init, root = 0)

    ## ---- X_star ----
    X_star_1t_current = (R_vec_current ** phi_vec_current) * g(Z_1t_current)

    ## ---- Y (Ns, Nt) ----
    Y_matrix_init = comm.bcast(Y_matrix_init, root = 0) # (Ns, Nt)
    Y_1t_current  = Y_matrix_init[:,rank]               # (Ns,)
    
    # initial imputation
    if start_iter == 1:
        X_1t_imputed = X_star_1t_current[miss_idx_1t] + \
                       scipy.stats.norm.rvs(loc = 0, scale = tau_current, size = len(miss_idx_1t), random_state = random_generator)
        Y_1t_imputed = qCGP(pRW(X_1t_imputed, phi_vec_current[miss_idx_1t], gamma_vec[miss_idx_1t], tau_current),
                            p, u_vec[miss_idx_1t], Scale_vec_current[miss_idx_1t], Shape_vec_current[miss_idx_1t])
        Y_1t_current[miss_idx_1t] = Y_1t_imputed
        assert 0 == len(np.where(np.isnan(Y_1t_current))[0])

        Y_1t_gathered = comm.gather(Y_1t_current, root = 0)
        if rank == 0: Y_trace[0, :, :] = np.array(Y_1t_gathered).T

    # Note:
    #   The censor/exceedance index NEED TO CHANGE whenever we do imputation
    censored_idx_1t_current = np.where(Y_1t_current <= u_vec)[0]
    exceed_idx_1t_current   = np.where(Y_1t_current  > u_vec)[0]

    ## ---- X_1t (Ns,) ----
    X_1t_current  = qRW(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                        phi_vec_current, gamma_vec, tau_current)
    dX_1t_current = dRW(X_1t_current, phi_vec_current, gamma_vec, tau_current)
   

    # %% Metropolis-Hasting Updates -----------------------------------------------------------------------------------
    # Metropolis-Hasting Updates
    comm.Barrier() # Blocking before the update starts

    if rank == 0:
        start_time = time.time()
        print('started on:', strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))

    # # With Jacobian
    # llik_1t_current = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
    #                                   R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
    #                                   X_1t_current, X_star_1t_current, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) + \
    #                   X_star_conditional_ll_1t(X_star_1t_current, R_vec_current, phi_vec_current, K_current,
    #                                            Z_1t_current)
    
    # Without Jacobian
    llik_1t_current = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                       R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
                                       X_1t_current, X_star_1t_current, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
                    + scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov = K_current)
    
    if np.isfinite(llik_1t_current): 
        llik_1t_current_gathered = comm.gather(llik_1t_current, root = 0)
        if rank == 0: loglik_trace[0, 0] = np.sum(llik_1t_current_gathered)
    else: print('initial likelihood non finite', 'rank:', rank)

    for iter in range(start_iter, n_iters):
        # %% Update St ------------------------------------------------------------------------------------------------
        ###########################################################
        #### ----- Update St ----- Parallelized Across Nt time ####
        ###########################################################
        for i in range(k):
            # propose new Stable St at knot i (No need truncation now?) -----------------------------------------------
            change_idx = np.array([i])

            S_proposal_log             = S_current_log.copy()
            S_proposal_log[change_idx] = S_current_log[change_idx] + np.sqrt(sigma_m_sq_St[i]) * random_generator.normal(0.0, 1.0, size = 1)
            
            R_vec_proposal             = wendland_weight_matrix @ np.exp(S_proposal_log)
            X_star_1t_proposal         = (R_vec_proposal ** phi_vec_current) * g(Z_1t_current)

            # Data Likelihood -----------------------------------------------------------------------------------------
            
            # # With Jacobian
            # llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
            #                                     R_vec_proposal, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
            #                                     X_1t_current, X_star_1t_proposal, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
            #                  + X_star_conditional_ll_1t(X_star_1t_proposal, R_vec_proposal, phi_vec_current, K_current,
            #                                             Z_1t_current)
            
            # Without Jacobian
            llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                R_vec_proposal, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
                                                X_1t_current, X_star_1t_proposal, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
                             + scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov = K_current)
            
            # Prior Density -------------------------------------------------------------------------------------------
            lprior_1t_current  = np.sum(scipy.stats.levy.logpdf(np.exp(S_current_log),  scale = gamma) + S_current_log)
            lprior_1t_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(S_proposal_log), scale = gamma) + S_proposal_log)

            # Update --------------------------------------------------------------------------------------------------
            r = np.exp(llik_1t_proposal + lprior_1t_proposal - llik_1t_current - lprior_1t_current)
            u = random_generator.uniform()
            if np.isfinite(r) and r >= u:
                num_accepted_St[i] += 1
                S_current_log       = S_proposal_log.copy()
                R_vec_current       = wendland_weight_matrix @ np.exp(S_current_log)
                X_star_1t_current   = (R_vec_current ** phi_vec_current) * g(Z_1t_current)
                llik_1t_current     = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        S_current_log_gathered = comm.gather(S_current_log, root = 0)
        if rank == 0: S_trace_log[iter,:,:]  = np.vstack(S_current_log_gathered).T
        
        comm.Barrier()

        # %% Update Zt ------------------------------------------------------------------------------------------------
        ###########################################################
        ####                 Update Zt                         ####
        ###########################################################
        for i in range(Ns):
            # propose new Zt at site i  -------------------------------------------------------------------------------
            idx                = np.array([i])
            Z_1t_proposal      = Z_1t_current.copy()
            Z_1t_proposal[idx] = Z_1t_current[idx] + np.sqrt(sigma_m_sq_Zt[i]) * random_generator.normal(0.0, 1.0, size = 1)
            X_star_1t_proposal = (R_vec_current ** phi_vec_current) * g(Z_1t_proposal)

            # Data Likelihood -----------------------------------------------------------------------------------------
            
            # # With Jacobian
            # llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
            #                                     R_vec_current, Z_1t_proposal, phi_vec_current, gamma_vec, tau_current,
            #                                     X_1t_current, X_star_1t_proposal, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
            #                  + X_star_conditional_ll_1t(X_star_1t_proposal, R_vec_current, phi_vec_current, K_current,
            #                                             Z_1t_proposal)
            
            # Without Jacobian
            llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                R_vec_current, Z_1t_proposal, phi_vec_current, gamma_vec, tau_current,
                                                X_1t_current, X_star_1t_proposal, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
                             + scipy.stats.multivariate_normal.logpdf(Z_1t_proposal, mean = None, cov = K_current)
            
            # Update --------------------------------------------------------------------------------------------------
            r = np.exp(llik_1t_proposal - llik_1t_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted_Zt[i] += 1
                Z_1t_current      = Z_1t_proposal.copy()
                X_star_1t_current = (R_vec_current ** phi_vec_current) * g(Z_1t_current)
                llik_1t_current   = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        Z_1t_current_gathered = comm.gather(Z_1t_current, root = 0)
        if rank == 0: Z_trace[iter,:,:]  = np.vstack(Z_1t_current_gathered).T
        
        comm.Barrier()

        # %% Update phi ------------------------------------------------------------------------------------------------
        ############################################################
        ####                 Update phi                         ####
        ############################################################
        for key in phi_block_idx_dict.keys():
            # Propose new phi_block at the change_indices -------------------------------------------------------------
            idx = np.array(phi_block_idx_dict[key])
            if rank == 0:
                phi_knots_proposal = phi_knots_current.copy()
                phi_knots_proposal[idx] += np.sqrt(sigma_m_sq[key]) * random_generator.multivariate_normal(np.zeros(len(idx)), Sigma_0[key])
            else:
                phi_knots_proposal = None
            phi_knots_proposal     = comm.bcast(phi_knots_proposal, root = 0)

            # Data Likelihood -----------------------------------------------------------------------------------------
            if not all(0 < phi < 1 for phi in phi_knots_proposal):
                llik_1t_proposal = np.NINF
            else:
                phi_vec_proposal       = gaussian_weight_matrix @ phi_knots_proposal
                X_star_1t_proposal     = (R_vec_current ** phi_vec_proposal) * g(Z_1t_current)
                X_1t_proposal = qRW(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                                    phi_vec_proposal, gamma_vec, tau_current)
                dX_1t_proposal = dRW(X_1t_proposal, phi_vec_proposal, gamma_vec, tau_current)

                # Without Jacobian
                llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                    R_vec_current, Z_1t_current, phi_vec_proposal, gamma_vec, tau_current,
                                                    X_1t_proposal, X_star_1t_proposal, dX_1t_proposal, censored_idx_1t_current, exceed_idx_1t_current) \
                                 + scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov = K_current)

            # Update --------------------------------------------------------------------------------------------------
            phi_accepted = False
            llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
            llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
            if rank == 0:
                llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(scipy.stats.beta.logpdf(phi_knots_current, a = 5, b = 5))
                llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))
                r = np.exp(llik_proposal - llik_current)
                if np.isfinite(r) and r >= random_generator.uniform():
                    num_accepted[key] += 1
                    phi_accepted       = True
            phi_accepted = comm.bcast(phi_accepted, root = 0)
            
            if phi_accepted:
                phi_knots_current = phi_knots_proposal.copy()
                phi_vec_current   = phi_vec_proposal.copy()
                X_star_1t_current = X_star_1t_proposal.copy()
                X_1t_current      = X_1t_proposal.copy()
                dX_1t_current     = dX_1t_proposal.copy()
                llik_1t_current   = llik_1t_proposal
        
        # Save --------------------------------------------------------------------------------------------------------
        if rank == 0: phi_knots_trace[iter,:] = phi_knots_current.copy()
        comm.Barrier()

        # %% Update rho ------------------------------------------------------------------------------------------------
        ############################################################
        ####                 Update rho                         ####
        ############################################################
        for key in range_block_idx_dict.keys():
            # Propose new range_block at the change_indices -----------------------------------------------------------
            idx = np.array(range_block_idx_dict[key])
            if rank == 0:
                range_knots_proposal = range_knots_current.copy()
                range_knots_proposal[idx] += np.sqrt(sigma_m_sq[key]) * random_generator.multivariate_normal(np.zeros(len(idx)), Sigma_0[key])
            else:
                range_knots_proposal = None
            range_knots_proposal     = comm.bcast(range_knots_proposal, root = 0)

            # Data Likelihood -----------------------------------------------------------------------------------------
            if not all(rho > 0 for rho in range_knots_proposal):
                llik_1t_proposal = np.NINF
            else:
                range_vec_proposal = gaussian_weight_matrix @ range_knots_proposal
                K_proposal = ns_cov(range_vec = range_vec_proposal,
                                    sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
                # Without Jacobian
                llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                    R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
                                                    X_1t_current, X_star_1t_current, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current) \
                                 + scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov = K_proposal)
            
            # Update --------------------------------------------------------------------------------------------------
            range_accepted = False
            llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
            llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
            if rank == 0:
                llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(range_knots_current, loc = 0, scale = 2))
                llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_proposal, loc = 0, scale = 2))
                r = np.exp(llik_proposal - llik_current)
                if np.isfinite(r) and r >= random_generator.uniform():
                    num_accepted[key] += 1
                    range_accepted     = True
            range_accepted = comm.bcast(range_accepted, root = 0)
            
            if range_accepted:
                range_knots_current = range_knots_proposal.copy()
                K_current           = K_proposal.copy()
                llik_1t_current     = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        if rank == 0: range_knots_trace[iter,:] = range_knots_current.copy()
        comm.Barrier()

        # %% Update tau ------------------------------------------------------------------------------------------------
        ############################################################
        ####                 Update tau                         ####
        ############################################################
        # Propose new tau ---------------------------------------------------------------------------------------------
        if rank == 0:
            tau_proposal = np.sqrt(sigma_m_sq['tau']) * scipy.stats.norm.rvs(loc = tau_current, random_state = random_generator)
        else:
            tau_proposal = None
        tau_proposal = comm.bcast(tau_proposal, root = 0)

        # Data Likelihood ---------------------------------------------------------------------------------------------
        if not tau_proposal > 0:
            llik_1t_proposal = np.NINF
        else:
            X_1t_proposal = qRW(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                                phi_vec_current, gamma_vec, tau_proposal)
            dX_1t_proposal = dRW(X_1t_proposal, phi_vec_current, gamma_vec, tau_proposal)

            # Without Jacobian
            llik_1t_proposal = Y_censored_ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_proposal,
                                                X_1t_proposal, X_star_1t_current, dX_1t_proposal, censored_idx_1t_current, exceed_idx_1t_current) \
                                + scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov = K_current)

        # Update ------------------------------------------------------------------------------------------------------
        tau_accepted = False
        llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
        llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
        if rank == 0:
            llik_current  = np.sum(llik_1t_current_gathered)  + np.log(dhalft(tau_current, nu = 1, mu = 0, sigma = 5))
            llik_proposal = np.sum(llik_1t_proposal_gathered) + np.log(dhalft(tau_proposal, nu = 1, mu = 0, sigma = 5)) if tau_proposal > 0 else np.NINF
            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['tau'] += 1
                tau_accepted         = True
        tau_accepted = comm.bcast(tau_accepted, root = 0)
        
        if tau_accepted:
            tau_current     = tau_proposal
            X_1t_current    = X_1t_proposal.copy()
            dX_1t_current   = dX_1t_proposal.copy()
            llik_1t_current = llik_1t_proposal
        
        # Save --------------------------------------------------------------------------------------------------------
        if rank == 0: tau_trace[iter,:] = tau_current
        comm.Barrier()

        # %% Update GPD ------------------------------------------------------------------------------------------------
        ############################################################
        ####                 Update GPD                         ####
        ############################################################


        # %% After iteration likelihood
        ######################################################################
        #### ----- Keeping track of likelihood after this iteration ----- ####
        ######################################################################

        llik_1t_current_gathered = comm.gather(llik_1t_current, root = 0)
        if rank == 0: loglik_trace[iter, 0] = np.sum(llik_1t_current_gathered)

        # # With the Jacobian
        # censored_ll_1t, exceed_ll_1t = Y_censored_ll_1t_detail(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
        #                                                        R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
        #                                                        X_1t_current, X_star_1t_current, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current)
        # D_gauss_ll_1t, log_J_1_1t, log_J_2_1t = X_star_conditional_ll_1t_detail(X_star_1t_current, R_vec_current, phi_vec_current, K_current, Z_1t_current)
        # censored_ll_gathered = comm.gather(censored_ll_1t, root = 0)
        # exceed_ll_gathered   = comm.gather(exceed_ll_1t,   root = 0)
        # D_gauss_ll_gathered  = comm.gather(D_gauss_ll_1t,  root = 0)
        # log_J_1_gathered     = comm.gather(log_J_1_1t,     root = 0)
        # log_J_2_gathered     = comm.gather(log_J_2_1t,     root = 0)
        # if rank == 0: loglik_detail_trace[iter, :] = np.sum(np.array([censored_ll_gathered, exceed_ll_gathered,
        #                                                               D_gauss_ll_gathered, log_J_1_gathered, log_J_2_gathered]), 
        #                                                     axis = 1)

        # Without the Jacobian
        censored_ll_1t, exceed_ll_1t = Y_censored_ll_1t_detail(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                               R_vec_current, Z_1t_current, phi_vec_current, gamma_vec, tau_current,
                                                               X_1t_current, X_star_1t_current, dX_1t_current, censored_idx_1t_current, exceed_idx_1t_current)
        D_gauss_ll_1t = scipy.stats.multivariate_normal.logpdf(Z_1t_current, mean = None, cov=K_current)
        censored_ll_gathered = comm.gather(censored_ll_1t, root = 0)
        exceed_ll_gathered   = comm.gather(exceed_ll_1t,   root = 0)        
        D_gauss_ll_gathered  = comm.gather(D_gauss_ll_1t,  root = 0)
        if rank == 0: loglik_detail_trace[iter, [0,1,2]] = np.sum(np.array([censored_ll_gathered, exceed_ll_gathered, D_gauss_ll_gathered]),
                                                                  axis = 1)
        
        comm.Barrier()

        # %% Adaptive Update tunings
        #####################################################
        ###### ----- Adaptive Update autotunings ----- ######
        #####################################################

        if iter % adapt_size == 0:

            gamma1 = 1 / ((iter/adapt_size + offset) ** c_1)
            gamma2 = c_0 * gamma1

            # St
            if norm_pareto == 'standard':
                for i in range(k):
                    r_hat              = num_accepted_St[i]/adapt_size
                    num_accepted_St[i] = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq_St[i]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq_St[i]   = np.exp(log_sigma_m_sq_hat)
                comm.Barrier()
                sigma_m_sq_St_list     = comm.gather(sigma_m_sq_St, root = 0)
            
            # Zt
            for i in range(Ns):
                r_hat              = num_accepted_Zt[i]/adapt_size
                num_accepted_Zt[i] = 0
                log_sigma_m_sq_hat = np.log(sigma_m_sq_Zt[i]) + gamma2 * (r_hat - r_opt)
                sigma_m_sq_Zt[i]   = np.exp(log_sigma_m_sq_hat)
            comm.Barrier()
            sigma_m_sq_Zt_list = comm.gather(sigma_m_sq_Zt, root = 0)

            # phi
            if rank == 0:
                for key in phi_block_idx_dict.keys():
                    start_idx          = phi_block_idx_dict[key][0]
                    end_idx            = phi_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(phi_knots_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

            # range
            if rank == 0:
                for key in range_block_idx_dict.keys():
                    start_idx          = range_block_idx_dict[key][0]
                    end_idx            = range_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(range_knots_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

            # tau
            if rank == 0:
                r_hat               = num_accepted['tau']/adapt_size
                num_accepted['tau'] = 0
                log_sigma_m_sq_hat  = np.log(sigma_m_sq['tau']) + gamma2 * (r_hat - r_opt)
                sigma_m_sq['tau']   = np.exp(log_sigma_m_sq_hat)
            
        comm.Barrier()

        # %% Midway Printing, Drawings, and Savings
        ##############################################
        ###    Printing, Drawings, Savings       #####
        ##############################################
        thin = 10
        if rank == 0:

            if iter % 10 == 0: print('iter', iter, 'elapsed: ', round(time.time() - start_time, 1), 'seconds')

            if iter % 30 == 0 or iter == n_iters-1: # we are not saving the counters, so this must be a multiple of adapt_size!

                # Saving ----------------------------------------------------------------------------------------------
                np.save('loglik_trace',      loglik_trace)
                np.save('S_trace_log',       S_trace_log)
                np.save('Z_trace',           Z_trace)
                np.save('phi_knots_trace',   phi_knots_trace)
                np.save('range_knots_trace', range_knots_trace)
                np.save('tau_trace',         tau_trace)

                with open('iter.pkl', 'wb')               as file: pickle.dump(iter, file)
                with open('sigma_m_sq.pkl', 'wb')         as file: pickle.dump(sigma_m_sq, file)
                with open('Sigma_0.pkl', 'wb')            as file: pickle.dump(Sigma_0, file)
                with open('sigma_m_sq_St_list.pkl', 'wb') as file: pickle.dump(sigma_m_sq_St_list, file)
                with open('sigma_m_sq_Zt_list.pkl', 'wb') as file: pickle.dump(sigma_m_sq_Zt_list, file)

                # Drawing ---------------------------------------------------------------------------------------------
                
                # ---- thinning ----
                xs       = np.arange(iter)
                xs_thin  = xs[0::thin] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...
                loglik_trace_thin              = loglik_trace[0:iter:thin,:]
                loglik_detail_trace_thin       = loglik_detail_trace[0:iter:thin,:]
                S_trace_log_thin               = S_trace_log[0:iter:thin,:,:]
                Z_trace_thin                   = Z_trace[0:iter:thin,:,:]
                phi_knots_trace_thin           = phi_knots_trace[0:iter:thin,:]
                range_knots_trace_thin         = range_knots_trace[0:iter:thin,:]
                tau_trace_thin                 = tau_trace[0:iter:thin,:]

                # ---- log-likelihood ----
                plt.subplots()
                plt.plot(xs_thin2, loglik_trace_thin)
                plt.title('traceplot for log-likelihood')
                plt.xlabel('iter thinned by '+str(thin))
                plt.ylabel('loglikelihood')
                plt.savefig('trace_loglik.pdf')
                plt.close()

                # ---- log-likelihood in detail ----
                plt.subplots()
                labels = ['censored_ll', 'exceed_ll', 'D_gauss_ll', 'log_J_1', 'log_J_2']
                for i in range(5):
                    plt.plot(xs_thin2, loglik_detail_trace_thin[:,i], label = labels[i])
                    plt.annotate(labels[i], xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
                plt.title('traceplot for detail log likelihood')
                plt.xlabel('iter thinned by '+str(thin))
                plt.ylabel('log likelihood')
                plt.legend(loc = 'upper left')
                plt.savefig('trace_detailed_loglik.pdf')
                plt.close()

                # ---- S_t ----

                for t in range(Nt):
                    label_by_knot = ['knot ' + str(knot) for knot in range(k)]
                    plt.subplots()
                    plt.plot(xs_thin2, S_trace_log_thin[:,:,t], label = label_by_knot)
                    plt.legend(loc = 'upper left')
                    plt.title('traceplot for log(St) at t=' + str(t))
                    plt.xlabel('iter thinned by '+str(thin))
                    plt.ylabel('log(St)s')
                    plt.savefig('trace_St'+str(t)+'.pdf')
                    plt.close()
                
                # ---- Z_t ---- (some randomly selected subset)
                for t in range(Nt):
                    selection = np.random.choice(np.arange(Ns), size = 10, replace = False)
                    selection_label = np.array(['site ' + str(s) for s in selection])
                    plt.subplots()
                    plt.plot(xs_thin2, Z_trace_thin[:,selection,t], label = selection_label)
                    plt.legend(loc = 'upper left')
                    plt.title('traceplot for Zt at t=' + str(t))
                    plt.xlabel('iter thinned by '+str(thin))
                    plt.ylabel('Zt')
                    plt.savefig('trace_Zt'+str(t)+'.pdf')
                    plt.close()
                
                # ---- phi ----
                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_thin2[-1], phi_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
                plt.xlabel('iter thinned by '+str(thin))
                plt.ylabel('phi')
                plt.legend(loc = 'upper left')
                plt.savefig('trace_phi.pdf')
                plt.close()

                # ---- range ----
                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, range_knots_trace_thin[:,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_thin2[-1], range_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for range')
                plt.xlabel('iter thinned by '+str(thin))
                plt.ylabel('range')
                plt.legend(loc = 'upper left')
                plt.savefig('trace_range.pdf')
                plt.close()
            
                # ---- tau ----
                plt.subplots()
                plt.plot(xs_thin2, tau_trace_thin, label = 'nugget std dev')
                plt.title('tau nugget standard deviation')
                plt.xlabel('iter thinned by '+str(thin))
                plt.ylabel('tau')
                plt.legend(loc='upper left')
                plt.savefig('trace_tau.pdf')
                plt.close()

            if iter == n_iters - 1:
                print(iter)
                end_time = time.time()
                print('elapsed: ', round(end_time - start_time, 1), 'seconds')
                print('FINISHED.')