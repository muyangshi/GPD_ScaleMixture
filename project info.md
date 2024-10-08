# Documentation

This is to document the GPD project. Including it's file structure and along with any preparation that needs to be down to run the sampler.


```
GPD_ScaleMixture/
│
├── data/ # storing datafiles and emulator files
│   │  
│   ├── cb_2018_us_state_20m/
│   │  
│   │    # data files
│   ├── <datafiles1>/
│   │     │   # (simulated data)
│   │     ├── simulated .npy files (e.g. Y_NA.npy, sites_xy.npy, etc.)
│   │     ├── data checking plots (e.g. .pdf QQplots)
│   │     └── simulated_data.RData (simulated combined)
│   ├── <datafiles2>/
│   │     │   # (real data)
│   │     └── e.g. JJA_precip_maxima_nonimputed.RData
│   │
│   │   # emulator objects/data
│   ├── <emulator>/
│   │     │
│   │     └── Neural-Net's activation, bias, and weights (e.g. NN_acts, NN_bs.pkl, NN_Ws.pkl, qRW_NN.keras )
│   │     
│   └──
│
├── scripts/
│   │
│   │   # sampler
│   ├── simulate_data.py             # for simulating non-stationary dataset
│   ├── simulate_data_stationary.py  # for simulating stationary dataset
│   │
│   │   # utilities functions
│   ├── p_cubature.py   # for using 2d numerical integral
│   ├── RW_inte_cpp.cpp # for using 1d numerical integral
│   ├── RW_inte_cpp.so
│   ├── RW_inte.py
│   ├── utilities.py
│   │
│   │   # likelihood emulations
│   ├── emulate_11_1t.py
│   │
│   │   # sampler
│   ├── proposal_cov.py
│   └── sampler.py
│
├── chains/ # storing (raw) traceplots
│
└── results/ # storing posterior summaries
```


## Preparation

This section includes some dependencies packages/modules to be downloaded and imported, instructions on a function library to be built, and specification on the dataset that feeds into the sampler. 

### Imports/Dependencies

**Parallel Computation**:

- openmpi
- openmpi-mpicxx

**Cpp**:

- GSL
- Boost

**Python**: (packages are managed through `Conda`)

- numpy
- scipy=1.11
- mpi4py=3.1.4
- gstools
- mpmath
- gmpy2
- numba
- rpy2
- matplotlib
    - Statemap `cb_2018_us_state_20m/` downloaded from [US census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)


### Utilities
Files:

- `RW_inte_cpp.cpp`
- `RW_inte_cpp.so`
- `RW_inte.py`
- `utilities.py`

Numerical Integration functions are written in the `RW_inte_cpp.cpp` script. This script is compiled to a shared library object `RW_inte_cpp.so` (as a function library), for example using the terminal command: 

```
g++ -I$GSL_INCLUDE -I$BOOST_INC -L$GSL_LIBRARY -L$BOOST_LIBRARY -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas
```

where `GSL_INCLUDE`, `BOOST_INCLUDE`, `GSL_LIBRARY`, and `BOOST_LIBRARY` are the corresponding `include/` and `lib/` folders for those packages.

Then these compiled functions are "packaged" into `RW_inte.py`, which is then imported in the set of all utilities/helper function in `utilities.py`.

### Dataset

#### Using a real dataset:

TODO

<!-- 
File:

- `JJA_precip_maxima_nonimputed.RData`

This is an example real dataset of central US summer time block-maxima precipitation. This `.RData` file contains the following items:

- `JJA_maxima_nonimputed`: an ($N_s$, $N_t$) matrix of type `double` of the non-imputed (contains NA) summer time block-maxima precipitation at the $N_s$ stations across $N_t$ times. Each column $t$ represents the $N_s$ observations at time $t$.
- `GEV_estimates`: an ($N_s$ $\times$ 4) `dataframe` of the marginal parameter estimates (by fitting a simple GEV) at each station. Each of the four columns represents the estimates for $\mu_0$, $\mu_1$, $\log(\sigma)$, and $\xi$ for each station. 
- `stations`: an ($N_s$ $\times$ 2) `dataframe` of the coordinates of each station. The first column is longitude and the second column represents latitude.
- `elev`: an ($N_s$,) vector of type `double` of the elevation of the geographic location of each station.

where $N_s$ denotes the number of stations and $N_t$ denotes the number of years recorded in the dataset. -->

#### Generating a Simulated Dataset:

TODO

<!-- If we are not performing an application on a real dataset, we could simulate a dataset to test the model.

File:

- `simulate_data.py`

This is a python script to generate a simulated dataset. 
 Within this file, specifies $N_s$, $N_t$, $\phi$ and/or $\rho$ surfaces as well as the marginal parameters ($\Beta's$ for the $\mu_0, \mu1$ as well as $\sigma$ and $\xi$) surfaces to fine tune the truth parameter setup. Some additional comments in the script might be helpful. 

To run this file:

```
python3 simulate_data.py <random_seed>
```
where `<random_seed>` is a seed used to randomly generate the data;
 e.g. `python3 simulate_data.py 2345` can be used to generate a simulated dataset using the random seed of 2345.


Outputs:

Running this script will generate the following dataset and/or plots:
 (assuming we have simulated a dataset of $N_t=24$ time replicates at $N_s = 300$ sites using simulate scenario $sc = 2$)
- `simulated_data.RData`: The generated dataset, which consists of the 4 items matching those described in the real dataset above.
- Additional formats/pieces of the same simulated dataset (for easier checking): 
    - `Y_full_sim_sc2_t24_s300.npy`: simulated observation $Y$ with all observations (no missing)
    - `miss_matrix_bool.npy`: simulated missing indicator matrix
    - `Y_miss_sim_sc2_t24_s300.npy`: simulated observation $Y$ after applying the `miss_matrix` (some observation become `NA`s as mimicing missing at random in data)
- `.png` Plots (for checking generated dataset):
    - QQPlots: QQplots will be generated for the Levy variables at the knots, the Pareto $W=g(Z)$ at the sites, and the process $X^*$ across the sites $s$ or time $t$. All qqplots are transformed to the uniform margin. Other than the QQPlot for $X^*$ across all sites at a specific time $t$ should deviates from 1:1 line (because of spatial correlation), the others should all approximately be linearly following diagonal line.
    - Histogram: MLE fitted GEV models on the dataset at each site $s$ across all time $t$ is pooled together into a histogram; the values of the $\mu, \sigma, \xi$ should roughly reflect that in the parameter surface setting. (Not precise but should/could serve as a quick check) -->


## Sampler

TODO

<!-- 
This section includes some information on running the sampler `MCMC.py` script, as well as some notes on the output (plots, traceplots, intermediate model states) that running this script will produce.

**Required Files** (in addition to the files described in the previous section):

- `proposal_cov.py`
- `MCMC.py`

**Optional Files** (saved model states from prior runs that `MCMC.py` will generate periodically):

- `iter.pkl`
- `sigma_m_sq_Rt_list.pkl`
- `sigma_m_sq.pkl`
- `Sigma_0.pkl`


`MCMC.py` is the main sampler file, and it uses a random-walk Metropolis algorithm using Log-Adaptive Proposal (LAP) as an adaptive tuning strategy (Shaby and Wells, 2010). This script takes in a dataset file (e.g. `JJA_precip_maxima_nonimputed.RData`) placed under the same directory. Additional dependencies are specified in the previous section. 
 
Sometimes (especially when running on clusters) we can't afford to have the sampler be continuously running until it finishes, and so we have to "chop" it into pieces and "daisychaining" the subsequent runs. This script will automatically create and save the traceplots for the variables/parameters, as well as saving the model states when the script is stopped (e.g. run into the time limit). When invoking this script, it will check if there are saved model states (the optional files) saved in the directory and will pick up from there.

This `MCMC.py` script is split into the following sections (more detailed comments are made within the script):

- Load the dataset
- Setup the spline smoothings
- Estimate the initial starting points for parameters
  - Plot the initially estimated parameter surfaces
- Specify the block-update structure for MCMC updating the parameters
- Metropolis-Hasting MCMC Loop:
  - Update $R_t$
  - Update $\phi$
  - Update $\rho$
  - Update $Y$ (imputations)
  - Update $\beta(\mu_0)$
  - Update $\beta(\mu_1)$
  - Update $\beta(\log(\sigma))$
  - Update $\beta(\xi)$
  - Update the adaptive metropolis strategy (periodically, once every certain \# of iterations)

The `proposal_cov.py` is the initial proposal covariance matrix $\Sigma_0$ for this LAP tuning strategy. 
Without specific knowledge on the covariance of the proposals, one can set the variables in the `proposal_cov.py` script to `None`, as this would make the sampler default to initialize with identity $I$ proposals.
This is only used when starting the chains fresh, as the later continuance/"daisychain" will load the proposal scalar variance and covariance from the `.pkl` files saved from previous runs.

To **run this sampler script**:

```
mpirun -n <Nt> python3 MCMC.py
```

where `<Nt>` is the number of time replicates in the dataset and hence (by the parallelization of the code) the number of cores used to invoke this parallelized job using mpi.

**Outputs**:

Running the sampler will generate the following results files

- Plots

    - Geo/spatial informations on the dataset:
        - `Plot_US.pdf`, `Plot_stations.pdf`: scatterplots of the stations (longitude, latitude) with overlaying state boundary
        - `Plot_station_elevation.pdf`: scatterplots of the stations with color coding their elevations
    - Initial Parameter Estimates:
        - `Plot_initial_heatmap_phi_surface.pdf`, `Plot_initial_heatmap_rho_surface.pdf`: heatmaps of the $\phi$ and $\rho$ surfaces coming from initial parameter estimation.
        - `Plot_initial_mu0_estimates.pdf`, `Plot_initial_mu1_estimates.pdf`, `Plot_initial_logsigma_estimates.pdf`, and `Plot_initial_ksi_estimates.pdf`: Comparison of the intial GEV fitted parameters at the sites versus the spline smoothed marginal parameters at the sites (color represents value)
    - Plots of the Traceplots:
      - Overall log-likelihoods: `Traceplot_loglik.pdf`, `Traceplot_loglik_detail.pdf`
      - Copula parameters: `Traceplot_Rt_<t>.pdf` (`<t>` in 1, ..., $N_t$), `Traceplot_phi.pdf`, and `Traceplot_range.pdf`
      - Marginal model coefficients and regularization terms: `Traceplot_<Beta_mu0_block_idx>.pdf` (for $\Beta$'s for $\mu_0$ in that block update), `Traceplot_<Beta_mu1_block_idx>.pdf`, `Traceplot_Beta_logsigma.pdf`, `Traceplot_Beta_ksi.pdf`, and `Traceplot_sigma_Beta_xx.pdf`.

- Traceplot `.npy` Matrix

    - The traceplot items are periodically saved (currently after every 50 iterations), including
        - log-likelihood trace: `loglik_trace.npy`, `loglik_detail_trace.npy`
        - copula parameter trace: `R_trace_log.npy`, `phi_knots_trace.npy`, `range_knots_trace.npy`, 
        - marginal model parameter trace: `Beta_mu0_trace.npy`, `Beta_mu1_trace.npy`, `Beta_logsigma_trace.npy`, `Beta_ksi_trace.npy` and their regularization hyper parameter trace `sigma_Beta_mu0_trace.npy`, `sigma_Beta_mu1_trace.npy`, `sigma_Beta_logsigma_trace.npy`
        - records on imputation (conditional gaussian draws) on `Y_trace.npy`

- Periodically saved model/chain states in `.pkl` pickles (to be picked up at each consecutive run)

    - `iter.pkl`: the number of iteration this chain has reached; a consecutive run will "restart" the chain at the last saved `iter.pkl` iteration)
    - Adapted proposal scalar variance and/or covariance matrix
        - the proposal scalar variance for the stable variables $R_t$'s: `sigma_m_sq_Rt_list.pkl`
        - the proposal scalar variance and covariance matrix for any other parameters: `sigma_m_sq.pkl`, `Sigma_0.pkl` -->


<!-- ## Posterior Summary

File:

- `posterior_and_diagnostics.py`

This is the posterior summary script, that summarizes the chains resulting from running the `MCMC.py` sampler. 

What does it do.

What does it produce.

To fun this file:
```
python3 posterior_and_diagnostics.py
``` -->

