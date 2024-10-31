# Notebook on GPD Project

## TODOs

- (By December) Start Coverage Analysis
  - without imputation
  - Figure out folder structures (write scripts to generate data and run jobs)
    - change the `savefolder` and `loadfolder` in the `simulate_data.py` and the `sampler.py` script respectively
  - [Alpine allocation](https://colostate.sharepoint.com/sites/Division_Research_Computing_and_Cyberinfrastructure/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FDivision%5FResearch%5FComputing%5Fand%5FCyberinfrastructure%2FShared%20Documents%2FGeneral%2FPUBLIC%2DWEB%2DCONTENT%2FAlpine%20Project%20Allocation%20Request%20Process%2Epdf&parent=%2Fsites%2FDivision%5FResearch%5FComputing%5Fand%5FCyberinfrastructure%2FShared%20Documents%2FGeneral%2FPUBLIC%2DWEB%2DCONTENT&p=true&ga=1)

- (By December) Speed up
  - Keep track of `X_star` and, at least, `X` separately (outside) as this will save the number of times we do `qRW`
  - pull the calculation of `X_star` and `X` outside of `ll_1t` function (use as argument)
  - make dedicated section to calculate `X_star` and `X` after each update



- will it be faster if the entire likelihood function is implemented in `Cpp`?

- emulation

# Meetings

## Oct. 29 Tuesday Meeting with Likun/Mark/Ben

- Training grid for the emulator
  - This is also computationally intensive, needs to be put on hold.


- Keep runing the sampler 
  - `recv(20) failed: Connection reset by peer (104)`
    - $S_t$, $Z_t$, $\gamma_k$, $\phi_k$ seems to be updating, so stopped it
  - [x] Try updating: $S_t$, $Z_t$, $\gamma_k$, $\phi_k$, $\rho_k$, $\tau$
    - 20241030
  - [ ] Try updating: $S_t$, $Z_t$, $\gamma_k$, $\phi_k$, $\rho_k$, $\tau$, $\theta_{GP}$

- imputation
  - ![alt text](notes/IMG_6060.jpeg)
  - We might actually also need to <mark>impute $Z_t$</mark> !?
    - <mark>check:</mark> the imputation should change both $Z_t$ and $Y_t$? because we need the smooth process $X_t^*$
      - then, when updating $Z_t$ in its own section, do we update both the missing and observed, or only the observed? (Previously in GEV we didn't have this problem because we do not update $Z_t$.) -- will this break MCMC?
      - current code is that we are updating all $s$ in $Z_t(s)$ when updating $Z_t$
  - Return imputed $(Z_t, Y_t)$ is enough
    - As long as we have $Y_t$ and we updated the $Z_t$, everything else can be calculated
  - <mark>check:</mark> because there is censoring, how should we perform predictive check on $Y_t(s)$ (predictive check won't show anything for the censored site)? 
    - Do we just do it on exceedance sites (from simulation)?
    - Do we do it on $X_t(s)$?


## Oct. 22 Meeting with Likun/Mark/Ben

- [x] Make the data $Y$ be from 10-day intervals.
  - Check the June 21 to Sept 21 definition of summer time
  - remove the 2 days or add additional 8 days
  - Using June 22 to Sept 19 to match 90 days
  - From the dataset, the scale and shape are estimated using spatially varying (temporal constant) thresholds.
    - Later in the sampler, the sampler creates a all-constant threshold

- Simulation study with threshold exeedance without marginals
  - Simulated dataset generates censored data. (reporting the threshold when observation is below the threshold). The censoring is performed using the distribution functions `pRW` on X.
  - For the realdata, however, 
    - censoring is performed in the sampler.py empirically.
    - Or it can happen in the data preprocessing (estimate $\sigma$ and $\xi$) to get spatially varying thresholds
  - Do MCMC for separately for each parameter (see who works)
    - [x] Update $S_t$
      - seems to work for $S_t$ when running only $S_t$ with $\gamma_k = 0.5$; using dataset `simulated_seed:2345_t:50_s:100_phi:nonstatsc2_rho:nonstat` radius = 4
      - [x] Try 50 sites, radius 3, will make it marginally faster for testing in progress -- seems to work
    - [x] Try update $\gamma_k$
    - [x] Try update both $S_t$ and $\gamma_k$
      - seems to be working 10/27
    - [x] Try update $S_t$, $Z_t$, $\gamma_k$
      - seems to be working 10/28
    - [x] Try update $S_t$, $Z_t$, $\gamma_k$, $\phi_k$
    - [-] Try update $S_t$, $Z_t$, $\gamma_k$, $\phi_k$, $\rho_k$
      - This one is probably harder to do with the just 50 sites? some knot does not contain many sites
      - potentially, increase the numebr of sites (if time permits)

- [x] "Unfix" $\gamma_k$
  - By changing the `gamma_at_knots`, `gamma_vec` ($\bar{\gamma}$) is calculated using Wendland basis matrix, and is used each time we call `qRW`
  - [x] Add the storage, update, and adaptive tuning for `gamma_at_knots`
    - [x] storage
    - [x] initialize
    - [x] update
      - [x] change corresponding `gamma_vec` to `gamma_vec_current`
      - [x] Need also modify the `ll_1t` function to add the `gamma_at_knot` in there, because `ll_1t` contains $p(S_t \mid \gamma_k)$ where $\gamma_k$ is no longer fixed
      - [x] Metropolis update -- <mark> discuss prior choice </mark>
      - saving
    - adaptive tuning
      - initialize
        - $\sigma_m^2$: `sigma_m_sq['gamma'] = list(np.diag(gamma_cov))` and `if rank == 0: with open('sigma_m_sq.pkl','rb') as file: sigma_m_sq = pickle.load(file)`
        - counter: `num_accepted['gamma'] = [0] * k_S`
      - tuning after $b$ iters
        - $\sigma_m^2$
        - counter

    - [x] plotting

  - [x] Change the `simulate_data.py` such that $\gamma_k$ is able to be spatially varying.
    - [x] Modify how $S_t$ is generated as now $\gamma$ is changed
    - [x] Modify the corresponding "hard coding truth for simulation" section in the `sampler.py`

- [x] Marginal model in the sampler

- [x] facilitate with posterior covariance matrix

## Oct. 15 Meeting with Likun/Ben

- Summarize (shrink) the data
  - truncate to JJA summertime data
  - for each season, split into subchunks, and take the max of each subchunk
  - fit GP to these

### Work

- [x] Separated knot_$S$, knot_$\phi$ and knot_$\rho$
- [x] Do not simplify likelihood calculation
  - e.g. $X_t$, `dRW`($X_t$);
  - easier for later emulator integration
  - exceedance/censored index can be simplified -- we can build an emulator just for the likelihood of the exceedances.

- [x] Summarize the likelihood into equations and document them here.
- [x] Incorporate $S_t$ likelihood into `ll_1t` function
  - [x] Change code updating $S_t$

- [x] Elevation from simulation data generation should not be negative
  - doesn't REALLY matter, but better if changed

### Question

- [x] Do we want to not fix $\gamma_k$ this time?
  - let's fix that at 1.
- [x] How to get site level MLE estimates for $\sigma$ and $\xi$ for GP?
  1. Ben: Have the threshold all constant for now.
  2. (Later) Likun: Make the threshold spatially varying, temporally constant
  3. (Later) Ben: Then smooth them, with splines?
      - Ben: We can't have a site at time t go above the threshold in some iterations and then below the threshold in some other iterations. It will break the MCMC.
- [x] How to estimate the initial nugget -- maybe from empirical semivariogram?
  - Ben: Certainly an option
- [x] Go over hierarchical model and full conditionals
  - [x] In the likelihood, <mark>do we need a piece for $R_t$, i.e. $p(R_t \mid \bar{\gamma})$</mark>? Or just put the $p(\bm{S}_t \mid \bm{\gamma})$? Are they the same?
    - Likun: the transformation is one-to-one, but we do not know $p(\bm{R_t} \mid \bar{\gamma})$ because some of them are correlated (not independent so can't be naively multiplied together).
  - [x] We directly update $S_t$ on log scale, so there's a jacobian; we also update $\sigma$ on a log scale, but we are actually update $\beta$, do we need a jacobian for the $\sigma$?
    - No because once $\beta$'s are know $\sigma$ is fixed; this is different from directly updating $\log \bm{S}_t$
    - Ben: STAN's online book as a section talking about this.
- In the MCMC updating $Z_t$, proposing $Z_t$ will surely change $X^*$ as $X^* = R^\phi g(Z)$. HOWEVER, <mark>do we need to change $X$ too based on $Z$ (or $R$) during update?</mark> i.e. do we need to keep track of our current $\epsilon$?
  - [x] After the update? (e.g. after accept proposed $Z$, we certainly change $X^*$, do we also change $X$?)
    - No
  - I think no because $X$ is marginal transformation from $Y$, which only depends are the marginal paraemter, $\phi$, ($\gamma$), and $\tau$.
- [x] Check that for updating $S_t$ and $Z_t$, because of assumed temporal independence, no need to gather likelihood and compare the sum; comparison of the individual ($ll_t$) is enough
- [x] Are we still conditioning on Time $t$? There is a lot more time replicates.

### Others

- 3 minute talk in Golden -- separate slides?
- spring 2025 TA and summer 2025 RA?
  - one semester in the 2025 - 2026 year RA
- [JCSDS](https://www.jconf-sds.com) 2025 7/11 - 13
- Supervise undergraduate students
- JSM 2025 topic-contributed session
  - Likun is doing Dynamic Space-Time modeling

## Oct.8 Meeting with Likun/Mark/Ben

- Daily data from Mark
  - Need to <mark>break the temporal independence</mark>
  - Maybe aggresively filter the data by time blocks, make $N_t$ to be roughly matching to 75.
    - Check in next meeting on how/what to do
- "task order":
  - Work on getting the sampler to work, first
  - Get a emulator, either for the `qRW` quantile function or for the `ll` likelihood function

- [x] Organize and Solidify the distribution functions
- Work on the sampler
  -  MPI oversubscribe

---

# Notes

## Thoughts

### Emulating the quantile funtion `qRW`

- consider trying out the RBFinterpolator from Scipy on the `qRW` function within a range, e.g. (0.95, 0.99999)
- consider the NN neural network on `qRW` function within range (0.95, 0.99999)


### Emulating the log-likelihood

- `emulate_ll_1t.py` fill the training X with LHS, then ppf to get the marginal Y from scale and shape
- `emulate_ll_1t_2.py` fill the training X with LHS, including the marginal Y that is also LHS'ly filled
  - had issue where $Y$ is out of support, leading to problem in $\log(\text{density})$, so created some filters

## Hierarchical Model and Likelihood

### Hierarcichal dependence model (and priors):

With index

$$
\begin{align*}
k &\in \{1, 2, \dots, K\} \quad \text{denoting number of knots (total $K$ knots)} \\
t &\in \{1, 2, \dots, T\} \quad \text{denoting time index (total $T$ time replicates)} \\
j &\in \{1, 2, \dots, D\} \quad \text{denoting site index (total $D$ locations)}
\end{align*}
$$

Hierarchical model written as:

$$
\begin{align*}
F_{Y \mid \bm{\theta}_{GP}, t}(Y_t(\bm{s}_j)) &= F_{X \mid \phi(\bm{s}_j), \bar{\gamma}(\bm{s}_j), \tau, t}(X_t(\bm{s}_j)) \\
&= F_{X \mid \phi(\bm{s}_j), \bar{\gamma}(\bm{s}_j), \tau, t}(R_t(\bm{s}_j)^{\phi(\bm{s}_j)}Z_t(\bm{s}_j) + \epsilon_t(\bm{s}_j)) \\
\bm{S}_t \mid \bm{\gamma} &\sim \text{Stable}(\alpha \equiv 0.5, \beta \equiv 1, \bm{\gamma}, \delta \equiv 0) \\
\bm{Z}_t \mid \bm{\rho} &\sim \text{MVN}(0, \bm{\Sigma}_{\bm{\rho}}) \\
\epsilon_t(\bm{s}_j) &\stackrel{\text{iid}}{\sim} \text{N}(0, \text{Var} = \tau^2)
\end{align*}
$$
priors:
$$
\begin{align*}
\phi_k &\sim \text{Beta}(5, 5) \\
\rho_k &\sim \text{halfNorm}(0, 2) \\
\gamma_k &\sim \textcolor{yellow}{\text{halfNorm}(0, 2)} \\
\tau &\sim \textcolor{yellow}{\text{halfT}_{\nu = 1}(\mu=0, \sigma=5)?}
\end{align*}
$$
marginal model:
$$
\begin{align*}
\log\sigma_t(\bm{s}_j) &\equiv \log\sigma(\bm{s}_j) = \beta_0^{(\sigma)} + \beta_1^{(\sigma)} \cdot \text{elev}(\bm{s}_j) \\
\xi_t(\bm{s}_j) &\equiv \xi(\bm{s}_j) = \beta_0^{(\xi)} + \beta_1^{(\xi)} \cdot \text{elev}(\bm{s}_j) \\
\beta_i^{(\theta)} &\sim \text{N}(0, \text{Var} = \sigma_{(\theta)}^2), \quad (\theta) \in \{(\sigma), (\xi)\} \\
\sigma_{(\theta)} &\sim \text{halfT}_{\nu = 2}(0, 1)
\end{align*}
$$

### Likelihood (pieces):
$$
\begin{align*}
p(\Theta | y) &\propto f(y | \Theta) \cdot \pi(\Theta = \left\{\bm{\phi}, \bm{\rho}, \bm{\gamma}, \tau, \bm{\sigma}, \bm{\xi}\right\}) \quad \text{i.e., assume $\theta_t(s) \equiv \theta(s)$ for $\theta \in \Theta$} \\
&= \prod_{t=1}^T \left[f(\bm{Y}_t \mid \Theta) \right] \cdot \pi(\Theta) \\
&= \prod_{t=1}^T\left[f(\bm{Y}_t \mid \textcolor{yellow}{\bm{S}_t}, \bm{Z}_t, \bm{\phi}, \bm{\gamma}, \bm{\rho}, \bm{\tau}, \bm{\sigma}, \bm{\xi}) \ \textcolor{yellow}{ p(\bm{S}_t \mid \bm{\gamma})} \ p(\bm{Z}_t \mid \bm{\rho}) \right] \pi(\bm{\phi}, \bm{\rho}, \bm{\gamma}, \tau, \bm{\sigma}, \bm{\xi})
\end{align*}
$$

where:

$$
\begin{align*}
f(\bm{Y}_t \mid \bm{S}_t, \bm{Z}_t, \bm{\phi}, \bm{\gamma}, \bm{\rho}, \bm{\tau}, \bm{\sigma}, \bm{\xi}) &= 
  \begin{cases}
  \Phi\left(F_{X \mid \phi(\bm{s}_j), \bar{\gamma}(\bm{s}_j), \tau}^{-1}(p) \mid X^*_t(\bm{s}_j), \tau\right) & \text{if } Y_t(\bm{s}_j)\leq u_t(\bm{s}_j)\\
  \\
  \varphi\left(X_t(\bm{s}_j) \mid X^*_t(\bm{s}_j),\tau\right)\frac{f_Y(Y_t(\bm{s}_j))}{f_X\left(X_t(\bm{s}_j)\right)}&\text{if } Y_t(\bm{s}_j)> u_t(\bm{s}_j)
  \end{cases} \\
  p(\bm{S}_t \mid \bm{\gamma}) &= f_{\text{Stable(0.5, 1, $\bm{\gamma}$, 0)}}(\bm{S}_t) \\
  \text{ as we propose and update $\log(\bm{S}_t)$ in MCMC}: & \\
  p(\bm{S}_t \mid \bm{\gamma}) d(\bm{S}_t) &= f_{\text{Stable(0.5, 1, $\bm{\gamma}$, 0)}}(\exp [\log (\bm{S}_t)]) \cdot \left\lvert \dfrac{d\bm{S}_t}{d\log\bm{S}_t} \right\rvert \cdot d(\log \bm{S}_t) \\
  &= f_{\text{Stable(0.5, 1, $\bm{\gamma}$, 0)}}(\bm{S}_t) \cdot \bm{S}_t\\
  p(\bm{Z}_t \mid \bm{\rho}) &= f_{\text{MVN}(\bm{0}, \bm{\Sigma}_{\bm{\rho}})}(\bm{Z}_t)
\end{align*}
$$
in which

- $\Phi(\ \cdot \mid X_t^*(s_i), \tau)$ 
- $\varphi(\ \cdot \mid X_t^*(s_i), \tau)$ 

respectively represents the distribution and the density function of $N(\mu = X_t^*(s_i), \text{sd} = \tau)$. 

## Distribution Functions

- Take derivative with respect to incomplete gamma function
  - [incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function)
    $$\gamma(s,x) = \int_0^xt^{s-1}e^{-t}dt$$
    $$\Gamma(s,x) = \int_x^\infty t^{s-1}e^{-t}dt$$
  - [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule)
    $$\dfrac{d}{dx}(\int_{a(x)}^{b(x)}f(\sout{x},t)dt) = f(\sout{x}, t=b(x)) \cdot \dfrac{d}{dx}b(x) - f(\sout{x}, t=a(x)) \cdot \dfrac{d}{dx}a(x) + \int_{a(x)}^{b(x)}\dfrac{\partial}{\partial x}f(\sout{x},t)dt$$

- Inverse function theorem 
  - the derivative of $F^{-1}(t)$ is equal to $1/F'(F^{-1}(t))$ as long as $F'(F^{-1}(t))\neq 0$.

### Standard Pareto 

---

Pareto distribution function
  $$F(x) = 1 - \left(\dfrac{x_m}{x}\right)^{\alpha}$$

  - $x_m$ = 1, minimum value, so support $[x_m, \infty)$
  - $\alpha = 1$, shape
Pareto density function
  $$f(x) = \dfrac{\alpha x_m^\alpha}{x^{\alpha+1}} \mathbb{1}(x \geq x_m)$$


#### CDF `pRW(x, phi_j, gamma_j)` no nugget

$$
F_{X^*_j}(x) = 1 - \left\{\sqrt{\frac{1}{\pi}}{\gamma\left(\frac{1}{2},\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)} + x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}\right\}
$$

- Ported from original paper Appendix B, modified such that 
  - $1/\phi_j$ should only stay on the <mark>denominator</mark>
  - $\bar{\gamma}_j$ should <mark>not</mark> be raised to the power of $\alpha$

- Details:
  $$
  \begin{equation*}
    \begin{split}
        1-F_{X^*_j}(x)&=P(R_j^{\phi_j}W_j>x)\\
        &=\int_0^\infty P(W_j>x/r^{\phi_j})f_{R_j}(r)dr\\
        &=\int_{x^{1/\phi_j}}^\infty f_{R_j}(r)dr+\int_0^{x^{1/\phi_j}} r^{\phi_j}f_{R_j}(r)/xdr\\
        &= \text{(lots of omited details using variable substitute, verified in oldest paper notebook)}\\
        &=1-\text{erfc}\left(\sqrt{\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}}\right)+x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)\\
        &=\sqrt{\frac{1}{\pi}}{\gamma\left(\frac{1}{2},\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)} + x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}
    \end{split}
  \end{equation*}
  $$

#### pdf `dRW(x, phi_j, gamma_j)` no nugget

$$
f_{X^*_j}(x)= x^{-2}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}
$$

- Details:
  $$
  \begin{align*}
  \dfrac{d}{dx}\left(-\sqrt{\dfrac{1}{\pi}}\gamma\left(\dfrac{1}{2}, \dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)\right) &= -\sqrt{\dfrac{1}{\pi}}\left(\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)^{-1/2} \exp\left(-\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)\left(-\dfrac{\bar{\gamma}_j}{2\phi x^{1/\phi+1}}\right) \\
  &= \sqrt{\dfrac{1}{\pi}}\textcolor{orange}{\left(\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)^{-1/2}} \exp\left(-\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)\left(\dfrac{\bar{\gamma}_j}{2\phi x^{1/\phi+1}}\right) \\
  \dfrac{d}{dx}\left(-x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)\right) &= \dfrac{1}{x^2}\sqrt{\dfrac{1}{\pi}} \left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j}\Gamma(\dfrac{1}{2} - \phi, \dfrac{\bar{\gamma}_j}{2x^{1/\phi}}) \\
  &- \dfrac{1}{x}\sqrt{\dfrac{1}{\pi}}\left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j} \left[-\left(\dfrac{\bar{\gamma}_j}{2x^{1/\phi}}\right)^{-\phi - 1/2}\exp\left(-\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)\left(-\dfrac{\bar{\gamma}_j}{2\phi x^{1/\phi+1}}\right)\right] \\
  &= \dfrac{1}{x^2}\sqrt{\dfrac{1}{\pi}} \left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j}\Gamma(\dfrac{1}{2} - \phi, \dfrac{\bar{\gamma}_j}{2x^{1/\phi}}) \\
  &- \textcolor{orange}{\dfrac{1}{x}}\sqrt{\dfrac{1}{\pi}}\textcolor{orange}{\left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j} \left(\dfrac{\bar{\gamma}_j}{2x^{1/\phi}}\right)^{-\phi - 1/2}} \exp\left(-\dfrac{\bar{\gamma}_j}{2x^{1/\phi_j}}\right)\left(\dfrac{\bar{\gamma}_j}{2\phi x^{1/\phi+1}}\right)
  \end{align*}
  $$


#### CDF `pRW(x, phi_j, gamma_j, tau)` with nugget

$$
1 - \left\{\bar{\Phi}(x) + \sqrt{\dfrac{1}{\pi}}\int_0^\infty\gamma\left(\dfrac{1}{2}, \dfrac{\bar{\gamma}_j}{2t^{1/\phi_j}}\right)\phi(x-t)dt + \sqrt{\dfrac{1}{\pi}}\left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j} \int_0^\infty \dfrac{1}{t} \Gamma \left(\dfrac{1}{2} - \phi_j, \dfrac{\bar{\gamma}_j}{2t^{\phi_j}}\right)\phi(x-t)dt \right\}
$$

  - $\Phi$ and $\phi$ are the distribution and density functions of $N(0, \tau^2)$
  - Details:
    - In Notability note page 63

    $$
    \begin{align*}
      1 - F_{X_j}(x) = \bar{F}_{X_j}(x) &= \int_{-\infty}^\infty P(X_j^* > x-\epsilon)\phi(\epsilon)d\epsilon \\
      &= \int_{-\infty}^xP(X_j^* > x - \epsilon)\phi(\epsilon)d\epsilon +\int_x^\infty P(X_j^* > x - \epsilon)\phi(\epsilon)d\epsilon \\
      &= \int_{-\infty}^xP(X_j^* > x - \epsilon)\phi(\epsilon)d\epsilon +\int_x^\infty 1 \cdot \phi(\epsilon)d\epsilon \\
      &= \bar{\Phi}(x) + \int_{-\infty}^x \bar{F}_{X_j^*}(x - \epsilon)\phi(\epsilon)d\epsilon \\
      &\text{define $t = x-\epsilon$} \\
      &\text{then $\epsilon \in (-\infty, x) \rightarrow t \in (\infty, 0)$} \\
      &= \bar{\Phi}(x) + \int_{t=\infty}^{t=0} \bar{F}_{X_j^*}(t)\phi(x-t)(-1)dt \text{\quad as $\quad \dfrac{dt}{d\epsilon}=-1$} \\
      &= \bar{\Phi}(x) + \int_{t=0}^{t=\infty} \bar{F}_{X_j^*}(t)\phi(x-t)dt \\
      &= \bar{\Phi}(x) + \int_0^\infty \left\{\sqrt{\dfrac{1}{\pi}}\gamma\left(\dfrac{1}{2}, \dfrac{\bar{\gamma}_j}{2t^{1/\phi_j}}\right) + \dfrac{1}{t} \sqrt{\dfrac{1}{\pi}} \left(\dfrac{\bar{\gamma_j}}{2}\right)^{\phi_j} \Gamma\left(\dfrac{1}{2} - \phi_j, \dfrac{\bar{\gamma}_j}{2t^{1/\phi_j}}\right) \right\} \phi(x-t)dt
    \end{align*}
    $$
    
    - Computation with gaussian density, using **definite** integral, in $\int_0^\infty \cdots \ \phi(x-t)dt$
      - $-38 \tau \leq x - t \leq 38 \tau$ as C/GSL's gaussian pdf vanishes after 38 SDs
      - $x-38\tau \leq t \leq x + 38\tau$ $\rightarrow$ $\max(0, x-38\tau) \leq t \leq x + 38\tau$
    - Do not transform to definite integral from 0 to 1
      - gives unstable integration

#### pdf `dRW(x, phi_j, gamma_j, tau)` with nugget

$$
f_{X_j}(x) = \sqrt{\dfrac{1}{\pi}}\left(\dfrac{\bar{\gamma}}{2}\right)^\phi \int_0^\infty \dfrac{1}{t^2} \Gamma\left(\dfrac{1}{2} - \phi_j, \dfrac{\bar{\gamma}}{2t^{1/\phi_j}} \right) \phi(x -t) dt
$$

- Details:
  $$
  \begin{align*}
  f_{X_j}(x) &= \int_{-\infty}^\infty f_{X_j^*}(x-\epsilon)\phi(\epsilon)d\epsilon \\
  &= \int_{-\infty}^x f_{X_j^*}(x-\epsilon)\phi(\epsilon)d\epsilon \text{$\quad$ as  $X_j^* > 0$} \\
  &= \int_0^\infty f_{X_j^*}(t)\phi(x-t)dt \text{$\quad$ same as before let $t = x - \epsilon$} \\
  &= \int_0^\infty \dfrac{1}{t^2} \sqrt{\dfrac{1}{\pi}} \left(\dfrac{\bar{\gamma}_j}{2}\right)^{\phi_j} \Gamma \left(\dfrac{1}{2} - \phi_j, \dfrac{\bar{\gamma}_j}{2t^{1/\phi_j}}\right)\phi(x-t)dt
  \end{align*}
  $$
  - In Notability note page 63
  - Same trick as before on definite integral bound with gaussian density inside
  - Do not transform to definite integral between 0 and 1
    - gives unstable integration


### Shifted (Type II) Pareto 
---

Pareto distribution function

  $$F(x) = 1 - \left(\dfrac{x_m}{x_m + x - \delta}\right)^{\alpha}$$
  - $\delta \equiv 0$, the support is
    $$x \geq \delta$$
  - $x_m = 1$ 
  - $\alpha = 1$
  - When $\delta = 0$, the Pareto distribution Type II is also known as the Lomax distribution

- all derivations are already done/shown in the emulator paper

#### CDF `pRW(x, phi_j, gamma_j)` no nugget

$$
F_{X_j^*}(x) = 1 - \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr 
$$

- Details:

  $$
  \begin{equation*}
  \begin{split}
      1-F_{X_j^*}(x)&=P(R_j^{\phi_j}W_j>x)\\
      &=\int_0^\infty P(W_j>x/r^{\phi_j})f_{R_j}(r)dr\\
      &=\int_0^\infty 1/(1+x/r^{\phi_j})f_{R_j}(r)dr\\
      &=\sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr      
  \end{split}
  \end{equation*}
  $$

  - numerical approximation for $F_{X_j^*}(x)$:

    $$F_{X_j^*}(x) \approx x\left(\frac{\bar{\gamma}_{j}}{2}\right)^{-\phi_j}\frac{\Gamma(\phi_j+1/2)}{\sqrt{\pi}} \text{ as } x \rightarrow 0$$

    - <mark>how?</mark> Proposition 2.1 in the original paper defines three different cases base on $\phi$ and $\alpha$
  

#### pdf `dRW(x, phi_j, gamma_j)` no nugget

$$
f_{X_j^*}(x)=\sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{0}^\infty \frac{r^{\phi_j-3/2}}{(x+r^{\phi_j})^2}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr
$$

#### CDF `pRW(x, phi_j, gamma_j, tau)` with Gaussian nugget

$$
F_{X_j}(x) = 1 - \left(\int_{x}^{\infty} \varphi(\epsilon)d\epsilon + \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{-\infty}^{x} \varphi(\epsilon)\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\epsilon+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}drd\epsilon \right)
$$

-  $\varphi$ is the Gaussian distribution of $\epsilon$, i.e. $N(0, \tau^2)$.

- Details:

  $$
  \begin{equation*}
  \begin{split}
    1-F_{X_j}(x) &= \int_{-\infty}^{\infty} P(X_j^* > x-\epsilon)\varphi(\epsilon)d\epsilon \\
    &= \int_{x}^{\infty} \varphi(\epsilon)d\epsilon+ \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{-\infty}^{x} \varphi(\epsilon)\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\epsilon+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}drd\epsilon
  \end{split}
  \end{equation*}
  $$




#### pdf `dRW(x, phi_j, gamma_j, tau)` with Gaussian nugget
  $$
  \begin{align*}
  f_{X_j}(x) &= \varphi(x) - \varphi(x) + 
    \textcolor{orange}{\sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}} \int_{-\infty}^x \varphi(\epsilon) \textcolor{orange}{\int_0^\infty \dfrac{r^{\phi_j-3/2}}{(x - \epsilon + r^{\phi_j})^2} \exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\} dr} d\epsilon \\
    &= \int_{-\infty}^x \varphi(\epsilon) \textcolor{orange}{f_{X_j^*}(x-\epsilon)} d\epsilon
  \end{align*}
  $$

- Details:

  $$
  \begin{align*}
    \dfrac{d}{dx} F_{X_j}(x) &= 
    - \left( 
    \dfrac{d}{dx} \int_{x}^{\infty} \varphi(\epsilon)d\epsilon + 
    \dfrac{d}{dx} \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{-\infty}^{x} \varphi(\epsilon)\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\epsilon+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}drd\epsilon
    \right) \\
    \text{part 1} &= \dfrac{d}{dx} \int_{x}^{\infty} \varphi(\epsilon) d\epsilon = -\varphi(x) \\
    \text{part 2} &= \dfrac{d}{dx} \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{-\infty}^{x} \varphi(\epsilon)\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\epsilon+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}drd\epsilon \\
    \text{(Use Leibniz)}  &= \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}} \cdot
    \left[ 
    \dfrac{d}{dx}(\textcolor{orange}{x}) \cdot \varphi(\textcolor{orange}{x}) \int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\textcolor{orange}{x}+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr \right. \\
    &\quad +
    \left.
    \int_{-\infty}^{x} \dfrac{\partial}{\partial x} \left(\varphi(\epsilon)\int_{0}^\infty \frac{r^{\phi_j-3/2}}{x-\epsilon+r^{\phi_j}}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr\right) d\epsilon
    \right] \\
    &= \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}} \cdot
    \left[
    \varphi(x) \int_0^\infty r^{- 3/2}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr
    +
    \int_{-\infty}^x \varphi(\epsilon) \int_0^\infty \dfrac{-r^{\phi_j-3/2}}{(x - \epsilon + r^{\phi_j})^2} \exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\} dr d\epsilon
    \right] \\
    &= \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}  \varphi(x) \cdot -\dfrac{\sqrt{2\pi}}{\sqrt{\bar{\gamma_j}}} \int_{u=\frac{\sqrt{\bar{\gamma_j}}}{\sqrt{2r}}|_{r=0}}^{u=\frac{\sqrt{\bar{\gamma_j}}}{\sqrt{2r}}|_{r=\infty}} \dfrac{2e^{-u^2}}{\sqrt{\pi}} du + \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}  \int_{-\infty}^x \varphi(\epsilon) \int_0^\infty \dfrac{-r^{\phi_j-3/2}}{(x - \epsilon + r^{\phi_j})^2} \exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\} dr d\epsilon \\
    &= \varphi(x) + \sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}  \int_{-\infty}^x \varphi(\epsilon) \int_0^\infty \dfrac{-r^{\phi_j-3/2}}{(x - \epsilon + r^{\phi_j})^2} \exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\} dr d\epsilon
  \end{align*}
  $$

