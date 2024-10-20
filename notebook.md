# Notebook on GPD Project

# Meetings

## Oct. 15 Meeting with Likun/Ben

- Summarize (shrink) the data into 10-day max
  - fit GP to the 10-day max

### Work

- [x] Separated knot_$S$, knot_$\phi$ and knot_$\rho$
- [x] Do not simplify likelihood calculation
  - e.g. $X_t$, `dRW`($X_t$);
  - easier for later emulator integration
  - exceedance/censored index can be simplified -- we can build an emulator just for the likelihood of the exceedances.

- [x] Summarize the likelihood into equations here

- [ ] Elevation from simulation data generation should not be negative
  - doesn't REALLY matter, but better if changed

### Question

- Do we want to not fix $\gamma_k$ this time?
- How to get site level MLE estimates for $\sigma$ and $\xi$ for GP?
- How to estimate the initial nugget -- maybe from empirical semivariogram
- Go over hierarchical model and full conditionals
- In the MCMC updating $Z_t$, proposing $Z_t$ will surely change $X^*$ as $X^* = R^\phi g(Z)$. HOWEVER, <mark>do we need to change $X$ too based on $Z$ (or $R$) during update?</mark> i.e. do we need to keep track of our current $\epsilon$?
  - After the update?
  - I think no because $X$ is marginal transformation from $Y$, which only depends are the marginal paraemter, $\phi$, ($\gamma$), and $\tau$.
- <mark>Check</mark> that for updating $R_t$ and $Z_t$, because of assumed temporal independence, no need to gather likelihood and compare the sum; comparison of the individual ($ll_t$) is enough

### Logistics

- 3 minute talk in Golden -- separate slides?
- spring 2025 TA and summer 2025 RA?
- [JCSDS](https://www.jconf-sds.com) 2025 7/11 - 13

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


- [ ] Simulation study with threshold exeedance without marginals
  - [ ] Do MCMC for separately for each parameter (see who works)
  - [ ] Completely Stationary data w.r.t. $\phi$ and $\rho$
  - [ ] Nonstationary data
- [ ] Imputation
  - [ ] Posterior Predicative Check
- [ ] Marginal model in the sampler

## Emulating the quantile funtion `qRW`

- consider trying out the RBFinterpolator from Scipy on the `qRW` function within a range, e.g. (0.95, 0.99999)
- consider the NN neural network on `qRW` function within range (0.95, 0.99999)


## Emulating the log-likelihood

- `emulate_ll_1t.py` fill the training X with LHS, then ppf to get the marginal Y from scale and shape
- `emulate_ll_1t_2.py` fill the training X with LHS, including the marginal Y that is also LHS'ly filled
  - had issue where $Y$ is out of support, leading to problem in $\log(\text{density})$, so created some filters

## Hierarchical Model and Likelihood

### Hierarcichal dependence model (and priors):
$$
\begin{align*}
F_{Y \mid \bm{\theta}_{GP}, t}(Y_t(\bm{s})) &= F_{X \mid \phi(\bm{s}), \bar{\gamma}(\bm{s}), \tau, t}(X_t(\bm{s})) \\
&= F_{X \mid \phi(\bm{s}), \bar{\gamma}(\bm{s}), \tau, t}(R_t(\bm{s})^{\phi(\bm{s})}Z_t(\bm{s}) + \epsilon_t(\bm{s})) \\
\bm{S}_t \mid \bm{\gamma} &\sim \text{Stable}(\alpha \equiv 0.5, \beta \equiv 1, \bm{\gamma}, \delta \equiv 0) \\
\bm{Z}_t \mid \bm{\rho} &\sim \text{MVN}(0, \bm{\Sigma}_{\bm{\rho}})
\end{align*}
$$
priors:
$$
\begin{align*}
\bm{\phi} &\sim \text{Beta}(5, 5) \\
\rho &\sim \text{halfNorm}(0, 2) \\
\tau &\sim \textcolor{yellow}{\text{halfNorm(0, 2)?}}
\end{align*}
$$
marginal model:
$$
\begin{align*}
\sigma_t(\bm{s}) &\equiv \sigma(\bm{s}) = \beta_0^{(\sigma)} + \beta_1^{(\sigma)} \cdot \text{elev}(\bm{s}) \\
\xi_t(\bm{s}) &\equiv \xi(\bm{s}) = \beta_0^{(\xi)} + \beta_1^{(\xi)} \cdot \text{elev}(\bm{s})
\end{align*}
$$

### Likelihood (pieces):
$$
\begin{align*}
p(\Theta | y) &\propto f(y | \Theta) \cdot \pi(\Theta = \left\{\bm{\phi}, \bm{\rho}, \bm{\gamma}, \tau, \bm{\sigma}, \bm{\xi}\right\}) \quad \text{i.e., assume $\theta_t(s) \equiv \theta(s)$ for $\theta \in \Theta$} \\
&= \prod_{t=1}^T \left[f(\bm{Y}_t \mid \Theta) \right] \cdot \pi(\Theta) \\
&= \prod_{t=1}^T\left[f(\bm{Y}_t \mid \bm{R}_t, \bm{Z}_t, \bm{\phi}, \bar{\bm{\gamma}}, \bm{\rho}, \bm{\tau}, \bm{\sigma}, \bm{\xi}) \ p(\bm{R}_t \mid \bar{\bm{\gamma}}) \ p(\bm{Z}_t \mid \bm{\rho}) \right] \pi(\bm{\phi}, \bm{\rho}, \bm{\gamma}, \tau, \bm{\sigma}, \bm{\xi})
\end{align*}
$$
where:
$$
\begin{align*}
f(\bm{Y}_t \mid \bm{S}_t, \bm{Z}_t, \bm{\phi}, \bm{\gamma}, \bm{\rho}, \bm{\tau}, \bm{\sigma}, \bm{\xi}) &= 
  \begin{cases}
  \Phi\left(F_{X \mid \phi(\bm{s}_i), \bar{\gamma}(\bm{s}_i), \tau}^{-1}(p) \mid X^*_t(\bm{s}_i), \tau\right) & \text{if } Y_t(\bm{s}_i)\leq u_t(\bm{s}_i)\\
  \\
  \varphi\left(X_t(\bm{s}_i) \mid X^*_t(\bm{s}_i),\tau\right)\frac{f_Y(Y_t(\bm{s}_i))}{f_X\left(X_t(\bm{s}_i)\right)}&\text{if } Y_t(\bm{s}_i)> u_t(\bm{s}_i)
  \end{cases} \\
  p(\bm{R}_t \mid \bar{\bm{\gamma}}) &= f_{\text{Stable(0.5, 1, $\bar{\bm{\gamma}}$, 0)}}(\bm{S}_t) \\
  p(\bm{Z}_t \mid \bm{\rho}) &= f_{\text{MVN}(\bm{0}, \bm{\Sigma}_{\bm{\rho}})}(\bm{Z}_t)
\end{align*}
$$
in which

- $\Phi(\ \cdot \mid X_t^*(s_i), \tau)$ 
- $\varphi(\ \cdot \mid X_t^*(s_i), \tau)$ 

respectively represents the cumulative distribution function and the density function of $N(\mu = X_t^*(s_i), \text{sd} = \tau)$. 

## Organize and Recalculate the distribution functions

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

