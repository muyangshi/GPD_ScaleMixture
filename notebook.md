# Notebook on GPD Project

# Notes

## Emulating the quantile funtion `qRW`

- consider trying out the RBFinterpolator from Scipy on the `qRW` function within a range, e.g. (0.95, 0.99999)
- consider the NN neural network on `qRW` function within range (0.95, 0.99999)


## Emulating the log-likelihood

- `emulate_ll_1t.py` fill the training X with LHS, then ppf to get the marginal Y from scale and shape
- `emulate_ll_1t_2.py` fill the training X with LHS, including the marginal Y that is also LHS'ly filled
  - had issue where $Y$ is out of support, leading to problem in $\log(\text{density})$, so created some filters


## Organize and Recalculate the distribution functions

- Take derivative with respect to incomplete gamma function
  - [incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function)
    $$\gamma(s,x) = \int_0^xt^{s-1}e^{-t}dt$$
    $$\Gamma(s,x) = \int_x^\infty t^{s-1}e^{-t}dt$$
  - [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule)
    $$\dfrac{d}{dx}(\int_{a(x)}^{b(x)}f(\sout{x},t)dt) = f(\sout{x}, t=b(x)) \cdot \dfrac{d}{dx}b(x) - f(\sout{x}, t=a(x)) \cdot \dfrac{d}{dx}a(x) + \int_{a(x)}^{b(x)}\dfrac{\partial}{\partial x}f(\sout{x},t)dt$$

- Inverse function theorem 
  - the derivative of $F^{-1}(t)$ is equal to $1/F'(F^{-1}(t))$ as long as $F'(F^{-1}(t))\neq 0$.

### Standard Pareto -----------------------------------------------------------

Pareto distribution function
  $$F(x) = 1 - \left(\dfrac{x_m}{x}\right)^{\alpha}$$
  - $x_m$ = 1, minimum value, so support $[x_m, \infty)$
  - $\alpha = 1$, shape
Pareto density function
  $$f(x) = \dfrac{\alpha x_m^\alpha}{x^{\alpha+1}} \mathbb{1}(x \geq x_m)$$


#### CDF `pRW(x, phi_j, gamma_j)` no nugget

$$F_{X_j}(x) = 1 - \left\{\sqrt{\frac{1}{\pi}}{\gamma\left(\frac{1}{2},\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)} + x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}\right\}$$

- Ported from original paper Appendix B, modified such that 
  - $1/\phi_j$ should only stay on the <mark>denominator</mark>
  - $\bar{\gamma}_j$ should <mark>not</mark> be raised to the power of $\alpha$

- Details:
  $$
  \begin{equation*}
    \begin{split}
        1-F_{X_j}(x)&=P(R_j^{\phi_j}W_j>x)\\
        &=\int_0^\infty P(W_j>x/r^{\phi_j})f_{R_j}(r)dr\\
        &=\int_{x^{1/\phi_j}}^\infty f_{R_j}(r)dr+\int_0^{x^{1/\phi_j}} r^{\phi_j}f_{R_j}(r)/xdr\\
        &= \text{(lots of ommited details using variable substitute, verified in oldest paper notebook)}\\
        &=1-\text{erfc}\left(\sqrt{\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}}\right)+x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)\\
        &=\sqrt{\frac{1}{\pi}}{\gamma\left(\frac{1}{2},\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)} + x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}
    \end{split}
  \end{equation*}
  $$

#### pdf `dRW(x, phi_j, gamma_j)` no nugget
$$f_{X_j}(x)= x^{-2}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}$$

Details:
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


#### CDF with nugget
  - In Notability note page 63

#### pdf with nugget
  - In Notability note page 63

### Shifted (Type II) Pareto --------------------------------------------------

- CDF
  $$F(x) = 1 - \left(\dfrac{x_m}{x_m + x - \delta}\right)^{\alpha}$$
  - $\delta \equiv 0$, the support is
    $$x \geq \delta$$
  - $x_m = 1$ 
  - $\alpha = 1$
  - When $\delta = 0$, the Pareto distribution Type II is also known as the Lomax distribution

- all derivations are already done/shown in the emulator paper

#### CDF no nugget

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

  $$F_{X_j^*}(x) \approx x\left(\frac{\bar{\gamma}_{j}}{2}\right)^{-\phi_j}\frac{\Gamma(\phi_j+1/2)}{\sqrt{\pi}} \text{ as } x \rightarrow 0 $$
  - how? Proposition 2.1 in the original paper defines three different cases base on $\phi$ and $\alpha$
  

#### pdf no nugget

$$
  f_{X_j^*}(x)=\sqrt{\frac{\bar{\gamma}_{j}}{2\pi}}\int_{0}^\infty \frac{r^{\phi_j-3/2}}{(x+r^{\phi_j})^2}\exp\left\{-\frac{\bar{\gamma}_{j}}{2r}\right\}dr
$$

#### CDF with nugget

  - Done in 2nd paper

#### pdf with nugget

  - Done in 2nd paper


---

# Meetings

## Oct.8 Meeting with Likun/Mark/Ben

- Daily data from Mark
  - Need to <mark>break the temporal independence</mark>
  - Maybe aggresively filter the data by time blocks, make $N_t$ to be roughly matching to 75.
    - Check in next meeting on how/what to do
- "task order":
  - Work on getting the sampler to work, first
  - Get a emulator, either for the `qRW` quantile function or for the `ll` likelihood function

- [ ] Organize and Solidify the distribution functions
- [ ] Work on the sampler