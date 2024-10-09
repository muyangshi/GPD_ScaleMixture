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
  - [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule)
- Inverse function theorem 
  - the derivative of $F^{-1}(t)$ is equal to $1/F'(F^{-1}(t))$ as long as $F'(F^{-1}(t))\neq 0$.

### Using standard Pareto

- Ported from original paper Appendix B, modified that 
  - $1/\phi_j$ should only stay on the denominator
  - $\bar{\gamma}_j$ should not be raised to the power of $\alpha$

#### CDF no nugget
$$
\begin{equation*}
    \begin{split}
        1-F_{X_j}(x)&=P(R_j^{\phi_j}W_j>x)=\int_0^\infty P(W_j>x/r^{\phi_j})f_{R_j}(r)dr\\
        &=\int_{x^{1/\phi_j}}^\infty f_{R_j}(r)dr+\int_0^{x^{1/\phi_j}} r^{\phi_j}f_{R_j}(r)/xdr\\
        &=1-\text{erfc}\left(\sqrt{\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}}\right)+x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)\\
        &=\sqrt{\frac{1}{\pi}}\textcolor{blue}{\gamma\left(\frac{1}{2},\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}+x^{-1}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\textcolor{blue}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}.
    \end{split}
\end{equation*}
$$

#### pdf no nugget
$$
\begin{equation*}
    f_{X_j}(x)= x^{-2}\sqrt{\frac{1}{\pi}}\left(\frac{\bar{\gamma}_{j}}{2}\right)^{\phi_j}\textcolor{blue}{\Gamma\left(\frac{1}{2}-\phi_j,\frac{\bar{\gamma}_{j}}{2x^{1/\phi_j}}\right)}.
\end{equation*}
$$

#### CDF with nugget
  - In Notability note page 63

#### pdf with nugget
  - In Notability note page 63

### Using shifted Pareto

#### CDF no nugget

  - Done in 2nd paper

#### pdf no nugget

  - Done in 2nd paper

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
  - [ ] Work on getting the sampler to work, first
  - Get a emulator, either for the `qRW` quantile function or for the `ll` likelihood function

- [ ] Organize and Solidify the distribution functions