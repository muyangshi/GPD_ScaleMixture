# Notebook on GPD Project

# Notes


### Emulating the quantile funtion `qRW`

- consider trying out the RBFinterpolator from Scipy on the `qRW` function within a range, e.g. (0.95, 0.99999)
- consider the NN neural network on `qRW` function within range (0.95, 0.99999)


### Emulating the log-likelihood

- `emulate_ll_1t.py` fill the training X with LHS, then ppf to get the marginal Y from scale and shape
- `emulate_ll_1t_2.py` fill the training X with LHS, including the marginal Y that is also LHS'ly filled
  - had issue where $Y$ is out of support, leading to problem in $\log(\text{density})$, so created some filters


---

# Meetings

## Oct.8 Meeting with Likun/Mark/Ben

- Daily data from Mark
- "Working order":
  - Get the sampler working first
  - Get a emulator, either for the `qRW` quantile function or for the `ll` likelihood function


