import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

B = 200
N = 20
# we looking for a mean of 0 and variance of 1 because it's from gaussian
# distribution
X = np.random.randn(N)

print(f"sample mean of X: {X.mean()}")

individual_estimates = np.empty(B)

for b in range(B):
    # take a sample and replace (smaple and replacement)
    sample = np.random.choice(X, size=N)
    individual_estimates[b] = sample.mean()  # we save the sample mean


bmean = individual_estimates.mean()
bstd = individual_estimates.std()
# lower limit for the CI(confidence Interveal)
lower = bmean + norm.ppf(0.025) * bstd
# upper limit for the CI(confidence Interveal)
upper = bmean + norm.ppf(0.975) * bstd

print(f'bootstrap mean f X:{bmean}')

lower2 = X.mean() + norm.ppf(0.025) * X.std() / np.sqrt(N)

upper2 = X.mean() + norm.ppf(0.975) * X.std() / np.sqrt(N)

plt.hist(individual_estimates, bins=20)
plt.axvline(x=lower, linestyle='--', color='g',
            label='lower bound for 95% CI (bootstrap)')
plt.axvline(x=upper, linestyle='--', color='g',
            label='upper bound for 95% CI (bootstrap)')
plt.axvline(
    x=lower2,
    linestyle='--',
    color='r',
    label='lower2 bound for 95% CI ')
plt.axvline(
    x=upper2,
    linestyle='--',
    color='r',
    label='upper2 bound for 95% CI ')
plt.legend()
plt.show()
