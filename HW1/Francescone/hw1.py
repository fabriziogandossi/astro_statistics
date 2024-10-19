import numpy as np
from numpy import array, random, linspace, pi, exp, sqrt, zeros_like, arange, argmax
from matplotlib import pyplot as plt, colormaps as cm


# Likelihood calculation for this problem
def likelihood_prior_calculation(
        distribution_H, random_tossesf,  total_amount, center, std_dev
):
    # Define the amount of heads for each series considered
    heads_tosses = sum(random_tossesf[:total_amount])
    # Likelihood calculation
    likelihoodf = (
        distribution_H**heads_tosses *
        (1 - distribution_H)**(total_amount - heads_tosses)
    )
    # Gaussian Prior calculation (could have used scipy,
    # but I preferred to explicit the calculation)
    gaussian_priorf = (
            (1 / (std_dev * sqrt(2 * pi))) *
            exp(-0.5 * ((distribution_H - center) / std_dev)**2)
    )
    return likelihoodf, gaussian_priorf


# Normalize function
def normalize(arr):
    return (arr - min(arr)) / max(arr)


# ## MAIN ## #
# Define the amount of tosses
n = 1000
# Setting the probability value for having head
# in the simulation to H=0.3
H = 0.3
# Simulate n tosses with a probability of having head H
rng = random.default_rng()
random_tosses = rng.binomial(1, H, n)
# List of amount of tosses to be considered
valid_array_of_tosses = array([1, 50, 100, 300, 700, 1000])
# Total tosses to analyze the map
array_of_tosses = arange(n + 1)
# Create the inferno colormap with len of
# array_of_tosses colors
cmap = cm.get_cmap('inferno')
# Generate equally spaced colors
num_colors = len(array_of_tosses) + 100
colors = [cmap(i / num_colors) for i in range(num_colors)]
# Figure definition
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
# Generate uniformly spaced distribution
# from 0-1 for H values
start_H = 0
end_H = 1
amount_H = 5000
H_distribution = linspace(start_H, end_H, amount_H)
# Sigma definition and H prior definition for part 2
H_priors = [0.5, 0.3, 0.7]
sigma = 0.1
# Arrays creation for the MAP analysis
map_no_prior = zeros_like(array_of_tosses, dtype=float)
dict_priors = dict()
dict_priors[H_priors[0]] = zeros_like(array_of_tosses, dtype=float)
dict_priors[H_priors[1]] = zeros_like(array_of_tosses, dtype=float)
dict_priors[H_priors[2]] = zeros_like(array_of_tosses, dtype=float)
# Iteration over the amount of tosses that must be explored
for tosses in array_of_tosses:
    for H_prior in H_priors:
        # Likelihood calculation
        likelihood, gaussian_prior = likelihood_prior_calculation(
            H_distribution, random_tosses, tosses, H_prior, sigma
        )
        # Posterior (in this case is the same of the likelihood,
        # no priors defined)
        posterior_part_1 = normalize(likelihood)
        # Calculation of posterior with a gaussian prior
        # distribution with standard deviation sigma
        # centered at H_prior
        posterior_part_2 = normalize(likelihood * gaussian_prior)
        # Assignment of H_hat for the current amount of tosses
        map_no_prior[tosses] = H_distribution[argmax(posterior_part_1)]
        dict_priors[H_prior][tosses] = H_distribution[argmax(posterior_part_2)]
        # Plot the posteriors distribution only if in the selected tosses
        if tosses in valid_array_of_tosses and H_prior == 0.5:
            # Plot pt1
            ax1.plot(
                H_distribution, posterior_part_1,
                color=colors[tosses], label=f'{tosses} tosses', alpha=0.7
            )
            # Plot pt2
            ax2.plot(
                H_distribution, posterior_part_2, color=colors[tosses],
                label=f'{tosses} tosses', alpha=0.7
            )
        #
# Add properties to the plots
ax1.set_title('Posterior Distribution after N tosses (no priors)')
ax1.set_xlabel('H (Probability of heads)')
ax1.set_ylabel('Normalized density')
ax1.axvline(H, linestyle=":", color="g", label=f"H={H}", alpha=0.7)
ax1.legend()
ax1.grid(True)
fig1.savefig("./plot_1.png")
#
ax2.set_title(
    r'Posterior Distribution after N tosses (gaussian priors '
    r'$\mu={}$, $\sigma={}$)'.format(H_priors[0], sigma)
)
ax2.set_xlabel('H (Probability of heads)')
ax2.set_ylabel('Normalized density')
ax2.axvline(H, linestyle=":", color="g", label=f"H={H}", alpha=0.7)
ax2.legend()
ax2.grid(True)
fig2.savefig("./plot_2.png")
#
ax3.set_title("MAP for faster convergence")
ax3.set_xlabel('Number of tosses')
ax3.set_ylabel(r'$\hat{H}$')
ax3.plot(
    array_of_tosses, map_no_prior, color=colors[100], 
    label=f"MAP uniform prior", alpha=0.7
)
ax3.plot(
    array_of_tosses, dict_priors[H_priors[0]], color=colors[200],
    label=f"MAP gaussian prior centered at {H_priors[0]}", alpha=0.7
)
ax3.plot(
    array_of_tosses, dict_priors[H_priors[1]], color=colors[500],
    label=f"MAP gaussian prior centered at {H_priors[1]}", alpha=0.7
)
ax3.plot(
    array_of_tosses, dict_priors[H_priors[2]], color=colors[800],
    label=f"MAP gaussian prior centered at {H_priors[2]}", alpha=0.7
)
ax3.axhline(H, linestyle=":", color="g", label=f"H={H}", alpha=0.7)
ax3.legend()
ax3.grid(True)
ax3.set_xscale("log")
fig3.savefig("./plot_3.png")
