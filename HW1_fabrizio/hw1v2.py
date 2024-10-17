########### tossing coins
'''
Consider the coin tossing example, discussed in the first lecture. Simulate
1000 tosses of the coins, setting H = 0.3. Consider a uniform prior and
update the posterior at each toss. Plot the resulting posterior after 1, 50,
100, 300, 700, 1000 tosses. Repeat the simulated experiment by setting
a Gaussian prior centered in H = 0.5, with standard deviation  = 0.1.
Do both posteriors converge a similar distribution in the end? What does
that mean? Which posterior converges faster and why?
'''
#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Parameters
H = 0.3  #probability of heads
N = 1000  #number of tosses
obs_tosses = [0, 1, 50, 100, 300, 700, 1000] #number of tosses for plotting

#Simulating tosses (1 = head, 0 = tail)
tosses = np.random.binomial(1, H, N)

#possible values of H
h_range = np.linspace(0, 1, 1000)

#first with a uniform prior (beta function with alpha=beta=1 or i just put nothing)
#in this case the posterior is just the likelihood, which follows a binomial distribution

#plotting all the distriubtions
for i in range(len(obs_tosses)):
    N_heads = np.sum(tosses[:obs_tosses[i]])
    posterior = (h_range ** N_heads) * ((1 - h_range) ** (obs_tosses[i] - N_heads))
    plt.plot(h_range, posterior/np.max(posterior), label=f"number of tosses: {obs_tosses[i]}", linewidth = 1)

#plotting the real value dashed
plt.axvline(x=0.3, color='grey', linestyle='--', label='true H=0.3', alpha = 1)

plt.legend() 
plt.xlabel("H")
plt.ylabel("P(H|data)")
plt.title("Distriution with a uniform prior")   
plt.show()

### now i just do the same but putting a gaussian prior

for i in range(len(obs_tosses)):
    N_heads = np.sum(tosses[:obs_tosses[i]])
    posterior = (h_range ** N_heads) * ((1 - h_range) ** (obs_tosses[i] - N_heads)) * norm.pdf(h_range, loc=0.5, scale=0.1)
    plt.plot(h_range, posterior/np.max(posterior), label=f"number of tosses: {obs_tosses[i]}", linewidth = 1)

plt.axvline(x=0.3, color='grey', linestyle='--', label='true H=0.3', alpha = 1)
plt.legend() 
plt.xlabel("H")
plt.ylabel("P(H|data)")
plt.title("Distriution with a gaussian prior")   
plt.show()

#####let's compute how fast the two distributions converge
uniform_max = [0]*N
gaussian_max = [0]*N

for i in range(N):
    N_heads = np.sum(tosses[:i])
    uniform = (h_range ** N_heads) * ((1 - h_range) ** (i - N_heads))
    gaussian = (h_range ** N_heads) * ((1 - h_range) ** (i - N_heads))*norm.pdf(h_range, loc=0.5, scale=0.1)

    uniform_max[i] = np.argmax(uniform)/1000
    gaussian_max[i] = np.argmax(gaussian)/1000

plt.plot(uniform_max, label = 'uniform')
plt.plot(gaussian_max, label = 'gaussian')
plt.legend()
plt.grid()
plt.xlabel('Number of tosses')
plt.ylabel('MAP')
plt.xscale('log')
plt.show()