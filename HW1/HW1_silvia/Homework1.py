#Coin tosses
import numpy as np
import matplotlib.pyplot as plt

H=0.3 #setting H to simulate the coin tosses
N=1000 #number of tosses
Ninteresting=[1,50,100,300,700,1000]  #array with the N we want to plot

tosses=np.random.binomial(1,H,N) #tosses simulated with binomial distribution with H=0.3
R=np.zeros_like(Ninteresting, dtype=int) #array to put heads count for Ninteresting


#Uniform prior
for i in range(len(Ninteresting)):  #iterating over the values Ninteresting
     
    R[i]=np.sum(tosses[:Ninteresting[i]])   #updating head count until Ninteresting[i] is reached

    Hdata=np.linspace(0,1,500)  #generating the x-values for the posterior
    Posterior=Hdata**R[i]*(1-Hdata)**(Ninteresting[i]-R[i]) #calculating the posterior
    
    Posterior=Posterior/np.max(Posterior) #normalization to have the maximum at 1

    plt.plot(Hdata,Posterior , '-',label=f'N={Ninteresting[i]},R={R[i]}') #plotting the posterior
    
plt.xlabel('H')
plt.ylabel('prob(H|data)')
plt.title('Posterior distribution: uniform prior')
plt.legend()
plt.show()

#Gaussian prior
for i in range(len(Ninteresting)): #iterating over the values Ninteresting
    
    R[i]=np.sum(tosses[:Ninteresting[i]])    #updating the head count until Ninteresting[i] is reached

    Hdata=np.linspace(0,1,500)
    prior = 1./(0.1*np.sqrt(2.*np.pi))*np.exp(-((Hdata - 0.5) ** 2) / (2 * (0.1 ** 2)))  #gaussian prior
    Posterior=Hdata**R[i]*(1-Hdata)**(Ninteresting[i]-R[i])*prior #calculating the posterior with the new prior
    
    Posterior=Posterior/np.max(Posterior) #normalization to have the maximum at 1

    plt.plot(Hdata,Posterior , '-',label=f'N={Ninteresting[i]},R={R[i]}')  #plotting the posterior
    
plt.xlabel('H')
plt.ylabel('prob(H|data)')
plt.title('Posterior distribution: gaussian prior')
plt.legend()
plt.show()



