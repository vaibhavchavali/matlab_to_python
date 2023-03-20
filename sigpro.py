import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import dft
# This script compares the angle between the highest eigenvector of a complex Wishart Sample Covariance Matrix and the ensemble "true" basis vector. The Random Matrix Theory (RMT) predictions are obtained using Theorem 4 of 
# D. Paul, “Asymptotics of Sample Eigenstructure for a Large Dimensional Spiked Covariance Model,” Statistica Sinica, vol. 17, no. 4, pp. 1617–1642, 2007.


# Functions 
# This function provides a complex Gaussian random vector with a "mean" and standard deviation "sigma"
# The function takes the following inputs:
#               nrow - Number of rows
#               ncol - Number of columns
#               mean - mean of the random variable/vector
#               sigma - standard deviation of the random vector 
# 
# The function returns the random variable/vector 
#           rand_var - of size nrow x ncol
#         
def proper_complex_gauss(nrow,ncol,mean,sigma):
    rand_var=mean+np.sqrt(sigma/2)*(np.random.standard_normal((nrow,ncol))+1j*np.random.standard_normal((nrow,ncol)))
    return rand_var


# This function returns the eigenvalues and the associated  eigenvectors of a square NxN matrix in an descending order 
# The function takes input arguments: 
#           R - square matrix of size N x N (possibly Hermitian)
# Ouput: 
#       eigen_vec_sorted - Eigen vectors sorted according to associated eigenvalues 
#       eigen_val_sorted - Eigen values sorted in a descending order             
def eig_sort_descending(R):
    eval,evec=np.linalg.eig(R)
    ind=np.flip(eval.argsort())
    eigen_val_sorted = eval[ind]
    eigen_vec_sorted = evec[:,ind]
    return eigen_vec_sorted,eigen_val_sorted

# This function replicates the Random Matrix Theory (RMT) results for the angle between the  eigenvector associated with the highest eigenvalue and the "true" signal basis vector. This is the result from Theorem 4 in [1]- D. Paul, “Asymptotics of Sample Eigenstructure for a Large Dimensional Spiked Covariance Model,” Statistica Sinica, vol. 17, no. 4, pp. 1617–1642, 2007.
#
#   Input:
#       N - Number of rows (in our case number of sensors)
#       L - Number of columns (in our case number of snapshots)
#       sig_power - the power of the spike (or signal)
#   Output: 
#       cos_square_RMT - The cosine-squared angle between the estimated and true eigenvector (normalized to unity)     
def RMT_predicted_cosine_square(N,L,sig_power):
    cos_square_RMT = np.zeros((np.size(sig_power),1))
    c=N/L
    for k in range(np.size(sig_power)):
        if sig_power[k] > np.sqrt(c):
            cos_square_RMT[k]=(1 - (c/(sig_power[k] )**2))/(1+ (c/(sig_power[k])))
        else:  
            cos_square_RMT[k]=0 
    return cos_square_RMT

# Variables 
N = 50 # Number of Sensors 
d = 1 # Spacing between the sensors (in m)
n = np.array(np.linspace(0,(N-1)*d, num=N)) # Sensor positions along z-axis
V=dft(N) # The DFT - matrix represents the "true basis vectors" for the signals


MC_trials = 10**3 # number of Monte Carlo trials
L = 100 # Number of snapshots in time 
signal_power = 10**(np.arange(-2, 2.1, 0.1)) # in decibels (powers of 10)
noise_power = 1 # Unity noise power 

c = N/L # The RMT ratio of number of rows/columns or specifically, sensors/snapshots
print('The RMT ratio of number of sensors/snapshots = '+str(c))
# plot the array locations 
test=np.ones((N,1))
plt.plot(n,test,'o',markerfacecolor="None",markeredgecolor='blue')
plt.xlabel('Sensor Locations (in m)')
plt.grid()
plt.title('Uniform Linear Array with N = '+str(N)+', d = '+str(d)+' m')
plt.show()
plt.close()



cos_square_val=np.zeros((np.size(signal_power),1))
for k in range(np.size(signal_power)):
    for trials in range(MC_trials):
        signal=proper_complex_gauss(1,L,0,signal_power[k]/N)
        noise=proper_complex_gauss(N,L,0,noise_power)
        x=np.outer(V[:,0],signal) + noise # This is the simulated data vector 
        Rx_estimate=np.inner(x,x.conj())/L # Sample Covariance Matrix estimate
        eigen_vec,eigen_val = eig_sort_descending(Rx_estimate)
        cos_square_val[k]= cos_square_val[k] + abs(np.inner(V[:,0]/np.sqrt(N),eigen_vec[:,0]))**2 # Storing the cosine-square values using simulated data

# print(['%.2f' % elem for elem in abs(eigen_val)])

mc_trial_avg_cosine_square = cos_square_val/MC_trials
SNR = 10*np.log10(signal_power)

cos_square_RMT = RMT_predicted_cosine_square(N,L,signal_power)
plt.plot(SNR,mc_trial_avg_cosine_square,'bo--',SNR,cos_square_RMT,'r*:',markerfacecolor="none")
plt.xlabel('Output Signal-to-Noise Ratio (SNR) in dB')
plt.ylabel ('cosine-square ')
plt.legend(['SCM - estimate','RMT - prediction'])
plt.grid()
plt.xlim((SNR[0],SNR[np.size(signal_power)-1]))
plt.ylim((0,1))
plt.title('Prediction of the RMT results with simulations when c'+str(c))
plt.show()
plt.close()
