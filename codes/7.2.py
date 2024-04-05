import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

#read .wav file 
input_signal, fs = sf.read('audio.wav') 

#sampling frequency of Input signal
sampl_freq = fs
print(sampl_freq)

#order of the filter
order = 4

#cutoff frquency 
cutoff_freq = 6000.0  

#digital frequency
Wn = 2 * cutoff_freq / sampl_freq  

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

# get partial fraction expansion
r, p, k = signal.residuez(b, a)
print(r)
print(p)
print(k)


#number of terms of the impulse response
sz = 100
sz_lin = np.arange(sz)

# Vectorized function to compute the impulse response
def rp_vec(x):
    return np.sum(r * p**x)

# Apply the vectorized function to sz_lin to compute impulse response
h1 = np.vectorize(rp_vec)(sz_lin)
k_add = np.pad(k, (0, sz - len(k)), 'constant', constant_values=(0, 0))
h = h1 + k_add

# Plotting
plt.stem(sz_lin, h.real)  
plt.xlabel('n')
plt.ylabel('h(n)')
plt.grid()
plt.savefig("h(n)_7.2.png")

