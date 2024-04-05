import soundfile as sf
from scipy import signal, fft
import numpy as np
from numpy.polynomial import Polynomial as P
from matplotlib import pyplot as plt

def myfiltfilt(b, a, input_signal):
    X = fft.fft(input_signal)
    w = np.linspace(0, 1, len(X) + 1)
    W = np.exp(2j*np.pi*w[:-1])
    B = (np.absolute(np.polyval(b, W)))**2
    A = (np.absolute(np.polyval(a, W)))**2
    Y = B*(1/A)*X
    return fft.ifft(Y).real

input_signal, fs = sf.read('audio.wav')

if input_signal.ndim == 2:
    input_signal = input_signal[:, 0]

sampl_freq = fs
order = 4
cutoff_freq = 6000.0
Wn = 2 * cutoff_freq / sampl_freq
b, a = signal.butter(order, Wn, 'low')
output_signal = signal.filtfilt(b, a, input_signal)
op1 = myfiltfilt(b, a, input_signal)

x_plt = np.arange(len(input_signal))
plt.plot(x_plt[8000:16000], output_signal[8000:16000], 'b.', label='Output by built-in function')
plt.plot(x_plt[8000:16000], op1[8000:16000], 'r.', label='Output by not using built in function')
plt.title("Verification of outputs of Audio Filter")
plt.grid()
plt.legend()
plt.savefig("Audio_Filter_verf.png")
plt.show()
