import soundfile as sf
from scipy import signal

#read .wav file 
input_signal,fs = sf.read('audio.wav') 

#sampling frequency of Input signal
sampl_freq=fs

#order of the filter
order=4   

#cutoff frquency 1kHz
cutoff_freq=6000 

#digital frequency
Wn=2*cutoff_freq/sampl_freq  

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order,Wn, 'low') 
print(a)
print(b)
#filter the input signal with butterworth filter
#output_signal = signal.filtfilt(b, a, input_signal)
output_signal = signal.lfilter(b, a, input_signal)

#write the output signal into .wav file
sf.write('Sound-with-Reduced-noise.wav', output_signal, fs) 
