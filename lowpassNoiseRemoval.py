# Lewis Shemery

import numpy as np
import scipy as sc
from scipy.signal import freqz
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math

def lowPassFilter(signal,fs):
    fc = 7500
    ft = fc/fs
    L = 101
    M = L-1
    h = [] # Low Pass Filter
    w = [] # Windowing values

    for i in range(L):
        if(i != M/2):
            weight = (np.sin(2*np.pi*ft*(i-M/2)))/(np.pi*(i-M/2))
            h = np.append(h,weight)
        else:
            weight = 2*ft
            h = np.append(h,weight)
    
    for i in range(L):
        hamming_weight = 0.54-(0.46*np.cos((2*np.pi*i)/M))
        w = np.append(w, hamming_weight)

    # Multiplying the original weights by the hamming weights
    c = np.multiply(h,w)
    filteredValues = np.convolve(signal,c)
    plotAndSave(h,c,filteredValues,fs)  
    
def plotAndSave(h,c,filteredValues,samplingRate):
    x,y = freqz(h,1) # Original frequency response
    hamming_x,hamming_y = freqz(c,1) # Frequency response with Hamming window

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    data = np.int32(filteredValues/np.max(np.abs(filteredValues)) * 2147483647)
    write("cleanMusic.wav", samplingRate, data)
    
    plt.plot(x,abs(y)) # Original
    plt.plot(hamming_x,abs(hamming_y)) # Windowed

    plt.gca().legend(('original','windowed'))
    plt.title("Frequency Response")
    plt.show()

def main():
    fs, signal = sc.io.wavfile.read("P_9_2.wav")
    lowPassFilter(signal,fs)     

main()