# Lewis Shemery

import numpy as np
import scipy as sc
from scipy.signal import freqz
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

def processFile(fn, offset) :
	fs, signal = read("P_9_2.wav")
	# applying the fft
	fft = np.fft.fft(signal)

	plt.figure()
	plt.plot(np.absolute(fft.real))
	plt.title('FFT Values')
	plt.show()

	median = len(fft.real)/2
	median = int(median)
	left = median-offset
	right = median+offset

	# setting the values around the midpoint to zero
	for i in range(left, right):
	    fft[i] = 0

	plt.figure()
	plt.plot(np.absolute(fft.real))
	plt.title('Cleaned FFT values')
	plt.show()

	# applying the inverse fft
	signal = np.fft.ifft(fft).real
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
	signal = np.int32(signal/np.max(np.abs(signal)) * 2147483647)

	write('cleanMusic.wav', fs, signal)

##############  main  ##############
if __name__ == "__main__":
    filename = "P_9_2.wav"
    offset = 20000

    # this function should be how your code knows the name of
    #   the file to process and the offset to use
    processFile(filename, offset)
