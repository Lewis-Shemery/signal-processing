# Lewis Shemery

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def applyShelvingFilter(inName, outName, g, fc) :
	signal, fs = sf.read(inName)
	original_fft = np.fft.fft(signal)
	length = len(original_fft)
	u = np.zeros(length)
	y = np.zeros(length)

	# Shelving filter
	theta = (2*np.pi*fc)/fs
	mu = 10**(g/20)
	gamma = (1-(4/1+mu)*np.tan(theta/2))/(1+(4/1+mu)*np.tan(theta/2))
	alpha = (1-gamma)/2

	# Difference equation
	for n in range(0, length):
	    u[n] = alpha*(signal[n]+signal[n-1])+gamma*u[n-1]
	    y[n] = signal[n]+(mu-1)*u[n]

	# Apply fft to filtered signal
	fft_filter = np.fft.fft(y)

	# Parameters for plotting
	x = np.arange(length)
	x = x*(fs/length)
	y_max = max(np.absolute(original_fft))+100 

	plt.figure(1)
	plt.subplot(1,2,1)
	plt.title('original signal')
	plt.ylim(0,y_max)
	plt.xlim(-200,fs/4)
	plt.xlabel('Hz')
	plt.plot(x,np.absolute(original_fft))

	plt.subplot(1,2,2)
	plt.title('filtered signal')
	plt.ylim(0,y_max)
	plt.xlim(-200,fs/4)
	plt.xlabel('Hz')
	plt.plot(x,np.absolute(fft_filter))

	plt.show()

	sf.write('shelvingOutput.wav', y, fs)

##########################  main  ##########################
if __name__ == "__main__" :
    inName = "P_9_1.wav"
    gain = -10  # can be positive or negative
                # WARNING: small positive values can greatly amplify the sounds
    cutoff = 300
    outName = "shelvingOutput.wav"

    applyShelvingFilter(inName, outName, gain, cutoff)
