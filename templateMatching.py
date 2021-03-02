# Lewis Shemery

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import feature
#from skimage.feature import match_template
from skimage.color import rgb2gray

def findImage(mainImage, template):
	img_main = mpimg.imread(mainImage)
	img_temp = mpimg.imread(template)

	# Convert to grayscale
	img_main_gray = rgb2gray(img_main)
	img_temp_gray = rgb2gray(img_temp)

	plt.figure(0)
	plt.gray()
	plt.title('ERBwideGraySmall')
	plt.imshow(img_main_gray)
	plt.show()

	plt.figure(1)
	plt.gray()
	plt.title('ERBwideTemplate')
	plt.imshow(img_temp_gray)
	plt.show()

	# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
	output = feature.match_template(img_main_gray, img_temp_gray)

	length = img_main_gray[0,:].size
	width = img_main_gray[:,0].size

	length_temp = img_temp_gray[0,:].size
	width_temp = img_temp_gray[:,0].size

	# Find location of largest correlation
	row = np.argmax(np.max(output, axis=1))
	column = np.argmax(np.max(output, axis=0))   

	# Remove template from larger image
	for i in range(row, row+length_temp):
	    for j in range(column, column+width_temp):
	        img_main_gray[i][j] = 0

	plt.figure(2)
	plt.gray()
	plt.title('ERBwide minus Template')
	plt.imshow(img_main_gray)
	plt.show()

	return(row, column)
    
#############  main  #############
# this function should be how your code knows the names of
# the images to process
# it will return the coordinates of where the template best fits

if __name__ == "__main__":
    mainImage = "ERBwideColorSmall.jpg"
    template = "ERBwideTemplate.jpg"
    r, c = findImage(mainImage, template)

    print("coordinates of match = (%d, %d)" % (r, c))
