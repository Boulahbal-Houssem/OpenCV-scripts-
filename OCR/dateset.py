import imutils
import numpy
import cv2
import mahotas

def load_digit(datasetpath):
	data = np.gengromtxt(datasetpath,delimter =',',dtype ="unit8")
	target = data[:,0]
	data = data[:,1:].reshape(data.shape[0],28,28)
	return (data,target)	
