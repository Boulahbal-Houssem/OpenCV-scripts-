import cv2
import numpy
# FaceDetector class used to encpasualte the detecting 
class FaceDetector:
	#one single parametre that points to the xml file path
	def __init__(self,faceCascadePath): 
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
	# detecting using haar classifier method
	def detect(self,image,scaleFactor =1.5,minNeighbors = 5,minSize = (30,30)) :
		ret = self.faceCascade.detectMultiScale(image,
			scaleFactor = scaleFactor,
			minNeighbors = minNeighbors, minSize = minSize,
			flags = cv2.CASCADE_SCALE_IMAGE)
		return ret


