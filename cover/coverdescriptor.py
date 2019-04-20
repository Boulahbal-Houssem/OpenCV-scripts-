import cv2
import numpy as np

class CoverDescriptor:
	def __init__(self):
		orb = cv2.ORB_create()

	def describe(self,image):
		kp = orb.detect(image,None)
		kp, des = orb.compute(image, kp)
		return (kp,des)

	
