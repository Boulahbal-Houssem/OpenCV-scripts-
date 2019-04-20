import cv2
import keras
class EyeDetectoin:
	def __init__(self,casacade_path):
		self.eye_cascade = cv2.CascadeClassifier(casacade_path)
	def eyeDetect(self,img):
		return self.eye_cascade.detectMultiScale(img)
print()