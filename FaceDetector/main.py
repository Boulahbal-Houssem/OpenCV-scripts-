import argparse
import cv2
from facedetector import FaceDetector

def main():
	# Loading the cascade xml file
	parser = argparse.ArgumentParser()

	parser.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")

	parser.add_argument("-v", "--video",action='store_true',
	help = "detection from WebCam")

	parser.add_argument("-i", "--image",
	help = "path to where the image resides")
	
	args = vars(parser.parse_args())
	img = cv2.imread(args["image"],1)
	if (args["video"]):
		# cap points to the WebCam
		cap = cv2.VideoCapture(0)
		while True:
			# Getting images from WebCam
			ret , img = cap.read()	
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# using FaceDetector class to detecte the face
			faceDetector = FaceDetector(args['face'])
			faces = faceDetector.detect(gray)
			# draw a box around faces
			for (x,y,w,h) in faces:
			    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			# draw image
			cv2.imshow('img',img)
			if cv2.waitKey(1) == ord('q'):
				cv2.destroyAllWindows()
				break
		cap.release()
	elif  img != None :
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# using FaceDetector class to detecte the face
		faceDetector = FaceDetector(args['face'])
		faces = faceDetector.detect(gray)
		# draw a box around faces
		for (x,y,w,h) in faces:
		    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		# draw image
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print("Please enter the correct arguments") 		


if __name__ == '__main__':
    main()
