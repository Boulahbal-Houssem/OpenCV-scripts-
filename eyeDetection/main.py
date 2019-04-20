import cv2
import numpy 
import EyeDetection
import argparse
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--eye", required = True,
	help = "path to where the face cascade resides")

	parser.add_argument("-v", "--video",action='store_true',
	help = "tracking from WebCam")
	args = vars(parser.parse_args())

	if not args['video']:
		print('camera not enabled')
		return False 
	cap = cv2.VideoCapture(0)
	while True:
		ret,frame = cap.read()
		if not ret:
			continue
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		eyeDetecotor =  EyeDetection.EyeDetectoin(args['eye'])
		eyes = 	eyeDetecotor.eyeDetect(gray)
		# draw a box around faces
		for (x,y,w,h) in eyes:
			img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			# draw image
		cv2.imshow('img',frame)
		
		if cv2.waitKey(1) == ord('q'):
			break



if __name__== "__main__" :
	main()
