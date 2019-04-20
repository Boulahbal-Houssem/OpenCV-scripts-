import cv2 
import numpy as np
import argparse
def main():
	parser = argparse.ArgumentParser(description = "Video Tracking script")
	parser.add_argument	("-v","--video",
				help = "path to the video [optional]")
	args = vars(parser.parse_args())
	blower = np.array([100, 67, 0], dtype = "uint8")
	bupper = np.array([255, 128, 50], dtype = "uint8")
	cap = args["video"] if args["video"] != None else int(0)
	cap = cv2.VideoCapture(cap)
	while True:
		done, frame = cap.read()
		if not done :
			continue 
		det = cv2.inRange(frame,blower,bupper)
		det = cv2.GaussianBlur(det , (3,3),0)
		_,cnts,_ = cv2.findContours(det.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		if len(cnts)>0:
			cnt = sorted ( cnts, key =cv2.contourArea , reverse = True)[0]
			rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
			cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)
		cv2.imshow("frame",frame)
		cv2.imshow("binary",det)
		if cv2.waitKey(1) == ord('q'):
			break
	

if __name__ == '__main__':
	main()
