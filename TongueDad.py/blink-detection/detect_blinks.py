# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyautogui

pyautogui.FAILSAFE = True
width, height = pyautogui.size()

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
'''ap' = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_arg'''

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_CLOSE_THRESH = 0
EYE_OPEN_THRESH = 0
EYE_CLOSE_CONSEC_FRAMES = 2
EYE_OPEN_CONSEC_FRAMES = 4

# initialize the frame counters and the total number of blinks
LCOUNTER = 0
LTOTAL = 0
RCOUNTER = 0
RTOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

fullcalibrated = False
closedcalibrated = False
opencalibrated = False
# loop over frames from the video stream
(x,y,w,h) = (0,0,0,0)
while True:
	key = cv2.waitKey(1) & 0xFF
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = cv2.flip(frame,1)
	frame = imutils.resize(frame, width=1000)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)


		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftBrow = shape[lbStart:lbEnd]
		rightBrow = shape[rbStart:rbEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftdist = (leftEye[4][1] - leftBrow[4][1])
		rightdist = (rightEye[0][0] - rightBrow[0][0])
		avgdist = (leftdist + rightdist)/2
		if not fullcalibrated:
			cv2.putText(frame, "Please stare at the screen and press c to calibrate", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, "EAR: {:.3f}".format(ear), (650, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if key == ord("c"):
				EYE_CLOSE_THRESH = ear - (ear*.090)
				EYE_OPEN_THRESH = avgdist + (avgdist * .05)
				fullcalibrated = True

		else:
			(x,y,w,h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# show the face number
			cv2.putText(frame, "Face", (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			leftBrowHull = cv2.convexHull(leftBrow)
			rightBrowHull = cv2.convexHull(rightBrow)
			cv2.drawContours(frame, [leftBrowHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightBrowHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_CLOSE_THRESH:
				LCOUNTER += 1

			else:
				if LCOUNTER >= EYE_CLOSE_CONSEC_FRAMES:
					LTOTAL += 1
					pyautogui.click(pyautogui.position(),button="left")
				LCOUNTER = 0

			if avgdist > EYE_OPEN_THRESH:
				RCOUNTER += 1
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if RCOUNTER >= EYE_OPEN_CONSEC_FRAMES:
					RTOTAL += 1
					pyautogui.click(pyautogui.position(),button="right")

				RCOUNTER = 0

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Left Clicks: {}".format(LTOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Right Clicks: {}".format(RTOTAL), (10, 70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Dist: {}".format(avgdist), (300, 70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	# show the frame
	cv2.imshow("Frame", frame)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
