import cv2
import numpy as np
import pandas as pd

eyes_cascade = cv2.CascadeClassifier('snapchat filter/classifier/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('snapchat filter/classifier/Nose18x15.xml')

cap = cv2.VideoCapture(0)
glasses = cv2.imread('snapchat filter/glasses.png', cv2.IMREAD_UNCHANGED)
mustache = cv2.imread('snapchat filter/mustache.png', cv2.IMREAD_UNCHANGED)

while True:

	ret, frame = cap.read()

	if ret == False:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	eyes = eyes_cascade.detectMultiScale(gray, 1.1, 15)
	noses = nose_cascade.detectMultiScale(gray, 1.1, 15)	

	for eye in eyes:
		x, y, w, h = eye
		glasses = cv2.resize(glasses, (w, h))
		# frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		for i in range(glasses.shape[0]):
			for j in range(glasses.shape[1]):
				if (glasses[i, j, 3] > 0):
					frame[i+y, j+x, :] = glasses[i, j, 0:3]

	for nose in noses:
		x, y, w, h = nose
		mustache = cv2.resize(mustache, (int(mustache.shape[1]/mustache.shape[0]*int(h/1.35)), int(h/1.35)))
		# frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		for i in range(mustache.shape[0]):
			for j in range(mustache.shape[1]):
				if (mustache[i][j][3] > 0):
					frame[int(i+y+.5*h)][int(j+x+w/2-mustache.shape[1]/2)] = mustache[i][j][0:3]

	cv2.imshow("FRAME", frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break