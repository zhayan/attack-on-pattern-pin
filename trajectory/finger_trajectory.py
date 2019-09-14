import argparse
import sys
from collections import deque

import cv2
import numpy as np

print(sys.argv)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

# Get four point by click
def getclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(phone_shape) < 4:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        phone_shape.append((x, y))



# Path of video
file = sys.argv[1]

 # Read video
video = cv2.VideoCapture(str(file))
 
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Box for finger (top-left.x, top-left.y, bottom-right.x, bottom-right.y)
if len(sys.argv) > 2:
    finger_box = sys.argv[2]
else:
    finger_box = cv2.selectROI(frame, False)

# shape of phone (top-left.x, top-left.y, top-right.x, top-right.y,
#                  bottom-left.x, bottom-left.y, bottom-right.x, bottom-right.y)
if len(sys.argv) > 3:
    phone_shape = sys.argv[3]
else:
    phone_shape = []
    cv2.namedWindow('GetPhone')
    cv2.setMouseCallback("GetPhone", getclick)
    while len(phone_shape) <4:
        cv2.imshow('GetPhone', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    print(phone_shape)

# Set up tracker.
# Instead of MIL, you can also use
 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[0]
 
if int(major_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

ok = tracker.init(frame, finger_box)

pts = deque(maxlen=64)

pts1 = np.float32(phone_shape)
pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, finger_box = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(finger_box[0]), int(finger_box[1]))
            p2 = (int(finger_box[0] + finger_box[2]), int(finger_box[1] + finger_box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            center = (int(finger_box[0] + finger_box[2] / 2), int(finger_box[1] + finger_box[3] / 2))
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        pts.appendleft(center)
        for i in range (1,len(pts)):
            if pts[i-1]is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i-1],pts[i],(0,0,225),thick)
		

        #out.write(frame)
        result = cv2.warpPerspective(frame, matrix, (500, 600))
 

        # Display result
        # cv2.namedWindow("Tracking",0);
        # cv2.resizeWindow("Tracking", 640, 480);
        cv2.imshow("Tracking", frame)
        cv2.imshow("Result", result)



        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
