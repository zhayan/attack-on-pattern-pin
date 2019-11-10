import argparse
import csv
import sys
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

# Path of video
VIDEO_PATH = sys.argv[1]

VIDEO_NAME = str(VIDEO_PATH).split("/")[-1].split(".")[0]

DETECTION_PATH = "results/"+VIDEO_NAME+"/box.txt"

# Read Result
DETECTION_RESULT = open(DETECTION_PATH).readline().split(",")
print(DETECTION_RESULT)
DETECTION_RESULT = [ int(i) for i in DETECTION_RESULT[0:9]]

frameNO = DETECTION_RESULT[0]
finger_box = tuple(DETECTION_RESULT[1:5])
phone_box = tuple(DETECTION_RESULT[5:9])

# Read video
video = cv2.VideoCapture(str(VIDEO_PATH))

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
while frameNO > 0:
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    frameNO -= 1

print(str(finger_box))
print(str(phone_box))

X = []
Y = []

np.array(X)
np.array(Y)

finger_center = (int(finger_box[0] + finger_box[2] / 2), int(finger_box[1] + finger_box[3] / 2))
phone_center = (int(phone_box[0] + phone_box[2] / 2), int(phone_box[1] + phone_box[3] / 2))
X.append(finger_center[0] - phone_center[0])
Y.append(finger_center[1] - phone_center[1])

# Set up tracker.
# you can use
 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
 
if int(major_ver) < 3:
    finger_tracker = cv2.Tracker_create(tracker_type)
    phone_tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        finger_tracker = cv2.TrackerBoosting_create()
        phone_tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        finger_tracker = cv2.TrackerMIL_create()
        phone_tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        finger_tracker = cv2.TrackerKCF_create()
        phone_tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        finger_tracker = cv2.TrackerTLD_create()
        phone_tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        finger_tracker = cv2.TrackerMedianFlow_create()
        phone_tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        finger_tracker = cv2.TrackerGOTURN_create()
        phone_tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        finger_tracker = cv2.TrackerMOSSE_create()
        phone_tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        finger_tracker = cv2.TrackerCSRT_create()
        phone_tracker = cv2.TrackerCSRT_create()

finger_ok = finger_tracker.init(frame, finger_box)
phone_ok = phone_tracker.init(frame, phone_box)

pts = deque(maxlen=64)

plt.axis([-100, 100, -100, 100])
plt.ion()


while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        finger_ok, finger_box = finger_tracker.update(frame)
        phone_ok, phone_box = phone_tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if finger_ok and phone_ok:
            # Tracking success
            p1 = (int(finger_box[0]), int(finger_box[1]))
            p2 = (int(finger_box[0] + finger_box[2]), int(finger_box[1] + finger_box[3]))
            p3 = (int(phone_box[0]), int(phone_box[1]))
            p4 = (int(phone_box[0] + phone_box[2]), int(phone_box[1] + phone_box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(frame, p3, p4, (255,0,0), 2, 1)
            finger_center = (int(finger_box[0] + finger_box[2] / 2), int(finger_box[1] + finger_box[3] / 2))
            phone_center = (int(phone_box[0] + phone_box[2] / 2), int(phone_box[1] + phone_box[3] / 2))
            X.append(-(finger_center[0] - phone_center[0]))
            Y.append(finger_center[1] - phone_center[1])
            #plt.cla()
            plt.scatter(-(finger_center[0] - phone_center[0]), (finger_center[1] - phone_center[1]))
            #plt.scatter(X, Y, s=20, alpha=.5)
            plt.pause(0.1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        pts.appendleft(finger_center)
        for i in range (1,len(pts)):
            if pts[i-1]is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 1.5)
            cv2.line(frame, pts[i-1],pts[i],(0,0,225),thick)

        cv2.imshow("Tracking", frame)

        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

plt.savefig("results/"+VIDEO_NAME+"/track.png")

rows= []

for i in range(len(X)):
    row = {"frame":i,"X":X[i],"Y":-Y[i]}
    rows.append(row)

with open("results/"+VIDEO_NAME+"/track.csv","w") as f:
    rst_csv = csv.DictWriter(f,["frame","X","Y"])
    rst_csv.writeheader()
    rst_csv.writerows(rows)

video.release()
cv2.destroyAllWindows()
