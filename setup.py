import os
import sys

VIDEO_PATH = sys.argv[1]

VIDEO_NAME = str(VIDEO_PATH).split("/")[-1].split(".")[0]


os.system("python3 ./detection/detection.py " + VIDEO_NAME)

os.system("python3 ./tracking/finger-phone.py " + VIDEO_PATH)

os.system("python3 ./trajectory/simplify.py " + VIDEO_PATH)
