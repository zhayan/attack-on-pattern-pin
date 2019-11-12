import os
import sys

VEDIO_PATH = sys.argv[1]

VIDEO_NAME = str(VIDEO_PATH).split(".")[0]


os.system("python3 ./detection/detection.py " + VEDIO_NAME)

os.system("python3 ./tracking/finger-phone.py " + VEDIO_PATH)

os.system("python3 ./trajectory/simplify.py " + VEDIO_PATH)
