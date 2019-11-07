import os
import sys

VEDIO_PATH = sys.argv[1]
VEDIO_NAME = str(VEDIO_PATH).split("/")[-1].split(".")[0]

os.system("python3 ./detection/detection.py" + VEDIO_PATH)
