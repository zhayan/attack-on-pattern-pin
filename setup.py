import os
import sys

VEDIO_PATH = sys.argv[1]

os.system("python3 ./detection/detection.py " + VEDIO_PATH)

os.system("python3 ./tracking/finger-phone.py " + VEDIO_PATH)

os.system("python3 ./trajectory/simplify.py " + VEDIO_PATH)
