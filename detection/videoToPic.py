# 视频分解图片
# 1 load 2 info 3 parse 4 imshow imwrite
import cv2
# import PIL.image as img
import time
import os

'''config'''
# Root directory of the project
ROOT_DIR = "/backup/xym1/test1/"
videopath = os.path.join(ROOT_DIR, "test_data/input_video")
outdir = os.path.join(ROOT_DIR, "test_data/input_images")
def main(videopath, step，nframe = 10，startnum = 0，videonum = 30):
    if os.path.isdir(path):
        for video in os.listdir(videopath):
            # 获取一个视频打开cap 1 file name
            cap = cv2.VideoCapture(videopath + video) 
            # 判断是否打开 
            is_opened = cap.isOpened  
            if not os.path.exists(outdir + video[0:11]):
                os.makedirs(outdir + videoname[0:11])
            # fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # w h
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(fps,width,height)
            i = 0
            while is_opened:
                if i == nframe * step:
                    break
                else:
                    i = i + 1
                (flag, frame) = cap.read()  # 读取每一张 flag frame
                if i % step == 0:
                    filename = outdir + videoname[0:11] + "/" + str(i + startnum) + '.png'
                    print(filename)
                    if flag:
                        cv2.imwrite(filename, frame)
                time.sleep(0.01)
            startnum += nframe
            if startnum > videonum * nframe:
                break
    elif os.path.isfile(path):
        # 获取一个视频打开cap 1 file name
        cap = cv2.VideoCapture(videopath) 

        #创建文件夹
        if not os.path.exists(outdir + video[0:11]):
            os.makedirs(outdir + videoname[0:11])

            # 判断是否打开 
            is_opened = cap.isOpened  
            
            while is_opened:
                if i == nframe * step:
                    break
                else:
                    i = i + 1
                (flag, frame) = cap.read()  # 读取每一张 flag frame
                if i % step == 0:
                    filename = outdir + videoname[0:11] + "/" + str(i + startnum) + '.png'
                    print(filename)
                    if flag:
                        cv2.imwrite(filename, frame)
                time.sleep(0.01)
            startnum += nframe
    else:
        videopath = videopath + ".mov"
        # 获取一个视频打开cap 1 file name
        cap = cv2.VideoCapture(videopath) 

        #创建文件夹
        if not os.path.exists(outdir + video[0:11]):
            os.makedirs(outdir + videoname[0:11])

            # 判断是否打开 
            is_opened = cap.isOpened  
            
            while is_opened:
                if i == nframe * step:
                    break
                else:
                    i = i + 1
                (flag, frame) = cap.read()  # 读取每一张 flag frame
                if i % step == 0:
                    filename = outdir + videoname[0:11] + "/" + str(i + startnum) + '.png'
                    print(filename)
                    if flag:
                        cv2.imwrite(filename, frame)
                time.sleep(0.01)
            startnum += nframe
    print('end!')
    


if __name__ == '__main__':
    inputname = "SSSM-000002.mov"
    step = 1
    main(inputname, step)
