#framer program
# 
# all imports 
import cv2    
import math   

frame_count = 0
videoFiles = ['gettinginto_1.mp4', 'gettinginto_2.mp4', 'gettingout_1.mp4', 'gettingout_2.mp4', 'gettinginto_3.mp4', 'neither_1.mp4', 'gettingout_3.mp4', 'gettingout_4.mp4', 'gettinginto_4.mp4']

for each in videoFiles:
    cap = cv2.VideoCapture(each)
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    #print(frameRate)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(math.floor(frameRate)/2) == 0): #getting two frames per second of video
            filename ="frame%d.jpg" % frame_count;frame_count+=1
            cv2.imwrite(filename, frame)
    cap.release()
print("Done!")