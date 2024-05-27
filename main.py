from ultralytics import YOLO
import cv2


#load yolov8 model

model = YOLO('yolov8n.pt')


#load video
video_path = './test_5.mp4'
cap = cv2.VideoCapture(video_path) 

ret = True


#read frames
while ret:
    ret, frame = cap.read()


    if ret:



#detect objects




#track objects
        results = model.track(frame, persist=True)  # persist helps to remember the tracked object



#plot results
#puts the rectangles cv2.rectangle cv2.text
        frame_ = results[0].plot()




#visualize
        cv2.imshow('frame', frame_)
#press q to exit the visualize
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break



### we can use which command to find out the path
##PAth to python3   /Users/daudiqbal/Library/Python/3.9/bin
###also path to pip3 and python3 at /usr/bin

### also path stored in /usr/local/opt

#######daudiqbal@Dauds-Air ~ % python3
##Python 3.9.6 (default, Mar 10 2023, 20:16:38) 
##[Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
##Type "help", "copyright", "credits" or "license" for more information.
##>>> import sys
##>>> print(sys.executable)
###/Library/Developer/CommandLineTools/usr/bin/python3//