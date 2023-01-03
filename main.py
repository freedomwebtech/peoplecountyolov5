import cv2
import torch
import numpy as np
from vidgear.gears import CamGear

points=[]
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)            
    
           


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#cap=cv2.VideoCapture('parking.mp4')
count=0

list=['person','truck','car']

stream = CamGear(source='https://www.youtube.com/watch?v=NWdiyp8MzpA', stream_mode = True, logging=True).start() # YouTube Video URL as input
area=[(621,340),(184,405),(193,427),(219,430),(246,538),(246,599),(842,560)]

while True:
    frame = stream.read()
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    results=model(frame)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2

        if 'person' in d:
            results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)    
               cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
               list.append([cx])
    l=len(list)           
    cv2.putText(frame,str(l),(50,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
   

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
