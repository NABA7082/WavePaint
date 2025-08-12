import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 100

script_dir = os.path.dirname(os.path.abspath(__file__))
folderPath = os.path.join(script_dir, "Header")
if not os.path.exists(folderPath):
    raise FileNotFoundError(f"❌ Header folder not found: {folderPath}")


myList=os.listdir(folderPath)
print("Found files:", myList)
print("Total files:", len(myList))


overlayList = []

for imPath in myList:
    image_path = os.path.join(folderPath, imPath)
    image = cv2.imread(image_path)
    
    overlayList.append(image)


header=overlayList[0]

drawColor = (0, 0, 0)  # Default color is red

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.handDetector(detectionCon=0.75, trackCon=0.75)
xp,yp=0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2.Find hand landmarks

    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    # print(lmlist)
    if len(lmlist) != 0:
       
       
        #x1,y1=lmlist[8][1],lmlist[8][2]  # Index finger tip
        #x2,y2=lmlist[12][1],lmlist[12][2]  # Middle finger tip

        x1,y1=lmlist[8][1],lmlist[8][2]  # Index finger tip
        x2,y2=lmlist[12][1],lmlist[12][2]  # Middle finger tip

        #3.check which fingers are up

        fingers = detector.fingersUp()
        print(fingers)
        #4.Selection-two fingers up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.putText(img, "Selection Mode", (x1+50, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #cv2.rectangle(img,(x1-20,y1-20),(x2+20,y2+20),(0,0,255),2,cv2.FILLED)

            print("Selection Mode")

            if y1<130:
                if 250 < x1 < 450:
                    header = overlayList[1]
                    drawColor = (255, 0, 255)

                elif  550<  x1 < 750:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2] 
                    drawColor = (0, 255, 0)    

                elif 1050 < x1 < 1200:
                    header = overlayList[3] 
                    drawColor=(0,0,0)  
            cv2.rectangle(img,(x1-20,y1-20),(x2+20,y2+20),drawColor,2,cv2.FILLED)
              

        #5.Drawing mode-Index finger up

        if fingers[1] and not fingers[2]:
            cv2.putText(img, "Drawing Mode", (x1+50, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==-0:

                xp, yp = x1, y1

            if drawColor == (0, 0, 0):  # Eraser mode
                cv2.line(img, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),drawColor, eraserThickness)  

            else:     
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        if all(fingers):
            cv2.putText(img, "Resetting", (x1+50, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            xp, yp = 0, 0

    frame_height, frame_width = img.shape[:2]

    # Resize header to match webcam width, fixed height (e.g. 125 pixels)
    header_height = 125
    header_resized = cv2.resize(header, (frame_width, header_height))

    # Overlay header at top of frame
    img[0:header_height, 0:frame_width] = header_resized

    img=cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Virtual Painter", img)
    # cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
   

