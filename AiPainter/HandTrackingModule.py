import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands        
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands( static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds=[4, 8, 12, 16, 20]

    def findHands(self,img):
        #img=cv2.flip(img,1)       
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
             for handlms in self.results.multi_hand_landmarks:
                 self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        return img 

    def findPosition(self, img, handNo=0, draw=True): 
        self.listHand = [] 

        if self.results.multi_hand_landmarks: 
            handlms = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(handlms.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.listHand.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                
        return self.listHand
    

    def fingersUp(self):
        fingers = []
    # Thumb
        if self.listHand[self.tipIds[0]][1] < self.listHand[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # Fingers
        for id in range(1, 5):
            if self.listHand[self.tipIds[id]][2] < self.listHand[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers
    

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            
            # print("Landmarks:", lmList)
            # print("Fingers up:", detector.fingersUp()) # Print landmark positions

         cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


