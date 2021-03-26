import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False,2,0.5,0.5) #False sometimes detect other times tracking based on confidence
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #just processing not detection
    #print(results.multi_hand_landmarks)
    #to check for multiple hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id==0:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)#to draw the hand

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_TRIPLEX,3,(255,0,255),3)


    cv2.imshow("Hand Track",img)
    key = cv2.waitKey(1)
    if key == ord('q'):  # once you enter 'q' the window will be destroyed
        break

cap.release()
cv2.destroyAllWindows()