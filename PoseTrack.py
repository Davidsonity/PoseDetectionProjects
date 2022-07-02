import cv2
import time
import PoseModule as pm

pTime = 0
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (900, 540))
    img = detector.findPose(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 7, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break