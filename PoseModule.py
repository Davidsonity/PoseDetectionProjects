import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation=False, smooth_segmentation=True,
                detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.segmentation,
                                     self.smooth_segmentation, self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self,img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            my_pose = self.results.pose_landmarks
            for id, lm in enumerate(my_pose.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmlist



def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (900, 540))
        img = detector.findPose(img)
        lmlist = detector.findPosition(img, draw=False)
        if len(lmlist) != 0:
            print(lmlist)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()