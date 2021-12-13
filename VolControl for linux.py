import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # пересчитывать ли постоянно детекшн
        self.maxHands = maxHands  # макс кол-во рук
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon  # правильность детекции руки
        self.trackCon = trackCon  # правильность трекинга

        self.mpHands = mp.solutions.hands  # делаем класс на основе mediapipe
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)  # создаем объект рук из медиапайп внутри нашего класса

        self.mpDraw = mp.solutions.drawing_utils  # для отображения на экране

    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    # в след функции фишка в том, что если в кадре появится новая рука, трекинг слетит на нее, потому
    # я переписал в 37 строке [-1] вместо [N] а так надо подумать как это правильно реализовать, чтобы тречилась именно та
    # которая нужна
    def findPos(self, img, N=0,
                draw=True):  # h,w,c лучше сделать глобальными, чтобы не пресчитывать их каждый раз с учетом того что они константы
        res = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[-1]  # tracking hand number N
            for id, lm in enumerate(
                    hand.landmark):  # в идеале прописать трекинг произвольного количества рук с учетом проверки на наличие на каждом шагу
                cx, cy = (int(lm.x * w), int(lm.y * h))  # getting cords in pixels
                res.append([id, cx, cy])  # may be no sense adding id
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return res


def norm(a):
    return int((a[0] ** 2 + a[1] ** 2) ** (0.5))


def volume(x):
    if x <= 25:
        return str(0) + '%'
    elif x >= 150:
        return str(100) + '%'
    else:
        return str(int((x / 1.5))) + '%'


def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:  # going in loop in video
        succes, img = cap.read()  # getting frame to img

        img = detector.findHands(img)
        if detector.findPos(img):
            pinkie = np.array([int(i) for i in detector.findPos(img)[20]])
            control = np.array([int(i) for i in detector.findPos(img)[17]])
            a1 = np.array([int(i) for i in detector.findPos(img)[4]])
            a2 = np.array([int(i) for i in detector.findPos(img)[8]])
            # print(detector.findPos(img)[4]) # к примеру выведем положение кончика большого пальца IRL
            if  pinkie[2] > control[2]:
                print(pinkie)
                os.system('amixer set Master ' +  str((volume(norm(a1[1:] - a2[1:])))))  # 25 min 250 max
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow('Image', img)
        if cv.waitKey(1) == 0xFF & ord('q'):
            break


if __name__ == '__main__':
    main()
