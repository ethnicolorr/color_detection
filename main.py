import time

import cv2
import numpy as np
import const
# from numba import njit


def keyboard(cap):
    match cv2.waitKeyEx(1):
        case 2424832:  # влево
            if const.x1 != 0:
                const.x1 = const.x1 - const.shift
                const.x2 = const.x2 - const.shift
        case 2490368:  # вверх
            if const.y1 != 0:
                const.y1 = const.y1 - const.shift
                const.y2 = const.y2 - const.shift
        case 2555904:  # вправо
            if const.x2 <= cap.get(3):
                const.x1 = const.x1 + const.shift
                const.x2 = const.x2 + const.shift
        case 2621440:  # вниз
            if const.y2 <= cap.get(4):  # down
                const.y1 = const.y1 + const.shift
                const.y2 = const.y2 + const.shift


def detectColor(hsv_frame):
    mask_green = cv2.inRange(hsv_frame, const.lower_green, const.upper_green)
    mask_purple = cv2.inRange(hsv_frame, const.lower_purple, const.upper_purple)
    mask_yellow = cv2.inRange(hsv_frame, const.lower_yellow, const.upper_yellow)
    sums = [np.sum(mask_green), np.sum(mask_purple), np.sum(mask_yellow)]
    ix = np.argmax(sums)
    if sums[ix] > 0:
        return matchColor(ix)
    else:
        return 'unknown'


# @njit
def matchColor(ix):
    match ix:
        case 0: return 'green'
        case 1: return 'purple'
        case 2: return 'yellow'


# @njit
def detectColorNp(hsv_frame):
    height, width, channels = hsv_frame.shape
    colors = []
    for w in range(0, width):
        for h in range(0, height):
            px = hsv_frame[h, w]
            if (const.lower_green < px).all() and (const.upper_green > px).all():
                colors.append(0)
            elif (const.lower_purple < px).all() and (const.upper_purple > px).all():
                colors.append(1)
            elif (const.lower_yellow < px).all() and (const.upper_yellow > px).all():
                colors.append(2)
    if len(colors) == 0:
        return 'unknown'
    else:
        counts = np.bincount(np.array(colors))
        return matchColor(np.argmax(counts))


def main(file):
    start = time.time()
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        hsv_frame = cv2.cvtColor(frame[const.y1: const.y2, const.x1:const.x2], cv2.COLOR_BGR2HSV)
        keyboard(cap)
        cv2.rectangle(frame, (const.x1, const.y1), (const.x2, const.y2), (0, 0, 255), 2)
        cv2.putText(frame, detectColor(hsv_frame), (const.x1 + 10, const.y1 + 80), 0, 0.6, (0, 0, 255), 2)
        cv2.imshow('{}'.format(file[5:-4]), frame)
        if cv2.waitKeyEx(1) == 27:
            break
    finish = time.time()
    return finish-start


time_spent = main("data/jaba.mp4")
print(time_spent)

