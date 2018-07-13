#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-22 20:39:27
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$

import dlib
import cv2
import numpy as np
import math
import time
from movefilter import ave_move

# fiterx = ave_move()
# fitery = ave_move()

predictor_path = "./shape_predictor_68_face_landmarks.dat"


# def distance(x1, y1, x2, y2):
#     return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

#     # 需要行，列


# def get_subface_mean(rect, img):
#     x, y, z, w = rect
#     sub_face = img[x:y, z:w, :]
#     val = np.mean(sub_face[:, :, 1])
#     return val


class getpulse(object):
    """docstring for getpluse"""

    def __init__(self):
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 256
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.freqs = []
        self.fft = []
        self.t0 = time.time()
        self.bpm = 0
        self.flag = True
        self.rois = np.zeros((10, 10))
        self.last_center = np.array([0, 0])

        self.slices = [[0]]
        self.samples = []

        self.x, self.y, self.z, self.w = 0, 0, 0, 0

        self.fx, self.fy = 0, 0

        self.predictor = dlib.shape_predictor(predictor_path)

        self.detector = dlib.get_frontal_face_detector()

        self.fiterx = ave_move()
        self.fitery = ave_move()

        self.fiter1 = ave_move()
        self.fiter2 = ave_move()
        self.fiter3 = ave_move()
        self.fiter4 = ave_move()

        self.fiter5 = ave_move()
        self.fiter6 = ave_move()
        self.fiter7 = ave_move()
        self.fiter8 = ave_move()
        self.fiter9 = ave_move()
        self.fiter10 = ave_move()
        self.fiter11 = ave_move()
        self.fiter12 = ave_move()

        self.roi = []

        self.left = True
        self.top = True
        self.right = True

        self.change = 20
        self.top_fps = 0
        self.right_fps = 0
        self.left_fps = 0
        self.top_flag = False
        self.left_flag = False
        self.right_flag = False

        # need (x1,y1),(x2,y2)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * (w-x), y + 0.5 * (h-y)])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

        # 需要行，列
    def get_subface_mean(self, rect, img):
        x, y, z, w = rect
        sub_face = img[x:y, z:w, :]
        val = np.mean(cv2.equalizeHist(sub_face[:, :, 1]))
        # val = np.mean(sub_face[:, :, 1])
        return val

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (w, h), col, 1)

    def get_rppg(self, title):

        # self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in

        if self.flag:
            self.x, self.y, self.z, self.w = 0, self.frame_in.shape[0], 0, self.frame_in.shape[1]
            self.flag = False

        img = self.frame_in[self.x:self.y, self.z:self.w, :]
        self.rois = img

        dets = self.detector(img, 1)

        if len(dets) != 1:

            self.fx, self.fy = 0, 0
            self.x, self.y, self.z, self.w = 0, self.frame_in.shape[0], 0, self.frame_in.shape[1]
            self.data_buffer = []
            self.times = []

            return
        for i, d in enumerate(dets):

            # cv2.rectangle(self.frame_out, (self.fx+d.right(), self.fy+d.top()), (self.fx+d.left(), self.fy+d.bottom()), (0, 0, 255))
            # if self.shift([self.fx+d.right(), self.fy + d.top(), self.fx+d.left(), self.fy+d.bottom()]) > 5:

            #     return
            shape = self.predictor(img, d)

            cx, cy = int(shape.part(27).x), int(shape.part(27).y)

            wEyeRight = self.distance(shape.part(36).x, shape.part(36).y, shape.part(39).x, shape.part(39).y)
            wEyeLeft = self.distance(shape.part(42).x, shape.part(42).y, shape.part(45).x, shape.part(45).y)
            wEye = 0.5*(wEyeLeft+wEyeRight)
            hNose = self.distance(shape.part(27).x, shape.part(27).y, shape.part(33).x, shape.part(33).y)

            wFaceRight = self.distance(shape.part(30).x, shape.part(30).y, shape.part(2).x, shape.part(2). y)

            wFaceLeft = self.distance(shape.part(30).x, shape.part(30).y, shape.part(14).x, shape.part(14). y)

            self.z = int(self.fx+cx-3.5*wEye)
            self.w = int(self.fx+cx+3.5*wEye)
            self.x = int(self.fy+cy-2.5*hNose)
            self.y = int(self.fy+cy+2.5*hNose)  # row
            # print("z:{},w:{},x:{},y:{}".format(self.z, self.w, self.x, self.y))
            self.z = int(self.fiterx.move_ave(self.z, 7))
            self.x = int(self.fitery.move_ave(self.x, 7))

            cv2.rectangle(self.frame_out, (self.z, self.x), (self.w, self.y), (0, 255, 0))

            self.fx, self.fy = self.z, self.x

            if self.z < 0 or self.w > self.frame_in.shape[1] or self.x < 0 or self.y > self.frame_in.shape[0]:
                # self.fx, self.fy, self.x, self.y, self.z, self.w = 0, 0, 0, self.frame_in.shape[1], 0, self.frame_in.shape[0]
                self.fx = 0
                self.fy = 0
                self.x, self.y, self.z, self.w = 0, self.frame_in.shape[0], 0, self.frame_in.shape[1]

                self.data_buffer = []
                self.times = []
                self.fiter1.clear()
                self.fiter2.clear()
                self.fiter3.clear()
                self.fiter4.clear()
                self.fiterx.clear()
                self.fitery.clear()

                self.fiter5.clear()
                self.fiter6.clear()
                self.fiter7.clear()
                self.fiter8.clear()
                self.fiter9.clear()
                self.fiter10.clear()
                self.fiter11.clear()
                self.fiter12.clear()

            if(int(wFaceRight) < int(3.5 * wEye) and int(wFaceLeft) < int(3.5 * wEye)):

                self.top_fps += 1

            else:
                if(int(wFaceLeft) > int(3.5 * wEye)):

                    self.left_fps += 1
                else:
                    self.right_fps += 1

            if self.top_fps >= self.change and self.top_fps > self.left_fps and self.top_fps > self.right_fps:
                self.top_fps = 0
                self.left_fps = 0
                self.right_fps = 0
                self.top_flag = True
                self.left_flag = False
                self.right_flag = False

            if self.left_fps >= self.change and self.left_fps > self.top_fps and self.left_fps > self.right_fps:
                self.left_fps = 0
                self.top_fps = 0
                self.right_fps = 0
                self.left_flag = True
                self.top_flag = False
                self.right_flag = False

            if self.right_fps >= self.change and self.right_fps > self.left_fps and self.right_fps > self.top_fps:
                self.right_fps = 0
                self.left_fps = 0
                self.top_fps = 0
                self.right_flag = True
                self.left_flag = False
                self.top_flag = False

            if self.top_flag:

                if self.top:
                    self.data_buffer, self.times = [], []
                    self.top = False
                    self.right = True
                    self.left = True

                self.roi = [int(self.fiter1.move_ave(cy - 0.5 * hNose - 38, 7)), int(self.fiter2.move_ave(cy - 0.5 * hNose - 10, 7)), int(self.fiter3.move_ave(cx - 50, 7)), int(self.fiter4.move_ave(cx + 50, 7))]

            if self.left_flag:
                if self.left:
                    self.data_buffer, self.times = [], []
                    self.left = False
                    self.right = True
                    self.top = True

                self.roi = [int(self.fiter5.move_ave(shape.part(31).y - 24, 7)), int(self.fiter6.move_ave(shape.part(31).y + 30, 7)), int(self.fiter7.move_ave(shape.part(45).x - 30, 7)), int(self.fiter8.move_ave(shape.part(45).x + 30, 7))]

            if self.right_flag:
                if self.right:
                    self.data_buffer, self.times = [], []
                    self.right = False
                    self.top = True
                    self.left = True

                self.roi = [int(self.fiter9.move_ave(shape.part(31).y - 24, 7)), int(self.fiter10.move_ave(shape.part(31).y + 30, 7)), int(self.fiter11.move_ave(shape.part(36).x - 30, 7)), int(self.fiter12.move_ave(shape.part(36).x + 30, 7))]
                # self.roi = [shape.part(31).y - 6, shape.part(31).y + 6, shape.part(36).x - 6,  shape.part(36).x + 6]

            for i in range(68):

                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 100, 255), 1)
                # cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 255), 1)

            # cv2.rectangle(self.frame_out, (self.fx+self.roi[2], self.fy+self.roi[0]), (self.fx+self.roi[3], self.fy+self.roi[1]), (0, 100, 255))
            if len(self.roi) != 0:
                cv2.rectangle(img, (self.roi[2], self.roi[0]), (self.roi[3], self.roi[1]), (0, 100, 255))

                val = self.get_subface_mean(self.roi, img)
                self.times.append(time.time() - self.t0)

                self.data_buffer.append(val)
                L = len(self.data_buffer)
                # print("l:{}".format(L))
                if L > self.buffer_size:
                    self.data_buffer = self.data_buffer[-self.buffer_size:]
                    self.times = self.times[-self.buffer_size:]
                    L = self.buffer_size

                processed = np.array(self.data_buffer)
                self.samples = processed
                if L > 10:
                    self.fps = float(L) / (self.times[-1] - self.times[0])
                    cv2.putText(self.frame_out, "fps:{}".format(self.fps), (10, 95), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 255))
                    even_times = np.linspace(self.times[0], self.times[-1], L)
                    interpolated = np.interp(even_times, self.times, processed)
                    interpolated = np.hamming(L) * interpolated

                    interpolated = (interpolated - np.mean(interpolated))/np.std(interpolated)

                    # interpolated = (interpolated - np.mean(interpolated))

                    raw = np.fft.rfft(interpolated)
                    self.fft = np.abs(raw)
                    self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)  # like k
                    freqs = 60. * self.freqs
                    idx = np.where((freqs > 50) & (freqs < 180))
                    # print(idx)
                    try:
                        pruned = self.fft[idx]  # 没有乘60

                    except Exception as e:
                        # self.fx, self.fy = 0, 0
                        return

                    # pruned = self.fft[idx]  # 没有乘60

                    pfreq = freqs[idx]  # 乘60
                    self.freqs = pfreq
                    self.fft = pruned
                    idx2 = np.argmax(pruned)
                    self.bpm = self.freqs[idx2]

                    gap = (self.buffer_size - L) / self.fps
                    if gap:
                        text = "(rPpG: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
                    else:
                        text = "(rPpG: %0.1f bpm)" % (self.bpm)

                    tsize = 1.25
                    # self.slices = [np.copy(img[:, :, 1])]

                    col = (0, 100, 255)
                    # cv2.rectangle(self.frame_out, (self.fx+d.right(), self.fy+d.top()), (self.fx+d.left(), self.fy+d.bottom()), (0, 0, 255))

                    # cv2.putText(self.frame_out, text,
                    #             (self.fx+d.left(), self.fy+d.top()), cv2.FONT_HERSHEY_PLAIN, tsize, col)
                    cv2.putText(self.frame_out, text,
                                (10, 115), cv2.FONT_HERSHEY_PLAIN, tsize, col)
                else:
                    return


# clicked = False


# titles = "Demo"


# def onMouse(event, x, y, flags, param):
#     global clicked
#     if event == cv2.EVENT_LBUTTONUP:
#         clicked = True


# cv2.namedWindow(titles)
# cv2.setMouseCallback(titles, onMouse)

# if __name__ == '__main__':

#     cap = cv2.VideoCapture(0)
#     app = getpulse()

#     while cap.isOpened() and not clicked and cv2.waitKey(1) == -1:

#         _, img = cap.read()

#         app.frame_in = img
#         app.get_rppg(titles)
#         cv2.imshow(titles, app.frame_out)

#     cap.release()
