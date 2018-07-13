#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-22 21:32:12
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$

import cv2

from imageprocess_cv2 import getpulse
from device import Camera
from interface import plotXY
clicked = False


titles = "Demo"


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cv2.namedWindow(titles)
cv2.setMouseCallback(titles, onMouse)
# cv2.namedWindow("detected")
# cv2.setMouseCallback("detected", onMouse)


class App(object):
    """docstring for App"""

    def __init__(self):
        self.cameras = []
        self.selected_cam = 0

        self.key = "q"
        self.app = getpulse()
        self.flag = False
        self.color = (0, 0, 255)
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"
        self.bpm_plot = False
        self.pressed = 0
        self.key_controls = {
            "d": self.disp_make_plot,
            "c": self.toggle_cam,
        }

        for i in range(3):
            self.camera = Camera(camera=i)
            if self.camera.valid or not len(self.cameras):
                self.cameras.append(self.camera)
            else:
                break

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def make_plot(self):
        plotXY([[self.app.times,
                 self.app.samples],
                [self.app.freqs,
                 self.app.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.app.slices[0],
               bpm=self.app.bpm_s,

               flag=self.app.flag3)

    def disp_make_plot(self):
        self.bpm_plot = True

    def key_handler(self):
        self.pressed = cv2.waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.release()
            self.flag = True
        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def loop(self):

        self.app.frame_in = self.cameras[self.selected_cam].get_frame()
        cv2.putText(
            self.app.frame_in, "Press 'C' to change camera   (current: %d)" % (self.selected_cam),
            (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, self.color)
        cv2.putText(
            self.app.frame_in, "Press 'D' to display plot signal",
            (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, self.color)
        cv2.putText(
            self.app.frame_in, "Press esc to exit",
            (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, self.color)

        self.app.get_rppg("title")
        cv2.imshow(titles, self.app.frame_out)
        # cv2.imshow("detected", self.app.rois)
        if self.bpm_plot:
            self.make_plot()
        self.key_handler()


if __name__ == '__main__':
    run = App()

    while True:
        if run.flag:
            break

        run.loop()
