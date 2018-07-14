#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-09 16:11:44
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$


import numpy as np


class move_ave(object):
    """docstring for move_agv"""

    def __init__(self, size):

        self.item = [0]*size

    def move_ave(self, data):
        self.item.pop(0)
        self.item.append(data)
        data_out = np.mean(self.item)
        return data_out


class ave_move(object):
    """docstring for ave_move"""

    def __init__(self):

        self.item = []

    def move_ave(self, data, size):
        if len(self.item) < size:
            self.item.append(data)
            data_out = np.mean(self.item)
            return data_out
        else:
            if len(self.item) == size:
                self.item.pop(0)
                self.item.append(data)
                data_out = np.mean(self.item)
                return data_out

    def clear(self):
        self.item = []


# a = move_ave(7)
# b = move_ave(7)
# k = []
# j = []

# c = ave_move()
# l = []

# for i in range(100, 500, 50):
#     l.append(c.move_ave(i, 7))
#     j.append(a.move_ave(i))
#     k.append(i)
#     # k.append(b.move_ave(i+1))

# print("k:{}".format(k))
# print("j:{}".format(j))
# print("l:{}".format(l))
