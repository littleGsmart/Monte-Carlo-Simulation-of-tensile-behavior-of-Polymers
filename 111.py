import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left as bl
import random as rd
import json
import math
import pickle
from tqdm import tqdm, trange
import tkinter.messagebox as msg
import tkinter as tk
import tkinter.filedialog
import pickle
import tkinter.simpledialog
import time
import matplotlib
import multiprocessing
import sys

sys.setrecursionlimit(1000000)

sq3 = np.sqrt(3)
plt.axis('equal')
with open("dict_600000_10.json", encoding="utf-8") as f:
    e_dict = json.load(f)
Tg = 263
Tm = 433
Td = 583
T_environment = 200
n_g_arr = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1., 1e8, 1e8, 1e8]
dTeproll = 0


class Ts:

    def __init__(self, Tg, Tm, Td, T_environment, dTeproll):
        self.Tg = Tg
        self.Tm = Tm
        self.Td = Td
        self.T_environment = T_environment
        self.dTeproll = dTeproll

    def call(self):
        return 'self.Tg = {}\nself.Tm = {}\nself.Td = {}\nself.T_environment = {}\nself.dTeproll = {}'.format(self.Tg,
                                                                                                              self.Tm,
                                                                                                              self.Td,
                                                                                                              self.T_environment,
                                                                                                              self.dTeproll)


aTs = Ts(Tg, Tm, Td, T_environment, dTeproll)


class hex_coordinate:
    def __init__(self, coordinate_as_array):
        self.x = coordinate_as_array[0]
        self.y = coordinate_as_array[1]

    def Transform_2_Ortho(self):
        x = self.x - 0.5 * self.y
        y = self.y * sq3 / 2
        return [x, y]

    def around(self, size):
        around_loaction = [hex_coordinate([self.x + 1, self.y]), hex_coordinate([self.x - 1, self.y]),
                           hex_coordinate([self.x, self.y + 1]), hex_coordinate([self.x, self.y - 1]),
                           hex_coordinate([self.x + 1, self.y + 1]), hex_coordinate([self.x - 1, self.y - 1])]
        pop_locate = []
        if self.x == size[0] - 1:
            pop_locate.append(0)
            pop_locate.append(4)
        if self.x == 0:
            pop_locate.append(1)
            pop_locate.append(5)
        if self.y == size[1] - 1:
            pop_locate.append(2)
            pop_locate.append(4)
        if self.y == 0:
            pop_locate.append(3)
            pop_locate.append(5)

        return np.delete(around_loaction, pop_locate).tolist()

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        x = dx - 0.5 * dy
        y = dy * sq3 / 2
        return math.sqrt(x * x + y * y)

    def tolist(self):
        return [self.x, self.y]

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        ans = hex_coordinate([x, y])
        return ans

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        ans = hex_coordinate([x, y])
        return ans


class hex_box:
    def __init__(self, coordinate):
        self.location = coordinate
        self.content = []

    def draw_box(self):
        [x, y] = self.location.Transform_2_Ortho()
        plt.plot([x - 0.5, x, x + 0.5, x + 0.5, x, x - 0.5, x - 0.5],
                 [y + sq3 / 6, y + sq3 / 3, y + sq3 / 6, y - sq3 / 6, y - sq3 / 3, y - sq3 / 6, y + sq3 / 6],
                 color='gray', linestyle='-.')
        plt.fill([x - 0.5, x, x + 0.5, x + 0.5, x, x - 0.5, x - 0.5],
                 [y + sq3 / 6, y + sq3 / 3, y + sq3 / 6, y - sq3 / 6, y - sq3 / 3, y - sq3 / 6, y + sq3 / 6],
                 color='gray', alpha=0.3)


class Kuhn:
    def __init__(self, coordinate):
        self.location = coordinate

    def grow(self, possible_location):
        location = int(rd.random() * len(possible_location))
        location = possible_location[location]
        return Kuhn(location)

    def __sub__(self, other):
        return self.location - other.location


class System:
    def __init__(self, size_arr):
        self.size = size_arr
        self.boxes = []
        self.lines = []
        for i in range(size_arr[0]):
            temp = []
            for j in range(size_arr[1]):
                temp.append(hex_box(hex_coordinate([i, j])))
            self.boxes.append(temp)
        self.boxes = np.array(self.boxes)

    def random_locate(self):
        x = int(rd.random() * self.size[0])
        y = int(rd.random() * self.size[1])
        return hex_coordinate([x, y])

    def line_generate(self, DP):
        length = 0
        line = []
        while not line:
            begin_location = self.random_locate()
            if len(self.boxes[begin_location.x, begin_location.y].content) <= 1:
                line.append(Kuhn(begin_location))
                length = 1
                self.boxes[begin_location.x, begin_location.y].content.append([len(self.lines), 0])

        while length < DP:
            around = line[length - 1].location.around(self.size)
            pop_list = []
            for i in range(len(around)):
                if len(self.boxes[around[i].x, around[i].y].content) >= 2:
                    pop_list.append(i)
            around = np.delete(around, pop_list)
            if len(around) == 0:
                if len(line) == 1:
                    self.line_generate(DP)
                    return 2
                else:
                    self.lines.append(line)
                return 1
            point = (line[length - 1].grow(around))
            line.append(point)
            self.boxes[point.location.x, point.location.y].content.append([len(self.lines), length])
            length = length + 1
        self.lines.append(line)

        return 0

    def line_generate_DP(self, DP):
        length = 0
        line = []
        while not line:
            begin_location = self.random_locate()
            if len(self.boxes[begin_location.x, begin_location.y].content) <= 1:
                line.append(Kuhn(begin_location))
                length = 1
                self.boxes[begin_location.x, begin_location.y].content.append([len(self.lines), 0])
            else:
                print([[begin_location.x, begin_location.y], self.boxes[begin_location.x, begin_location.y].content])

        def next_point(around_arr):
            pop_list = []
            for i in range(len(around_arr)):
                if len(self.boxes[around_arr[i].x, around_arr[i].y].content) >= 2:
                    pop_list.append(i)
            pop_around = np.delete(around_arr, pop_list)
            if len(pop_around) == 0:
                next_location = int(rd.random() * len(around_arr))
                next_location = around_arr[next_location]
                next_arr = next_location.around(self.size)
                # print('next')
                return next_point(next_arr)
            else:
                location = int(rd.random() * len(around_arr))
                location = around_arr[location]
            return Kuhn(location)

        for length in trange(1, DP):
            around = line[length - 1].location.around(self.size)
            point = next_point(around)
            line.append(point)
            self.boxes[point.location.x, point.location.y].content.append([len(self.lines), length])
        self.lines.append(line)

        return 0

    def draw_Lines(self, line_num='all'):
        draw_shitmontain = np.zeros(self.size)
        if line_num == 'all':
            for line in tqdm(self.lines):
                X = []
                Y = []
                for i in line:
                    draw_shitmontain[i.location.x, i.location.y] += 1
                    [x, y] = i.location.Transform_2_Ortho()
                    if draw_shitmontain[i.location.x, i.location.y] == 2:
                        x = x - 0.25
                        y = y + 0.15
                    X.append(x)
                    Y.append(y)
                plt.plot(X, Y, marker='o')
        else:
            for j in tqdm(line_num):
                X = []
                Y = []
                for i in self.lines[j]:
                    draw_shitmontain[i.location.x, i.location.y] += 1
                    [x, y] = i.location.Transform_2_Ortho()
                    if draw_shitmontain[i.location.x, i.location.y] == 2:
                        x = x - 0.25
                        y = y + 0.15
                    X.append(x)
                    Y.append(y)
                plt.plot(X, Y, marker='o')

    def rdpoint(self,line_num = 'all'):
        if line_num == 'all':
            i = int(rd.random() * len(self.lines))
        else:
            i = int(line_num)
        j = int(rd.random() * len(self.lines[i]))
        return [i, j]

    def point_motive(self, point_num):
        origin_location = self.lines[point_num[0]][point_num[1]].location
        directions = origin_location.around(self.size)
        target_location = directions[int(rd.random() * len(directions))]
        rd_energy = e_dict[0][bl(e_dict[1], rd.random())] * aTs.T_environment

        energy_g = n_g_arr[len(self.boxes[target_location.x][target_location.y].content)] * aTs.Tg
        # energy_m
        direction_1 = []
        energy_m = 0
        for i in origin_location.around(self.size):
            box = self.boxes[i.x, i.y]
            if box.content:
                for point in box.content:
                    if point[1] == 0:
                        direction_1.append(
                            (self.lines[point[0]][point[1]] - self.lines[point[0]][point[1] + 1]).tolist())
                    elif point[1] == (len(self.lines[point[0]]) - 1):
                        direction_1.append(
                            (self.lines[point[0]][point[1]] - self.lines[point[0]][point[1] - 1]).tolist())
                    else:
                        direction_1.append(
                            (self.lines[point[0]][point[1]] - self.lines[point[0]][point[1] + 1]).tolist())
                        direction_1.append(
                            (self.lines[point[0]][point[1]] - self.lines[point[0]][point[1] - 1]).tolist())

        if point_num[1] == 0:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            direction_00 = (origin_location - latter_location).tolist()
            if direction_00 in direction_1:
                energy_m = aTs.Tm
        elif point_num[1] == (len(self.lines[point_num[0]]) - 1):
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            direction_01 = (origin_location - former_location).tolist()
            if direction_01 in direction_1:
                energy_m = aTs.Tm

        else:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            direction_00 = (origin_location - latter_location).tolist()
            direction_01 = (origin_location - former_location).tolist()
            if (direction_00 in direction_1) or (direction_01 in direction_1):
                energy_m = aTs.Tm

        # energy_d
        if point_num[1] == 0:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            d_distance = 0.5 * (pow(target_location.distance(latter_location), 2) - pow(
                origin_location.distance(latter_location), 2))
            energy_d = d_distance * aTs.Td
        elif point_num[1] == (len(self.lines[point_num[0]]) - 1):
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            d_distance = 0.5 * (pow(target_location.distance(former_location), 2) - pow(
                origin_location.distance(former_location), 2))
            energy_d = d_distance * aTs.Td
        else:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            d_distance = 0.5 * (pow(target_location.distance(latter_location), 2) - pow(origin_location.distance(
                latter_location), 2) + pow(target_location.distance(former_location), 2) - pow(
                origin_location.distance(former_location), 2))
            energy_d = d_distance * aTs.Td

        blocking_energy = energy_m + energy_d + energy_g
        if rd_energy > blocking_energy:
            self.lines[point_num[0]][point_num[1]].location = target_location
            self.boxes[origin_location.x][origin_location.y].content.remove(point_num)
            self.boxes[target_location.x][target_location.y].content.append(point_num)
            return 1

        return 0

        # print([[origin_location.x,origin_location.y],[target_location.x,target_location.y],rd_energy,energy_d,energy_g,energy_m,direction_1])

    def add_line(self, arr):
        aLine = []
        for i in range(len(arr[0])):
            point = Kuhn(hex_coordinate([arr[0][i], arr[1][i]]))
            aLine.append(point)
            self.boxes[point.location.x, point.location.y].content.append([len(self.lines), i])
        self.lines.append(aLine)

    def calc_rd(self, line_num):
        aLine = self.lines[line_num]
        len_aLine = len(aLine)
        average = hex_coordinate([0, 0])
        ave_distance = 0
        for i in aLine:
            average = average + i.location
        average = hex_coordinate([average.x / len_aLine, average.y / len_aLine])
        for i in aLine:
            ave_distance = ave_distance + pow(average.distance(i.location), 2)
        ave_distance = ave_distance / len_aLine
        ave_distance = math.sqrt(ave_distance)
        return ave_distance


def change_Te(newTe):
    aTs.T_environment = newTe[0]
    aTs.dTeproll = newTe[1]
    print(aTs.call())


# -------------------------------------


def myrun(aSys):
    print('fuckU')
    aSys.line_generate_DP(30000)
    # print(aSys.lines[-1])
    return aSys.lines[-1]

def mydiedai(aSys,line_num):
    [aSys.point_motive(aSys.rdpoint(line_num)) for i in range(100)]
    return [line_num,aSys.lines[line_num]]


if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(12)
    result = []
    test_Sys = System([500, 500])
    test_man_list = multiprocessing.Manager().list()
    test_man_list.append(test_Sys)
    for i in range(10):
        r = pool.apply_async(func=myrun, args=(test_man_list[0],))
        result.append(r)

    test_Sys.lines = [i.get() for i in result]

    pool.close()
    pool.join()
    # test_man_list[0].draw_Lines()
    test_Sys.draw_Lines()
    plt.show()

    t11 = time.time()
    pool2 = multiprocessing.Pool(12)
    result = []

    for j in range(100):
        test_man_list[0] = (test_Sys)
        for i in range(len(test_Sys.lines)):
            r = pool2.apply_async(func=mydiedai, args=(test_man_list[0],i,))
            result.append(r)

        test_Sys.lines = [ii[1] for ii in sorted([i.get() for i in result],key=lambda x:x[0])]

    pool.close()
    pool.join()
    # test_man_list[0].draw_Lines()
    test_Sys.draw_Lines()
    plt.show()

