import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left as bl
import random as rd
import json
import math
from tqdm import tqdm, trange
import tkinter.messagebox as msg
import tkinter as tk
import tkinter.filedialog
import pickle
import tkinter.simpledialog
import matplotlib
import sys
import rich.progress
import threading
import time

sys.setrecursionlimit(1000000)

sq3 = np.sqrt(3)
plt.axis('equal')
with open("dict_600000_10.json", encoding="utf-8") as f:
    e_dict = json.load(f)
Tg = 201
Tm = 403
Td = 473  # NR数据
Tp = 2 * Td
T_environment = 400
n_g_arr = [0, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 22, 23,
           24, 25, 26, 27, 28, 29,
           30, 31, 1e8, 1e8, 1e8]
dTeproll = 0

output = 500

record_Rd = 1
rudong = 0
diedai_continue = 0
plt.ion()
auto_clean = 0
CDF = 1

box_area = [[60 - (5 * sq3 / 3), 45 - (10 * sq3 / 3)], [60 + (5 * sq3 / 3), 45 + (10 * sq3 / 3)],
            [90 + (5 * sq3 / 3), 45 + (10 * sq3 / 3)], [90 - (5 * sq3 / 3), 45 - (10 * sq3 / 3)]]
gen_area = box_area  # [[60, 0], [60, 90], [90, 90], [90, 0]]
new_sys_size = [150, 90]
max_y = 22.5 * sq3 + 5
min_y = 22.5 * sq3 - 5
max_x = 67.5
min_x = 37.5

rec_boundary = [[], [], []]
rec_Rd_bar = []


# gjw = 'sb'


# 创建一个900,1500的体系

class Ts:

    def __init__(self, Tg, Tm, Td, Tp, T_environment, dTeproll):
        self.Tg = Tg
        self.Tm = Tm
        self.Td = Td
        self.Tp = Tp
        self.T_environment = T_environment
        self.dTeproll = dTeproll

    def call(self):
        return 'self.Tg = {}\nself.Tm = {}\nself.Td = {}\nself.T_environment = {}\nself.dTeproll = {}'.format(self.Tg,
                                                                                                              self.Tm,
                                                                                                              self.Td,
                                                                                                              self.T_environment,
                                                                                                              self.dTeproll)


class PEF:  # Poteneial energy field
    def __init__(self, xx, x, y, x0, ky, Fx):
        self.x = x
        self.x0 = x0
        self.ky = ky
        self.Fx = Fx
        self.kx = 0
        self.b_x0 = 0
        self.y = y
        self.xx = xx
        self.boundary = [0, 0, 0, 0]

    def potential_ene(self, plot):
        ene = 0
        plot = plot.Transform_2_Ortho()
        if self.x != 0:
            ene = self.x * (0.5 * self.kx * plot[0] * plot[0] - self.kx * self.b_x0 * plot[0])

        if self.y != 0:
            ene = ene + self.ky * (-max(plot[1], max_y) + min(plot[1], min_y))

        if self.xx != 0:
            ene = ene + self.ky * (-max(plot[0], max_x) + min(plot[0], min_x))

        return ene


aTs = Ts(Tg, Tm, Td, Tp, T_environment, dTeproll)
sysPEF = PEF(0, 0, 0, 0, 1000, 200)


class hex_coordinate:
    def __init__(self, coordinate_as_array):
        self.x = coordinate_as_array[0]
        self.y = coordinate_as_array[1]

    def Transform_2_Ortho(self):
        x = self.x - 0.5 * self.y
        y = self.y * sq3 / 2
        return [x, y]

    def around(self, area):
        around_loaction = [hex_coordinate([self.x + 1, self.y]), hex_coordinate([self.x - 1, self.y]),
                           hex_coordinate([self.x, self.y + 1]), hex_coordinate([self.x, self.y - 1]),
                           hex_coordinate([self.x + 1, self.y + 1]), hex_coordinate([self.x - 1, self.y - 1])]

        for point_arr in around_loaction:
            if not pnpoly(area, point_arr.tolist()):
                around_loaction.remove(point_arr)

        return around_loaction

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
        self.packing_containing_num = 0

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
        self.cross_linked = []

    def grow(self, possible_location):
        location = int(rd.random() * len(possible_location))
        location = possible_location[location]
        return Kuhn(location)

    def __sub__(self, other):
        return self.location - other.location


class packing:
    def __init__(self, coordinate):
        self.location = coordinate


def pnpoly(vertices, testp):  # 检查点是否在区域内
    n = len(vertices)
    j = n - 1
    res = False
    for i in range(n):
        if (vertices[i][1] > testp[1]) != (vertices[j][1] > testp[1]) and \
                testp[0] < (vertices[j][0] - vertices[i][0]) * (testp[1] - vertices[i][1]) / (
                vertices[j][1] - vertices[i][1]) + vertices[i][0]:
            res = not res
        j = i
    return res


def draw_area(area):
    for num in range(len(area) - 1):
        x = [area[num][0], area[num + 1][0]]
        y = [area[num][1], area[num + 1][1]]
        plt.plot(x, y, color='black')
    x = [area[len(area) - 1][0], area[0][0]]
    y = [area[len(area) - 1][1], area[0][1]]
    plt.plot(x, y, color='black')


class System:
    def __init__(self, size_arr):
        self.size = size_arr
        self.boxes = []
        self.lines = []
        for i in trange(size_arr[0]):
            temp = []
            for j in range(size_arr[1]):
                temp.append(hex_box(hex_coordinate([i, j])))
            self.boxes.append(temp)
        self.boxes = np.array(self.boxes)
        self.total_cross_linking = 0
        self.crosslinking_Kuhns = []
        self.packing = []

    def random_locate(self, myarea):
        while True:
            x = int(rd.random() * self.size[0])
            y = int(rd.random() * self.size[1])
            if pnpoly(myarea, [x, y]):
                return hex_coordinate([x, y])

    # def line_generate(self, DP):
    #     length = 0
    #     line = []
    #     while not line:
    #         begin_location = self.random_locate()
    #         if len(self.boxes[begin_location.x, begin_location.y].content) <= 1:
    #             line.append(Kuhn(begin_location))
    #             length = 1
    #             self.boxes[begin_location.x, begin_location.y].content.append([len(self.lines), 0])
    #
    #     while length < DP:
    #         around = line[length - 1].location.around(self.size)
    #         pop_list = []
    #         for i in range(len(around)):
    #             if len(self.boxes[around[i].x, around[i].y].content) >= 2:
    #                 pop_list.append(i)
    #         around = np.delete(around, pop_list)
    #         if len(around) == 0:
    #             if len(line) == 1:
    #                 self.line_generate(DP)
    #                 return 2
    #             else:
    #                 self.lines.append(line)
    #             return 1
    #         point = (line[length - 1].grow(around))
    #         line.append(point)
    #         self.boxes[point.location.x, point.location.y].content.append([len(self.lines), length])
    #         length = length + 1
    #     self.lines.append(line)
    #
    #     return 0

    def line_generate_DP(self, DP, size='all'):
        if size == 'all':
            mysize = [[0, 0], [self.size[0], 0], self.size, [0, self.size[1]]]
        else:
            mysize = size
        length = 0
        line = []
        while not line:
            begin_location = self.random_locate(mysize)
            # if len(self.boxes[begin_location.x, begin_location.y].content) <= 1:
            line.append(Kuhn(begin_location))
            length = 1
            self.boxes[begin_location.x, begin_location.y].content.append([len(self.lines), 0])

        def next_point(around_arr, mysize):
            pop_list = []
            # for i in range(len(around_arr)):
            #     if len(self.boxes[around_arr[i].x, around_arr[i].y].content) >= 2:
            #         pop_list.append(i)
            pop_around = np.delete(around_arr, pop_list)
            if len(pop_around) == 0:
                next_location = int(rd.random() * len(around_arr))
                next_location = around_arr[next_location]
                next_arr = next_location.around(mysize)
                # print('next')
                return next_point(next_arr, mysize)
            else:
                location = int(rd.random() * len(around_arr))
                location = around_arr[location]
            return Kuhn(location)

        for length in range(1, DP):
            around = line[length - 1].location.around(mysize)
            point = next_point(around, mysize)
            line.append(point)
            self.boxes[point.location.x, point.location.y].content.append([len(self.lines), length])
        self.lines.append(line)

        return 0

    def packing_generate(self, num, size='all', distribution='rd'):
        if size == 'all':
            mysize = [[0, 0], [self.size[0], 0], self.size, [0, self.size[1]]]
        else:
            mysize = size
        for i in range(num):
            begin_location = self.random_locate(mysize)
            self.boxes[begin_location.x, begin_location.y].packing_containing_num += 1

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
                    elif draw_shitmontain[i.location.x, i.location.y] == 3:
                        x = x + 0.25
                        y = y - 0.15
                    elif draw_shitmontain[i.location.x, i.location.y] == 4:
                        x = x - 0.25
                        y = y - 0.15
                    elif draw_shitmontain[i.location.x, i.location.y] == 5:
                        x = x + 0.25
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

        for box_lines in tqdm(self.boxes):
            for box in box_lines:
                if box.packing_containing_num >= 1:
                    [x, y] = box.location.Transform_2_Ortho()
                    x += 0.3
                    y -= 0.3
                    plt.plot([x], [y], marker='s', color='black')
                    if box.packing_containing_num >= 2:
                        plt.text(x, y, str(box.packing_containing_num))

    def rdpoint(self):
        i = int(rd.random() * len(self.lines))
        j = int(rd.random() * len(self.lines[i]))
        return [i, j]

    def point_motive(self, point_num, area='all'):

        if area == 'all':
            myarea = [[0, 0], [self.size[0], 0], self.size, [0, self.size[1]]]
        else:
            myarea = area
        origin_location = self.lines[point_num[0]][point_num[1]].location
        directions = origin_location.around(myarea)
        target_location = directions[int(rd.random() * len(directions))]
        rd_energy = e_dict[0][bl(e_dict[1], rd.random())] * aTs.T_environment

        energy_g = n_g_arr[len(self.boxes[target_location.x][target_location.y].content)] * aTs.Tg
        # energy_m
        direction_1 = []
        energy_m = 0
        for i in origin_location.around(myarea):
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

        # Cross_linking
        energy_cross_linking = 0
        if self.lines[point_num[0]][point_num[1]].cross_linked:
            for linked_group_num in self.lines[point_num[0]][point_num[1]].cross_linked:
                other_Kuhn = self.crosslinking_Kuhns[linked_group_num[0]][1 - linked_group_num[1]]

                other_location = other_Kuhn.location
                d_distance = 0.5 * (pow(target_location.distance(other_location), 2) - pow(
                    origin_location.distance(other_location), 2))
                energy_cross_linking += d_distance * aTs.Td

        # PEF

        energy_pull = sysPEF.potential_ene(target_location) - sysPEF.potential_ene(origin_location)

        # Packing

        pack_num = 0
        for location in origin_location.around(myarea):
            pack_num += self.boxes[location.x, location.y].packing_containing_num
        pack_num += self.boxes[origin_location.x, origin_location.y].packing_containing_num
        # for location in target_location.around(myarea):
        #     pack_num -= self.boxes[location.x, location.y].packing_containing_num
        # pack_num -= self.boxes[target_location.x, target_location.y].packing_containing_num  # 第二种物理吸附：允许链段在表面的滑动
        energy_pack = aTs.Tp * pack_num

        blocking_energy = energy_m + energy_d + energy_g + energy_cross_linking + energy_pack
        motive_energy = rd_energy + energy_pull

        if motive_energy > blocking_energy:
            self.lines[point_num[0]][point_num[1]].location = target_location
            self.boxes[origin_location.x][origin_location.y].content.remove(point_num)
            self.boxes[target_location.x][target_location.y].content.append(point_num)

            dire = target_location - origin_location
            if self.boxes[origin_location.x, origin_location.y].packing_containing_num and self.boxes[origin_location.x, origin_location.y].packing_containing_num <= 5:
                self.boxes[target_location.x, target_location.y].packing_containing_num += 1
                self.boxes[origin_location.x, origin_location.y].packing_containing_num -= 1  # 填料随链移动
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

    def calc_rd_bar(self):
        total_rd = 0
        for line in range(len(self.lines)):
            total_rd = total_rd + self.calc_rd(line)
        total_rd = total_rd / len(self.lines)
        return total_rd

    def calc_boundary(self):
        boundary = []
        for line in self.lines:
            for dot in line:
                axis = dot.location.Transform_2_Ortho()
                if not boundary:
                    boundary = [axis[0], axis[0], axis[1], axis[1]]
                else:
                    boundary[0] = min(axis[0], boundary[0])
                    boundary[1] = max(axis[0], boundary[1])
                    boundary[2] = min(axis[1], boundary[2])
                    boundary[3] = max(axis[1], boundary[3])
        return boundary

    def find_a_Kuhn_arr(self):
        aKuhn_loc = self.rdpoint()
        aKuhn = self.lines[aKuhn_loc[0]][aKuhn_loc[1]]
        return [aKuhn, aKuhn_loc]

    def crosslinking(self):
        aKuhn_found_arr = self.find_a_Kuhn_arr()
        box_Kuhn_arr = self.boxes[aKuhn_found_arr[0].location.x][aKuhn_found_arr[0].location.y].content
        while len(box_Kuhn_arr) == 1:
            aKuhn_found_arr = self.find_a_Kuhn_arr()
            box_Kuhn_arr = self.boxes[aKuhn_found_arr[0].location.x][aKuhn_found_arr[0].location.y].content
            # print('1')
        # print('2')

        aKuhn = aKuhn_found_arr[0]
        target_Kuhn = aKuhn_found_arr[1]
        # print('3')
        while target_Kuhn == aKuhn_found_arr[1]:
            # print('4')
            target_Kuhn = box_Kuhn_arr[int(rd.random() * len(box_Kuhn_arr))]
            # global gjw
            # gjw = target_Kuhn
        # print('5')
        target_Kuhn = self.lines[target_Kuhn[0]][target_Kuhn[1]]
        aKuhn.cross_linked.append([self.total_cross_linking, 0])
        target_Kuhn.cross_linked.append([self.total_cross_linking, 1])
        self.crosslinking_Kuhns.append([aKuhn, target_Kuhn])
        self.total_cross_linking += 1


def change_Te(newTe):
    aTs.T_environment = newTe[0]
    aTs.dTeproll = newTe[1]
    print(aTs.call())


# -----------以上为MC_toolbox内容-----------------

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def 拉取体系():
    global Sys
    Sys_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with rich.progress.open(Sys_root, 'rb') as file:
        Sys = pickle.loads(file.read())
    msg.showinfo(title='读取已完成', message='已读取完成')


def 绘制体系():
    if box_area:
        def tohex(list):
            return hex_coordinate(list)

        hex_arr = []
        for point in box_area:
            hex_arr.append(tohex(point).Transform_2_Ortho())
        draw_area(hex_arr)
    try:
        draw_box_or_not = msg.askyesno(title='是否需要绘制边框', message='请问是否需要绘制六边形框？')
        if draw_box_or_not:
            for i in tqdm(Sys.boxes.flatten()):
                i.draw_box()
        Sys.draw_Lines()
        plt.axis('equal')
        plt.show()
    except:
        msg.showerror(title='错误', message='绘制错误，请检查体系是否加载正确')


def 创建新体系():
    if new_sys_size:
        创建新体系_2(new_sys_size[0], new_sys_size[1])
        return 0
    top_1 = tk.Toplevel()
    top_1.title('新模型构建向导')
    top_1.geometry('900x900')

    L_1 = tk.Label(top_1, text="请输入体系长度：")
    L_1.pack(side="top", padx=13, pady=3)
    E_1 = tk.Entry(top_1)
    E_1.pack(side="top", padx=13, pady=3)
    L_2 = tk.Label(top_1, text="请输入体系宽度：")
    L_2.pack(side="top", padx=13, pady=3)
    E_2 = tk.Entry(top_1)
    E_2.pack(side="top", padx=13, pady=3)
    B_1 = tk.Button(top_1, text='下一步', command=lambda: 创建新体系_2(E_1.get(), E_2.get()))
    B_1.pack(side="top", padx=13, pady=3)


def 创建新体系_2(x, y):
    if msg.askyesno(title='是否需要生成链', message='请问是否需要生成链？'):

        top_1 = tk.Toplevel()
        top_1.title('新模型构建向导')
        top_1.geometry('900x900')

        L_1 = tk.Label(top_1, text="生成链的数量：")
        L_1.pack(side="top", padx=13, pady=3)
        E_1 = tk.Entry(top_1)
        E_1.pack(side="top", padx=13, pady=3)
        L_2 = tk.Label(top_1, text="最长链的长度")
        L_2.pack(side="top", padx=13, pady=3)
        E_2 = tk.Entry(top_1)
        E_2.pack(side="top", padx=13, pady=3)
        B_1 = tk.Button(top_1, text='下一步', command=lambda: 创建新体系_3(x, y, E_1.get(), E_2.get()))
        B_1.pack(side="top", padx=13, pady=3)
    else:
        创建新体系_3(x, y, 0, 0)


def 创建新体系_3(x, y, line_num, DP):
    if msg.askyesno(title='是否确定生成模型', message='请问是否确定生成模型？'):
        global Sys
        # try:
        Sys = System([int(x), int(y)])
        if line_num == 0:
            return 0
        for i in trange(int(line_num)):
            Sys.line_generate_DP(int(DP), gen_area)
        msg.showinfo(title='生成已完成', message='模型已生成完成')
        # except:
        #     msg.showerror(title='生成失败', message='模型生成失败')


def 保存体系():
    Sys_root = tkinter.filedialog.asksaveasfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    if '.pkl' in Sys_root:
        pass
    else:
        Sys_root += '.pkl'
    out_put = open(Sys_root, 'wb')
    saved_obj = pickle.dumps(Sys)
    out_put.write(saved_obj)
    out_put.close()
    msg.showinfo(title='保存已完成', message='已保存完成')


def 开始迭代(轮数, 单轮次数):
    def diedai():
        global rec_Rd_bar, T_environment
        for j in trange(轮数):
            sysPEF.boundary = Sys.calc_boundary()
            sysPEF.kx = sysPEF.Fx / (sysPEF.boundary[1] - sysPEF.boundary[0])
            sysPEF.b_x0 = 0.5 * (sysPEF.boundary[1] + sysPEF.boundary[0])
            rec_boundary[0].append(sysPEF.boundary[0])
            rec_boundary[1].append(sysPEF.boundary[1])
            rec_boundary[2].append(sysPEF.boundary[1] - sysPEF.boundary[0])
            for i in range(单轮次数):
                Sys.point_motive(Sys.rdpoint(), 'all')

            if record_Rd:
                rec_Rd_bar.append(Sys.calc_rd_bar())

            if not diedai_continue:
                msg.showinfo(title='迭代已中止', message='迭代已中止')
                return 0

            aTs.T_environment += aTs.dTeproll

    global th_diedai
    global diedai_continue
    diedai_continue = 1 - diedai_continue
    th_diedai = threading.Thread(target=diedai)
    th_diedai.setDaemon(True)
    th_diedai.start()


def 绘制均方末端距():
    plt.plot(range(len(rec_Rd_bar)), rec_Rd_bar)
    plt.axis('auto')
    plt.show()


def 保存均方末端距():
    rd_root = tkinter.filedialog.asksaveasfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    if '.pkl' in rd_root:
        pass
    else:
        rd_root += '.pkl'
    out_put = open(rd_root, 'wb')
    saved_obj = pickle.dumps(rec_Rd_bar)
    out_put.write(saved_obj)
    out_put.close()


def 读取均方末端距():
    global rec_Rd_bar
    rd_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with open(rd_root, 'rb') as file:
        rec_Rd_bar = pickle.loads(file.read())


def 看看链的蠕动():
    def aRun(单轮次数=1000):
        global rec_Rd_bar
        rec_Rd_bar = []
        for i in range(len(Sys.lines)):
            rec_Rd_bar.append([])
        while rudong:
            sysPEF.boundary = Sys.calc_boundary()
            sysPEF.kx = sysPEF.Fx / (sysPEF.boundary[1] - sysPEF.boundary[0])
            sysPEF.b_x0 = 0.5 * (sysPEF.boundary[1] + sysPEF.boundary[0])
            rec_boundary[0].append(sysPEF.boundary[0])
            rec_boundary[1].append(sysPEF.boundary[1])
            rec_boundary[2].append(sysPEF.boundary[1] - sysPEF.boundary[0])
            Sys.point_motive(Sys.rdpoint())
            if auto_clean:
                plt.clf()
            Sys.draw_Lines()
            mypause(0.005)
            if record_Rd:
                for i in range(len(Sys.lines)):
                    rec_Rd_bar[i].append(Sys.calc_rd(i))
            aTs.T_environment += aTs.dTeproll
            # t_gap = (t2-t1) * 1.1
            time.sleep(0.05)
            # print(666)

    global th

    global rudong
    rudong = 1 - rudong

    th = threading.Thread(target=aRun, args=())
    th.setDaemon(True)
    th.start()


def 清除画板():
    plt.clf()


def 自动清除画板():
    global auto_clean
    auto_clean = 1 - auto_clean
    aDict = {1: '已启动自动清除画板',
             0: '已关闭自动清除画板'}
    msg.showinfo(title=aDict[auto_clean], message=aDict[auto_clean])


def 更改环境温度(newTe):
    global T_environment, dTeproll
    change_Te(newTe)
    msg.showinfo(title='更改环境温度', message='更改环境温度已成功，当前温度：环境温度：{}，每轮变化环境温度：{}'.format(aTs.T_environment, aTs.dTeproll))


def 绘制当前体系分布():
    global CDF
    CDF = 1 - CDF
    longest = len(max(Sys.lines, key=len))
    len_map = [0] * longest
    for i in Sys.lines:
        len_map[len(i) - 1] += 1
    if CDF:
        for i in range(1, longest):
            len_map[i] += len_map[i - 1]
    plt.axis('auto')
    plt.plot(range(1, longest + 1), len_map)


def 记录均方末端距():
    global record_Rd, rec_Rd_bar, rec_boundary
    if record_Rd == 0:
        record_Rd = 1
        L_当前状态['text'] = '当前状态：记录均方末端距'
    else:
        record_Rd = 0
        rec_Rd_bar = []
        rec_boundary = [[], [], []]
        L_当前状态['text'] = '当前状态：不记录均方末端距'


def 测试按钮1():
    print(Sys.calc_boundary())


def 测试按钮2():
    sysPEF.x = 1
    sysPEF.xx = 0
    msg.showinfo(title='体系有变', message='开启拉伸')


def 测试按钮3():
    sysPEF.y = 1
    sysPEF.xx = 1
    msg.showinfo(title='体系有变', message='开启压缩')


def 测试按钮4():
    for i in rec_boundary:
        plt.plot(range(len(i)), i)


def 测试按钮5():
    for i in trange(50):
        Sys.crosslinking()
    msg.showinfo(title='体系有变', message='发生了50次交联，100个点被连接了！')


def 测试按钮6():
    Sys.packing_generate(100, box_area)
    msg.showinfo(title='体系有变', message='体系中加入了100个填料！')


# -----------以上为MC_Tktoolbox内容-----------------

Sys = System([1, 1])

root = tk.Tk()
root.title("聚合物拉伸行为的格点法模拟")
root.geometry("900x1800")
B_载入已有体系 = tk.Button(root, text='载入已有体系', command=拉取体系)
B_载入已有体系.pack(side='top', padx=13, pady=3)
B_绘制体系 = tk.Button(root, text='绘制体系', command=绘制体系)
B_绘制体系.pack(side='top', padx=13, pady=3)
B_创建新体系 = tk.Button(root, text='创建新体系', command=创建新体系)
B_创建新体系.pack(side='top', padx=13, pady=3)
B_保存体系 = tk.Button(root, text='保存体系', command=保存体系)
B_保存体系.pack(side='top', padx=13, pady=3)
B_记录均方末端距 = tk.Button(root, text='记录均方末端距', command=记录均方末端距)
B_记录均方末端距.pack(side='top', padx=13, pady=3)
L_当前状态 = tk.Label(root, text='当前状态：记录均方末端距')
L_当前状态.pack(side='top', padx=13, pady=3)
L_环境温度 = tk.Label(root, text='环境温度：')
L_环境温度.pack(side='top', padx=13, pady=3)
E_环境温度 = tk.Entry(root)
E_环境温度.pack(side='top', padx=13, pady=3)
B_更改环境温度 = tk.Button(root, text='更改环境温度', command=lambda: 更改环境温度(list(map(int, (E_环境温度.get()).split(' ')))))
B_更改环境温度.pack(side='top', padx=13, pady=3)
L_迭代轮数 = tk.Label(root, text='迭代轮数：')
L_迭代轮数.pack(side='top', padx=13, pady=3)
E_迭代轮数 = tk.Entry(root)
E_迭代轮数.pack(side='top', padx=13, pady=3)
L_单轮次数 = tk.Label(root, text='单轮次数：')
L_单轮次数.pack(side='top', padx=13, pady=3)
E_单轮次数 = tk.Entry(root)
E_单轮次数.pack(side='top', padx=13, pady=3)
B_开始迭代 = tk.Button(root, text='开始迭代', command=lambda: 开始迭代(int(E_迭代轮数.get()), int(E_单轮次数.get())))
B_开始迭代.pack(side='top', padx=13, pady=3)
B_绘制均方末端距 = tk.Button(root, text='绘制均方末端距', command=绘制均方末端距)
B_绘制均方末端距.pack(side='top', padx=13, pady=3)
B_保存均方末端距 = tk.Button(root, text='保存均方末端距', command=保存均方末端距)
B_保存均方末端距.pack(side='top', padx=13, pady=3)
B_读取均方末端距 = tk.Button(root, text='读取均方末端距', command=读取均方末端距)
B_读取均方末端距.pack(side='top', padx=13, pady=3)
B_看看链的蠕动 = tk.Button(root, text='看看链的蠕动', command=看看链的蠕动)
B_看看链的蠕动.pack(side='top', padx=13, pady=3)
L_看看链的蠕动 = tk.Label(root, text='再点一下可以不看了')
L_看看链的蠕动.pack(side='top', padx=13, pady=0)
B_清除画板 = tk.Button(root, text='清除画板', command=清除画板)
B_清除画板.pack(side='top', padx=13, pady=3)
B_自动清除画板 = tk.Button(root, text='自动清除画板', command=自动清除画板)
B_自动清除画板.pack(side='top', padx=13, pady=3)

E_调试命令 = tk.Entry(root)
E_调试命令.pack(side='top', padx=13, pady=3)
B_调试命令 = tk.Button(root, text='调试命令', command=lambda: eval(E_调试命令.get()))
B_调试命令.pack(side='top', padx=13, pady=3)

B_当前温度体系 = tk.Button(root, text='当前温度体系', command=lambda: msg.showinfo(title='温度体系', message=aTs.call()))
B_当前温度体系.pack(side='top', padx=13, pady=3)
B_绘制当前体系分布 = tk.Button(root, text='绘制当前体系分布', command=绘制当前体系分布)
B_绘制当前体系分布.pack(side='top', padx=13, pady=3)

B_测试按钮1 = tk.Button(root, text='测试按钮1', command=测试按钮1)
B_测试按钮1.pack(side='top', padx=13, pady=3)
B_测试按钮2 = tk.Button(root, text='测试按钮2', command=测试按钮2)
B_测试按钮2.pack(side='top', padx=13, pady=3)
B_测试按钮3 = tk.Button(root, text='测试按钮3', command=测试按钮3)
B_测试按钮3.pack(side='top', padx=13, pady=3)
B_测试按钮4 = tk.Button(root, text='测试按钮4', command=测试按钮4)
B_测试按钮4.pack(side='top', padx=13, pady=3)
B_测试按钮5 = tk.Button(root, text='测试按钮5', command=测试按钮5)
B_测试按钮5.pack(side='top', padx=13, pady=3)
B_测试按钮6 = tk.Button(root, text='测试按钮6', command=测试按钮6)
B_测试按钮6.pack(side='top', padx=13, pady=3)

plt.show()

root.mainloop()
