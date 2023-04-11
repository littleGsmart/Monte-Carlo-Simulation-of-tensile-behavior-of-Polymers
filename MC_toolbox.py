import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left as bl
import random as rd
import json
import math
import pickle
from tqdm import tqdm
from multiprocessing import Pool

sq3 = np.sqrt(3)
plt.axis('equal')
with open("dict_600000_10.json", encoding="utf-8") as f:
    e_dict = json.load(f)
Tg = 263
Tm = 433
Td = 583
T_environment = 400
n_g_arr = [0, 1, 1e8, 1e8, 1e8, 1e8, 1e8]


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
        for i in tqdm(range(size_arr[0])):
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
        begin_location = self.random_locate()
        line = []
        while not line:
            if len(self.boxes[begin_location.x, begin_location.y].content) <= 2:
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
            for j in line_num:
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

    def rdpoint(self):
        i = int(rd.random() * len(self.lines))
        print(i)
        print(self.lines)
        j = int(rd.random() * len(self.lines[i]))
        return [i, j]

    def point_motive(self, point_num):
        origin_location = self.lines[point_num[0]][point_num[1]].location
        directions = origin_location.around(self.size)
        target_location = directions[int(rd.random() * len(directions))]
        rd_energy = e_dict[0][bl(e_dict[1], rd.random())] * T_environment

        energy_g = n_g_arr[len(self.boxes[target_location.x][target_location.y].content)] * Tg
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
                energy_m = Tm
        elif point_num[1] == (len(self.lines[point_num[0]]) - 1):
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            direction_01 = (origin_location - former_location).tolist()
            if direction_01 in direction_1:
                energy_m = Tm

        else:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            direction_00 = (origin_location - latter_location).tolist()
            direction_01 = (origin_location - former_location).tolist()
            if (direction_00 in direction_1) or (direction_01 in direction_1):
                energy_m = Tm

        # energy_d
        if point_num[1] == 0:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            d_distance = 0.5 * (pow(target_location.distance(latter_location), 2) - pow(
                origin_location.distance(latter_location), 2))
            energy_d = d_distance * Td
        elif point_num[1] == (len(self.lines[point_num[0]]) - 1):
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            d_distance = 0.5 * (pow(target_location.distance(former_location), 2) - pow(
                origin_location.distance(former_location), 2))
            energy_d = d_distance * Td
        else:
            latter_location = self.lines[point_num[0]][point_num[1] + 1].location
            former_location = self.lines[point_num[0]][point_num[1] - 1].location
            d_distance = 0.5 * (pow(target_location.distance(latter_location), 2) - pow(origin_location.distance(
                latter_location), 2) + pow(target_location.distance(former_location), 2) - pow(
                origin_location.distance(former_location), 2))
            energy_d = d_distance * Td

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

Sys = System([500, 500])

def cmyf(ii,aSys):

    aSys.point_motive(aSys.rdpoint())

if __name__ == '__main__':
    Sys = System([500, 500])

    with open("500_500_500lines_maxDP500_cystalmethod1_31B.pkl", 'rb') as file:
        Sys = pickle.loads(file.read())

    # for i in range(500):
    #     print(a.line_generate(500))
    print(len(min(Sys.lines, key=len)))

    # for i in a.boxes.flatten():
    #     i.draw_box()

    # a.add_line([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
    #             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])

    # out_put = open("5lines_DP50.pkl", 'wb')
    # tree_str = pickle.dumps(a)
    # out_put.write(tree_str)
    # out_put.close()

    print(Sys.calc_rd(0))
    print(len(Sys.lines[0]))
    Sys.draw_Lines()
    plt.show()
    cc = []





    P = Pool(processes=1)

    for i in range(len(Sys.lines)):
        cc.append([])
    for j in range(3000):
        # for i in a.boxes.flatten():
        #     i.draw_box()

        # 这里计算很快
        res = [P.apply_async(func=cmyf, args=(ii,)) for ii in range(60000)]

        # 主要是看这里
        result = [ii.get(timeout=200) for ii in tqdm(res)]

        for i in range(len(Sys.lines)):
            cc[i].append(Sys.calc_rd(i))

        print('\r{}'.format(j), end='')

        # a.draw_Lines()
        # plt.show()

    print(cc)
    Sys.draw_Lines()
    plt.axis('equal')
    plt.show()

    # if input() == '1':
    import time

    name = str(time.time()) + '.pkl'
    out_put = open(name, 'wb')
    saved_obj = pickle.dumps(Sys)
    out_put.write(saved_obj)
    out_put.close()

    for i in range(len(cc)):
        plt.plot(range(3000), cc[i])
    plt.show()
