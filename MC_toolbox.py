import matplotlib.pyplot as plt
import numpy as np
import random as rd

sq3 = np.sqrt(3)
plt.axis('equal')


class hex_coordinate:
    def __init__(self, coordinate_as_array):
        self.x = coordinate_as_array[0]
        self.y = coordinate_as_array[1]

    def Transform_2_Ortho(self):
        x = self.x - 0.5 * self.y
        y = self.y * sq3 / 2
        return [x, y]

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        ans = hex_coordinate([x, y])
        return ans


# def hex_coor_plot(hex_coor_arr):
#     plots = np.zeros((len(hex_coor_arr),2))
#     x = 0
#     for i in hex_coor_arr:
#         [plots[x,0],plots[x,1]] = i.Transform_2_Ortho()
#         x = x + 1
#     print(plots)
#     print(plots[:,0])
#     print(plots[:,1])
#     plt.plot(plots[:,0],plots[:,1],color = 'gray', linestyle = '-.')

class hex_box:
    def __init__(self, coordinate):
        self.location = coordinate

    def draw_box(self):
        [x, y] = self.location.Transform_2_Ortho()
        plt.plot([x - 0.5, x, x + 0.5, x + 0.5, x, x - 0.5, x - 0.5],
                 [y + sq3 / 6, y + sq3 / 3, y + sq3 / 6, y - sq3 / 6, y - sq3 / 3, y - sq3 / 6, y + sq3 / 6],
                 color='gray', linestyle='-.')
        plt.fill([x - 0.5, x, x + 0.5, x + 0.5, x, x - 0.5, x - 0.5],
                 [y + sq3 / 6, y + sq3 / 3, y + sq3 / 6, y - sq3 / 6, y - sq3 / 3, y - sq3 / 6, y + sq3 / 6],
                 color='gray', alpha = 0.3)


a = [hex_box(hex_coordinate([0, 0])), hex_box(hex_coordinate([0, 1])), hex_box(hex_coordinate([0, 2])),
     hex_box(hex_coordinate([1, 0])), hex_box(hex_coordinate([1, 1])), hex_box(hex_coordinate([1, 2])),
     hex_box(hex_coordinate([2, 0])), hex_box(hex_coordinate([2, 1])), hex_box(hex_coordinate([2, 2]))]
for i in a:
    i.draw_box()
plt.show()
