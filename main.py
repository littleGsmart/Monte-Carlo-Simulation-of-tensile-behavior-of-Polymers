import MC_toolbox as mc
import pickle
mc.T_environment = 0
a = mc.System([20, 20])

# with open("1681094190.6071975.pkl", 'rb') as file:
#     a = pickle.loads(file.read())

for i in range(5):
    print(a.line_generate(50))
print(len(min(a.lines,key=len)))

# for i in a.boxes.flatten():
#     i.draw_box()

# a.add_line([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
#             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])

# out_put = open("5lines_DP50.pkl", 'wb')
# tree_str = pickle.dumps(a)
# out_put.write(tree_str)
# out_put.close()

# print(a.calc_rd(0))
# print(len(a.lines[0]))
a.draw_Lines()
mc.plt.show()
cc = []
for i in range(len(a.lines)):
    cc.append([])
for j in range(50):
    # for i in a.boxes.flatten():
    #     i.draw_box()

    for i in range(10000):
        a.point_motive(a.rdpoint())

    for i in range(len(a.lines)):
        cc[i].append(a.calc_rd(i))

    print('\r{}'.format(j), end='')

    # a.draw_Lines()
    # plt.show()

print(cc)
a.draw_Lines()
mc.plt.axis('equal')
mc.plt.show()

# # if input() == '1':
# import time
# name = str(time.time()) + '.pkl'
# out_put = open(name, 'wb')
# saved_obj = pickle.dumps(a)
# out_put.write(saved_obj)
# out_put.close()


for i in range(len(cc)):
    mc.plt.plot(range(50), cc[i])
mc.plt.show()
