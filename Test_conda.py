import math
import json
from bisect import bisect_left as bl
import random as rd
import time




# dict_6000_10 = {}
# list0 = [[],[]]
# for i in range(600000):
#     x = i / 60000
#     y = p(x)+1
#     list0[0].append(x)
#     list0[1].append(y)
# list0[0].append(1e3)
# with open("dict_600000_10.json", "w", encoding='utf-8') as f:
#     json.dump(list0, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
# #
with open("dict_600000_10.json", encoding="utf-8") as f:
    json_file = json.load(f)

a = time.time()
x = []
y = []
zz = {}
for i in range(100000):
    x.append(i)
    t = json_file[0][bl(json_file[1], rd.random())]
    if t ==1000: t = 10
    if t in zz.keys():
        zz[t] = zz[t]+1
    else:
        zz.update({t:1})
import matplotlib.pyplot as plt

plt.plot(zz.keys(),zz.values())
plt.show()
print(time.time() - a)
