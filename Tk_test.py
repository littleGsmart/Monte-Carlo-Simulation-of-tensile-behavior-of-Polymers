from multiprocessing import Pool
from tqdm import tqdm
import time
import math
class a:
    def __init__(self):
        pass

    def myf(self,num):
        result = 0
        for i in range(1000000):
            result += i
        return result
c = a()

def cmyf(i):
    c.myf(i)

if __name__ == '__main__':
    value_x= range(2000)
    P = Pool(processes=1)

    # 这里计算很快
    res = [P.apply_async(func=cmyf, args=(i, )) for i in value_x]

    # 主要是看这里
    result = [i.get(timeout=2) for i in tqdm(res)]

    print(result)