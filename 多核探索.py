from multiprocessing import Process
from MC_toolbox import *

class MyProcess(Process):
    # 重写run()方法

    def run(self):
        n = 1
        while n < 60000:
            Sys.point_motive(Sys.rdpoint())
            print('进程名:{} n的值：{}'.format(self.name, n))
            n += 1


if __name__ == '__main__':
    p1 = MyProcess(name='李诺')
    p2 = MyProcess(name='夜寒')

    with open("500_500_500lines_maxDP500_cystalmethod1_31B.pkl", 'rb') as file:
        Sys = pickle.loads(file.read())
    for i in range(10000):
        Sys.point_motive(Sys.rdpoint())
    print('ok')
    p1.start()
