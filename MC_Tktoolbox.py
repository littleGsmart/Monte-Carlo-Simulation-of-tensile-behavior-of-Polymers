import tkinter.messagebox as msg
from MC_toolbox import *
import tkinter as tk
import tkinter.filedialog
import pickle
import tkinter.simpledialog
import time
from multiprocessing import Pool
import threading

Sys = System([1, 1])
record_Rd = 1
run_or_not = 1


def 拉取体系():
    global Sys
    Sys_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with open(Sys_root, 'rb') as file:
        Sys = pickle.loads(file.read())


def 绘制体系():
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


def 还不太会多线程(DP):
    Sys.line_generate(int(DP))


def 创建新体系_3(x, y, line_num, DP):
    if msg.askyesno(title='是否确定生成模型', message='请问是否确定生成模型？'):
        global Sys

        Sys = System([int(x), int(y)])
        if line_num == 0:
            return 0
        for i in tqdm(range(int(line_num))):
            还不太会多线程(DP)
            # th = threading.Thread(target=还不太会多线程, args=(DP,))
            # th.setDaemon(True)  # 守护线程
            # th.start()
        msg.showinfo(title='生成已完成', message='模型已生成完成')


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


def 开始迭代(轮数, 单轮次数):
    global record, run_or_not
    run_or_not = 1
    record = []
    for i in range(len(Sys.lines)):
        record.append([])

    print(result)
    for j in tqdm(range(轮数)):

        for i in range(单轮次数):
            Sys.point_motive(Sys.rdpoint())

        if record_Rd:
            for i in range(len(Sys.lines)):
                record[i].append(Sys.calc_rd(i))

        # if not run_or_not:
        #     print('函数已停止运行')
        #     return -1

        # print('\r{}'.format(j), end='')


def 绘制均方末端距():
    for i in tqdm(record):
        plt.plot(range(len(i)), i)
    plt.axis('auto')
    plt.show()


def 保存均方末端距():
    rd_root = tkinter.filedialog.asksaveasfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    if '.pkl' in rd_root:
        pass
    else:
        rd_root += '.pkl'
    out_put = open(rd_root, 'wb')
    saved_obj = pickle.dumps(record)
    out_put.write(saved_obj)
    out_put.close()


def 读取均方末端距():
    global record
    rd_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with open(rd_root, 'rb') as file:
        record = pickle.loads(file.read())

#
# def 停止迭代():
#     global run_or_not
#     run_or_not = 0
