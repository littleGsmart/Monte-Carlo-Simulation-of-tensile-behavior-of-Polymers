from MC_toolbox import *


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


import threading

Sys = System([1, 1])
record_Rd = 1
rudong = 0
diedai_continue = 0
plt.ion()
auto_clean = 0
CDF = 1

anArea = [[0,10],[350,10],[350,0],[450,0],[450,50],[350,50],[350,40],[0,40]]
GenArea = [[0,10],[100,10],[100,40],[0,40]]


def 拉取体系():
    global Sys
    Sys_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with rich.progress.open(Sys_root, 'rb') as file:
        Sys = pickle.loads(file.read())
    msg.showinfo(title='读取已完成', message='已读取完成')


def 绘制体系():
    def tohex(list):
        return hex_coordinate(list)
    area = anArea
    hex_arr = []
    for point in area:
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
            Sys.line_generate_DP(int(DP),GenArea)
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
        global record, T_environment
        record = []
        for i in range(len(Sys.lines)):
            record.append([])
        for j in trange(轮数):

            for i in range(单轮次数):
                Sys.point_motive(Sys.rdpoint(),anArea)

            if record_Rd:
                for i in range(len(Sys.lines)):
                    record[i].append(Sys.calc_rd(i))

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
    for i in record:
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


def 看看链的蠕动():
    def aRun(单轮次数=1000):
        global record
        record = []
        for i in range(len(Sys.lines)):
            record.append([])
        niganma = 0
        while rudong:
            t1 = time.time()
            niganma += 1
            for isss in trange(单轮次数):
                print('\r第{}线程'.format(str(niganma)), end='')
                Sys.point_motive(Sys.rdpoint(),anArea)
            if auto_clean:
                plt.clf()
            Sys.draw_Lines()
            mypause(0.005)
            t2 = time.time()
            if record_Rd:
                for i in range(len(Sys.lines)):
                    record[i].append(Sys.calc_rd(i))
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
    len_map = [0]* longest
    for i in Sys.lines:
        len_map[len(i)-1] += 1
    if CDF:
        for i in range(1,longest):
            len_map[i] += len_map[i-1]
    plt.axis('auto')
    plt.plot(range(1,longest+1),len_map)
