import tkinter.messagebox as msg
from MC_toolbox import *
import tkinter as tk
import tkinter.filedialog
import pickle
import tkinter.simpledialog


def 拉取体系():
    global Sys
    Sys_root = tkinter.filedialog.askopenfilename(title="请选择已有模型", filetypes=[('pkl文件', '.pkl')])
    with open(Sys_root, 'rb') as file:
        Sys = pickle.loads(file.read())


def 绘制体系():
    try:
        draw_box_or_not = msg.askyesno(title='是否需要绘制边框', message='请问是否需要绘制六边形框？')
        if draw_box_or_not:
            for i in Sys.boxes.flatten():
                i.draw_box()
        Sys.draw_Lines()
        plt.axis('equal')
        plt.show()
    except:
        msg.showerror(title='错误', message='绘制错误，请检查体系是否加载正确')


def 创建新模型():
    global para_arr
    para_arr = []
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
    B_1 = tk.Button(top_1, text='下一步', command=lambda :创建新模型_2(E_1.get(),E_2.get()))
    B_1.pack(side="top", padx=13, pady=3)



def 创建新模型_2(x,y):
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
        B_1 = tk.Button(top_1, text='下一步', command=lambda :创建新模型_3(x,y,E_1.get(),E_2.get()))
        B_1.pack(side="top", padx=13, pady=3)
    else:创建新模型_3(x,y,0,0)


def 创建新模型_3(x,y,line_num,DP):
    if msg.askyesno(title='是否确定生成模型',message='请问是否确定生成模型？'):
        pass

