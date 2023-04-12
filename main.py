from MC_Tktoolbox import *

def 记录均方末端距():
    global record_Rd
    if record_Rd == 0:
        record_Rd = 1
        L_当前状态['text'] = '当前状态：记录均方末端距'
    else:
        record_Rd = 0
        L_当前状态['text'] = '当前状态：不记录均方末端距'

root = tk.Tk()
root.title("聚合物拉伸行为的格点法模拟")
root.geometry("900x1800")
B_载入已有体系= tk.Button(root, text='载入已有体系', command=拉取体系)
B_载入已有体系.pack(side='top', padx=13, pady=3)
B_绘制体系= tk.Button(root, text='绘制体系', command=绘制体系)
B_绘制体系.pack(side='top', padx=13, pady=3)
B_创建新体系 = tk.Button(root,text='创建新体系', command=创建新体系)
B_创建新体系.pack(side='top', padx=13, pady=3)
B_保存体系 = tk.Button(root,text='保存体系', command=保存体系)
B_保存体系.pack(side='top', padx=13, pady=3)
B_记录均方末端距 = tk.Button(root,text='记录均方末端距', command=记录均方末端距)
B_记录均方末端距.pack(side='top', padx=13, pady=3)
L_当前状态 = tk.Label(root,text = '当前状态：记录均方末端距')
L_当前状态.pack(side='top', padx=13, pady=3)
L_环境温度 = tk.Label(root,text = '环境温度：')
L_环境温度.pack(side='top', padx=13, pady=3)
E_环境温度 = tk.Entry(root)
E_环境温度.pack(side='top', padx=13, pady=3)
B_更改环境温度 = tk.Button(root,text='更改环境温度', command=lambda :更改环境温度(list(map(int,(E_环境温度.get()).split(' ')))))
B_更改环境温度.pack(side='top', padx=13, pady=3)
L_迭代轮数 = tk.Label(root,text = '迭代轮数：')
L_迭代轮数.pack(side='top', padx=13, pady=3)
E_迭代轮数 = tk.Entry(root)
E_迭代轮数.pack(side='top', padx=13, pady=3)
L_单轮次数 = tk.Label(root,text = '单轮次数：')
L_单轮次数.pack(side='top', padx=13, pady=3)
E_单轮次数 = tk.Entry(root)
E_单轮次数.pack(side='top', padx=13, pady=3)
B_开始迭代 = tk.Button(root,text='开始迭代', command=lambda :开始迭代(int(E_迭代轮数.get()),int(E_单轮次数.get())))
B_开始迭代.pack(side='top', padx=13, pady=3)
B_绘制均方末端距 = tk.Button(root,text='绘制均方末端距', command=绘制均方末端距)
B_绘制均方末端距.pack(side='top', padx=13, pady=3)
B_保存均方末端距 = tk.Button(root,text='保存均方末端距', command=保存均方末端距)
B_保存均方末端距.pack(side='top', padx=13, pady=3)
B_读取均方末端距 = tk.Button(root,text='读取均方末端距', command=读取均方末端距)
B_读取均方末端距.pack(side='top', padx=13, pady=3)
B_看看链的蠕动 = tk.Button(root,text='看看链的蠕动', command=看看链的蠕动)
B_看看链的蠕动.pack(side='top', padx=13, pady=3)
L_看看链的蠕动 = tk.Label(root,text = '再点一下可以不看了')
L_看看链的蠕动.pack(side='top', padx=13, pady=0)
B_清除画板 = tk.Button(root,text='清除画板', command=清除画板)
B_清除画板.pack(side='top', padx=13, pady=3)
B_自动清除画板 = tk.Button(root,text='自动清除画板', command=自动清除画板)
B_自动清除画板.pack(side='top', padx=13, pady=3)

E_调试命令 = tk.Entry(root)
E_调试命令.pack(side='top', padx=13, pady=3)
B_调试命令 = tk.Button(root,text='调试命令', command=lambda :eval(E_调试命令.get()))
B_调试命令.pack(side='top', padx=13, pady=3)

B_当前温度体系 = tk.Button(root,text='当前温度体系', command=lambda :msg.showinfo(title='温度体系',message=aTs.call()))
B_当前温度体系.pack(side='top', padx=13, pady=3)
B_绘制当前体系分布 = tk.Button(root,text='绘制当前体系分布', command=绘制当前体系分布)
B_绘制当前体系分布.pack(side='top', padx=13, pady=3)

plt.show()

root.mainloop()

